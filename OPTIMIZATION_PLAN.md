# ExLlamaV3 Block Sparse MLP Optimization Plan

## Current State (run_bszN with CUDA Graphs)

**Benchmark results:** 12% throughput improvement over baseline in mixed prefill/decode scenarios.

### Current Flow
```
For each token i in batch [0..bsz):
    1. gate_mgemm(yi) → interm_g        [num_experts_per_tok matmuls]
    2. up_mgemm(yi) → interm_u          [num_experts_per_tok matmuls]  
    3. activation(interm_g, interm_u) → interm_a
    4. down_mgemm(interm_a, weights) → out_d  [reduction across experts]
    5. copy(out_d[0] → out_final[i])
    (6. shared_experts if present)
```

**Total kernels per MoE layer:** `bsz × (4 + 1)` = 5 kernels per token (graph-captured)

### Bottlenecks Identified

1. **Sequential Token Processing**: Even with CUDA graphs, tokens processed one-by-one
2. **Per-Token Copy**: `copy_row_gr` called for every token
3. **Redundant Input Reads**: Gate and Up projections read same input
4. **Graph Cache Fragmentation**: Separate graph per batch size

---

## Proposed Optimizations (Ordered by Effort/Impact)

### Level 1: Quick Wins (Low Effort)

#### 1.1 Power-of-2 Graph Caching ✅ (Already applied)
```cpp
// Round up batch size to power of 2 for better cache reuse
int cached_bsz = 1;
while (cached_bsz < bsz) cached_bsz *= 2;
```
**Impact:** Reduces graph recompilation, especially in variable-batch scenarios.

#### 1.2 Prefetch Expert Weights
Add `__prefetch_global_l2` for next token's expert weights during current computation.

**Impact:** ~2-5% improvement from better memory access patterns.

---

### Level 2: Moderate Effort (~15-25% improvement)

#### 2.1 Fused Gate+Up Kernel ✅ (Implemented)
Created `exl3_fused_gate_up_mgemm_gr` that processes both gate and up projections
in a single kernel launch, reducing kernel launch overhead and improving cache locality.

**Implementation Details:**
- New kernel template `exl3_fused_mgemm_kernel` in `exl3_gemm_kernel.cuh`
- New C++ wrapper `exl3_fused_gate_up_mgemm_gr` in `exl3_gemm.cu`
- Added `EXL3_FUSED_MGEMM_ARGS` macro in `exl3_kernel_map.cuh`
- Updated `blocksparse_mlp.cpp` to use fused kernel with fallback for incompatible configs
- CUDA graph patching updated for both `run_bsz1` and `run_bszN` paths

**Current Status:**
- Gate and Up projections fused into single kernel when `gate_K == up_K` and flags match
- Kernel launch count reduced from 2 to 1 for gate+up phase
- **Hadamard Reuse:** When gate and up experts share the same `suh` pointer, the kernel
  computes the input Hadamard once and reuses it for both projections (runtime fast-path)
- Python-side aliasing in `block_sparse_mlp.py` deduplicates identical `suh` tensors so
  the pointer-equality check succeeds when gate/up were quantized from the same Hessian

**Expected Impact:** ~15-20% from reduced kernel launches + Hadamard reuse.

#### 2.2 Direct Output Write ✅ (Already Done)
Eliminate `copy_row_gr` by writing down projection directly to `out_final`:

```cpp
// Modify down mgemm to use strided output
// out_final_slice = out_final.slice(0, i, i+1).unsqueeze(0)
exl3_mgemm_gr(interm_a, down_ptrs, out_final_slice, ..., graph);
```

**Challenge:** Need to handle the view/slice without breaking graph capture.

**Impact:** ~5-10% from eliminated memory copies.

---

### Level 3: Higher Effort (~30-50% improvement)

#### 3.1 True Batched Token Processing
Resize intermediate buffers to hold all tokens × experts:

```cpp
// New buffer shapes (preallocated for max_bsz)
interm_g_big: [num_experts_per_tok * max_bsz, 1, intermediate_size]
interm_u_big: [num_experts_per_tok * max_bsz, 1, intermediate_size]

// Flatten indices for all tokens
indices_flat: [1, num_experts_per_tok * bsz]  // [tok0_exp0, tok0_exp1, tok1_exp0, ...]

// Single mgemm call for entire batch
exl3_mgemm_gr(y_all, gate_ptrs, interm_g_big, ..., indices_flat, ...);
```

**Implementation:**
1. Add larger intermediate buffer allocation in `load_local()`
2. Create index flattening utility
3. Modify graph recording to handle batch-all-at-once pattern
4. Handle expert reduction (down proj) with proper indexing

**Impact:** ~30-40% from drastically reduced kernel launches.

#### 3.2 Expert-Aware Batching
Sort tokens by expert, batch tokens sharing same expert:

```python
# Conceptual algorithm:
# 1. Collect all (token_idx, expert_idx) pairs from selected_experts
# 2. Group by expert_idx
# 3. For each unique expert with N tokens:
#    - Run larger matmul: [N, 1, hidden] @ expert_weights
# 4. Scatter results back to token positions
```

**Pros:** 
- Maximizes GPU utilization by running larger matmuls
- Best for high-concurrency scenarios

**Cons:**
- Complex index management
- May not benefit small batches (overhead > savings)

**Impact:** ~40-50% for batches of 4+ tokens with expert overlap.

---

### Level 4: Advanced Optimizations

#### 4.1 Persistent Kernel Approach
Keep a persistent kernel running that processes tokens from a queue:
- Eliminates all kernel launch overhead
- Complex synchronization required

#### 4.2 Expert Weight Caching in L2
Pin frequently-used expert weights in L2 cache using CUDA memory pools.

#### 4.3 Tensor Core Utilization Optimization
Ensure matmul shapes align with tensor core requirements (multiples of 16/32).

---

## Recommended Implementation Order

1. **Level 1.1** ✅ Power-of-2 graph caching (done)
2. **Level 2.2** ✅ Direct output write (done)
3. **Level 2.1** ✅ Fused Gate+Up kernel (done)
4. **Level 3.1** True batched processing (major rewrite - not recommended for concurrency≤4)

## Benchmarking Strategy

For each optimization, measure:
- Continuous load test (mixed prefill/decode) - primary metric
- Batch mode tests at concurrency 1, 2, 4, 8
- Memory usage delta
- Graph recompilation frequency

---

## Files to Modify

| File | Changes |
|------|---------|
| `blocksparse_mlp.cpp` | Buffer allocation, graph logic |
| `blocksparse_mlp.h` | New buffer members |
| `exl3_gemm.cu` | Fused gate+up kernel |
| `exl3_gemm.cuh` | New function declarations |
| `exl3_gemm_kernel.cuh` | Dual-output kernel variant |
| `block_sparse_mlp.py` | Buffer sizing, Python interface |

