#include <Python.h>
#include "blocksparse_mlp.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../activation.cuh"
#include "../add.cuh"

std::tuple<at::Tensor, at::Tensor> blocksparse_mlp_routing(
    int bsz,
    const py::object& cfg,
    const at::Tensor& y,
    const py::dict& params
)
{
    bool activate_all = false;
    if (params.contains("activate_all_experts"))
        activate_all = params["activate_all_experts"].cast<bool>();

    at::Tensor gate_tensor = cfg.attr("gate_tensor").cast<at::Tensor>();
    int64_t num_experts = cfg.attr("num_experts").cast<int64_t>();
    int64_t num_exp_per_tok = cfg.attr("num_experts_per_tok").cast<int64_t>();

    if (!activate_all && bsz == 1)
    {
        at::Tensor router_logits_bsz1 = cfg.attr("router_logits_bsz1").cast<at::Tensor>();
        at::Tensor routing_weights_bsz1 = cfg.attr("routing_weights_bsz1").cast<at::Tensor>();
        at::Tensor selected_experts_bsz1 = cfg.attr("selected_experts_bsz1").cast<at::Tensor>();

        at::matmul_out(router_logits_bsz1, y, gate_tensor);
        at::topk_out
        (
            routing_weights_bsz1,
            selected_experts_bsz1,
            router_logits_bsz1,
            num_exp_per_tok,
            -1,
            true,
            false
        );

        at::softmax_out(routing_weights_bsz1, routing_weights_bsz1, -1);
        return {selected_experts_bsz1, routing_weights_bsz1};
    }
    else
    {
        int64_t k = activate_all ? num_experts : num_exp_per_tok;

        at::Tensor router_logits = at::matmul(y, gate_tensor);

        auto topk_result = at::topk(router_logits, k, -1);
        at::Tensor routing_weights = std::get<0>(topk_result);
        at::Tensor selected_experts = std::get<1>(topk_result);

        routing_weights = at::softmax(routing_weights, -1);

        return {selected_experts, routing_weights};
    }
}

void BC_BlockSparseMLP::run_bsz1_gr
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights,
    Graph* graph
)
{
    py::gil_scoped_release _;
    const at::Tensor& yi = y.unsqueeze(0);

    exl3_mgemm_gr
    (
        yi,
        gate_ptrs_trellis,
        interm_g,
        gate_ptrs_suh,
        yh,
        gate_ptrs_svh,
        selected_experts,
        {},
        gate_K,
        -1,
        gate_mcg,
        gate_mul1,
        min_expert,
        max_expert,
        0,
        {},
        graph
    );

    exl3_mgemm_gr
    (
        yi,
        up_ptrs_trellis,
        interm_u,
        up_ptrs_suh,
        yh,
        up_ptrs_svh,
        selected_experts,
        {},
        up_K,
        -1,
        up_mcg,
        up_mul1,
        min_expert,
        max_expert,
        0,
        {},
        graph
    );

    if (act_silu)
        silu_mul_gr(interm_g, interm_u, interm_a, graph);
    else if (act_gelu)
        gelu_mul_gr(interm_g, interm_u, interm_a, graph);

    exl3_mgemm_gr
    (
        interm_a,
        down_ptrs_trellis,
        out_d,
        down_ptrs_suh,
        interm_a,
        down_ptrs_svh,
        selected_experts,
        routing_weights,
        down_K,
        -1,
        down_mcg,
        down_mul1,
        min_expert,
        max_expert,
        0,
        {},
        graph
    );

    if (shared_experts)
    {
        shared_experts->run_bsz1_gr(yi, out_d_sh.value(), graph);
        if (shared_gate)
        {
            add_sigmoid_gate_proj_gr(out_d_sh.value(), yi, out_d, shared_gate->weight, graph);
        }
        else
        {
            add_gr(out_d, out_d_sh.value(), out_d, graph);
        }
    }
}

void BC_BlockSparseMLP::run_bsz1
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    c10::cuda::CUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    #define USE_GRAPH
    #ifndef USE_GRAPH

        run_bsz1_gr(y, selected_experts, routing_weights, nullptr);

    #else

        if (!graph_bsz1.ready)
        {
            graph_bsz1.capture_begin();
            run_bsz1_gr(y, selected_experts, routing_weights, &graph_bsz1);
            graph_bsz1.capture_end();
        }

        auto args = std::vector<PPTR>
        {
            PPTR(GP_mgemm_A,            (void*) y.data_ptr()),
            PPTR(GP_mgemm_indices,      (void*) selected_experts.data_ptr()),
            PPTR(GP_end,                nullptr),
            PPTR(GP_mgemm_A,            (void*) y.data_ptr()),
            PPTR(GP_mgemm_indices,      (void*) selected_experts.data_ptr()),
            PPTR(GP_end,                nullptr),
        };

        if (shared_experts && shared_gate)
        {
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                         nullptr));
            args.push_back(PPTR(GP_mgemm_A,                     (void*) y.data_ptr()));
            args.push_back(PPTR(GP_add_sigmoid_gate_proj_y,     (void*) y.data_ptr()));
            args.push_back(PPTR(GP_add_sigmoid_gate_proj_z,     (void*) out_d.data_ptr()));
        }
        else if (shared_experts)
        {
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                         nullptr));
            args.push_back(PPTR(GP_mgemm_A,                     (void*) y.data_ptr()));
            args.push_back(PPTR(GP_add_x,                       (void*) out_d.data_ptr()));
            args.push_back(PPTR(GP_add_z,                       (void*) out_d.data_ptr()));
        }
        else
        {
            args.push_back(PPTR(GP_mgemm_C,                     (void*) out_d.data_ptr()));
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
        }

        graph_bsz1.launch(args, stream);

    #endif
}

void BC_BlockSparseMLP::run_bszN_gr
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights,
    int bsz,
    Graph* graph
)
{
    // Process each token in the batch sequentially using the optimized mgemm kernels
    // This moves the Python loop into C++ and enables CUDA graph capture
    for (int i = 0; i < bsz; ++i)
    {
        // Slice input for this token: y is [bsz, hidden], we need [1, 1, hidden] for mgemm
        at::Tensor yi = y.slice(0, i, i + 1).unsqueeze(0);
        at::Tensor idx_i = selected_experts.slice(0, i, i + 1);
        at::Tensor w_i = routing_weights.slice(0, i, i + 1);

        // Gate projection
        exl3_mgemm_gr
        (
            yi,
            gate_ptrs_trellis,
            interm_g,
            gate_ptrs_suh,
            yh,
            gate_ptrs_svh,
            idx_i,
            {},
            gate_K,
            -1,
            gate_mcg,
            gate_mul1,
            min_expert,
            max_expert,
            0,
            {},
            graph
        );

        // Up projection
        exl3_mgemm_gr
        (
            yi,
            up_ptrs_trellis,
            interm_u,
            up_ptrs_suh,
            yh,
            up_ptrs_svh,
            idx_i,
            {},
            up_K,
            -1,
            up_mcg,
            up_mul1,
            min_expert,
            max_expert,
            0,
            {},
            graph
        );

        // Activation function
        if (act_silu)
            silu_mul_gr(interm_g, interm_u, interm_a, graph);
        else if (act_gelu)
            gelu_mul_gr(interm_g, interm_u, interm_a, graph);

        // Down projection with expert weights reduction:
        // - Use out_d as scratch for per-expert outputs
        // - Write the final reduced sum directly into out_final[i] (C_red) to avoid a copy
        at::Tensor out_i = out_final.slice(0, i, i + 1).unsqueeze(0);
        exl3_mgemm_gr
        (
            interm_a,
            down_ptrs_trellis,
            out_d,
            down_ptrs_suh,
            interm_a,
            down_ptrs_svh,
            idx_i,
            w_i,
            down_K,
            -1,
            down_mcg,
            down_mul1,
            min_expert,
            max_expert,
            0,
            out_i,
            graph
        );

        // Handle shared experts if present (before copy, so result accumulates in out_d)
        if (shared_experts)
        {
            TORCH_CHECK(out_d_sh.has_value(), "shared_experts set but out_d_sh buffer is missing");
            shared_experts->run_bsz1_gr(yi, out_d_sh.value(), graph);
            if (shared_gate)
            {
                add_sigmoid_gate_proj_gr(out_d_sh.value(), yi, out_i, shared_gate->weight, graph);
            }
            else
            {
                add_gr(out_i, out_d_sh.value(), out_i, graph);
            }
        }
    }
}

at::Tensor BC_BlockSparseMLP::run_bszN
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    c10::cuda::CUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    py::gil_scoped_release _;

    int bsz = y.size(0);
    int hidden = y.size(1);

    // Store hidden size for buffer operations
    if (hidden_size == 0) hidden_size = hidden;

    // Ensure output buffer is large enough
    if (out_final_max_bsz < bsz)
    {
        // Allocate generously to minimize reallocations
        // Round up to power of 2 for better graph cache reuse
        int new_max = 8;
        while (new_max < bsz) new_max *= 2;
        out_final_max_bsz = new_max;
        out_final = torch::empty(
            {out_final_max_bsz, hidden},
            torch::TensorOptions().dtype(out_d.dtype()).device(y.device())
        );
        // MUST invalidate all captured graphs: they contain the old out_final pointer
        graphs_bszN.clear();
    }

    #define USE_GRAPH_BSZN
    #ifndef USE_GRAPH_BSZN

        // Direct C++ execution without graph (still much faster than Python loop)
        run_bszN_gr(y, selected_experts, routing_weights, bsz, nullptr);

    #else

        // Get or create graph for this batch size
        Graph& g = graphs_bszN[bsz];

        if (!g.ready)
        {
            g.capture_begin();
            run_bszN_gr(y, selected_experts, routing_weights, bsz, &g);
            g.capture_end();
        }

        // Build parameter update list for the graph
        // For each token in batch, we have: gate mgemm, up mgemm, activation, down mgemm, copy
        // The mgemm calls record: GP_mgemm_A, GP_mgemm_C, GP_mgemm_indices, GP_mgemm_weights, GP_end
        
        // Calculate required size and reserve capacity to avoid reallocations
        // Per token:
        //   gate:   3 (A, indices, end)
        //   up:     3 (A, indices, end)
        //   down:   3 (indices, weights, end)  // outputs reduced sum directly to out_final row (via C_red)
        //   shared: 4 (A, add_y, add_z, end) or (A, add_x, add_z, end)
        int args_per_token = 9 + (shared_experts ? 4 : 0);
        int required_size = bsz * args_per_token;
        if (graph_args.capacity() < required_size)
            graph_args.reserve(required_size * 2);  // Reserve 2x to reduce future reallocations
        graph_args.clear();

        for (int i = 0; i < bsz; ++i)
        {
            // Get pointers for this token's data
            void* y_ptr = (void*)((half*)y.data_ptr() + i * hidden);
            void* idx_ptr = (void*)((int64_t*)selected_experts.data_ptr() + i * selected_experts.size(1));
            void* w_ptr = (void*)((half*)routing_weights.data_ptr() + i * routing_weights.size(1));
            void* out_ptr = (void*) ((char*) out_final.data_ptr() + (int64_t) i * (int64_t) hidden * (int64_t) out_final.element_size());

            // Gate mgemm params
            graph_args.push_back(PPTR(GP_mgemm_A, y_ptr));
            graph_args.push_back(PPTR(GP_mgemm_indices, idx_ptr));
            graph_args.push_back(PPTR(GP_end, nullptr));

            // Up mgemm params
            graph_args.push_back(PPTR(GP_mgemm_A, y_ptr));
            graph_args.push_back(PPTR(GP_mgemm_indices, idx_ptr));
            graph_args.push_back(PPTR(GP_end, nullptr));

            // Activation doesn't need param updates (uses fixed intermediate buffers)

            // Down mgemm params (includes weights for reduction)
            graph_args.push_back(PPTR(GP_mgemm_indices, idx_ptr));
            graph_args.push_back(PPTR(GP_mgemm_weights, w_ptr));
            graph_args.push_back(PPTR(GP_end, nullptr));

            // Handle shared experts if present (comes before copy in run_bszN_gr)
            if (shared_experts)
            {
                if (shared_gate)
                {
                    // shared_experts mgemm + add_sigmoid_gate_proj
                    graph_args.push_back(PPTR(GP_mgemm_A, y_ptr));
                    graph_args.push_back(PPTR(GP_add_sigmoid_gate_proj_y, y_ptr));
                    graph_args.push_back(PPTR(GP_add_sigmoid_gate_proj_z, out_ptr));
                    graph_args.push_back(PPTR(GP_end, nullptr));
                }
                else
                {
                    // shared_experts mgemm + add
                    graph_args.push_back(PPTR(GP_mgemm_A, y_ptr));
                    graph_args.push_back(PPTR(GP_add_x, out_ptr));
                    graph_args.push_back(PPTR(GP_add_z, out_ptr));
                    graph_args.push_back(PPTR(GP_end, nullptr));
                }
            }

            // No copy needed: down mgemm writes reduced output directly into out_final
        }

        g.launch(graph_args, stream);

    #endif

    return out_final.slice(0, 0, bsz);
}
