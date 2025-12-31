#pragma once

#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                nullptr,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
        (A_, B, C_, size_m_, size_k, size_n, locks);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh)
            grid.sync();
    }

    if (svh)
    {
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    nullptr,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    nullptr,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
}

#define MAX_INDICES 128

__device__ int64_t v_indices[128];
__device__ half v_weights[128];
__device__ int bszm_sync;

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_mgemm_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg::this_grid();

    // Pack indices within min_index <= idx < max_index

    if (min_index >= 0)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            int j = 0;
            for (int i = 0; i < bszm; ++i)
            {
                int idx = B_indices[i];
                if (idx >= min_index && idx < max_index)
                {
                    v_indices[j] = idx - min_index;
                    if (B_weights) v_weights[j] = B_weights[i];
                    j++;
                }
            }
            bszm_sync = j;
            for (; j < bszm; ++j)
            {
                v_indices[j] = -1;
            }
        }
        __threadfence();
        grid.sync();
        B_indices = v_indices;
        if (B_weights) B_weights = v_weights;
        bszm = bszm_sync;
    }

    for (int i = 0; i < bszm; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int mat_index = -1;
        const uint16_t* B = nullptr;
        if (j >= bszm) j = -1;
        else
        {
            mat_index = B_indices ? (int) B_indices[j] : j;
            if (mat_index >= 0)
            {
                B = B_list[mat_index];
            }
        }

        // Had and input scales

        if (B)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh = suh_list[mat_index];
            const half* A_ = bszm_in == 1 ? A : A + j * size_m * size_k;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh + (this_warp * 128) % size_k,
                    nullptr,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
        grid.sync();

        // Matmul

        int size_m_ = size_m;
        half* A_ = A_had + j * size_m * size_k;
        void* C_;
        if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
        else                  C_ = (void*) (((half*) C) + j * size_m * size_n);

        while (size_m_ > 0)
        {
            if (B)
            {
                int lock_offs = blockIdx.z * size_n / 128;

                exl3_gemm_kernel_inner
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
                (A_, B, C_, size_m_, size_k, size_n, locks + lock_offs);
             }

            A_ += 16 * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
            else                  C_ = (void*) (((half*) C_) + 16 * size_n);
            size_m_ -= 16;
            grid.sync();
        }

        // Had and output scales

        if (B)
        {
            int total_warps = size_m * size_n / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* svh = svh_list[mat_index];
            float scale = 0.088388347648f;  // 1/sqrt(128)
            if (B_weights) scale *= __half2float(B_weights[j]);

            if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
            else                  C_ = (void*) (((half*) C) + j * size_m * size_n);

            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner
                    (
                        ((const float*) C_) + this_warp * 128,
                        ((float*) C_) + this_warp * 128,
                        nullptr,
                        svh + (this_warp * 128) % size_n,
                        scale
                    );
                else
                    had_hf_r_128_inner
                    (
                        ((const half*) C_) + this_warp * 128,
                        ((half*) C_) + this_warp * 128,
                        nullptr,
                        svh + (this_warp * 128) % size_n,
                        scale
                    );
            }
        }
    }

    if (B_weights)
        grid.sync();

    // Final reduction
    if (B_weights && blockIdx.z == 0)
    {
        int total_warps = size_m * size_n / 32;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        int this_lane = threadIdx.x % 32;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            // If C_red is provided, write the reduced (weighted) sum there instead of into C[0].
            // This lets callers keep C as a scratch buffer while emitting the final output
            // directly into a separate destination (e.g. a per-token output row).
            if constexpr (c_fp32)
            {
                float* C__ = ((float*) C) + this_warp * 32 + this_lane;
                float* C___ = C__;
                float sum = 0.0f;
                for (int j = 0; j < bszm; ++j)
                {
                    sum += *C___;
                    C___ += size_m * size_n;
                }
                float* C_dst = (C_red ? ((float*) C_red) : ((float*) C)) + this_warp * 32 + this_lane;
                *C_dst = sum;
            }
            else
            {
                half* C__ = ((half*) C) + this_warp * 32 + this_lane;
                half* C___ = C__;
                half sum = {};
                for (int j = 0; j < bszm; ++j)
                {
                    sum = __hadd(sum, *C___);
                    C___ += size_m * size_n;
                }
                half* C_dst = (C_red ? ((half*) C_red) : ((half*) C)) + this_warp * 32 + this_lane;
                *C_dst = sum;
            }
        }
    }
}

// Fused gate+up mgemm kernel
// Processes both gate and up projections in a single kernel launch to reduce overhead
// and improve cache locality. For each expert, we do:
// 1. Gate Hadamard + Gate GEMM
// 2. Up Hadamard + Up GEMM (reusing A_had buffer)
template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_fused_mgemm_kernel(EXL3_FUSED_MGEMM_ARGS)
{
    auto grid = cg::this_grid();

    // Pack indices within min_index <= idx < max_index
    if (min_index >= 0)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            int j = 0;
            for (int i = 0; i < bszm; ++i)
            {
                int idx = B_indices[i];
                if (idx >= min_index && idx < max_index)
                {
                    v_indices[j] = idx - min_index;
                    j++;
                }
            }
            bszm_sync = j;
            for (; j < bszm; ++j)
            {
                v_indices[j] = -1;
            }
        }
        __threadfence();
        grid.sync();
        B_indices = v_indices;
    }

    int bszm_effective = (min_index >= 0) ? bszm_sync : bszm;

    for (int i = 0; i < bszm_effective; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int mat_index = -1;
        const uint16_t* B_gate = nullptr;
        const uint16_t* B_up = nullptr;

        if (j < bszm_effective)
        {
            mat_index = B_indices ? (int) B_indices[j] : j;
            if (mat_index >= 0)
            {
                B_gate = B_gate_list[mat_index];
                B_up = B_up_list[mat_index];
            }
        }

        // ============ GATE PROJECTION ============

        if (B_gate)
        {
            // Hadamard transform on input for gate
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh_gate = suh_gate_list[mat_index];
            const half* A_ = A;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh_gate + (this_warp * 128) % size_k,
                    nullptr,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
        grid.sync();

        // Gate GEMM
        if (B_gate)
        {
            int size_m_ = size_m;
            half* A_ = A_had + j * size_m * size_k;
            void* C_;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_gate) + j * size_m * size_n_gate);
            else                  C_ = (void*) (((half*) C_gate) + j * size_m * size_n_gate);

            while (size_m_ > 0)
            {
                int lock_offs = blockIdx.z * size_n_gate / 128;

                exl3_gemm_kernel_inner
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
                (A_, B_gate, C_, size_m_, size_k, size_n_gate, locks + lock_offs);

                A_ += 16 * size_k;
                if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n_gate);
                else                  C_ = (void*) (((half*) C_) + 16 * size_n_gate);
                size_m_ -= 16;
                grid.sync();
            }

            // Output Hadamard for gate
            {
                int total_warps = size_m * size_n_gate / 128;
                int warps_grid = gridDim.x * blockDim.x / 32;
                int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

                const half* svh_gate = svh_gate_list[mat_index];

                if constexpr (c_fp32) C_ = (void*) (((float*) C_gate) + j * size_m * size_n_gate);
                else                  C_ = (void*) (((half*) C_gate) + j * size_m * size_n_gate);

                for(; this_warp < total_warps; this_warp += warps_grid)
                {
                    if constexpr (c_fp32)
                        had_ff_r_128_inner
                        (
                            ((const float*) C_) + this_warp * 128,
                            ((float*) C_) + this_warp * 128,
                            nullptr,
                            svh_gate + (this_warp * 128) % size_n_gate,
                            0.088388347648f
                        );
                    else
                        had_hf_r_128_inner
                        (
                            ((const half*) C_) + this_warp * 128,
                            ((half*) C_) + this_warp * 128,
                            nullptr,
                            svh_gate + (this_warp * 128) % size_n_gate,
                            0.088388347648f
                        );
                }
            }
        }
        grid.sync();

        // ============ UP PROJECTION ============
        //
        // NOTE: `suh` is a *pre-scale* (applied before the Hadamard), so we can only reuse the
        // already-computed A_had from the gate path if the pointers are identical (same tensor).
        //
        // If `suh_gate == suh_up`, we can skip the second Hadamard entirely and reuse A_had.
        bool same_suh = false;
        if (B_gate && B_up)
        {
            same_suh = (suh_gate_list[mat_index] == suh_up_list[mat_index]);
        }

        if (B_up && !same_suh)
        {
            // Hadamard transform on input for up (reusing A_had buffer)
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh_up = suh_up_list[mat_index];
            const half* A_ = A;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh_up + (this_warp * 128) % size_k,
                    nullptr,
                    0.088388347648f
                );
        }
        grid.sync();

        // Up GEMM
        if (B_up)
        {
            int size_m_ = size_m;
            half* A_ = A_had + j * size_m * size_k;
            void* C_;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_up) + j * size_m * size_n_up);
            else                  C_ = (void*) (((half*) C_up) + j * size_m * size_n_up);

            while (size_m_ > 0)
            {
                // Use a different lock region for up to avoid conflicts with gate
                int lock_offs = blockIdx.z * size_n_up / 128 + gridDim.z * size_n_gate / 128;

                exl3_gemm_kernel_inner
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
                (A_, B_up, C_, size_m_, size_k, size_n_up, locks + lock_offs);

                A_ += 16 * size_k;
                if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n_up);
                else                  C_ = (void*) (((half*) C_) + 16 * size_n_up);
                size_m_ -= 16;
                grid.sync();
            }

            // Output Hadamard for up
            {
                int total_warps = size_m * size_n_up / 128;
                int warps_grid = gridDim.x * blockDim.x / 32;
                int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

                const half* svh_up = svh_up_list[mat_index];

                if constexpr (c_fp32) C_ = (void*) (((float*) C_up) + j * size_m * size_n_up);
                else                  C_ = (void*) (((half*) C_up) + j * size_m * size_n_up);

                for(; this_warp < total_warps; this_warp += warps_grid)
                {
                    if constexpr (c_fp32)
                        had_ff_r_128_inner
                        (
                            ((const float*) C_) + this_warp * 128,
                            ((float*) C_) + this_warp * 128,
                            nullptr,
                            svh_up + (this_warp * 128) % size_n_up,
                            0.088388347648f
                        );
                    else
                        had_hf_r_128_inner
                        (
                            ((const half*) C_) + this_warp * 128,
                            ((half*) C_) + this_warp * 128,
                            nullptr,
                            svh_up + (this_warp * 128) % size_n_up,
                            0.088388347648f
                        );
                }
            }
        }

        // Sync before next iteration
        if (i + gridDim.z < bszm_effective)
            grid.sync();
    }
}