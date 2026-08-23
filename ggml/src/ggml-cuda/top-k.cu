#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE


// Two-stage top-k for wide rows.
//
// The paths below select one row at a time, or sort the whole row to keep k of
// it. That is fine for the sampler, which asks for one row, but a speculative
// selector asks for top-k over the whole vocabulary at every block position at
// once. Taking each tile's top-k first cannot drop a winner: a global top-k
// element has at most k-1 larger elements anywhere, so at most k-1 inside its
// own tile. What survives is small enough for the existing bitonic argsort to
// finish the job for every row in one launch.
//
// The tile stage packs the value and the column into one 64-bit key, so a plain
// max reduction resolves both the ordering and the tie-break, and every key in a
// tile is distinct - which is what makes "clear the entry equal to the running
// maximum" remove exactly one element.

#define TOPK_TILE  4096
#define TOPK_BLOCK 256
#define TOPK_CAND  1024  // argsort_f32_i32_cuda_bitonic's row limit

static __global__ void topk_tile(const float * src, float * cand_val, int * cand_idx,
                                 const int ncols, const int ntiles, const int k) {
    __shared__ uint64_t smem[TOPK_BLOCK];

    const int     row     = blockIdx.x / ntiles;
    const int     tile    = blockIdx.x % ntiles;
    const float * row_ptr = src + (size_t) row * ncols;

    uint64_t keys[TOPK_TILE / TOPK_BLOCK];
#pragma unroll
    for (int i = 0; i < TOPK_TILE / TOPK_BLOCK; ++i) {
        const int col = tile * TOPK_TILE + threadIdx.x + i * TOPK_BLOCK;
        uint32_t  b   = col < ncols ? __float_as_uint(row_ptr[col]) : 0;
        b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        keys[i] = col < ncols ? (((uint64_t) b << 32) | (uint32_t) (ncols - 1 - col)) : 0;
    }

    const size_t out = ((size_t) row * ntiles + tile) * k;
    for (int j = 0; j < k; ++j) {
        uint64_t local = 0;
#pragma unroll
        for (int i = 0; i < TOPK_TILE / TOPK_BLOCK; ++i) {
            local = max(local, keys[i]);
        }
        smem[threadIdx.x] = local;
        __syncthreads();
        for (int s = TOPK_BLOCK / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + s]);
            }
            __syncthreads();
        }
        const uint64_t best = smem[0];
        if (threadIdx.x == 0) {
            const int col = ncols - 1 - (int) (best & 0xFFFFFFFFu);
            cand_val[out + j] = best ? row_ptr[col] : -INFINITY;
            cand_idx[out + j] = best ? col : 0;
        }
#pragma unroll
        for (int i = 0; i < TOPK_TILE / TOPK_BLOCK; ++i) {
            if (keys[i] == best) {
                keys[i] = 0;
            }
        }
        __syncthreads();
    }
}

// The argsort ranks candidates; turn its positions back into columns.
static __global__ void topk_unmap(const int * cand_idx, const int * order, int * dst,
                                  const int ncand, const int k) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        dst[(size_t) blockIdx.x * k + i] = cand_idx[(size_t) blockIdx.x * ncand + order[(size_t) blockIdx.x * ncand + i]];
    }
}

static bool ggml_cuda_top_k_tiled(ggml_cuda_pool & pool, const float * src, int * dst,
                                  const int ncols, const int nrows, const int k,
                                  cudaStream_t stream) {
    const int ntiles = (ncols + TOPK_TILE - 1) / TOPK_TILE;
    const int ncand  = ntiles * k;
    // Narrow rows do not have enough tiles for the reduction to pay off, and the
    // survivors have to fit the bitonic argsort.
    if (ntiles < 4 || ncand > TOPK_CAND) {
        return false;
    }

    ggml_cuda_pool_alloc<float> cand_val(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>   cand_idx(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>   order   (pool, (size_t) nrows * ncand);

    topk_tile<<<nrows * ntiles, TOPK_BLOCK, 0, stream>>>(
            src, cand_val.get(), cand_idx.get(), ncols, ntiles, k);
    argsort_f32_i32_cuda_bitonic(cand_val.get(), order.get(), ncand, nrows,
            GGML_SORT_ORDER_DESC, stream);
    topk_unmap<<<nrows, TOPK_BLOCK, 0, stream>>>(cand_idx.get(), order.get(), dst, ncand, k);
    return true;
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    if (ggml_cuda_top_k_tiled(pool, src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
