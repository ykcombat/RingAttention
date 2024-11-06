#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <numeric>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <vector>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "kernel_traits.h"
#include "ring.h"
#include "utils.h"

using FP = float;
using FPC = cute::half_t;

void check_nvshmem_init()
{
    if (nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED)
    {
        nvshmem_init();
    }
}

template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  // TODO: Aligned的话smem的计算是否有问题
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};


template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        // NOTE: 4表示4个线程, 因为在SM80_16x8x16_F32F16F16F32_TN中,
        // 每组每行就是4个线程处理8个value的, 每个线程处理2个value
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    // NOTE: 遍历tensor每行, 记录到summary中
    // reduce 当前thread的max 
    thread_reduce_<zero_init>(tensor, summary, op);
    // NOTE: 二分法对summary[]进行reduce
    // reduce thread间的max
    quad_allreduce_(summary, summary, op);
}


template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

namespace ring{

    // copy from S to D with tiled_copy
    // TODO: 需要支持causal模式的的跳过拷贝
    template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                Tensor<Engine1, Layout1> &D) {
        CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
        CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
        CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
        CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

        #pragma unroll
        for (int m = 0; m < size<1>(S); ++m) {
            // TODO: 原版处这里identity_MN是用来跳过大块的block的, predicate用于跳过block内的拷贝
            // TODO: 添加predicate逻辑, 用于跳过无用拷贝
            // if (get<0>(identity_MN(0, m, 0)) < max_MN)
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        }
    }

    template <int N>
    CUTE_HOST_DEVICE
    void cp_async_wait() {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    }
}

template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
    // "int_tuple.hpp(74): error: conversion to inaccessible base class"
    // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};


// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    // NOTE: scores来自acc_s: Q@K.T
    // acc_s用来存储QK和softmax的结果[seqlen, seqlen]
    // acc_o用来存储softmax(QK)结果的分子部分, 用于rescale
    // 流式计算不断用当前分块计算的结果scors来rescale

    if (Is_first) {
        // NOTE: 优化, 第一次softmax不需要rescale, 只需要记录分子, max, sum
        reduce_max</*zero_init=*/true>(scores, scores_max);
        scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // 记录上一次的max
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        // TODO: reduce的实现学习一下
        // NOTE: 计算新max到scores_max
        // reduce_max包含步:
        //  1. 求当前thread内max: 遍历
        //  2. reduce thread间的max: 使用shift技巧reduce
        reduce_max</*zero_init=*/false>(scores, scores_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            // NOTE: 辅助变量: 当前max
            float scores_max_cur = scores_max(mi);
            // NOTE: 计算旧score的rescale值
            // NOTE: 因为QK(影响max)计算时没有考虑softmax_scale, 所以这里要补上
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            // NOTE: rescale旧分母部分
            scores_sum(mi) *= scores_scale;
            // NOTE: 旧分子部分rescale
            // acc_o_rowcol.shape = (nrow, ncol)
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        // NOTE: 计算新分子部分: 对所有scores进行rescale
        scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        // NOTE: 累加新分母
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        // NOTE:利用新分子来累加新分母
        //  1. 线程内累加: 遍历
        //  2. 线程间累加: 使用shift技巧reduce
        reduce_sum(scores, scores_sum_cur);
        // NOTE: 新分母累加到旧分母
        #pragma unroll
  



template <typename Fragment>
inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
  Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
    for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
  }
  return acc_fp16;
}

template<typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
    // TD [2023-08-13]: Same error as above on Cutlass 3.2
    // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
    //                    get<0, 1>(l),
    //                    get<1, 1, 1>(l));
    return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                       get<1>(get<0>(l)),
                       get<1>(get<1>(get<1>(l))));
};

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    // NOTE: 构造smem -> reg拷贝的目的地址寄存器对象
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// NOTE: A矩阵已经在寄存器中的gemm封装
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
    // NOTE: 符合M N K描述: A[M, K] @ B[N, K] = C[M, N]
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    // NOTE: retile 成拷贝需要的大小
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}



std::vector<FPC*> nvshmem_create_tensor_list(const std::vector<int64_t> &shape)
{
    check_nvshmem_init();

    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) * sizeof(FPC);
    int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int rank = nvshmem_my_pe();
    int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(rank));
    std::vector<FPC *> ptrs;
    ptrs.reserve(local_world_size);
    void *ptr = nvshmem_malloc(size);
    CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
    int rank_offset = rank - local_rank;
    for (int i = 0; i < local_world_size; i++)
    {
        // runs this call nvshmem failure, don't know why
        //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank, NVSHMEM_TEAM_WORLD)
        int rank_global = i + rank_offset;
        if (rank == rank_global)
        {
            ptrs.emplace_back((FPC*)ptr);
        }
        else
        {
            void *rptr = nvshmem_ptr(ptr, rank_global);
            ptrs.emplace_back((FPC*)rptr);
        }
    }

    return ptrs;
}

void set_params_fprop(Ring_fwd_params &params,
                      // device pointers
                      FPC* q,
                      FPC* k,
                      FPC* v,
                      float* out,
                      int npes,
                      int mype,
                      int* seqlen,
                      int dim,
                      int max_seq) {

  memset(&params, 0, sizeof(params));
  // TODO: get ptr
  params.q_ptr = (void*)q;

  params.k_ptr = (void*)k;
  params.v_ptr = (void*)v;

  params.out_ptr = (void*)out;
  params.npes = npes;
  params.mype = mype;
  int* d_seqlen;
  CUDA_CHECK(cudaMalloc((void**)&d_seqlen,npes*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_seqlen,seqlen,npes*sizeof(int),cudaMemcpyHostToDevice));
  params.seqlen = d_seqlen;
  params.h_seqlen = seqlen;
  params.dim = dim;
  params.softmax_scale = 1.f;

  void* k_buffer = nvshmem_malloc(max_seq*dim*sizeof(FPC));
  params.k_buffer = k_buffer;
  void* v_buffer = nvshmem_malloc(max_seq*dim*sizeof(FPC));
  params.v_buffer = v_buffer;
}


template<typename Kernel_traits, typename Params>
__global__ void ring_attention_kernel(Params params){
  using namespace cute;
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
  using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  extern __shared__ char smem_[];
  using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);
  int mype = params.mype;
  int npes = params.npes;
  int m_block = blockIdx.x;
  int tidx = threadIdx.x;
  Tensor Q = make_tensor(
    make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr)),
    make_shape(params.seqlen[mype], params.dim),
    make_stride(params.dim, Int<1>{})
  );

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});


  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
  Tensor scores_sum = make_fragment_like(scores_max);

  clear(rAccOut);

    Tensor LocalK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr)),
        make_shape(params.seqlen[mype], params.dim),
        make_stride(params.dim, Int<1>{})
    );
    Tensor LocalV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr)),
        make_shape(params.seqlen[mype], params.dim),
        make_stride(params.dim, Int<1>{})
    );

  for(int i = 0; i < params.npes; i++){
    // __syncthreads();
    // if(!tidx)nvshmem_barrier_all();
    // __syncthreads();

    Tensor CurrentK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.k_buffer)),
        make_shape(params.seqlen[(mype + i)%npes], params.dim),
        make_stride(params.dim, Int<1>{})
    );
    Tensor CurrentV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.v_buffer)),
        make_shape(params.seqlen[(mype + i)%npes], params.dim),
        make_stride(params.dim, Int<1>{})
    );

    //cp.async global -> shared
    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));
    Tensor gK = local_tile(CurrentK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    Tensor gV = local_tile(CurrentV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    typename Kernel_traits::BufferTiledCopyKV buffer_tiled_copy_kv;
    auto buffer_thr_copy_kv = buffer_tiled_copy_kv.get_thread_slice(tidx);
    if(i==0){
        Tensor gKBuffer = local_tile(LocalK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
        Tensor gVBuffer = local_tile(LocalV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
        Tensor tCKrK = buffer_thr_copy_kv.partition_S(gKBuffer(_, _, 0));
        Tensor tCKgK = buffer_thr_copy_kv.partition_D(gK);
        Tensor tCVrV = buffer_thr_copy_kv.partition_S(gVBuffer(_, _, 0));
        Tensor tCVgV = buffer_thr_copy_kv.partition_D(gV);
        cute::copy(buffer_tiled_copy_kv,tCKrK,tCKgK);
        cute::copy(buffer_tiled_copy_kv,tCVrV,tCVgV);
    }

    // if(!tidx&&!mype)print_tensor(local_tile(gK,make_tile(4,4),make_coord(0,0)));



    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)

      // NOTE: 准备拷贝Q, K, V到smem的copy对象
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // TODO: 拷贝时转置
    // NOTE: smem->reg拷贝Vt
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    ring::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    ring::copy(gmem_tiled_copy_QKV, tKgK, tKsK);

    cute::cp_async_fence();


    const int n_block_min = 0;
    // NOTE: 1. mask between N BLOCKs if is causal mode
    int n_block_max = cute::ceil_div(params.seqlen[(mype + i)%npes], kBlockN);


    // copy.async remote->local

    for (int nbi = n_block_min; nbi < n_block_max; nbi++) {
      auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

      clear(rAccScore);

    //   if(!mype&&!tidx)print_tensor(local_tile(gK, make_tile(8,8), make_coord(0,0)));

      // 等待Q, K的gmem -> smem拷贝完成, 即Q, K就绪
      // wait<0>表示等待还剩0个未完成
      ring::cp_async_wait<0>();
      __syncthreads();

      // gemm的同时异步加载V
      gV = local_tile(CurrentV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
      if(i==0){
        Tensor gVBuffer = local_tile(LocalV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
        Tensor tCVrV = buffer_thr_copy_kv.partition_S(gVBuffer(_, _, 0));
        Tensor tCVgV = buffer_thr_copy_kv.partition_D(gV);
        cute::copy(buffer_tiled_copy_kv,tCVrV,tCVgV);
      }

      tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
      // 异步加载V到smem
      ring::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
      // 发起异步拷贝
      cute::cp_async_fence();

      // O = Q@K.T
      // NOTE: 加载smem中的数据到reg再做gemm, **加载期间执行retile**
      gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
          smem_thr_copy_Q, smem_thr_copy_K
      );


      Tensor scores = make_tensor(rAccScore.data(), convert_layout_acc_rowcol(rAccScore.layout()));

      // NOTE: 2. mask within N BLOCKs
      // if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
      //   mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
      // }

      // NOTE: 等待V加载完成, 为下个K加载准备初始状态
      ring::cp_async_wait<0>();
      __syncthreads();

      // advance K
      if (nbi != n_block_max - 1) {
        gK = local_tile(CurrentK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
        if(i==0){
            Tensor gKBuffer = local_tile(LocalK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
            Tensor tCKrK = buffer_thr_copy_kv.partition_S(gKBuffer(_, _, 0));
            Tensor tCKgK = buffer_thr_copy_kv.partition_D(gK);
            cute::copy(buffer_tiled_copy_kv,tCKrK,tCKgK);
        }
        tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
        ring::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        cute::cp_async_fence();
      }

      // 计算softmax
      // NOTE: rAccOut记录softmax后所有的分子
      nbi == 0 && i==0 ? softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
        softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

      // 实际执行QK @ V
      // (score AKA rAccScore): QK[M, N] @ V[N, dim]
      // NOTE: DABC: F32F16F16F32, convert D type(F32) to A type(F16)
      // TODO: convert_type目前写死
      Tensor rP = convert_type_f32_to_f16(rAccScore);
      // NOTE: Convert from layout C to layout A
      Tensor tOrP = make_tensor(rP.data(), convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));
      //if(!tidx&&!mype)print_tensor(tOrP);

      gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }
    int block_offset = blockIdx.x * params.dim * kBlockN * params.seqlen[(mype + i + 1)%npes]/params.seqlen[mype] * sizeof(FPC);
    nvshmemx_getmem_block(params.k_buffer + block_offset,
        params.k_ptr + block_offset,
        kBlockN * params.dim * sizeof(FPC) * params.seqlen[(mype + i + 1)%npes]/params.seqlen[mype],
        (mype+i+1)%npes);
    nvshmemx_getmem_block(params.v_buffer + block_offset,
        params.v_ptr + block_offset,
        kBlockN * params.dim * sizeof(FPC) * params.seqlen[(mype + i + 1)%npes]/params.seqlen[mype],
        (mype+i+1)%npes);
  }
    Tensor acc_o_rowcol = make_tensor(rAccOut.data(), convert_layout_acc_rowcol(rAccOut.layout()));
    // for row
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float scale = inv_sum;
        // for col
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }

    Tensor rO = convert_type_f32_to_f16(rAccOut);
    // 复用sQ的smem做sO的拷出
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)

    // Partition sO to match the accumulator partitioning
    // TODO: review
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // NOTE: 先拷贝到smem
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor O = make_tensor(
    make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.out_ptr)),
    make_shape(params.seqlen[mype], params.dim),
    make_stride(params.dim, Int<1>{}));
    Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    ring::copy(gmem_tiled_copy_O, tOrO, tOgO);

}


template<typename Kernel_traits>
void run_ring_fwd(Ring_fwd_params &params, cudaStream_t stream) {
  // TODO: check if works: default stream = 0
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  const int num_m_block = (params.h_seqlen[params.mype] + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  constexpr size_t smem_size = size_t(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  dim3 grid(num_m_block);
  dim3 block(Kernel_traits::kNThreads);

//   printf("params.seqlen[%d]=%d\n", params.mype, params.h_seqlen[params.mype]);

//   if(!params.mype){
//     printf("shmem_size=%d\n", smem_size);
//     printf("grid_size=%d\n", num_m_block);
//     printf("block_size=%d\n", Kernel_traits::kNThreads);
//   }



  auto kernel = &ring_attention_kernel<Kernel_traits, Ring_fwd_params>;
  // NOTE: smem过大时需要设置
  if (smem_size >= 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  void *args[] = {&params};
  nvshmemx_collective_launch((const void *)kernel, grid, block, args, smem_size,0);

  // TODO: stream
//   kernel<<<grid, block, smem_size>>>(params);
}

void ring_attention(FPC* q, FPC* k, FPC* v, float* out, int npes, int mype, int* seqlen, int dim, int max_seq) {
  const int Bm = 64;
  const int Bn = 64;
  const int Warps = 4;
  const int DIM = 64;


  Ring_fwd_params params;
  set_params_fprop(params, q, k, v, out, npes, mype, seqlen, dim, max_seq);

  using Traits = Ring_fwd_traits<DIM, Bm, Bn, Warps, FPC>;

  run_ring_fwd<Traits>(params, 0);

  // Wait until kernel finish.
  cudaDeviceSynchronize();
}
template<typename d_type>
void printDeviceFloat(d_type * out, int m, int n){
    d_type * h_out = (d_type * )malloc(m * n * sizeof(d_type));
    CUDA_CHECK(cudaMemcpy(h_out, out, m * n * sizeof(d_type), cudaMemcpyDeviceToHost));
    Tensor Cute = make_tensor(h_out, make_shape(m, n), make_stride(n, 1));
    print_tensor(local_tile(Cute, make_tile(8,8), make_coord(0,0)));
}

void fillWithNum(FPC * mat, int m, int n, int mype){
    FPC* h = (FPC*)malloc(m*n*sizeof(FPC));
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            // h[i*n + j] = mype/35.0 + i/130.0 + j/110.0;
            h[i*n + j] = i/130.0 + j/110.0 + i*j/600.0;
            // h[i*n + j] = 1;
        }
    }
    CUDA_CHECK(cudaMemcpy(mat, h, m*n*sizeof(FPC), cudaMemcpyHostToDevice));
}

void initQKV(FPC* q, FPC* k, FPC* v, int m, int n, int mype){
    fillWithNum(q,m,n,mype);
    fillWithNum(k,m,n,mype);
    fillWithNum(v,m,n,mype);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(){
  int dim = 64;
  int max_seq = 8192;
  int seqlen[] ={8192, 8192};
  nvshmem_init();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  CUDA_CHECK(cudaSetDevice(mype));
  FPC* q;
  CUDA_CHECK(cudaMalloc(&q, seqlen[mype] *dim*sizeof(FPC)));
  FPC* k = (FPC*)nvshmem_malloc(max_seq*dim*sizeof(FPC));
  FPC* v = (FPC*)nvshmem_malloc(max_seq *dim*sizeof(FPC));
  initQKV(q,k,v,seqlen[mype],dim, mype);
  float* out;
  CUDA_CHECK(cudaSetDevice(mype));
  CUDA_CHECK(cudaMalloc(&out, seqlen[mype] *dim *sizeof(float)));

  int warmup = 50;
  for(int i = 0; i < warmup; i++){
    ring_attention(q, k, v, out, npes, mype, seqlen, dim, max_seq);
  }

  int test_loop = 50;
  cudaEvent_t start, stop;
  float elapsedTime = 0.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for(int i = 0; i < test_loop; i++){
    ring_attention(q, k, v, out, npes, mype, seqlen, dim, max_seq);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  for(int i = 0; i < npes; i++){
    if(mype == i)printf("mype: %d success\n", mype);
    nvshmem_barrier_all();
  }
  if(!mype)printDeviceFloat(out, seqlen[mype], dim);
  if(!mype)printf("elapsedtime=%f\n", elapsedTime/test_loop);
}