#pragma once

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct Ring_kernel_traits
{

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = uint32_t;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
    using ValLayoutMNK = Layout<Shape<_1, _2, _1>>;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using ValLayoutMNK = Layout<Shape<_1, _2, _2>>;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t,
          typename Base = Ring_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Ring_fwd_traits : public Base
{
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;

    // TODO: review
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>, // 4x1x1 or 8x1x1 thread group
        // NOTE: cutlass v3.3
        // typename Base::ValLayoutMNK>; // 1x2x1 or 1x2x2 value group for 16x16x16 MMA and LDSM
        // cutlass v3.4
        Tile<Int<16 * kNWarps>, _16, _16>>;

    using SmemLayoutAtomQ = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                                 // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                                                 Layout<Shape<_8, Int<kBlockKSmem>>,
                                                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // This has to be kBlockN and not 8, otherwise we get wrong results for d=128
    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>,
                                                      Stride<_1, Int<kBlockKSmem>>>;
    using SmemLayoutAtomVtransposed = decltype(composition(Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));
    // Maybe the VtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));
    // using SmemLayoutVtransposedNoSwizzle = decltype(SmemLayoutVtransposed{}.layout_fn());

    using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                                 Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                                                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQCount = size(SmemLayoutQ{});
    static constexpr int kSmemKVCount = size(SmemLayoutKV{}) * 2;
    static constexpr int kSmemQSize = kSmemQCount * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKVCount * sizeof(Element);
    // TODO:
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    // TODO: review

    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy>;
    using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per read
    using BufferTiledCopyKV = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                                    GmemLayoutAtom{},
                                                    Layout<Shape<_1, _8>>{}));
    using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                                    GmemLayoutAtom{},
                                                    Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store
    static constexpr int kGmemThreadsPerRowP = kBlockN / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRowP == 0, "kNThreads must be a multiple of kGmemThreadsPerRowP");
    using GmemLayoutAtomP = Layout<Shape<Int<kNThreads / kGmemThreadsPerRowP>, Int<kGmemThreadsPerRowP>>,
                                   Stride<Int<kGmemThreadsPerRowP>, _1>>;

    using GmemTiledCopyP = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                                    GmemLayoutAtomP{},
                                                    Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape<_16, _8>, // Thread layout, 8 threads per row
               Stride<_8, _1>>,
        Layout<Shape<_8, _16>, // Thread layout, 16 threads per row
               Stride<_16, _1>>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                                                         GmemLayoutAtomOaccum{},
                                                         Layout<Shape<_1, _4>>{})); // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                                                            GmemLayoutAtomRotcossin{},
                                                            Layout<Shape<_1, _4>>{})); // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                                                GmemLayoutAtomRotcossin{},
                                                                Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per load
};
