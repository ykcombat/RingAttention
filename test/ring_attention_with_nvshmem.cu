#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <vector>
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <nvshmemx.h>
#include <nvshmem.h>

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void check_nvshmem_init() {
  if(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED){
    nvshmem_init();
  }
}

std::vector<torch::Tensor> nvshmem_create_tensor_list(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
  check_nvshmem_init();
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  FLUX_CHECK(size != 0);
  int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int rank = nvshmem_my_pe();
  int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  std::vector<torch::Tensor> tensors;
  tensors.reserve(local_world_size);
  void *ptr = nvshmem_malloc(size);
  CUDA_CHECK(cudaMemset(ptr, 0, size)); // memset the allocated buffer
  FLUX_CHECK(ptr != nullptr);
  int rank_offset = rank - local_rank;
  for (int i = 0; i < local_world_size; i++) {
    // runs this call nvshmem failure, don't know why
    //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank, NVSHMEM_TEAM_WORLD)
    int rank_global = i + rank_offset;
    if (rank == rank_global) {
      tensors.emplace_back(
          at::from_blob(ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu));
    } else {
      void *rptr = nvshmem_ptr(ptr, rank_global);
      FLUX_CHECK(rptr != nullptr) << "rank " << rank;
      tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
    }
  }

  return tensors;
}

int main(){
    nvshmem_create_tensor({2,2},c10::ScalarType);
}

