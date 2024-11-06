#include "utils.h"
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
#include <nvshmemx.h>
#include <nvshmem.h>



void check_nvshmem_init()
{
    if (nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED)
    {
        nvshmem_init();
    }
}

std::vector<float*> nvshmem_create_tensor_list(const std::vector<int64_t> &shape)
{
    check_nvshmem_init();

    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()) * sizeof(float);
    int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int rank = nvshmem_my_pe();
    int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::vector<float *> ptrs;
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
            ptrs.emplace_back(ptr);
        }
        else
        {
            void *rptr = nvshmem_ptr(ptr, rank_global);
            ptrs.emplace_back(rptr);
        }
    }

    return ptrs;
}