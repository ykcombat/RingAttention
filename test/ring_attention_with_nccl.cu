#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>

#define CUDA_CHECK(cmd) do {                        \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCL_CHECK(cmd) do {                        \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

void Gemm(
    const float* A, const float* B, float* result,
    const int m, const int k, const int n,
    const bool ta, const bool tb
)
{
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    cublasOperation_t transa = ta?CUBLAS_OP_T: CUBLAS_OP_N;
    cublasOperation_t transb = tb?CUBLAS_OP_T: CUBLAS_OP_N;
    const float alpha = 1.0;
    const float beta = 0.0;
    int ldb = transb?k:n;
    int lda = transa?m:k;
    CUBLAS_CHECK(cublasSgemm(cublasH,transb,transa, n, m, k, &alpha, B, ldb, A, lda, &beta, result, n));
}

void synchronizeDevice(int nDev, cudaStream_t* s){
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamSynchronize(s[i]));
  }
}

__global__ void exponent(float* mat, int m, int n){
  int local_m = threadIdx.x / n;
  int local_n = threadIdx.x % n;
  if(local_m > m || local_n > n)return;
  mat[local_m * n + local_n] = exp(mat[local_m * n + local_n]);
}

__global__ void accumulate(float* dest, float*a, int length){
  if(threadIdx.x >= length)return;
  for(int i = 0; i < length; i++){
    dest[threadIdx.x] += a[threadIdx.x + i];
  }

}

__global__ void vectorAdd(float * dest, float* a, int length){
  if(threadIdx.x >= length)return;
  dest[threadIdx.x] += a[threadIdx.x];
}


int main(int argc, char* argv[])
{
  ncclComm_t comms[4];


  //managing 2 devices
  int nDev = 2;
  int fullN = 4;
  int d = 512;
  int n = fullN / nDev;
  int size = d * n;
  int devs[2] = { 0, 1};


  //allocating and initializing device buffers
  float** current_q = (float**)malloc(nDev * sizeof(float*));
  float** current_k = (float**)malloc(nDev * sizeof(float*));
  float** current_v = (float**)malloc(nDev * sizeof(float*));
  float** next_k = (float**)malloc(nDev * sizeof(float*));
  float** next_v = (float**)malloc(nDev * sizeof(float*));
  float** max = (float**)malloc(nDev * sizeof(float*));
  float** current_qkt = (float**)malloc(nDev * sizeof(float*));
  float** current_eqktv = (float**)malloc(nDev * sizeof(float*));
  float** up = (float**)malloc(nDev * sizeof(float*));
  float** down = (float**)malloc(nDev * sizeof(float*));

  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  cudaStream_t* nccl_s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMalloc((void**)current_q + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)current_k + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)current_v + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)next_k + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)next_v + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)max + i, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)current_qkt + i, n*n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)current_eqktv + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)up + i, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)down + i, n*n * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(s+i));
    CUDA_CHECK(cudaStreamCreate(nccl_s+i));
    CUDA_CHECK(cudaMemset(current_q[i], 0, size*sizeof(float)));
    CUDA_CHECK(cudaMemset(next_k[i], 0, size*sizeof(float)));
    CUDA_CHECK(cudaMemset(next_v[i], 0, size*sizeof(float)));
    CUDA_CHECK(cudaMemset(up[i], 0, size*sizeof(float)));
    CUDA_CHECK(cudaMemset(down[i], 0, n*sizeof(float)));
  }


  //initializing NCCL
  NCCL_CHECK(ncclCommInitAll(comms, nDev, devs));


   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCL_CHECK(ncclGroupStart());
  for(int i = 0; i < nDev; i++){
    synchronizeDevice(nDev, nccl_s);
    for(int j = 0; j < nDev; j++){
      CUDA_CHECK(cudaSetDevice(j));
      CUDA_CHECK(cudaMemcpyAsync(current_k[j], next_k[j], size * sizeof(float), cudaMemcpyDeviceToDevice, s[j]));
      CUDA_CHECK(cudaMemcpyAsync(current_v[j], next_v[j], size * sizeof(float), cudaMemcpyDeviceToDevice, s[j]));
      CUDA_CHECK(cudaStreamSynchronize(s[j]));
      NCCL_CHECK(ncclSend(current_k[j], size, ncclFloat32, (j+1)%nDev, comms[j], nccl_s[j]));
      NCCL_CHECK(ncclSend(current_v[j], size, ncclFloat32, (j+1)%nDev, comms[j], nccl_s[j]));
      NCCL_CHECK(ncclRecv(next_k[j], size, ncclFloat32, (j+nDev-1)%nDev, comms[j], nccl_s[j]));
      NCCL_CHECK(ncclRecv(next_v[j], size, ncclFloat32, (j+nDev-1)%nDev, comms[j], nccl_s[j]));
    }
    // synchronizeDevice(nDev, s);
    for(int j = 0; j < nDev; j++){
      CUDA_CHECK(cudaSetDevice(j));
      Gemm(current_q[j], current_k[j], current_qkt[j], n, d, n, false, true);
      // printf("device %d has invoked gemm1\n", j);
      //e指数
      exponent<<<1, 1024, 0, s[j]>>>(current_qkt[j], n, n);
      // float * host;
      // CUDA_CHECK(cudaMallocHost(&host, n*n*sizeof(float)));
      // CUDA_CHECK(cudaMemcpy(host,current_qkt[j],n*n*sizeof(float),cudaMemcpyDeviceToHost));
      // for(int ii = 0; ii < n; ii++){
      //   for(int jj = 0; jj < n; jj++){
      //     printf("%.3f ", host[ii * n + jj]);
      //   }
      //   printf("\n");
      // }
      Gemm(current_qkt[j], current_v[j], current_eqktv[j], n, n, d, false, false);
      vectorAdd<<<1,1024,0,s[j]>>>(up[j],current_eqktv[j], size);
      accumulate<<<1,2,0,s[j]>>>(down[j], current_qkt[j], n);

      //add
      // printf("device %d has invoked gemm2\n", j);
    }
  }
  NCCL_CHECK(ncclGroupEnd());


  //synchronizing on CUDA streams to wait for completion of NCCL operation
  synchronizeDevice(nDev, s);

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}