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


#define CUDA_CHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
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
    printf("tb = %d\n", tb);
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    cublasOperation_t transa = ta?CUBLAS_OP_T: CUBLAS_OP_N;
    cublasOperation_t transb = tb?CUBLAS_OP_T: CUBLAS_OP_N;
    const float alpha = 1.0;
    const float beta = 0.0;
    CUBLAS_CHECK(cublasSgemm(cublasH,transb,transa, n, m, k, &alpha, B, k, A, k, &beta, result, n));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

void printMatrix(float (*matrix)[4], int row, int col) {
    for(int i=0;i<row;i++)
    {
        std::cout << std::endl;
        std::cout << " [ ";
        for (int j=0; j<col; j++) {
         std::cout << matrix[i][j] << " ";
        }
        std::cout << " ] ";
    }
    std::cout << std::endl;
}

int main(){
    int M = 2;
    int K = 3;
    int N = 4;
    float h_A[M][K]={ {1,2,3}, {4,5,6} };
    float h_B[N][K]={ {1,2,3},{4,5,6}, {7,8,9},{10,11,12} };
    float h_C[M][4] = {0};

    float *d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,M*K*sizeof(float));
    cudaMalloc((void**)&d_b,K*N*sizeof(float));
    cudaMalloc((void**)&d_c,M*N*sizeof(float));
    cudaMemcpy(d_a,&h_A,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&h_B,K*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_c,0,M*N*sizeof(float));

    Gemm(d_a,d_b,d_c,M,K,N,false,true);
    cudaMemcpy(h_C,d_c,M*N*sizeof(float),cudaMemcpyDeviceToHost);

    printMatrix(h_C, M, N);
    return 0;
}