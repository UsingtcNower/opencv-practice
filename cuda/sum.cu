#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

void sum_gold(int* a, int n, int* sum)
{
    *sum = 0;
    for (int i = 0; i < n; ++i)
        *sum += a[i];
}

#define THREAD_BLOCK_SIZE 256
#define MUL(a,b) ((a)*(b))
#define MUD(a,b,c) (MUL(a,b)+(c))

__global__ void getPartialSum_GPU(int* a, int n, int* partialSum)
{
    __shared__ int subSum[THREAD_BLOCK_SIZE];

    subSum[threadIdx.x] = 0;
    __syncthreads();

    for (int i = MUD(blockIdx.x, blockDim.x, threadIdx.x); i < n; i += MUL(gridDim.x,blockDim.x)) {
        subSum[threadIdx.x] += a[i];
    }
    __syncthreads();

#if 0
    for (int stride = THREAD_BLOCK_SIZE / 2; stride > 0; stride = stride >> 1) {
        if (threadIdx.x < stride) {
            subSum[threadIdx.x] += subSum[threadIdx.x + stride];
            __syncthreads();
        }
    }
    partialSum[blockIdx.x] = subSum[0];
#else
    if (threadIdx.x == 0) {
        for (int i = 0; i < THREAD_BLOCK_SIZE;i++) {
            partialSum[blockIdx.x] += subSum[i];
        }
    }
#endif

}

#define ARRAY_SIZE (1<<20LL)

int main()
{
    int *h_a = (int*)malloc(ARRAY_SIZE*sizeof(int));
    int n = ARRAY_SIZE;
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) {
        h_a[i] = rand() % 2;
    }
    int h_sum = 0;
    sum_gold(h_a, ARRAY_SIZE, &h_sum);

    int* d_a;
    checkCudaErrors(cudaMalloc((void**)&d_a, ARRAY_SIZE*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_a, h_a, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 dimBlock(THREAD_BLOCK_SIZE);
    dim3 dimGrid(ARRAY_SIZE/THREAD_BLOCK_SIZE);
    int* dPartial_sum;
    checkCudaErrors(cudaMalloc((void**)&dPartial_sum, dimGrid.x*sizeof(int)));
    getPartialSum_GPU << <dimGrid, dimBlock >> >(d_a, ARRAY_SIZE, dPartial_sum);

    int* hPartial_sum = (int *)malloc(sizeof(int)*dimGrid.x);
    checkCudaErrors(cudaMemcpy(hPartial_sum, dPartial_sum, dimGrid.x*sizeof(int), cudaMemcpyDeviceToHost));

    int d_sum = 0;
    for (int i = 0; i < dimGrid.x; ++i) {
        d_sum += hPartial_sum[i];
    }

    printf("Check Result: %d\n", d_sum == h_sum);
    //getchar();
}