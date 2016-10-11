#include <cuda_runtime.h>
#include <helper_cuda.h>

int main(int argv, char** argc)
{
    int devId;
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDevice(&devId));

}