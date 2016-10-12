#include <cuda_runtime.h>
#include <helper_cuda.h>

int main(int argv, char** argc)
{
    int devId;
    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDevice(&devId));
    checkCudaErrors(cudaGetDeviceProperties(&devProp, devId));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devId, devProp.name, devProp.major, devProp.minor);
}