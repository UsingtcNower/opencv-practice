#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>

texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
uint* pImage;

__device__ uint float4Touint(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x));
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));

    return uint(rgba.x*255.0f) | (uint(rgba.y*255.0f) << 8) | (uint(rgba.z*255.0f) << 16) | (uint(rgba.w*255.0f)<<24);
}

__global__ void dilate(uint *od, int w, int h, int r)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h) {
        return;
    }

    float4 center = tex2D(rgbaTex, x, y);
    float4 t = center;
    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            float4 curPix = tex2D(rgbaTex, x + i, y + i);
            if (t.x < curPix.x)
                t.x = curPix.x;
            if (t.y < curPix.y)
                t.y = curPix.y;
            if (t.z < curPix.z)
                t.z = curPix.z;
            if (t.w < curPix.w)
                t.w = curPix.w;
        }
    }
    od[y*w + x] = float4Touint(t);
}

int main(int argc, char **argv)
{
    int width, height;
    char *imagePath = "";
    LoadBMPFile((uchar4 **)&pImage, &width, &height, imagePath);
    if (!pImage) {
        printf("failed top load bmp.\n");
        exit(-1);
    }

    int devId = findCudaDevice(argc, (const char **)argv);

    // result 
    uint* dData = NULL;
    checkCudaErrors(cudaMalloc((void **)&dData, width*height*sizeof(uint)));

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc<uchar4>();
    cudaArray *cuArray = NULL;
    checkCudaErrors(cudaMallocArray(&cuArray, channelDesc, width, height));
    checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, pImage, width*height*sizeof(uint), cudaMemcpyHostToDevice));

    rgbaTex.addressMode[0] = cudaAddressModeWrap;
    rgbaTex.addressMode[1] = cudaAddressModeWrap;

    checkCudaErrors(cudaBindTextureToArray(rgbaTex, cuArray, channelDesc);

}