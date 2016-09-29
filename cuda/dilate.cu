#include <helper_cuda.h>
#include <helper_math.h>
#include <helper_functions.h>
#include <opencv2/opencv.hpp>

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
    int radius = 3;
    char *imagePath = "lena.png";
    cv::Mat image0 = cv::imread(imagePath);
    cv::Mat image(image0.cols, image0.rows, CV_8UC4);
    cv::cvtColor(image0, image, CV_RGB2RGBA);
    if (image.empty()) {
        printf("failed to read image.\n");
        exit(-1);
    }
    assert(image.channels() == 4);
    width = image.cols;
    height = image.rows;

    int devId = findCudaDevice(argc, (const char **)argv);

    // device memory for result 
    uint* dData = NULL;
    checkCudaErrors(cudaMalloc((void **)&dData, width*height*sizeof(uint)));

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray = NULL;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));
    checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, image.data, width*height*sizeof(uint), cudaMemcpyHostToDevice));

    rgbaTex.addressMode[0] = cudaAddressModeWrap;
    rgbaTex.addressMode[1] = cudaAddressModeWrap;

    checkCudaErrors(cudaBindTextureToArray(rgbaTex, cuArray, channelDesc));

    dim3 dimBlock(16,16);
    dim3 dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y);
    dilate <<<dimGrid, dimBlock, 0>>>(dData, width, height, radius);

    checkCudaErrors(cudaDeviceSynchronize());

    // host memory
    checkCudaErrors(cudaMemcpy(image.data, dData, width*height*sizeof(uint), cudaMemcpyDeviceToHost));

    // save
    cv::imwrite("me_dilate3.bmp", image);

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
}