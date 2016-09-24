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
    LoadBMPFile();
}