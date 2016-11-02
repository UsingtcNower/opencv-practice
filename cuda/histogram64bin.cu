#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <opencv2/opencv.hpp>

#define SHARED_MEMORY_BANKS 16
#define THREADBLOCK_SIZE SHARED_MEMORY_BANKS*4

cv::Mat histImg;

__global__ void vectorAddKernel(float* A, float* B, float* C, int numCnt)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < numCnt) {
        C[i] = A[i] + B[i];
    }
}

// partialHist[64][GridDim.x]
__global__ void hist64binKernel(uchar* drData, int* partialHist, int dataCount)
{
    __shared__ int s_hist[64*THREADBLOCK_SIZE];
    for (int i = 0; i < 64; ++i) {
        s_hist[i*THREADBLOCK_SIZE + threadIdx.x] = 0;
    }
    __syncthreads();
    int threadPos = blockDim.x*blockIdx.x + threadIdx.x;
    if (threadPos < dataCount) {
        s_hist[drData[threadPos] * THREADBLOCK_SIZE + threadIdx.x] ++;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 64; ++i) {
            int sum = 0;
            for (int j = 0; j < blockDim.x; j++) {
                sum += s_hist[i*THREADBLOCK_SIZE+j];
            }
            partialHist[i*gridDim.x + blockIdx.x] = sum;
        }
    }
}

__global__ void hist64binMerge(int* partialHist, int* hist, int histCount)
{
    __shared__ int data[256];
    int sum = 0;
    for (int i = threadIdx.x; i < histCount; i += 256) {
        sum += partialHist[blockIdx.x*histCount+i];
    }
    data[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int i = 0; i < 256; ++i)
            sum += data[i];
        hist[blockIdx.x] = sum;
    }
}

int main(int argc, char** argv)
{
    int histSize = 64;
    cv::Mat img0 = cv::imread("lena.png");
    if (img0.empty()) {
        printf("error imread.\n");
        return -1;
    }
    assert(img0.channels() == 3);
    cv::Mat img(img0.cols, img0.rows, CV_8UC1);
    cv::cvtColor(img0, img, CV_RGB2GRAY);

    // init
    int devId = findCudaDevice(argc, (const char**)argv);
    uchar* drData = NULL;
    checkCudaErrors(cudaMalloc((void**)&drData, img.cols*img.rows*sizeof(uchar)));
    checkCudaErrors(cudaMemcpy(drData, img.data, img.cols*img.rows*sizeof(uchar), cudaMemcpyHostToDevice));
    int* d_hist = NULL;
    checkCudaErrors(cudaMalloc((void**)&d_hist, histSize*sizeof(int)));
    dim3 dimBlock(THREADBLOCK_SIZE, 1);
    dim3 dimGrid((img.cols*img.rows+dimBlock.x-1)/dimBlock.x, 1);
    int* partialHist = NULL;
    checkCudaErrors(cudaMalloc((void**)&partialHist, dimGrid.x*histSize*sizeof(int)));
    
    
    // calculate
    hist64binKernel <<<dimGrid, dimBlock >>>(drData, partialHist, img.cols*img.rows);
    cudaDeviceSynchronize();
    getLastCudaError("hist64binKernel");
    // -->> paritalHist[64][dimGrid.x]
    hist64binMerge <<< histSize, 256>>>(partialHist, d_hist, dimGrid.x);
    cudaDeviceSynchronize();
    getLastCudaError("hist64binMerge");

    int* hist = (int*)malloc(histSize*sizeof(int));
    checkCudaErrors(cudaMemcpy(hist, d_hist, histSize*sizeof(int), cudaMemcpyDeviceToHost));
    // draw
    int hist_w = 512;
    int hist_h = 400;
    histImg = cv::Mat(hist_w, hist_h, CV_8UC3, cv::Scalar(0,0,0));
    int bin_w = cvRound((double)hist_w/histSize);

    for (int i = 0; i < histSize; ++i)
    {
        hist[i] = hist[i] * hist_h / 256;
    }
    for (int i = 1; i < histSize; ++i) {
        cv::line(histImg, cv::Point((i-1)*bin_w, hist_h-hist[i-1]),
                          cv::Point(i*bin_w, hist_h-hist[i]),
                          cv::Scalar(255, 0, 0), 2, 8, 0);
    }
    cv::imwrite("histImg.png", histImg);
    return 0;
}