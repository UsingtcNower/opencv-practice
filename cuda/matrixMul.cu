#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <time.h>

typedef struct{
	int width,height;
	int pitch;
	int* pdata;
} Matrix;

void matrixMul_Gold(Matrix A, Matrix B, Matrix C)
{
	for(int y=0;y<C.height;y++) {
		for(int x=0;x<C.width;++x) {
			int sum = 0;
			for(int i=0;i<A.width;++i)
				sum += A.pdata[y*A.width+i]*B.pdata[i*B.width+x];
			C.pdata[y*C.width+x] = sum;
		}
	}
}

__global__ void matrixMul(Matrix d_A, Matrix d_B, Matrix d_C)
{
	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x<d_C.width && y<d_C.height) {
		int sum = 0;
		for(int i=0;i<d_A.width;++i)
			sum += d_A.pdata[y*d_A.width+i]*d_B.pdata[i*d_B.width+x];
		d_C.pdata[y*d_C.width+x] = sum;
	}	
}

int main(int argc, char** argv)
{
	Matrix A;
	A.width = 360;
	A.height = 640;
	A.pdata = (int*)malloc(A.width*A.height*sizeof(int));

	Matrix B;
	B.width = 1280;
	B.height = 360;
	B.pdata = (int*)malloc(B.width*B.height*sizeof(int));

	Matrix C;
	C.width = B.width;
	C.height = A.height;
	C.pdata = (int*)malloc(C.width*C.height*sizeof(int));

	srand((unsigned)time(NULL));
	for(int y=0;y<A.height;++y)
		for(int x=0;x<A.width;++x)
			A.pdata[y*A.width+x] = rand()%256;
	for(int y=0;y<B.height;++y)
		for(int x=0;x<B.width;++x)
			B.pdata[y*B.width+x] = rand()%256;
	matrixMul_Gold(A,B,C);

	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	checkCudaErrors(cudaMalloc((void**)&d_A.pdata, d_A.width*d_A.height*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_A.pdata, A.pdata, A.width*A.height*sizeof(int), cudaMemcpyHostToDevice));

	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	checkCudaErrors(cudaMalloc((void**)&d_B.pdata, d_B.width*d_B.height*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_B.pdata, B.pdata, B.width*B.height*sizeof(int), cudaMemcpyHostToDevice));

	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	checkCudaErrors(cudaMalloc((void**)&d_C.pdata, d_C.width*d_C.height*sizeof(int)));

	Matrix h_C;
	h_C.width = C.width;
	h_C.height = C.height;
	h_C.pdata = (int*)malloc(h_C.width*h_C.height*sizeof(int));

	dim3 dimBlock(16, 16);
	dim3 dimGrid((d_C.width+dimBlock.x-1)/dimBlock.x, (d_C.height+dimBlock.y-1)/dimBlock.y);
	matrixMul<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);
	
	checkCudaErrors(cudaMemcpy(h_C.pdata, d_C.pdata, d_C.width*d_C.height*sizeof(int), cudaMemcpyDeviceToHost));

	bool check = true;
	for(int y=0;y<h_C.height;++y)
		for(int x=0;x<h_C.width;++x)
			if(h_C.pdata[y*h_C.width+x]!=C.pdata[y*h_C.width+x]) {
				check = false;
				break;
			}

    cudaFree(d_A.pdata);
    cudaFree(d_B.pdata);
    cudaFree(d_C.pdata);

	printf("Check Result: %d\n", check);
	return 0;
}
