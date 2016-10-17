//#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include <stdlib.h>
#include <time.h>

typedef struct{
	int width,height;
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
	return 0;
}
