#include <stdio.h>

__global__ void (int *a, int *b, int *c){
	*c = *a + *b;
}


int main(){
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	printf("%d\n", size);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a = 2;
	b = 3;

	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1,1>>>(d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	printf("%d\n", c );

	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}