// int i = threadIdx.x + blockDim.x * blockIdx.x
#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>

#define N 32
#define BLOCK_SIZE 16

using namespace std;

__global__ void transpose(float *in, float *out, int width) {
    __shared__ float t_mat[BLOCK_SIZE][BLOCK_SIZE];

    int i = threadIdx.x + BLOCK_SIZE * blockIdx.x ;
    int j = threadIdx.y + BLOCK_SIZE * blockIdx.y ;
    

    if (i < width && j < width) {
        t_mat[threadIdx.y][threadIdx.x] = in[j * width + i];
    }

    __syncthreads();

    int trans_i = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    int trans_j = threadIdx.y + BLOCK_SIZE * blockIdx.x;


    if (trans_i < width && trans_j < width) {
        out[trans_j * width + trans_i] = t_mat[threadIdx.x][threadIdx.y];
    }
}


int main() {

    int width = N;
    float *d_in, *d_out;
    float *h_in = (float*)malloc(width*width*sizeof(float));
    float *h_out = (float*)malloc(width*width*sizeof(float));

    for (int i = 0; i < width*width; i++) {
        h_in[i] = i+1;
    }

    cudaMalloc((void**)&d_in, width*width*sizeof(float));
    cudaMemcpy(d_in, h_in, width*width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_out, width*width*sizeof(float));
    
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

      
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    transpose<<<grid_size, block_size>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    cudaMemcpy(h_out, d_out, width*width*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<width;i++){
        for(int j=0;j<width;j++)
            cout << h_out[i*width+j] << " ";
        cout << endl;
    }


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_stop);
    cout << "GPU time: " << milliseconds << " ms" << endl;


    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
} 