#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>
#include "image_process.h"

#define N 32
#define BLOCK_SIZE 16

using namespace std;

/* __global__ void transpose(float *in, float *out, int width) {
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
} */


__global__ void gpu_blurGRAY(unsigned char* input, unsigned char* output, int width, int height, int grid) {

    __shared__ float image_mat[BLOCK_SIZE][BLOCK_SIZE];

    int y = threadIdx.y + BLOCK_SIZE * blockIdx.y;
    int x = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    if (x < width && y < height) {
        image_mat[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    if (x < width && y < height) {
        int blur_sum = 0;
        int count = 0;
        for (int grid_y = -grid; grid_y < grid; grid_y++) {
            for (int grid_x = -grid; grid_x < grid; grid_x++) {
                int blur_y = y + grid_y;
                int blur_x = x + grid_x;
                if (blur_y >= 0 && blur_x >= 0 && blur_y < height && blur_x < width) {
                    blur_sum += image_mat[blur_y * width + blur_x];
                    count++;
                }
            }
        }
        output[y * width + x] = blur_sum / count;
    }  
}

void gpu_wrapper_blurGRAY(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid) {

    cout << "Executing gpu_blurGRAY kernel" << endl;
    unsigned char *d_input, *d_output;
    int size = width*height*sizeof(unsigned char);
    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    gpu_blurGRAY<<<grid_size, block_size>>>(d_input, d_output, width, height, grid);

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cout << "Freing memory in gpu" << endl;
}


void cpu_blurGRAY(unsigned char*& input, unsigned char*& output, int width, int height, int grid) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int blur_sum = 0;
            int count = 0;
            for (int grid_y = -grid; grid_y < grid; grid_y++) {
                for (int grid_x = -grid; grid_x < grid; grid_x++) {
                    int blur_y = y + grid_y;
                    int blur_x = x + grid_x;
                    if (blur_y >= 0 && blur_x >= 0 && blur_y < height && blur_x < width) {
                        blur_sum += input[blur_y * width + blur_x];
                        count++;
                        }
                }
            }
            output[y * width + x] = blur_sum / count;
        }  
    }
} 
    
void cpu_blurBGR(unsigned char*& input, unsigned char*& output, int width, int height, int grid) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int blur_sum_B = 0;
            int blur_sum_G = 0;
            int blur_sum_R = 0;
            int count = 0;
            for (int grid_y = -grid; grid_y < grid; grid_y++) {
                for (int grid_x = -grid; grid_x < grid; grid_x++) {
                    int blur_y = y + grid_y;
                    int blur_x = x + grid_x;
                    if (blur_y >= 0 && blur_x >= 0 && blur_y < height && blur_x < width) {
                        int in_index = (blur_y * width + blur_x) * 3;
                        blur_sum_B += input[in_index + 0];
                        blur_sum_G += input[in_index + 1];
                        blur_sum_R += input[in_index + 2];
                        count++;
                        }
                }
            }
            int out_index = (y * width +x) * 3;
            output[out_index + 0] = blur_sum_B / count;
            output[out_index + 1] = blur_sum_G / count;
            output[out_index + 2] = blur_sum_R / count;
        }  
    }
} 


/* 
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
 */