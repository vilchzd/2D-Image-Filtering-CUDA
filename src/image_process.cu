#include <iostream>   
#include <cstdlib>   
#include <iomanip>
#include <cuda_runtime.h>
#include "image_process.h"


#define BLOCK_SIZE 32 //threads per block 

using namespace std;


__global__ void gpu_blurGRAY(unsigned char* input, unsigned char* output, int width, int height, int grid) {

    __shared__ unsigned char tile[BLOCK_SIZE + 2 * 50][BLOCK_SIZE + 2 * 50];

    int y = threadIdx.y + BLOCK_SIZE * blockIdx.y; //global pixel positions
    int x = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    
    int shared_x = threadIdx.x + grid; //maps center pixel
    int shared_y = threadIdx.y + grid;
    int shared_width = BLOCK_SIZE + 2 * 50;

    if (x < width && y < height) {
        tile[shared_y][shared_x] = input[y * width + x];
    }

    for (int dy = -grid; dy <= grid; dy++) {
        for (int dx = -grid; dx <= grid; dx++)  {
            int gy = y + dy;
            int gx = x + dx;
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                tile[shared_y + dy][shared_x + dx] = input[gy * width + gx];
            } else {
                tile[shared_y + dy][shared_x + dx] = 0; // zero padding
            }
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        int blur_sum = 0;
        int count = 0;
        for (int grid_y = -grid; grid_y <= grid; grid_y++) {
            for (int grid_x = -grid; grid_x <= grid; grid_x++) {
                int blur_y = shared_y + grid_y;
                int blur_x = shared_x + grid_x;
                if (blur_y >= 0 && blur_x >= 0 && blur_y < shared_width && blur_x < shared_width) {
                    blur_sum += tile[blur_y][blur_x];
                    count++;
                }
            }
        }
        output[y * width + x] = blur_sum / count;
    }  
}


__global__ void gpu_blurBGR(unsigned char* input, unsigned char* output, int width, int height, int grid) {

    __shared__ unsigned char tile[(BLOCK_SIZE + 2 * 48) * (BLOCK_SIZE + 2 * 48) * 3];

    int y = threadIdx.y + BLOCK_SIZE * blockIdx.y; //global pixel positions
    int x = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    int shared_x = threadIdx.x + grid; //maps center pixel
    int shared_y = threadIdx.y + grid;
    int shared_width = BLOCK_SIZE + 2 * 48;

    if (x < width && y < height) {
        int in_index = (y * width + x) * 3;
        int sh_index = (shared_y * shared_width + shared_x) * 3;
        tile[sh_index + 0] = input[in_index + 0]; // B
        tile[sh_index + 1] = input[in_index + 1]; // G
        tile[sh_index + 2] = input[in_index + 2]; // R
    }

    for (int dy = -grid; dy <= grid; dy++) {
        for (int dx = -grid; dx <= grid; dx++)  {
            int gy = y + dy;
            int gx = x + dx;
            int sh_index = ((shared_y + dy) * shared_width + (shared_x + dx)) * 3;

            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                int in_index = (gy * width + gx) * 3;
                tile[sh_index + 0] = input[in_index + 0];
                tile[sh_index + 1] = input[in_index + 1];
                tile[sh_index + 2] = input[in_index + 2];
            } else {
                tile[sh_index + 0] = 0; // zero padding
                tile[sh_index + 1] = 0;
                tile[sh_index + 2] = 0; 
            }
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        int blur_sum_B = 0;
        int blur_sum_G = 0;
        int blur_sum_R = 0;
        int count = 0;

        for (int grid_y = -grid; grid_y <= grid; grid_y++) {
            for (int grid_x = -grid; grid_x <= grid; grid_x++) {
                int blur_y = shared_y + grid_y;
                int blur_x = shared_x + grid_x;
                if (blur_y >= 0 && blur_x >= 0 && blur_y < shared_width && blur_x < shared_width) {
                    int sh_index = (blur_y * shared_width + blur_x) * 3;
                    blur_sum_B += tile[sh_index + 0];
                    blur_sum_G += tile[sh_index + 1];
                    blur_sum_R += tile[sh_index + 2];
                    count++;
                }
            }
        }
        int out_index = (y * width + x) * 3;
        output[out_index + 0] = blur_sum_B / count;
        output[out_index + 1] = blur_sum_G / count;
        output[out_index + 2] = blur_sum_R / count;
    }  
}


/* __global__ void gpu_blurBGR(unsigned char* input, unsigned char* output, int width, int height, int grid) {
    // Shared memory for a tile with halo
    __shared__ unsigned char tile[(BLOCK_SIZE + 2 * 48) * (BLOCK_SIZE + 2 * 48) * 3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    int shared_width = BLOCK_SIZE + 2 * grid;
    int shared_x = tx + grid;
    int shared_y = ty + grid;

    // Step 1: load center pixel
    if (x < width && y < height) {
        int in_idx = (y * width + x) * 3;
        int sh_idx = (shared_y * shared_width + shared_x) * 3;
        tile[sh_idx + 0] = input[in_idx + 0];
        tile[sh_idx + 1] = input[in_idx + 1];
        tile[sh_idx + 2] = input[in_idx + 2];
    }

    // Step 2: load halo pixels (each thread loads some neighbors)
    for (int dy = -grid; dy <= grid; dy++) {
        for (int dx = -grid; dx <= grid; dx++) {
            int gx = x + dx;
            int gy = y + dy;
            int sh_idx = ((shared_y + dy) * shared_width + (shared_x + dx)) * 3;

            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                int in_idx = (gy * width + gx) * 3;
                tile[sh_idx + 0] = input[in_idx + 0];
                tile[sh_idx + 1] = input[in_idx + 1];
                tile[sh_idx + 2] = input[in_idx + 2];
            } else {
                tile[sh_idx + 0] = 0;
                tile[sh_idx + 1] = 0;
                tile[sh_idx + 2] = 0;
            }
        }
    }

    __syncthreads();

    // Step 3: compute blur
    if (x < width && y < height) {
        int sum_B = 0, sum_G = 0, sum_R = 0;
        int count = 0;

        for (int dy = -grid; dy <= grid; dy++) {
            for (int dx = -grid; dx <= grid; dx++) {
                int sh_idx = ((shared_y + dy) * shared_width + (shared_x + dx)) * 3;
                sum_B += tile[sh_idx + 0];
                sum_G += tile[sh_idx + 1];
                sum_R += tile[sh_idx + 2];
                count++;
            }
        }

        int out_idx = (y * width + x) * 3;
        output[out_idx + 0] = sum_B / count;
        output[out_idx + 1] = sum_G / count;
        output[out_idx + 2] = sum_R / count;
    }
}
 */


void gpu_wrapper_blurGRAY(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid) {

    cout << "Executing gpu_blurGRAY kernel" << endl;
    unsigned char *d_input, *d_output;
    int size = width*height*sizeof(unsigned char);
    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1)/BLOCK_SIZE, (height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    gpu_blurGRAY<<<grid_size, block_size>>>(d_input, d_output, width, height, grid);

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cout << "Freing memory in gpu" << endl;
}

void gpu_wrapper_blurBGR(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid) {

    cout << "Executing gpu_BGR kernel" << endl;
    unsigned char *d_input, *d_output;
    int size = width * height * sizeof(unsigned char) * 3;
    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1)/BLOCK_SIZE, (height + BLOCK_SIZE - 1)/BLOCK_SIZE);

    gpu_blurBGR<<<grid_size, block_size>>>(d_input, d_output, width, height, grid);

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cout << "Freing memory in gpu" << endl;
}


/* void cpu_blurGRAY(unsigned char*& input, unsigned char*& output, int width, int height, int grid) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int blur_sum = 0;
            int count = 0;
            for (int grid_y = -grid; grid_y <= grid; grid_y++) {
                for (int grid_x = -grid; grid_x <= grid; grid_x++) {
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
}  */
    
/* void cpu_blurBGR(unsigned char*& input, unsigned char*& output, int width, int height, int grid) {

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int blur_sum_B = 0;
            int blur_sum_G = 0;
            int blur_sum_R = 0;
            int count = 0;
            for (int grid_y = -grid; grid_y <= grid; grid_y++) {
                for (int grid_x = -grid; grid_x <= grid; grid_x++) {
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
            int out_index = (y * width + x) * 3;
            output[out_index + 0] = blur_sum_B / count;
            output[out_index + 1] = blur_sum_G / count;
            output[out_index + 2] = blur_sum_R / count;
        }  
    }
}  */

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