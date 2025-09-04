#include <iostream>   
#include <cstdlib>   
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


using namespace std;

__global__ void matrix(float *in, float *out, int width, int height) {
    extern __shared__ float mat[];
    int x = threadIdx.x;
    int y = threadIdx.y;
    mat[x * width + y] = in[x * width + y]; 
    __syncthreads();
 
    float blur_sum = 0;
    float count = 0;
    float average = 0;

    for (int grid_y = -1; grid_y < 2; grid_y++) {
        for (int grid_x = -1; grid_x < 2; grid_x++) {
            int blur_x = x + grid_x;
            int blur_y = y + grid_y;
            if (blur_x >= 0 && blur_x < width && blur_y >= 0 && blur_y < height) {
                blur_sum += mat[blur_x * width + blur_y];
                count++;
            }
        }
    }
    average = blur_sum / count;
    mat[x * width + y] = average;
    out[x * width + y] = mat[x * width + y];             
} 


int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    const int width = 5, height = 5;
    float *d_in, *d_out; 
    float *h_in = (float*)malloc(height*width*sizeof(float));
    float *h_out = (float*)malloc(height*width*sizeof(float));
    cudaMalloc((void**)&d_in,width*height*sizeof(float));
    cudaMalloc((void**)&d_out,width*height*sizeof(float));
    
    for (int i = 0; i < height*width; i++) {
        h_in[i] = i*10+10;
    }

    cudaMemcpy(d_in, h_in, width*height*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 grid_size(1);
    dim3 block_size(width,height);
    size_t shm_size = width * height * sizeof(float);
    matrix<<<grid_size,block_size, shm_size>>>(d_in, d_out, 5 , 5);
    
    cudaMemcpy(h_out, d_out, width*height*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int y = 0; y < height; y++) {
        for (int i = 0; i < width; i++) {
            cout << h_out[y*width + i] << " ";
        }
        cout << endl;
    }
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}

/* __global__ void read_m(float *in, float *out, int w, int h) {
    const int height = 5;
    const int width = 5;
    __shared__ float matrix[height][width];

    int y = threadIdx.y;
    int x = threadIdx.x;
    if ()
    

    if (i < width && j < width) {
        matrix[threadIdx.y][threadIdx.x] = in[j * width + i];
    }

    __syncthreads();
    if (trans_i < width && trans_j < width) {
        out[trans_j * width + trans_i] = matrix[threadIdx.x][threadIdx.y];
    }
}
*/