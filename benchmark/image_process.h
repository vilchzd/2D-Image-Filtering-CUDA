#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include <string>

#define BLOCK_SIZE 32 // Threads per block 
#define MAX_GRID_SIZE 5 // Grid size

//CPU Funtions
void image_process(const std::string& file_name, unsigned char*& input, unsigned char*& output, int& width, int& height, int& channels);
void cpu_blurGRAY(unsigned char*& input, unsigned char*& output, int width, int height, int grid); 
void cpu_blurBGR(unsigned char*& input, unsigned char*& output, int width, int height, int grid); 
void show_image(unsigned char*& input, unsigned char*& output, int width, int height, int channels); 

//GPU Funtions
void gpu_wrapper_blurGRAY(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid);
void gpu_wrapper_blurBGR(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid);

#endif 
