#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include <string>
#include <vector>
using namespace std;

#define TARGET_CHANNELS 3 // Desired image output use 1 for GREYSCALE or 3 for BGR 
#define BLOCK_SIZE 32     // Threads per block (MAX 32 for 2D block)
#define GRID_SIZE  1      // Grid size (MAX_GRID_SIZE = 48)

//CPU Funtions
void image_process(const string& file_name, unsigned char*& input, unsigned char*& output, int& width, int& height, int& channels);
void cpu_blurGRAY(unsigned char*& input, unsigned char*& output, int width, int height, int grid); 
void cpu_blurBGR(unsigned char*& input, unsigned char*& output, int width, int height, int grid); 
void show_image(unsigned char*& input, unsigned char*& output, int width, int height, int channels); 
float stddev(const vector<float>& times, float mean);

//GPU Funtions
void gpu_wrapper_blurGRAY(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid);
void gpu_wrapper_blurBGR(unsigned char*& h_input, unsigned char*& h_output, int width, int height, int grid);

#endif 
