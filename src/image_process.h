#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include <string>

//CPU Funtions
void image_process(const std::string& file_name, unsigned char*& input, unsigned char*& output, int& width, int& height, int& channels);
void cpu_blurGRAY(unsigned char*& input, unsigned char*& output, int width, int height, int grid); 
void cpu_blurBGR(unsigned char*& input, unsigned char*& output, int width, int height, int grid); 
void show_image(unsigned char*& input, unsigned char*& output, int width, int height, int channels); 

//GPU Funtions

#endif 
