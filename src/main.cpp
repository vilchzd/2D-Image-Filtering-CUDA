#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "image_process.h"

using namespace std;

int main() {

    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    int target_channels = TARGET_CHANNELS;
    int grid = GRID_SIZE;
    int width, height;
    
    string file_name = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\park.png";

    image_process(file_name, input, output, width, height, target_channels);
    std::cout << std::fixed << std::setprecision(2) << "Preforming " 
              << ((1.0 * width * height * ((2 * grid) * (2 * grid) + 1) * target_channels) / 1'000'000'000.0)
              << " billion operations using " << (2*grid)+1 << "x" << (2*grid)+1 << " blur kernel" << std::endl; 
    //(target_channels == 1 ? cpu_blurGRAY : cpu_blurBGR)(input, output, width, height, grid);
    (target_channels == 1 ?  gpu_wrapper_blurGRAY :  gpu_wrapper_blurBGR)(input, output, width, height, grid);
    show_image(input, output, width, height, target_channels); 
 
    cv::waitKey(0);
    
    delete[] input;
    delete[] output;
    return 0;
    
}