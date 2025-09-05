#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "image_process.h"

using namespace std;

int main() {

    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    int width, height;
    int channels = 4;
    int grid = 4;
    string file_name = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\love.png";

    image_process(file_name, input, output, width, height, channels); 
    (channels == 1 ? cpu_blurGRAY : cpu_blurBGR)(input, output, width, height, grid);
    show_image(input, output, width, height, channels); 
 
    cv::waitKey(0);
    
    delete[] input;
    delete[] output;
    return 0;
}