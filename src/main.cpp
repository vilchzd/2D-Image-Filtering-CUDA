#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "image_process.h"

using namespace std;

int main() {

    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    int width, height, size;
    int channels = 3;
    int grid = 10;
    string file_name = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\love.png";

    image_process(file_name, input, output, width, height, channels); 

    if (channels > 1) { 

        cpu_blurBGR(input, output, width, height, grid);
        cv::Mat input_image(height, width, CV_8UC3, input);
        cv::Mat image_out(height, width, CV_8UC3, output);

        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::resizeWindow("Input", 500, 500);
        cv::imshow("Input", input_image);

        cv::namedWindow("Blur", cv::WINDOW_NORMAL);
        cv::resizeWindow("Blur", 500, 500);
        cv::imshow("Blur", image_out);

    } else {

        cpu_blurGRAY(input, output, width, height, grid);
        cv::Mat input_image(height, width, CV_8UC1, input);
        cv::Mat image_out(height, width, CV_8UC1, output);

        cv::namedWindow("Input", cv::WINDOW_NORMAL);
        cv::resizeWindow("Input", 500, 500);
        cv::imshow("Input", input_image);

        cv::namedWindow("Blur", cv::WINDOW_NORMAL);
        cv::resizeWindow("Blur", 500, 500);
        cv::imshow("Blur", image_out);
    } 
    
    cv::waitKey(0);
    
    delete[] input;
    delete[] output;
    return 0;
}