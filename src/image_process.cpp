#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "image_process.h"

using namespace cv;

void image_process(const std::string& file_name, unsigned char*& input, unsigned char*& output, int& width, int& height, int& channels) {
    
    cv::Mat image = cv::imread(file_name, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        exit(1);
    }

    std::cout << "Channels: " << image.channels() << std::endl;
    if (channels == 1 && image.channels() != 1) {
        cv::cvtColor(image, image, (image.channels() == 4) ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
        std::cout << "Converted "<< image.channels() << "-channel image to 1-channel GRAY." << std::endl;
    } 
    else if (channels == 3 && image.channels() != 3) {
        cv::cvtColor(image, image, (image.channels() == 4) ? cv::COLOR_BGRA2BGR : cv::COLOR_GRAY2BGR);
        std::cout << "Converted "<< image.channels() << "-channel image to 3-channel BGR." << std::endl;
    } else {
        std::cerr << "Unsupported image format: " << channels << std::endl;
        exit(1);
    }

    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    width = image.cols;
    height = image.rows;
    int size = image.rows * image.cols * image.channels();

    input = new unsigned char[size];
    output = new unsigned char[size];
    unsigned char* temp = new unsigned char[size];
    std::memcpy(input, image.data, size); 
    std::memcpy(output, input, size); 
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
    int channels = 3;
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
                        int in_index = (blur_y * width + blur_x) * channels;
                        blur_sum_B += input[in_index + 0];
                        blur_sum_G += input[in_index + 1];
                        blur_sum_R += input[in_index + 2];
                        count++;
                        }
                }
            }
            int out_index = (y * width +x) * channels;
            output[out_index + 0] = blur_sum_B / count;
            output[out_index + 1] = blur_sum_G / count;
            output[out_index + 2] = blur_sum_R / count;
        }  
    }
} 
