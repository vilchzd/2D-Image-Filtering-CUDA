#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "image_process.h"

using namespace cv;

void image_process(const std::string& file_name, unsigned char*& input, unsigned char*& output) {
    //cv::Mat image = cv::imread(file_name, cv::IMREAD_UNCHANGED);
    cv::Mat image = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return;
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    int width = image.cols;
    int height = image.rows;
    int size = image.rows * image.cols * image.channels();

    input = new unsigned char[size];
    output = new unsigned char[size];
    std::memcpy(input, image.data, size); 
    std::memcpy(output, input, size); 

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[y * width + x] = input[y * width + x] / 2;
        }
    }
/*     cv::Mat image_out = image.clone();
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            uchar pixel = image.at<uchar>(x, y);
            image_out.at<uchar>(x,y) = pixel / 2;
        }
    } */
    cv::Mat input_image(height, width, CV_8UC1, input);
    cv::Mat image_out(height, width, CV_8UC1, output);
 
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", 500, 500);
    cv::imshow("Input", input_image);

    cv::namedWindow("Reduced noise", cv::WINDOW_NORMAL);
    cv::resizeWindow("Reduced noise", 500, 500);
    cv::imshow("Reduced noise", image_out);

    cv::waitKey(0);
    return;
}
