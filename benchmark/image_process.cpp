#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "image_process.h"

using namespace cv;
using namespace std;

void image_process(const string& file_name, unsigned char*& input, unsigned char*& output, int& width, int& height, int& channels) {
    
    Mat image = imread(file_name, IMREAD_UNCHANGED);
    if (image.empty()) {
        cerr << "Failed to load image!" << endl;
        exit(1);
    }

    cout << "Image size: " << image.cols << "x" << image.rows << ", Channels: " << image.channels() << endl;
    if (channels == 1 && image.channels() != 1) {
        cout << "Converted "<< image.channels() << "-channel image to 1-channel GRAY" << endl;
        cvtColor(image, image, (image.channels() == 4) ? COLOR_BGRA2GRAY : COLOR_BGR2GRAY);
        
    }  else if ((channels == 3 || channels == 4) && image.channels() != 3) {
        cout << "Converted "<< image.channels() << "-channel image to 3-channel BGR" << endl;
        cvtColor(image, image, (image.channels() == 4) ? COLOR_BGRA2BGR : COLOR_GRAY2BGR);
        
    } else if ((channels == 1 && image.channels() == 1) || (channels == 3 && image.channels() == 3)) {
        cout << "Image loadead in "<< image.channels() << "-channels" << endl;

    } else {
        cerr << "Unsupported image format: " << channels << endl;
        exit(1);
    }
 
    channels = image.channels();
    width = image.cols;
    height = image.rows;
    int size = image.rows * image.cols * image.channels();
    input = new unsigned char[size];
    output = new unsigned char[size];
    memcpy(input, image.data, size); 
    memcpy(output, input, size); 
    
}

void cpu_blurGRAY(unsigned char*& input, unsigned char*& output, int width, int height, int grid) {

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
} 
    
void cpu_blurBGR(unsigned char*& input, unsigned char*& output, int width, int height, int grid) {

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
} 

void show_image(unsigned char*& input, unsigned char*& output, int width, int height, int channels) {

    int type = CV_8UC(channels);
    cv::Mat input_image(height, width, type, input);
    cv::Mat image_out(height, width, type, output);
    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", 500, 500);
    cv::imshow("Input", input_image);
    cv::namedWindow("Blur", cv::WINDOW_NORMAL);
    cv::resizeWindow("Blur", 500, 500);
    cv::imshow("Blur", image_out);

}

float stddev(const vector<float>& times, float mean) {

    float sum = 0;
    for (float t : times) {
        sum += (t - mean) * (t - mean);
    }
    return sqrt(sum / times.size());

}
