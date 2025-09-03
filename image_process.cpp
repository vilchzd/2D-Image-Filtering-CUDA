#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    //cv::Mat image = cv::imread("a.png", cv::IMREAD_UNCHANGED);
    cv::Mat image = cv::imread("penguin.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return 1;
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;
    
    cv::Mat image_out = image.clone();


    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            uchar pixel = image.at<uchar>(x, y);
            image_out.at<uchar>(x,y) = pixel / 2;
        }
    }

    cv::imwrite("dimmed.png", image_out); // save result
    cv::imshow("Dimmed", image_out);
    cv::waitKey(0);
}
