#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "image_process.h"

using namespace std;
using namespace chrono;

int main() {

    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    int width, height;
    int target_channels = 3;
    int grid = MAX_GRID_SIZE;
    string file_name = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\love.png";

    image_process(file_name, input, output, width, height, target_channels);
    std::cout << std::fixed << std::setprecision(2) << "Preforming " 
              << ((1.0 * width * height * ((2 * grid) * (2 * grid) + 1) * target_channels) / 1'000'000'000.0)
              << " billion operations..." << std::endl; 

    int j = 0;
    for (int i = 0; i < 10; i++) {  

        auto start_1 = high_resolution_clock::now();
        (target_channels == 1 ? cpu_blurGRAY : cpu_blurBGR)(input, output, width, height, grid);
        auto stop_1 = high_resolution_clock::now();
        
        auto duration_1 = duration_cast<milliseconds>(stop_1 - start_1);
        cout << "Time elapsed on cpu execution: " << duration_1.count() << " ms" << endl;

        auto start = high_resolution_clock::now();
        (target_channels == 1 ?  gpu_wrapper_blurGRAY :  gpu_wrapper_blurBGR)(input, output, width, height, grid);
        auto stop = high_resolution_clock::now();
        
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Time elapsed on gpu execution: " << duration.count() << " ms" << endl;
        j =+ duration.count();
        //cout << "GPU speedup is x" << duration_1.count() / duration.count() << " times faster than CPU"<< endl;
    }

    cout << "Average gpu" << j/10 << endl;
    //show_image(input, output, width, height, target_channels); 
    //cv::waitKey(0);
    
    delete[] input;
    delete[] output;
    return 0;
    
}

/* 
const int runs = 15;
std::vector<int> grid_sizes = {8, 16, 32, 48};

for (int g : grid_sizes) {
    cout << "\n=== Benchmarking grid size = " << g << " ===" << endl;

    long long total_cpu = 0, total_gpu = 0;

    for (int i = 0; i < runs; i++) {
        auto start_1 = high_resolution_clock::now();
        (target_channels == 1 ? cpu_blurGRAY : cpu_blurBGR)(input, output, width, height, g);
        auto stop_1 = high_resolution_clock::now();
        auto duration_1 = duration_cast<milliseconds>(stop_1 - start_1);
        total_cpu += duration_1.count();

        auto start = high_resolution_clock::now();
        (target_channels == 1 ? gpu_wrapper_blurGRAY : gpu_wrapper_blurBGR)(input, output, width, height, g);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        total_gpu += duration.count();
    }

    cout << "Average CPU time: " << (total_cpu / runs) << " ms\n";
    cout << "Average GPU time: " << (total_gpu / runs) << " ms\n";
    cout << "Average speedup: x" << (total_cpu / total_gpu) << "\n";
}
*/