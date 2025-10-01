#include <iostream>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>

#include "image_process.h"

using namespace std;
using namespace chrono;

int main() {

    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    int target_channels = TARGET_CHANNELS;
    int grid = GRID_SIZE;
    int width, height;
    
    string file_name = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\media\\tiger.png";

    image_process(file_name, input, output, width, height, target_channels);
    cout << fixed << setprecision(2) << "Preforming " 
              << ((1.0 * width * height * ((2 * grid) * (2 * grid) + 1) * target_channels) / 1'000'000'000.0)
              << " billion operations using a " << (2*grid)+1 << "x" << (2*grid)+1 << " blur kernel" << endl; 

    cout << std::setprecision(3) << "Computing benchmark results for block size " 
         << BLOCK_SIZE << " and a " << (2*grid)+1 << "x" << (2*grid)+1 << " grid size" << endl;

    //----------------------------------------GPU WARMUP-------------------------------------------------//
    (target_channels == 1 ?  gpu_wrapper_blurGRAY :  gpu_wrapper_blurBGR)(input, output, width, height, grid);

    vector<float> cpu_speed, gpu_speed;
    int runs = 2;

    for (int i = 0; i < runs; i++) {  

        auto start_cpu = high_resolution_clock::now();
        (target_channels == 1 ? cpu_blurGRAY : cpu_blurBGR)(input, output, width, height, grid);
        auto stop_cpu = high_resolution_clock::now();
        
        float ms_cpu = chrono::duration<float, milli>(stop_cpu - start_cpu).count();
        cpu_speed.push_back(ms_cpu);
        
        auto start_gpu = high_resolution_clock::now();
        (target_channels == 1 ?  gpu_wrapper_blurGRAY :  gpu_wrapper_blurBGR)(input, output, width, height, grid);
        auto stop_gpu = high_resolution_clock::now();
        
        float ms_gpu = chrono::duration<float, milli>(stop_gpu - start_gpu).count();
        gpu_speed.push_back(ms_gpu);

    }

    //---------------------End to end GPU preformance------------------------------//

    // Compute totals and averages
    float total_cpu = accumulate(cpu_speed.begin(), cpu_speed.end(), 0.0f);
    float total_gpu = accumulate(gpu_speed.begin(), gpu_speed.end(), 0.0f);
    float avg_cpu = total_cpu / cpu_speed.size();
    float avg_gpu = total_gpu / gpu_speed.size();

    float std_cpu = stddev(cpu_speed, avg_cpu);
    float std_gpu = stddev(gpu_speed, avg_gpu);
    

    cout << std::setprecision(3) << "\n" << string(15,'=') << " Benchmark results for " << runs << " runs with block size " 
         << BLOCK_SIZE << " and grid size " << (2*grid)+1 << "x" << (2*grid)+1 << " " << string(15,'=') << endl;
    cout << "Average CPU: (" << total_cpu / runs << " +/- " << std_cpu << ") ms" << endl;
    cout << "Average GPU: (" << total_gpu / runs << " +/- " << std_gpu << ") ms" << endl;
    cout << "GPU speedup: x" << avg_cpu / avg_gpu << endl;
    cout << string(97, '=');
        
    
    delete[] input;
    delete[] output;
    return 0;
    
}

