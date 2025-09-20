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

    auto compute_stddev = [](const vector<float>& times, float mean) {
        float sum = 0;
        for (float t : times) sum += (t - mean) * (t - mean);
        return sqrt(sum / times.size());
    };

    float std_cpu = compute_stddev(cpu_speed, avg_cpu);
    float std_gpu = compute_stddev(gpu_speed, avg_gpu);
    

    cout << std::setprecision(3) << "\n" << string(15,'=') << " Benchmark results for " << runs << " runs with block size " 
         << BLOCK_SIZE << " and grid size " << (2*grid)+1 << "x" << (2*grid)+1 << " " << string(15,'=') << endl;
    cout << "Average CPU: (" << total_cpu / runs << " +/- " << std_cpu << ") ms" << endl;
    cout << "Average GPU: (" << total_gpu / runs << " +/- " << std_gpu << ") ms" << endl;
    cout << "GPU speedup: x" << avg_cpu / avg_gpu << endl;
    cout << string(79, '=');
        
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