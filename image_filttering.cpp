#include <iostream>
using namespace std;

void cpu_blur(int height, int width, int* in, int* out) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int blur_sum = 0;
            int count = 0;
            int average = 0;
            for (int grid_y = -1; grid_y < 2; grid_y++) {
                for (int grid_x = -1; grid_x < 2; grid_x++) {
                    int blur_y = y + grid_y;
                    int blur_x = x + grid_x;
                    if (blur_y >= 0 && blur_x >= 0 && blur_y < height && blur_x < width) {
                        blur_sum += in[blur_y * width + blur_x];
                        count++;
                        }
                }
            }
            average = blur_sum / count;
            out[y * width + x] = average;
        }  
    }      
    for (int yo = 0; yo < height; yo++) {
        for (int xo = 0; xo < width; xo++) {
            cout << out[yo * width + xo] << " ";
        }
        cout << endl;
    }
}

int main() {
    const int width = 5, height = 5;
    int input[width * height] = {
        10, 20, 30, 40, 50,
        60, 70, 80, 90,100,
        110,120,130,140,150,
        160,170,180,190,200,
        210,220,230,240,250
    };
    int output[width * height];
    cpu_blur(height, width, input, output);
    return 0;
}
