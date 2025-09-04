#include <iostream>
#include <string>
#include "image_process.h"

using namespace std;

int main() {
    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    string file_name = "C:\\Users\\dievi\\Desktop\\2D-Image-Filtering-CUDA\\sylva.png";
    image_process(file_name, input, output); 
    delete[] input;
    delete[] output;
    return 0;
}