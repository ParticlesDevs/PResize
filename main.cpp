//
// Created by eszdman on 15.02.23.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <HalideRuntime.h>
#include <HalideBuffer.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>
#include "resize.h"
using namespace Halide::Runtime;
int main(){
    int channels,x,y;
    int resizeTest = 4;
    auto dataldr = stbi_load_16("./in5.jpg",&x,&y,&channels,3);
    auto buf = Halide::Runtime::Buffer<uint16_t>::make_interleaved(dataldr,x,y,3);
    std::cout<<"resize:"<<x/resizeTest<<" "<<y/resizeTest<<std::endl;
    auto output = Halide::Runtime::Buffer<uint16_t>::make_interleaved(x/resizeTest,y/resizeTest,3);
    output = output.cropped({{0,x/resizeTest},{0,y/resizeTest}});
    resize(buf,resizeTest,0.4f,output);
    output.copy_to_host();

    auto cp = new uint16_t[(x/resizeTest)*(y/resizeTest)*3];
    memcpy(cp,output.data(),(x/resizeTest)*(y/resizeTest)*3*sizeof(uint16_t));
    stbi_write_jpg("test.jpg",x/resizeTest,y/resizeTest,3,stbi__convert_16_to_8(cp,x/resizeTest,y/resizeTest,3),95);
}