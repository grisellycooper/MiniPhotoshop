#include <iostream>
#include <string>
#include "include/image.h"

int main(int argc, char* argv[]){
    /// Time counting
	clock_t start, end;
    double globalTime = 0.0;
    
    /// Read & Write image path
    std::string inputImagePath, outputImagePath;
    
    inputImagePath = argv[1];           /// Input path        
    //inputImagePath = "../media/cabana.bmp";
    
    /// Read image
    start = clock();
    Image *image(new Image(inputImagePath));
    end = clock();
    std::cout<<"Reading file: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;

    /// Display input image
    image->showImage();

    /// get Image Histogram
    start = clock();
    image->getImageHistograms();
    end = clock();
    std::cout<<"Getting RGB Histograms: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    

    //image->showHistogram();
    
    //** GrayScale **//
    /*unsigned char *gs = new unsigned char[image->getImageSize()];
    start = clock();
    image->grayScale(gs);
    end = clock();
    std::cout<<"Converting to GrayScale: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(gs);    */

    //** Gamma **//
    /*float gamma = 4.0f;
    unsigned char *gred = new unsigned char[image->getImageSize()];    
    unsigned char *ggreen = new unsigned char[image->getImageSize()];    
    unsigned char *gblue = new unsigned char[image->getImageSize()];    
    start = clock();
    image->gamma(gred, ggreen, gblue, gamma);
    end = clock();
    std::cout<<"Gamma: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(gred, ggreen, gblue);  */

    //** Binary **//
    /*int threshold = 120;
    unsigned char *binary = new unsigned char[image->getImageSize()];    
    start = clock();
    image->binary(gs, binary, threshold);
    end = clock();
    std::cout<<"Binary: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(binary);  */

    
    //** Sobel Filter / Detector de bordes **//
    /*unsigned char *sobel = new unsigned char[image->getImageSize()];
    start = clock();
    image->sobel(gs, sobel);
    end = clock();
    std::cout<<"Sobel Filtering: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(sobel);*/

    //** Maximun Filter **//
    /*unsigned char *max_red = new unsigned char[image->getImageSize()];    
    unsigned char *max_green = new unsigned char[image->getImageSize()];    
    unsigned char *max_blue = new unsigned char[image->getImageSize()];    
    int k = 6; 
    start = clock();
    image->maximo(max_red, max_green, max_blue, k);
    end = clock();
    std::cout<<"Max Filter: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(max_red, max_green, max_blue);   */

    //** Equalization **//
    
}