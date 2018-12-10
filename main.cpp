#include <iostream>
#include "include/image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main(int argc, char* argv[]){
    /// Time counting
	clock_t start, end;
    double globalTime = 0.0;
    
    /// Read & Write image path
    std::string inputImagePath, outputImagePath;
    
    //inputImagePath = argv[1];           /// Input path        
    inputImagePath = "../media/Garfield-Portada.bmp";
    /// Read image
    start = clock();
    Image *image(new Image(inputImagePath));
    end = clock();
    std::cout<<"Reading file: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;

    image->showImage();
    //image->showHistogram();

    //unsigned char *gs = new unsigned char[image->getImageSize()];
    //start = clock();
    //image->grayScale(gs);
    //end = clock();
    //std::cout<<"Converting to GrayScale: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    //image->showImage(gs);    

    //** Sobel Filter / Detector de bordes **//
    //unsigned char *sobel = new unsigned char[image->getImageSize()];
    //unsigned char sobel[image->getImageSize()] = {0};
    //start = clock();
    //image->sobel(gs, sobel);
    //end = clock();
    //std::cout<<"Sobel Filtering: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    //image->showImage(sobel);

    //** Maximun Filter **//
    unsigned char *max_red = new unsigned char[image->getImageSize()];    
    unsigned char *max_green = new unsigned char[image->getImageSize()];    
    unsigned char *max_blue = new unsigned char[image->getImageSize()];    
    int k = 6; 
    start = clock();
    image->maximo(max_red, max_green, max_blue, k);
    end = clock();
    std::cout<<"Max Filter: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(max_red, max_green, max_blue);   
}

/*#include <QApplication>
#include <QPushButton>

int main(int argc, char **argv)
{
    QApplication app (argc, argv);

    QPushButton button ("Hello world !");
    button.show();

    return app.exec();
}*/
