#include <iostream>
#include <string>
#include "include/image.h"

///**************** CUDA useful functiions *****************///
/// Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
    printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
    printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
exit(EXIT_FAILURE);}}


///////////////////////////////////////////////////////////////
/// GPU functions to launch kernels                         ///
///////////////////////////////////////////////////////////////
extern "C" void  executeKernelTransfGamma( 
	unsigned char* h_inred, unsigned char* h_ingreen, unsigned char* h_inblue,
    unsigned char* h_outred, unsigned char* h_outgreen, unsigned char* h_outblue,
	unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outred, unsigned char* d_outgreen, unsigned char* d_outblue,
	float *gamma);


///////////////////////////////////////////////////////////////
/// CPU functions                                           ///
///////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////
/// Main function                                           ///
///////////////////////////////////////////////////////////////
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

    //** GrayScale **//
    unsigned char *gs = new unsigned char[image->getImageSize()];
    start = clock();
    image->grayScale(gs);
    end = clock();
    std::cout<<"Converting to GrayScale: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(gs);    

    //** Sobel Filter / Detector de bordes **//
    unsigned char *sobel = new unsigned char[image->getImageSize()];
    start = clock();
    image->sobel(gs, sobel);
    end = clock();
    std::cout<<"Sobel Filtering: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    image->showImage(sobel);

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

    //********* CUDA things **********//
    /// init device
	//cudaSetDevice(0);
	//cudaDeviceSynchronize();
	//cudaThreadSynchronize();
    
    //** Transformacion Gamma **//
    
    int sizeImage = image->getImageSize();
    size_t sizePixelsArray = sizeImage * sizeof(unsigned char);

    /// Allocate memory           
    /// Host: Initial RGB values. Output RGB values
    unsigned char *h_inred = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_ingreen = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_inblue = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_outred = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_outgreen = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_outblue = (unsigned char *)malloc(sizePixelsArray);    
    float *h_gamma = (float *)malloc(sizeof(float));

    image->getRGBs(h_inred, h_ingreen, h_inblue);

    /// Device: Initial RGB values. Output RGB values
	unsigned char *d_inred, *d_ingreen, *d_inblue;
    unsigned char *d_outred, *d_outgreen, *d_outblue;
    float *d_gamma;

    CUDA_CALL(cudaMalloc((void**) &d_inred, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_ingreen, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_inblue, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_outred, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_outgreen, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_outblue, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_gamma, sizeof(float)));
    
    /// Copy host to device
    CUDA_CALL(cudaMemcpy(d_inred, h_inred, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_ingreen, h_ingreen, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_inblue, h_inblue, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_outred, h_outred, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_outgreen, h_outgreen, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_outblue, h_outblue, cudaMemcpyHostToDevice));

    /// Execute Kernel


    /// Free space
    free(h_inred);
    free(h_ingreen);
    free(h_inblue);
    free(h_outred);
    free(h_outgreen);
    free(h_outblue);

    CUDA_CALL(cudaFree(d_inred));
    CUDA_CALL(cudaFree(d_ingreen));
    CUDA_CALL(cudaFree(d_inblue));
    CUDA_CALL(cudaFree(d_outred));
    CUDA_CALL(cudaFree(d_outgreen));
    CUDA_CALL(cudaFree(d_outblue));
}