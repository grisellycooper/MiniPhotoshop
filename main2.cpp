#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "include/image.h"

#define bCuda = true

///**************** CUDA useful functiions *****************///
/// Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
    printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
    printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
exit(EXIT_FAILURE);}}


///////////////////////////////////////////////////////////////
/// GPU functions to launch kernels                         ///
///////////////////////////////////////////////////////////////
extern "C" void  executeKernelInvert( 
	unsigned char* h_outred, unsigned char* h_outgreen, unsigned char* h_outblue,
	unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outred, unsigned char* d_outgreen, unsigned char* d_outblue,
	int imageSize, size_t sizePixelsArray);

extern "C" void  executeKernelGrayScale( 
	unsigned char* h_outgs, unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outgs, int imageSize, size_t sizePixelsArray);

extern "C" void  executeKernelBinary( 
	unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue, unsigned char* h_outbinary, 
    unsigned char* d_outbinary, int imageSize, size_t sizePixelsArray, int threshold);


///////////////////////////////////////////////////////////////
/// CPU functions                                           ///
///////////////////////////////////////////////////////////////
void cudaInvert(Image *image){
    //********* CUDA things **********//
    /// init device
	//cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();
        
    int imageSize = image->getImageSize();
    size_t sizePixelsArray = imageSize * sizeof(unsigned char);

    /// Allocate memory           
    /// Host: Initial RGB values. Output RGB values
    unsigned char *h_inred = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_ingreen = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_inblue = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_outred = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_outgreen = (unsigned char *)malloc(sizePixelsArray);
    unsigned char *h_outblue = (unsigned char *)malloc(sizePixelsArray);    

    image->getRGBs(h_inred, h_ingreen, h_inblue);

    /*printf("\n -host\n");
    for(int i = 0; i < 5 ; i++){
        printf("%d %d %d\n", (int)h_inred[i], (int)h_ingreen[i], (int)h_inblue[i] ); 
    }*/

    /// Device: Initial RGB values. Output RGB values
	unsigned char *d_inred, *d_ingreen, *d_inblue;
    unsigned char *d_outred, *d_outgreen, *d_outblue;
    //float *d_gamma;

    CUDA_CALL(cudaMalloc((void**) &d_inred, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_ingreen, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_inblue, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_outred, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_outgreen, sizePixelsArray));
    CUDA_CALL(cudaMalloc((void**) &d_outblue, sizePixelsArray));
    //CUDA_CALL(cudaMalloc((void**) &d_gamma, sizeof(float)));
    
    /// Copy host to device
    CUDA_CALL(cudaMemcpy(d_inred, h_inred, sizePixelsArray, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_ingreen, h_ingreen, sizePixelsArray, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_inblue, h_inblue, sizePixelsArray, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_outred, h_outred, sizePixelsArray, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_outgreen, h_outgreen, sizePixelsArray, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_outblue, h_outblue, sizePixelsArray, cudaMemcpyHostToDevice));
    
    /// Execute Kernel
    executeKernelInvert(h_outred, h_outgreen, h_outblue, d_inred, d_ingreen, d_inblue, 
                        d_outred, d_outgreen, d_outblue, imageSize, sizePixelsArray);

    image->showImage(h_outred, h_outgreen, h_outblue); 
    
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

// void cudaGrayScale(Image *image){
//     //********* CUDA things **********//
//     /// init device
// 	//cudaSetDevice(0);
// 	cudaDeviceSynchronize();
// 	cudaThreadSynchronize();
        
//     int imageSize = image->getImageSize();
//     size_t sizePixelsArray = imageSize * sizeof(unsigned char);
//     float gamma = 4.0f;

//     /// Allocate memory           
//     /// Host: Initial RGB values. Output RGB values
//     unsigned char *h_inred = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_ingreen = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_inblue = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_outred = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_outgreen = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_outblue = (unsigned char *)malloc(sizePixelsArray);    
//     //float *h_gamma = (float *)malloc(sizeof(float));

//     image->getRGBs(h_inred, h_ingreen, h_inblue);

//     /*printf("\n -host\n");
//     for(int i = 0; i < 5 ; i++){
//         printf("%d %d %d\n", (int)h_inred[i], (int)h_ingreen[i], (int)h_inblue[i] ); 
//     }*/

//     /// Device: Initial RGB values. Output RGB values
// 	unsigned char *d_inred, *d_ingreen, *d_inblue;
//     unsigned char *d_outred, *d_outgreen, *d_outblue;
//     //float *d_gamma;

//     CUDA_CALL(cudaMalloc((void**) &d_inred, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_ingreen, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_inblue, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_outred, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_outgreen, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_outblue, sizePixelsArray));
//     //CUDA_CALL(cudaMalloc((void**) &d_gamma, sizeof(float)));
    
//     /// Copy host to device
//     CUDA_CALL(cudaMemcpy(d_inred, h_inred, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_ingreen, h_ingreen, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_inblue, h_inblue, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_outred, h_outred, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_outgreen, h_outgreen, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_outblue, h_outblue, sizePixelsArray, cudaMemcpyHostToDevice));
    
//     /// Execute Kernel
//     executeKernelTransfGamma(h_inred, h_ingreen, h_inblue, h_outred, h_outgreen, h_outblue, 
//         d_inred, d_ingreen, d_inblue, d_outred, d_outgreen, d_outblue, gamma, imageSize, sizePixelsArray);

//     image->showImage(h_outred, h_outgreen, h_outblue); 
//     /// Free space
//     free(h_inred);
//     free(h_ingreen);
//     free(h_inblue);
//     free(h_outred);
//     free(h_outgreen);
//     free(h_outblue);

//     CUDA_CALL(cudaFree(d_inred));
//     CUDA_CALL(cudaFree(d_ingreen));
//     CUDA_CALL(cudaFree(d_inblue));
//     CUDA_CALL(cudaFree(d_outred));
//     CUDA_CALL(cudaFree(d_outgreen));
//     CUDA_CALL(cudaFree(d_outblue));
// }

// void cudaBinary(Image *image){
//     //********* CUDA things **********//
//     /// init device
// 	//cudaSetDevice(0);
// 	cudaDeviceSynchronize();
// 	cudaThreadSynchronize();
        
//     int imageSize = image->getImageSize();
//     size_t sizePixelsArray = imageSize * sizeof(unsigned char);
//     int threshold = 120;

//     /// Allocate memory           
//     /// Host: Initial RGB values. Output RGB values
//     unsigned char *h_inred = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_ingreen = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_inblue = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_outred = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_outgreen = (unsigned char *)malloc(sizePixelsArray);
//     unsigned char *h_outblue = (unsigned char *)malloc(sizePixelsArray);    
//     //float *h_gamma = (float *)malloc(sizeof(float));

//     image->getRGBs(h_inred, h_ingreen, h_inblue);

//     /*printf("\n -host\n");
//     for(int i = 0; i < 5 ; i++){
//         printf("%d %d %d\n", (int)h_inred[i], (int)h_ingreen[i], (int)h_inblue[i] ); 
//     }*/

//     /// Device: Initial RGB values. Output RGB values
// 	unsigned char *d_inred, *d_ingreen, *d_inblue;
//     unsigned char *d_outred, *d_outgreen, *d_outblue;
//     //float *d_gamma;

//     CUDA_CALL(cudaMalloc((void**) &d_inred, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_ingreen, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_inblue, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_outred, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_outgreen, sizePixelsArray));
//     CUDA_CALL(cudaMalloc((void**) &d_outblue, sizePixelsArray));
//     //CUDA_CALL(cudaMalloc((void**) &d_gamma, sizeof(float)));
    
//     /// Copy host to device
//     CUDA_CALL(cudaMemcpy(d_inred, h_inred, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_ingreen, h_ingreen, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_inblue, h_inblue, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_outred, h_outred, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_outgreen, h_outgreen, sizePixelsArray, cudaMemcpyHostToDevice));
//     CUDA_CALL(cudaMemcpy(d_outblue, h_outblue, sizePixelsArray, cudaMemcpyHostToDevice));
    
//     /// Execute Kernel
//     executeKernelTransfGamma(h_inred, h_ingreen, h_inblue, h_outred, h_outgreen, h_outblue, 
//         d_inred, d_ingreen, d_inblue, d_outred, d_outgreen, d_outblue, gamma, imageSize, sizePixelsArray);

//     image->showImage(h_outred, h_outgreen, h_outblue); 
//     /// Free space
//     free(h_inred);
//     free(h_ingreen);
//     free(h_inblue);
//     free(h_outred);
//     free(h_outgreen);
//     free(h_outblue);

//     CUDA_CALL(cudaFree(d_inred));
//     CUDA_CALL(cudaFree(d_ingreen));
//     CUDA_CALL(cudaFree(d_inblue));
//     CUDA_CALL(cudaFree(d_outred));
//     CUDA_CALL(cudaFree(d_outgreen));
//     CUDA_CALL(cudaFree(d_outblue));
// }

///////////////////////////////////////////////////////////////
/// Main function                                           ///
///////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
    /// Time counting
	clock_t start, end;
    
    std::string inputImagePath;                 /// Input image path
    int operation;                              /// Type of operation will be applied
    
    operation = 0;                              /// Default value 
    if(argc > 1){
        /// Get option
        if(argv[1] != NULL)
            operation = atoi(argv[1]);           /// Getting from command line                       
    }
    
    inputImagePath = "../media/Garfield-Portada.bmp";   /// Default input path    
    if(argc > 2){
        /// Get image input
        if(argv[2] != NULL)
            inputImagePath = argv[2];           /// Getting from command line                           
    }

    /// Check a valid input image path
    if(!std::ifstream(inputImagePath)){
        std::cout<<"File '"<<inputImagePath <<"' was not found! "<<std::endl;     
        return -1;
    }

    /// Read file
    start = clock();
    Image *image(new Image(inputImagePath));    
    end = clock();
    std::cout<<"Reading file: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    
    /// Display input image
    image->showImage();
    //image->showHistogram();
        
    /// Get Image Histogram
    start = clock();
    image->getImageHistograms();
    end = clock();
    std::cout<<"Getting RGB Histograms: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    
    /// Available operations
    /** 1 -> INVERT image colors
     *  2 -> Get GRAYSCALE image
     *  3 -> Get BINARY image (Require extra argument 'Threshold' // Default value: 120)
     *  4 -> Equalization
     *  5 -> Sobel Filter / Border detector
     *  6 -> Max Filter / Blur image (Require extra argument 'Radius' // Default value: 6)
     *  Default -> Show all operations
     **/

    /// Basic Operations are implemented in CUDA (Invert, GrayScale, Binary) 
    switch (operation)
    {
        case 1:
        {            
            //** Invert **//            
            if(bCuda){
                cudaInvert(image);
            }
            else{
                unsigned char *ired = new unsigned char[image->getImageSize()];    
                unsigned char *igreen = new unsigned char[image->getImageSize()];    
                unsigned char *iblue = new unsigned char[image->getImageSize()];    
                start = clock();
                image->invert(ired, igreen, iblue);
                end = clock();
                std::cout<<"Invert: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
                image->showImage(ired, igreen, iblue);
                delete(ired);
                delete(igreen);
                delete(iblue);
            }            
        }
        break;
    
        case 2:
        {
            //** GrayScale **//
            unsigned char *gs = new unsigned char[image->getImageSize()];
            start = clock();
            image->grayScale(gs);
            end = clock();
            std::cout<<"Converting to GrayScale: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(gs);    
            delete(gs);
        }
        break;

        case 3:
        {
            //** Binary **//
            int threshold = 120;               /// Default input path    
            
            /// Get extra argument if exists
            if(argv[3] != NULL)
                threshold = atoi(argv[3]);      /// Getting from command line               

            unsigned char *gs = new unsigned char[image->getImageSize()];                        
            unsigned char *binary = new unsigned char[image->getImageSize()];    
            start = clock();                    
            image->grayScale(gs);              /// Before Binary, get Grayscale
            image->binary(gs, binary, threshold);
            end = clock();
            std::cout<<"Binary: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(binary);
            delete(binary);
        }
        break;

        case 4:
        {
            //** Equalization **//
            unsigned char *eq_red = new unsigned char[image->getImageSize()];    
            unsigned char *eq_green = new unsigned char[image->getImageSize()];    
            unsigned char *eq_blue = new unsigned char[image->getImageSize()]; 
            start = clock();
            image->equalization(eq_red,eq_green, eq_blue);
            end = clock();
            std::cout<<"Equalization: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(eq_red, eq_green, eq_blue);
            delete(eq_red);
            delete(eq_green);
            delete(eq_blue);
        }
        break;

        case 5:
        {
            //** Sobel Filter / Border detector **//
            unsigned char *gs = new unsigned char[image->getImageSize()];
            unsigned char *sobel = new unsigned char[image->getImageSize()];
            start = clock();
            image->grayScale(gs);              /// Before, get Grayscale
            image->sobel(gs, sobel);
            end = clock();
            std::cout<<"Sobel Filtering: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(sobel);
            delete(sobel);
        }
        break;

        case 6:
        {
            //** Maximun Filter **//
            int k = 6;                          /// Default input path    
            
            /// Get extra argument if exists
            if(argv[3] != NULL)
                k = atoi(argv[3]);              /// Getting from command line      

            unsigned char *max_red = new unsigned char[image->getImageSize()];    
            unsigned char *max_green = new unsigned char[image->getImageSize()];    
            unsigned char *max_blue = new unsigned char[image->getImageSize()];    
            
            start = clock();
            image->maximo(max_red, max_green, max_blue, k);
            end = clock();
            std::cout<<"Max Filter: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(max_red, max_green, max_blue);   
            delete(max_red);
            delete(max_green);
            delete(max_blue);
        }
        break;

        default:
        {
            //** Show all operations **//
            //** Invert **//            
            unsigned char *ired = new unsigned char[image->getImageSize()];    
            unsigned char *igreen = new unsigned char[image->getImageSize()];    
            unsigned char *iblue = new unsigned char[image->getImageSize()];    
            start = clock();
            image->invert(ired, igreen, iblue);
            end = clock();
            std::cout<<"Invert: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(ired, igreen, iblue);
            delete(ired);
            delete(igreen);
            delete(iblue);

            //** GrayScale **//
            unsigned char *gs = new unsigned char[image->getImageSize()];
            start = clock();
            image->grayScale(gs);
            end = clock();
            std::cout<<"Converting to GrayScale: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(gs);   

            //** Binary **//
            int threshold = 120;               /// Default value    
            unsigned char *binary = new unsigned char[image->getImageSize()];    
            start = clock();                    
            image->binary(gs, binary, threshold);
            end = clock();
            std::cout<<"Binary: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(binary);
            delete(binary);     

            //** Equalization **//
            unsigned char *eq_red = new unsigned char[image->getImageSize()];    
            unsigned char *eq_green = new unsigned char[image->getImageSize()];    
            unsigned char *eq_blue = new unsigned char[image->getImageSize()]; 
            start = clock();
            image->equalization(eq_red,eq_green, eq_blue);
            end = clock();
            std::cout<<"Equalization: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(eq_red, eq_green, eq_blue);
            delete(eq_red);
            delete(eq_green);
            delete(eq_blue);

            //** Sobel Filter / Border Detector **//
            unsigned char *sobel = new unsigned char[image->getImageSize()];
            start = clock();
            image->sobel(gs, sobel);
            end = clock();
            std::cout<<"Sobel Filtering: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(sobel);
            delete(sobel);

            //** Maximun Filter **//
            int k = 6;                          /// Default value                
            unsigned char *max_red = new unsigned char[image->getImageSize()];    
            unsigned char *max_green = new unsigned char[image->getImageSize()];    
            unsigned char *max_blue = new unsigned char[image->getImageSize()]; 
            start = clock();
            image->maximo(max_red, max_green, max_blue, k);
            end = clock();
            std::cout<<"Max Filter: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
            image->showImage(max_red, max_green, max_blue);   
            delete(max_red);
            delete(max_green);
            delete(max_blue);        
        }
        break;
    }   
}