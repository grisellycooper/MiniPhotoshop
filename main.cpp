#include <iostream>
#include <string>
#include <fstream>
#include "include/image.h"

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