#ifndef _IMAGEPROCESSING_KERNEL
#define _IMAGEPROCESSING_KERNEL

//#include <helper_math.h>
//#include <helper_functions.h>
#include <cstdio>
#include "../include/timer.h"


///**************** CUDA useful functiions *****************///
/// Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
    printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
    printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
exit(EXIT_FAILURE);}}


__global__ void invert(unsigned char *d_inred, unsigned char *d_ingreen, unsigned char *d_inblue, 
    unsigned char *d_outred, unsigned char *d_outgreen, unsigned char *d_outblue, int imageSize) {  
	// Global thread index
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	    	  
	if(threadID < imageSize) {
		d_outred[threadID] = (unsigned char) 255 - d_inred[threadID];        
        d_outgreen[threadID] = (unsigned char) 255 - d_ingreen[threadID];
        d_outblue[threadID] = (unsigned char) 255 - d_inblue[threadID];       
	}
}

__global__ void grayscale(unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outgs, int imageSize) {  
	// Global thread index
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	    	  
	if(threadID < imageSize) {
        d_outgs[threadID] = 0.21*d_inred[threadID] + 0.72*d_ingreen[threadID] + 0.07*d_inblue[threadID];		
	}
}

extern "C" void  executeKernelInvert( 
	unsigned char* h_outred, unsigned char* h_outgreen, unsigned char* h_outblue,
	unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outred, unsigned char* d_outgreen, unsigned char* d_outblue,
	int imageSize, size_t sizePixelsArray)
{   
    /// We're working with 1D size for blocks and grids

    /// Get the maximun block size from our device 
    /*cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);*/
    //cudaDevProp prop;
    //int threadsPerBlock = prop.maxThreadsPerBlock;
    int threadsPerBlock = 128;
	printf("MaxThreadsPerBlock:  %d \n", threadsPerBlock);
	
    int gridSize = (imageSize + threadsPerBlock-1)/threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock);

    /*printf("Device in\n");
    for(int i = 0; i < imageSize ; i++){
        printf("%d %d %d\n", (int)d_inred[i], (int)d_ingreen[i], (int)d_inblue[i] ); 
    }

    printf("Device out\n");
    for(int i = 0; i < imageSize ; i++){
        printf("%d %d %d\n", (int)d_outred[i], (int)d_outgreen[i], (int)d_outblue[i] ); 
    }*/
    
    invert<<<gridSize, threadsPerBlock>>>(d_inred, d_ingreen, d_inblue, d_outred, d_outgreen, d_outblue, imageSize);

    CUDA_CALL(cudaMemcpy(h_outred, d_outred, sizePixelsArray,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_outgreen, d_outgreen, sizePixelsArray,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_outblue, d_outblue, sizePixelsArray,cudaMemcpyDeviceToHost));

    /*printf("\nAfter\n");
    for(int i = 0; i < imageSize ; i++){
        printf("%d %d %d\n", h_outred[i], h_outgreen[i], h_outblue[i] ); 
    }*/
}

extern "C" void  executeKernelGrayScale( 
	unsigned char* h_outgs, unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outgs, int imageSize, size_t sizePixelsArray){

    int threadsPerBlock = 128;
	printf("MaxThreadsPerBlock:  %d \n", threadsPerBlock);
	
    int gridSize = (imageSize + threadsPerBlock-1)/threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock);

    grayscale<<<gridSize, threadsPerBlock>>>(d_inred, d_ingreen, d_inblue, d_outgs, imageSize);

    CUDA_CALL(cudaMemcpy(h_outgs, d_outgs, sizePixelsArray,cudaMemcpyDeviceToHost));    
}

/*extern "C" void  executeKernelBinary( 
    unsigned char* h_outgs, unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outgs, int imageSize, size_t sizePixelsArray){

    int threadsPerBlock = 128;
    printf("MaxThreadsPerBlock:  %d \n", threadsPerBlock);
        
    int gridSize = (imageSize + threadsPerBlock-1)/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock);
    
    grayscale<<<gridSize, threadsPerBlock>>>(d_inred, d_ingreen, d_inblue, d_outgs, imageSize);
    binary<<<gridSize, threadsPerBlock>>>(d_outgs, d_outbinary, imageSize);
}*/

#endif