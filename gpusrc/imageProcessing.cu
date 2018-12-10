#ifndef _IMAGEPROCESSING_KERNEL
#define _IMAGEPROCESSING_KERNEL

//#include <helper_math.h>
//#include <helper_functions.h>
#include <cstdio>
#include "timer.h"


///**************** CUDA useful functiions *****************///
/// Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
    printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
    printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
exit(EXIT_FAILURE);}}


__global__ void transfGamma(unsigned char *d_inred, unsigned char *d_ingreen, unsigned char *d_inblue,
                            unsigned char *d_outred, unsigned char *d_outgreen, unsigned char *d_outblue, 
                            float gamma) {
  
	// Global thread index
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	    	  
	if(threadID < d_size) {
		d_outred[threadID] = (unsigned char) 255*powf((d_inred[threadID]/255),gamma);
        d_outgreen[threadID] = (unsigned char) 255*powf((d_ingreen[threadID]/255),gamma);
        d_outblue[threadID] = (unsigned char) 255*powf((d_inblue[threadID]/255),gamma);        
	}
}

extern "C" void  executeKernelTransfGamma( 
	unsigned char* h_inred, unsigned char* h_ingreen, unsigned char* h_inblue,
    unsigned char* h_outred, unsigned char* h_outgreen, unsigned char* h_outblue,
	unsigned char* d_inred, unsigned char* d_ingreen, unsigned char* d_inblue,
    unsigned char* d_outred, unsigned char* d_outgreen, unsigned char* d_outblue,
	float gamma, int imageSize, size_t sizePixelsArray)
{   
    /// We're working with 1D size for blocks and grids

    /// Get the maximun block size from our device 
	cudaDevProp prop;
	int threadsPerBlock = prop.maxThreadsPerBlock;
	printf("MaxThreadsPerBlock:  %d \n", threadsPerBlock);
	
    int gridSize = (imageSize + threadsPerBlock-1)/threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock);

    transfGamma<<gridSize, threadsPerBlock>>(d_inred, d_ingreen, d_inblue, d_outred, d_outgreen, d_outblue, gamma);

    CUDA_CALL(cudaMemcpy(h_outred, d_outred, sizePixelsArray,cudaMemcpyDeviceToHost));
}
