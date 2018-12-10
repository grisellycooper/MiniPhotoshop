#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////////////////
/// This class deals specifically with BMP files            ///
///////////////////////////////////////////////////////////////

//** BMP FILE HEADER **//
struct BMPSignature
{
    unsigned char data[2];      //contain 'BM' string
    BMPSignature() { data[0] = data[1] = 0; }
};

struct BMPHeader
{
    unsigned int fileSize;      //size in bytes
    unsigned short reserved1;   //reserved; must be 0
    unsigned short reserved2;   //reserved; must be 0
    unsigned int dataOffset;    //the offset in bytes from the BMPHeader to the bitmap bits

    BMPHeader(): fileSize(0),  reserved1(0), reserved2(0), dataOffset(0) { }
}; 

//** FILE HEADER **//
struct InfoHeader
{
    unsigned int    bmpSize;             /// number of bytes required
    unsigned int    bmpWidth;            /// width in pixels
    unsigned int    bmpHeight;           /// height in pixels
                                         /// if positive, it is a bottom-up DIB and its origin is the lower left corner
                                         /// if negative, it is a top-down DIB and its origin is the upper left corner
    unsigned short  bmpPlanes;           /// number of color planes; must be 1
    unsigned short  bmpBitCount;         /// number of bit per pixel; must be 1, 4, 8, 16, 24, or 32
    unsigned int    bmpCompression;      /// type of compression
    unsigned int    bmpSizeImage;        /// size of image in bytes
    int             bmpXPelsPerMeter;    /// number of pixels per meter in x axis
    int             bmpYPelsPerMeter;    /// number of pixels per meter in y axis
    
    ///both set to 256, while 16 and 24-bit images will set them to 0
    unsigned int    bmpClrUsed;          /// number of colors used
    unsigned int    bmpClrImportant;     /// number of colors that are important
    InfoHeader(): bmpSize(0), bmpWidth(0), bmpHeight(0), bmpPlanes(0), bmpBitCount(0), bmpCompression(0), bmpSizeImage(0), bmpXPelsPerMeter(0), bmpYPelsPerMeter(0), bmpClrUsed(0), bmpClrImportant(0) { }
};

//** PIXEL STRUCTURE **//
struct Pixel
{
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

class Image
{
    private:
        BMPSignature signature;                 /// BMP File Info 
        BMPHeader fileHeader;
        InfoHeader infoHeader; 
        int width;                              /// Image features
        int height;
        int depth; 
        unsigned char* reds;                    /// Image data
        unsigned char* greens;
        unsigned char* blues;
        int histoR[256];
        int histoG[256];
        int histoB[256];
        cv::Mat originalImage;     
   
    public:
        Image(std::string name);                     /// To read
        Image(int _width, int _height);         /// To write    
        ~Image();
        
        void ReadHeader(std::ifstream &fin, BMPSignature &sig, BMPHeader &header);
        void ReadInfoHeader(std::ifstream &fin, InfoHeader &infoH);
        int getImageSize();
        int getImageWidth();
        int getImageHeight();
        void getImageHistograms(); 
        void getRGBs(unsigned char* _reds, unsigned char* _greens, unsigned char* _blues);               
        void saveImage(std::string savePath);
        void showImage();
        void showImage(unsigned char* img);
        void showImage(unsigned char *_reds, unsigned char *_greens, unsigned char *_blues);
        void showHistogram();
        int equalization(unsigned char* outred, unsigned char* outgreen, unsigned char* outblue); 
        int gamma(unsigned char* outred, unsigned char* outgreen, unsigned char* outblue, float gamma);        
        int grayScale(unsigned char* out);
        int binary(unsigned char* inGS, unsigned char* out, int threshold);                
        int sobel(unsigned char* inGS, unsigned char* out);
        int maximo(unsigned char* outred, unsigned char* outgreen, unsigned char* outblue, int k);               
};
