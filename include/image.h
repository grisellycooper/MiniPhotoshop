#include <string>
#include <vector>

using namespace std;

//** FILE HEADER **//
struct BMPSignature
{
    unsigned char data[2]; //equal to the string 'BM'
    BMPSignature() { data[0] = data[1] = 0; }
};

//#pragma pack(push)  // push current alignment to stack
//#pragma pack(1)     // set alignment to 1 byte boundary

struct BMPHeader
{
    unsigned int fileSize;      //size in bytes
    unsigned short reserved1;   //reserved; must be 0
    unsigned short reserved2;   //reserved; must be 0
    unsigned int dataOffset;    //the offset in bytes from the BMPHeader to the bitmap bits

    BMPHeader(): fileSize(0),  reserved1(0), reserved2(0), dataOffset(0) { }
}; 

//#define BF_TYPE 0x4D42             //BM

//** FILE HEADER **//
struct infoHeader
{
    unsigned int    bmpSize;             //number of bytes required
    unsigned int    bmpWidth;            //width in pixels
    unsigned int    bmpHeight;           //height in pixels
                                         //if positive, it is a bottom-up DIB and its origin is the lower left corner
                                         //if negative, it is a top-down DIB and its origin is the upper left corner
    unsigned short  bmpPlanes;           //number of color planes; must be 1
    unsigned short  bmpBitCount;         //number of bit per pixel; must be 1, 4, 8, 16, 24, or 32
    unsigned int    bmpCompression;      //type of compression
    unsigned int    bmpSizeImage;        //size of image in bytes
    int             bmpXPelsPerMeter;    //number of pixels per meter in x axis
    int             bmpYPelsPerMeter;    //number of pixels per meter in y axis
    
    //both set to 256, while 16 and 24-bit images will set them to 0
    unsigned int    bmpClrUsed;          //number of colors used
    unsigned int    bmpClrImportant;     //number of colors that are important
    infoHeader(): bmpSize(0), bmpWidth(0), bmpHeight(0), bmpPlanes(0), bmpBitCount(0), bmpCompression(0), bmpSizeImage(0), bmpXPelsPerMeter(0), bmpYPelsPerMeter(0), bmpClrUsed(0), bmpClrImportant(0) { }
};

struct Pxl
{
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

class Pixel
{
    private:
        int red;
        int green;
        int blue;        

    public:
        Pixel(int _red, int _green, int _blue);
        Pixel(Pixel *pixel);

        int getRed();
        int getGreen();
        int getBlue();
        string getRGB();
        void setRGB(int _red, int _green, int _blue);
};

class Image
{
    private:
        BMPSignature sig; 
        BMPHeader fileheader;
        infoHeader infoheader; 
        int width;
        int height;
        int depth; 
        int* reds;
        int* greens;
        int* blues;      
        vector<Pixel*> pixelsList;

    public:
        //Image(const char* name);              /// To read
        Image(string name);                     /// To read
        Image(int _width, int _height);         /// To write    
        ~Image();

        void saveImage(string savePath);
        int getImageSize();
        int getImageWidth();
        int getImageHeight();        
        vector<Pixel*> getPixelsList();
        Pixel* getPixel(int index);
        Pixel* getRandomPixel();        
};
