#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "image.h"

using namespace std;
///////////////FILE HEADER
//the file type

struct BMPSignature
{
    unsigned char data[2]; //equal to the string 'BM'
    BMPSignature() { data[0] = data[1] = 0; }
};

#pragma pack(push)  // push current alignment to stack
#pragma pack(1)     // set alignment to 1 byte boundary

struct BMPHeader
{
    unsigned int fileSize;      //size in bytes
    unsigned short reserved1;   //reserved; must be 0
    unsigned short reserved2;   //reserved; must be 0
    unsigned int dataOffset;    //the offset in bytes from the BMPHeader to the bitmap bits

    BMPHeader(): fileSize(0),  reserved1(0), reserved2(0), dataOffset(0) { }
};
 

#define BF_TYPE 0x4D42             //BM

/////////////FILE INFO HEADER
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
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

void ReadHeader(ifstream &fin, BMPSignature &sig, BMPHeader &header)
{
    if(!fin)
        return;

    fin.seekg(0, ios::beg); //skipping the  first 'BM'
    fin.read((char*) &sig, sizeof(sig));
    fin.read((char*) &header, sizeof(header));
}

void ReadInfoHeader(ifstream &fin, infoHeader &infoH)
{
    if (!fin)
        return;

    fin.seekg(0, ios::beg);
    fin.read((char*) &infoH, sizeof(infoH));
}

void PrintHeader(BMPSignature sig, BMPHeader header)
{
    cout << "BMP HEADER"    << endl;
    cout << "Signature  : " << sig.data[0] << sig.data[1] << endl;
    cout << "File Size  : " << header.fileSize << " byte(s)" << endl;
    cout << "Reserved1  : " << header.reserved1 << endl;
    cout << "Reserved2  : " << header.reserved2 << endl;
    cout << "Data Offset: " << header.dataOffset << " byte(s)" << endl;
}
 
void PrintInfoH(infoHeader infoH)
{
    cout << endl;
    cout << "INFO HEADER"                   << endl;
    cout << "Size: "                        << infoH.bmpSize << " byte(s)" << endl;
    cout << "Width: "                       << infoH.bmpWidth << " pixel(s)" << endl;
    cout << "Height: "                      << infoH.bmpHeight << " pixel(s)" << endl;
    cout << "Planes: "                      << infoH.bmpPlanes << endl;
    cout << "Bit Count: "                   << infoH.bmpBitCount << endl;
    cout << "Type of Compression: "         << infoH.bmpCompression << endl;
    cout << "Size of Image: "               << infoH.bmpSizeImage << " byte(s)" << endl;
    cout << "Pixels per Meter in X Axis: "  << infoH.bmpXPelsPerMeter << endl;
    cout << "Pixels per Meter in Y Axis: "  << infoH.bmpYPelsPerMeter << endl;
    cout << "Colors Used: "                 << infoH.bmpClrUsed << endl;
    cout << "Important Colours: "           << infoH.bmpClrImportant << endl;
}

int ConvertRGBToInt(unsigned char R, unsigned char G, unsigned char B)
{
	int ReturnInt = 0;
	unsigned char Padding = 0;
	unsigned char buffer[4];
	buffer[0] = R;
	buffer[1] = G;
	buffer[2] = B;
	buffer[3] = Padding;
	memcpy((char*)&ReturnInt, buffer, 4); //Copy 4 bytes from our char buffer to our ReturnInteger
	return ReturnInt;
}

Pixel::Pixel(int _red, int _green, int _blue)
{
    red = _red;
    green = _green;
    blue = _blue;       
}

Pixel::Pixel(Pixel *pixel)
{
    red = pixel->red;
    green = pixel->green;
    blue = pixel->blue;
}

int Pixel::getRed()
{
    return red;
}

int Pixel::getGreen()
{
    return green;
}

int Pixel::getBlue()
{
    return blue;
}

string Pixel::getRGB()
{
    return to_string(red) + " " + to_string(green) + " " + to_string(blue);
}

void Pixel::setRGB(int _red, int _green, int _blue)
{
    red = _red;
    green = _green;
    blue = _blue;
}

Image::Image(int _width, int _height)
{
    width = _width;
    height = _height;
    for (int i=0; i < (width*height); i++)
    {
        pixelsList.push_back(new Pixel(0, 0, 0));
    }
}

Image::Image(string imageDir)
{    
    int Color;
    ifstream fin(imageDir, ios::binary);    
    BMPSignature sig;    
    fin.read((char*)&sig, sizeof(sig));    
    if (sig.data[0] == 'B' && sig.data[1] == 'M') { 
        cout<<"File BMP found in: "<< imageDir<<endl;       
        BMPHeader fileheader;
        infoHeader infoheader;
        fin.read((char*)&fileheader, sizeof(fileheader));
        fin.read((char*)&infoheader, sizeof(infoheader));

        PrintHeader(sig, fileheader);
        PrintInfoH(infoheader);
        cout<<endl;

        width = infoheader.bmpWidth;
        height = infoheader.bmpHeight;
        cout<<"Size: "<<width <<" * " <<height<<endl;
    
        fin.seekg(fileheader.dataOffset, fin.beg);
        int PaddingBytesPerRow = (4 - ((width * 3) % 4)) % 4;
        Pxl pxl;
        int cc = 0;
        //image.resize(width*height);

        //Instanciate RGB arrays
        reds = new int[width*height];
        greens = new int[width*height];
        blues = new int[width*height];

        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                fin.read((char*)&pxl, sizeof(pxl));
                
                cout <<cc <<" - RGB" <<" " << (int)pxl.blue <<" "<< (int)pxl.green <<" "<< (int)pxl.red<<endl;
                //pixelsList.push_back(new Pixel((int)pxl.blue, (int)pxl.green, (int)pxl.red));
                reds[cc] = (int)pxl.red;
                greens[cc] = (int)pxl.green;
                blues[cc] = (int)pxl.blue;
                cc++;
            }
            fin.seekg(PaddingBytesPerRow, fin.cur);
        }

        /*Pxl pix;
        fin.read((char*)&pix, sizeof(pix));
        cout<<" -> "<<pix.red<<" "<<pix.green<<" "<<pix.blue<<endl;*/

        /*
        int PaddingBytesPerRow = (3 - ((width* 3) % 3)) % 3;
        cout<<"PaddingBytesPerRow: "<<PaddingBytesPerRow<<endl;
                
        
        int cc = 0;

        pixelsList.resize(width*height);
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                fin.read((char*)&pix, sizeof(pix));
                cc = x + ((height - 1) - y) * width;
                cout<<cc<<" -> "<<(int)pix.red<<" "<<(int)pix.green<<" "<<pix.blue<<endl;
            }
            fin.seekg(PaddingBytesPerRow, fin.cur);
        }*/
    }    
    fin.close();

    /*BMPSignature sig;
    BMPHeader hdr;
    infoHeader ihdr;

    ReadHeader(fin, sig, hdr);
    ReadInfoHeader(fin, ihdr);
    PrintHeader(sig, hdr);
    PrintInfoH(ihdr);
*/
    /*FILE* img = fopen(imageDir, "rb");   //read the file
    
    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, img); // read the 54-byte header

    // extract image height and width from header 
    width = *(int*)&header[18];     
    height = *(int*)&header[22];     
    
    int row_padded = (width*3 + 3) & (~3);
    cout << "Padding: "<< row_padded <<endl;*/
    //unsigned char* data = new unsigned char[row_padded];
    //unsigned char tmp;

    /*for(int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, img);
        for(int j = 0; j < width*3; j += 3)
        {
            pixelsList.push_back(new Pixel((int)data[j+2], (int)data[j+1], (int)data[j]));
         
            /*if(i<2 && j < 9){
                cout << "R: "<< (int)data[j] << " G: " << (int)data[j+1]<< " B: " << (int)data[j+2]<< endl;
            }*/            
    //    }
        
    //}

    //delete data;
}


Image::~Image()
{
    for (int i = 0; i < width * height; i++)
    {
        delete pixelsList[i];
    }
}

void Image::saveImage(string name)
{
    cout <<"aa"<<endl;
    /*int x, y, r, g, b;
    int index = 0;
    FILE *f;
    unsigned char *img = NULL;
    int filesize = 54 + 3*width*height;  //w is your image width, h is image height, both int    
    img = (unsigned char *)malloc(3*width*height);
    memset(img,0,3*width*height);
    Pixel *pixel = NULL;

    cout <<"bb"<<endl;
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            x=i; 
            y=(height-1)-j;
            pixel = pixelsList[index];            
            /*r = red[i][j]*255;
            g = green[i][j]*255;
            b = blue[i][j]*255;
            if (r > 255) r=255;
            if (g > 255) g=255;
            if (b > 255) b=255;*/
            /*img[(x+y*width)*3+2] = (unsigned char)(reds[index]);
            img[(x+y*width)*3+1] = (unsigned char)(greens[index]);
            img[(x+y*width)*3+0] = (unsigned char)(blues[index]);
            index++;
        }
    }
    cout <<"cc"<<endl;
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       width);
    bmpinfoheader[ 5] = (unsigned char)(       width>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       width>>16);
    bmpinfoheader[ 7] = (unsigned char)(       width>>24);
    bmpinfoheader[ 8] = (unsigned char)(       height);
    bmpinfoheader[ 9] = (unsigned char)(       height>> 8);
    bmpinfoheader[10] = (unsigned char)(       height>>16);
    bmpinfoheader[11] = (unsigned char)(       height>>24);

    f = fopen("img.bmp","wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(int i=0; i<height; i++)
    {
        fwrite(img+(width*(height-i-1)*3),3,width,f);
        fwrite(bmppad,1,(4-(width*3)%4)%4,f);
    }

    free(img);
    fclose(f);
    /*ofstream image(name);
    Pixel *pixel = NULL;
    if (image)
    {
        image << "P3" << "\n";
        image << width << " " << height << "\n";
        image << depth << "\n";
        for (int y = 0; y < width; y++)
        {
            for (int x = 0; x < height; x++)
            {
                pixel = pixelsList[height * y + x];
                image << pixel->getRGB() << " ";
            }
            image << "\n";
        }
    } else {
        cout << name << "Something went wrong with the file " <<name << "\n";
    }*/
    // mimeType = "image/bmp";

    unsigned char file[14] = {
        'B','M', // magic
        0,0,0,0, // size in bytes
        0,0, // app data
        0,0, // app data
        40+14,0,0,0 // start of data offset
    };
    unsigned char info[40] = {
        40,0,0,0, // info hd size
        0,0,0,0, // width
        0,0,0,0, // heigth
        1,0, // number color planes
        24,0, // bits per pixel
        0,0,0,0, // compression is none
        0,0,0,0, // image bits size
        0x13,0x0B,0,0, // horz resoluition in pixel / m
        0x13,0x0B,0,0, // vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
        0,0,0,0, // #colors in pallete
        0,0,0,0, // #important colors
        };

    int w=width;
    int h=height;

    int padSize  = (4-(w*3)%4)%4;    
    int sizeData = w*h*3 + h*padSize;
    int sizeAll  = sizeData + sizeof(file) + sizeof(info);

    file[ 2] = (unsigned char)( sizeAll    );
    file[ 3] = (unsigned char)( sizeAll>> 8);
    file[ 4] = (unsigned char)( sizeAll>>16);
    file[ 5] = (unsigned char)( sizeAll>>24);

    info[ 4] = (unsigned char)( w   );
    info[ 5] = (unsigned char)( w>> 8);
    info[ 6] = (unsigned char)( w>>16);
    info[ 7] = (unsigned char)( w>>24);

    info[ 8] = (unsigned char)( h    );
    info[ 9] = (unsigned char)( h>> 8);
    info[10] = (unsigned char)( h>>16);
    info[11] = (unsigned char)( h>>24);

    info[20] = (unsigned char)( sizeData    );
    info[21] = (unsigned char)( sizeData>> 8);
    info[22] = (unsigned char)( sizeData>>16);
    info[23] = (unsigned char)( sizeData>>24);

    stream.write( (char*)file, sizeof(file) );
    stream.write( (char*)info, sizeof(info) );

    unsigned char pad[3] = {0,0,0};

    for ( int y=0; y<h; y++ )
    {
        for ( int x=0; x<w; x++ )
        {
            long red = lround( 255.0 * waterfall[x][y] );
            if ( red < 0 ) red=0;
            if ( red > 255 ) red=255;
            long green = red;
            long blue = red;

            unsigned char pixel[3];
            pixel[0] = blue;
            pixel[1] = green;
            pixel[2] = red;

            stream.write( (char*)pixel, 3 );
        }
        stream.write( (char*)pad, padSize );
}
}

Pixel* Image::getRandomPixel()
{
    return pixelsList[rand() % (width * height)];    
}

vector<Pixel*> Image::getPixelsList()
{
    return pixelsList;
}

Pixel* Image::getPixel(int index)
{
    return pixelsList[index];
}

int Image::getImageSize()
{
    return width * height;
}

int Image::getImageWidth()
{
    return width;
}

int Image::getImageHeight()
{
    return height;
}