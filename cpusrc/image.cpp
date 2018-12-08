#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "../include/image.h"

using namespace std;

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
    ifstream fin(imageDir, ios::binary);    
    //BMPSignature sig;    
    fin.read((char*)&sig, sizeof(sig));    
    if (sig.data[0] == 'B' && sig.data[1] == 'M') { 
        cout<<"File BMP found in: "<< imageDir<<endl;       
        //BMPHeader fileheader;
        //infoHeader infoheader;
        fin.read((char*)&fileheader, sizeof(fileheader));
        fin.read((char*)&infoheader, sizeof(infoheader));

        //PrintHeader(sig, fileheader);
        //PrintInfoH(infoheader);
        cout<<endl;

        width = infoheader.bmpWidth;
        height = infoheader.bmpHeight;
        cout<<"Size: "<<width <<" * " <<height<<endl;
    
        fin.seekg(fileheader.dataOffset, fin.beg);
        int PaddingBytesPerRow = (3 - ((width * 3) % 3)) % 3; //not sure
        //cout<<"Padding: "<<PaddingBytesPerRow <<endl;
        Pxl pxl;
        int index = 0;
        //image.resize(width*height);

        //Instanciate RGB arrays
        reds = new int[width*height];
        greens = new int[width*height];
        blues = new int[width*height];

        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                fin.read((char*)&pxl, sizeof(pxl));
                
                //cout <<index <<" - RGB" <<" " << (int)pxl.red <<" "<< (int)pxl.green <<" "<< (int)pxl.blue<<endl;
                //pixelsList.push_back(new Pixel((int)pxl.blue, (int)pxl.green, (int)pxl.red));
                reds[index] = (int)pxl.red;
                greens[index] = (int)pxl.green;
                blues[index] = (int)pxl.blue;
                index++;
            }
            fin.seekg(PaddingBytesPerRow, fin.cur);
        }
    }    
    fin.close();   
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
    /*ofstream fout(name, ios::out | ios::binary);    
    fout.write((char*)&sig, sizeof(sig));   
    fout.write((char*)&fileheader, sizeof(fileheader));
    fout.write((char*)&infoheader, sizeof(infoheader));

    int PaddingBytesPerRow = (4 - ((width * 3) % 4)) % 4;
    Pxl pxl;
    int index = 0;
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {            
            index = (y*width)+x;
            pxl.blue = blues[index];
            pxl.red = reds[index];
            pxl.green = greens[index];

            //cout <<index <<" - RGB" <<" " << (int)pxl.red <<" "<< (int)pxl.green <<" "<< (int)pxl.blue<<endl;

            fout.write((char *)&pxl, sizeof(pxl));            
        }
        //fout.seekp(PaddingBytesPerRow * sizeof(unsigned char), std::ios::cur);
        unsigned char i = 0x00;
        for (int pad = 0; pad < PaddingBytesPerRow; pad++) {

            fout.write((char *)&i, sizeof(i));            
           
        }
    }
    fout.close();  */
    ofstream image(name);
    //Pixel *pixel = NULL;
    //Pxl pxl;
    int r,g,b = 0;
    
    if (image)
    {
        image << "P3" << "\n";
        image << width << " " << height << "\n";
        image << 255 << "\n";
        for (int y = height-1; y >= 0; y--)
        {
            for (int x = 0; x <width; x++)
            {
                r = reds[y * width + x];
                g = greens[y * width + x];
                b = blues[y * width + x];
                image << r << " " << g << " " << b <<" ";
            }
            image << "\n";
        }
    } else {
        cout << name << "Something went wrong with the file " <<name << "\n";
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