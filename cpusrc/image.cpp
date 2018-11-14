#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "image.h"

using namespace std;

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

Image::Image(const char* imageDir)
{
    FILE* img = fopen(imageDir, "rb");   //read the file
    
    unsigned char header[54];
    fread(header, sizeof(unsigned char), 54, img); // read the 54-byte header

    // extract image height and width from header 
    width = *(int*)&header[18];     
    height = *(int*)&header[22];     
    
    int row_padded = (width*3 + 3) & (~3);
    cout << "Padding: "<< row_padded <<endl;
    unsigned char* data = new unsigned char[row_padded];
    unsigned char tmp;

    for(int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, img);
        for(int j = 0; j < width*3; j += 3)
        {
            pixelsList.push_back(new Pixel((int)data[j+2], (int)data[j+1], (int)data[j]));
         
            /*if(i<2 && j < 9){
                cout << "R: "<< (int)data[j] << " G: " << (int)data[j+1]<< " B: " << (int)data[j+2]<< endl;
            }*/            
        }
        
    }

    delete data;

    /*int size = 3 * width * height;
    unsigned char* data = new unsigned char[size]; 

    fread(data, sizeof(unsigned char), size, img); // read the rest of the data at once
    fclose(img);

    for(int i = 0; i < size; i += 3)
    {
        pixelsList.push_back(new Pixel(data[i+2], data[i+1], data[i]));
        if ( i < 10){
            std::cout<<"1: " <<*(int*)&data[i]<<" 2: "<<*(int*)&data[i+1]<<" 3: "<<*(int*)&data[i+4] <<std::endl;
        }
    }*/
    
    /*ifstream image(imageDir);
    if (image)
    {
        string type;
        image >> type;
        if (type == "P3")
        {
            int red;
            int green;
            int blue;
            image >> width;
            image >> height;
            image >> depth;
            for (int i = 0; i < (width * height); i++)
            {
                image >> red;
                image >> green;
                image >> blue;
                pixelsList.push_back(new Pixel(red, green, blue));
            }
        } else {
            cout << "The format file must be .ppm type P3" << endl;
        }
    } else {
        cout << "File not found!" << endl;
    }*/
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
    int x, y, r, g, b;
    int index = 0;
    FILE *f;
    unsigned char *img = NULL;
    int filesize = 54 + 3*width*height;  //w is your image width, h is image height, both int

    img = (unsigned char *)malloc(3*width*height);
    memset(img,0,3*width*height);
    Pixel *pixel = NULL;

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
            img[(x+y*width)*3+2] = (unsigned char)(pixel->getRed());
            img[(x+y*width)*3+1] = (unsigned char)(pixel->getGreen());
            img[(x+y*width)*3+0] = (unsigned char)(pixel->getBlue());
            index++;
        }
    }

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