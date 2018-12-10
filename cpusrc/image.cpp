#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../include/image.h"
#include <omp.h>

using namespace std;
using namespace cv;

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

Image::Image(int _width, int _height)
{
    width = _width;
    height = _height;
    /*for (int i=0; i < (width*height); i++)
    {
        pixelsList.push_back(new Pixel(0, 0, 0));
    }*/
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
        reds = new unsigned char[width*height];
        greens = new unsigned char[width*height];
        blues = new unsigned char[width*height];

        for (unsigned int y = 0, y_=height-1; y < height && y_>=0; y++,y_--) {
            for (unsigned int x = 0; x < width; x++) {
                fin.read((char*)&pxl, sizeof(pxl));
                index = (y_ * width) + x;
                //cout <<index <<" ";
                //cout <<index <<" - RGB" <<" " << (int)pxl.red <<" "<< (int)pxl.green <<" "<< (int)pxl.blue<<endl;
                //pixelsList.push_back(new Pixel((int)pxl.blue, (int)pxl.green, (int)pxl.red));
                reds[index] = pxl.red;
                greens[index] = pxl.green;
                blues[index] = pxl.blue;
                //index++; 
            }
            fin.seekg(PaddingBytesPerRow, fin.cur);
        }
    }    
    fin.close();   
}

Image::~Image()
{
    delete reds;
    delete greens;
    delete blues;    
}

void Image::showImage()
{
    //Display image with OpenCV    
    Mat chan[3] = {
        Mat(height,width,CV_8UC1,blues),
        Mat(height,width,CV_8UC1,greens),
        Mat(height,width,CV_8UC1,reds)};
       
    originalImage;
    merge(chan,3,originalImage);
    if(!originalImage.data ) {
        cout <<"Something went wrong with image!" << endl ;
        return;
    }
  
    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", originalImage);        
    
    waitKey(0);
}

void Image::showImage(unsigned char *img)
{
    Mat image = Mat(height, width,CV_8UC1, img);
    //Display image with OpenCV    
    if(!image.data ) {
        cout <<"Something went wrong when displaying image!" << endl ;
        return;
    }
  
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", image);        
    
    waitKey(0);
}

void Image::showImage(unsigned char *_reds, unsigned char *_greens, unsigned char *_blues)
{
    //Display image with OpenCV    
    Mat chan[3] = {
        Mat(height,width,CV_8UC1,_blues),
        Mat(height,width,CV_8UC1,_greens),
        Mat(height,width,CV_8UC1,_reds)};
       
    Mat resultImage;
    merge(chan,3,resultImage);
    if(!resultImage.data ) {
        cout <<"Something went wrong with result image!" << endl ;
        return;
    }
  
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", resultImage);        
    
    waitKey(0);
}

void Image::showHistogram()
{    
  /// Separate the image in 3 places ( B, G and R )
  vector<Mat> bgr_planes;
  split( originalImage, bgr_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", WINDOW_NORMAL );
  imshow("calcHist Demo", histImage );

  waitKey(0);

  return ;
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

int Image::grayScale(unsigned char* out)
{
    int size = width * height;
    int i;
//#	pragma omp parallel for num_threads(2) default(none) private(i, size) shared(reds,greens,blues) schedule(dynamic,4)
    for(i = 0; i < size; i++){
        out[i] = 0.21*reds[i] + 0.72*greens[i] + 0.07*blues[i];       //Precise grayScale
    }
}

int Image::sobel(unsigned char* in, unsigned char* out)
{
    //sobel filter
    int gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    int gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    
    unsigned char matrix[height+2][width+2] = {0};

    for (int i = 0; i<height+2;i++){
        for(int j = 0; j<width+2;j++){
            matrix[i][j] = 0;
            //cout<<(int)matrix[i][j]<<" ";
        }
        //cout<<endl;
    }

    int index;
    for (int i = 0; i<height;i++){
        for(int j = 0; j<width;j++){
            index = width*i+j;
            matrix[i+1][j+1] = in[index];
            //cout<<(int)matrix[i+1][j+1]<<" ";
        }
        //cout<<endl;
    }

    /*for (int i = 0; i<height+2;i++){
        for(int j = 0; j<width+2;j++){
            cout<<(int)matrix[i][j]<<" ";
        }
        cout<<endl;
    }*/

    //cout<<endl;
    int x, y, xy;
    for (int h = 1; h < height+1; h++){
        for (int w = 1; w < width+1; w++){
            x, y = 0;
            index = (width*(h-1))+(w-1);
			x =(matrix[h-1][w-1]*gx[0][0])+(matrix[h][w-1]*gx[1][0])+(matrix[h+1][w-1]*gx[2][0])
                +(matrix[h-1][w+1]*gx[0][2])+(matrix[h][w+1]*gx[1][2])+(matrix[h+1][w+1]*gx[2][2]); 
            y =(matrix[h-1][w-1]*gy[0][0])+(matrix[h-1][w]*gy[0][1])+(matrix[h-1][w+1]*gy[0][2])
                +(matrix[h+1][w-1]*gy[2][0])+(matrix[h+1][w]*gy[2][1])+(matrix[h+1][w+1]*gy[2][2]);
            xy = abs(x)+abs(y);
            //xy = sqrt(pow(x,2)+pow(y,2));   
            //cout<<x<<" ";
            //out[index] = xy>255 ? 255 : xy;
            if(xy < 0) xy = 0;
            if(xy > 255) xy = 255;  
            out[index] = xy;  
            //cout<<(int)out[index]<<" ";
        }
    }

    /*for (int i = 0; i<height;i++){
        for(int j = 0; j<width;j++){
            cout<<(int)out[i*width+j]<<" ";
        }
        cout<<endl;
    }*/
    return 0;
}

int Image::maximo(unsigned char* max_red, unsigned char* max_green, unsigned char* max_blue, int k)
{
    //** original matrices are augmented and filled with 0 **//
    //** they're augmented acording to parameter k **//
    int nHeight = height + 2*k;
    int nWidth = width + 2*k;

    unsigned char mred[nHeight][nWidth];
    unsigned char mgreen[nHeight][nWidth];
    unsigned char mblue[nHeight][nWidth];

    for (int i = 0; i<nHeight;i++){
        for(int j = 0; j<nWidth;j++){
            mred[i][j] = 0;            
            mgreen[i][j] = 0;            
            mblue[i][j] = 0;            
        }
    }
    
    //** Copy array to matrix **// 
    int index; //1D Array index
    for (int i = 0; i<height;i++){
        for(int j = 0; j<width;j++){
            index = width*i+j;
            mred[i+k][j+k] = reds[index];
            mgreen[i+k][j+k] = greens[index];
            mblue[i+k][j+k] = blues[index];
            //cout<<(int)matrix[i+1][j+1]<<" ";
        }
        //cout<<endl;
    }
    //cout<<endl;
    
    vector<int> tmpr, tmpg, tmpb;
    //** Convolve matrix **//    
    for (int h = k; h < nHeight-k; h++){
        for (int w = k; w < nWidth-k; w++){            
            index = (width*(h-k))+(w-k);
            for(int r=k; r>0;r--){
                tmpr.push_back((int)mred[h-r][w-r]);
                tmpg.push_back((int)mgreen[h-r][w-r]);
                tmpb.push_back((int)mblue[h-r][w-r]);
            }
            max_red[index] = *max_element(tmpr.begin(), tmpr.end());  
            max_green[index] = *max_element(tmpg.begin(), tmpg.end());  
            max_blue[index] = *max_element(tmpb.begin(), tmpb.end());  
            
            tmpr.clear();
            tmpg.clear();
            tmpb.clear();
        }
    }

    return 0;    
}