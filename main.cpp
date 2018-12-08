#include <iostream>
#include "include/image.h"

int main(int argc, char* argv[]){
    /// Time counting
	clock_t start, end;
    double globalTime = 0.0;
    
    /// Read & Write image path
    std::string inputImagePath, outputImagePath;
    
    inputImagePath = argv[1];           /// Input path        

    /// Read image
    start = clock();
    Image *image(new Image(inputImagePath));
    end = clock();
    std::cout<<"Reading file: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< std::endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;

    /*std::cout<< image->getPixel(1)->getRGB() <<std::endl;
    std::cout<< image->getPixel(2)->getRGB() <<std::endl;
    std::cout<< image->getPixel(50)->getRGB() <<std::endl;
    std::cout<< image->getImageHeight() <<std::endl;
    std::cout<< image->getImageWidth() <<std::endl; */ 

    /// Set the output file path
    //std::string saveImageAs = "../out/" +std::to_string(4) +"_" + inputImagePath.substr(9,(inputImagePath.size()));
    //std::cout<<saveImageAs;
    //string saveImageAs = "../out/" + inputImagePath.substr(9,(inputImagePath.size()));
    string saveImageAs = "output.ppm";
    cout<<saveImageAs <<endl;

    image->saveImage(saveImageAs);
}

/*#include <QApplication>
#include <QPushButton>

int main(int argc, char **argv)
{
    QApplication app (argc, argv);

    QPushButton button ("Hello world !");
    button.show();

    return app.exec();
}*/
