#include "Histogram.h"


Histogram::Histogram(){

}


Histogram::~Histogram(){}

void
Histogram::loadFile(char* filepath, int channels){
    int width, height, bpp;
    unsigned char* rgb;
    rgb = stbi_load( filepath, &width, &height, &bpp, 0 );
    // rgb is now three bytes per pixel, width*height size. Or NULL if load failed.
    // Do something with it...

    printf("bpp:    %8i\n", bpp);
    printf("Width:  %8i\n", width);
    printf("Height: %8i\n", height);

    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            printf("%3i", rgb[x+y*width]);
            if((x+1)%3==0){
                printf(" ");
            } else if(x<width-1){
                printf("|");
            }
        }
        printf("\n");
    }

    stbi_image_free( rgb );
    return;
}

void
Histogram::calcHist(char* rgb_data){

}