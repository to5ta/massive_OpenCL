#include <iostream>
#include "../shared/ansi_colors.h"

#define STB_IMAGE_IMPLEMENTATION

#include "Histogram.h"


int main() {

    Histogram histo;

//    histo.loadFile("colorful_image_small.jpg", 3);
//    histo.loadFile("../colorful_image_small.jpg", 3);
//    histo.loadFile("../color_test_mini.jpg", 3);
//    histo.loadFile("../color_test_mini.png", 3);


//    histo.loadFile("../gradient_bw.png", 3);
//    histo.loadFile("../ease_bw.png", 3);
//    histo.loadFile("../half_ease_large.png", 3);
//    histo.loadFile("../gradient_small.png", 3);


//    histo.loadFile("../stripes_bw.png", 3);
    histo.loadFile("../hist2parts.png", 3);
    histo.calcHist();

    std::cout << ANSI_COLOR_BRIGHTGREEN << "Aufgabe 1 OK!" << ANSI_COLOR_RESET << std::endl;
    return 0;
}