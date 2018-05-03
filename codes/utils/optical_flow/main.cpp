#include <of.h>
#include <COpticFlowPart.h>
#include <iostream>
#include <fstream>
#include <string>

/*
Read a file containing the path of the first image, the path of the second
image and the path of the output file (optical flow image). Both input files
are used to create the optical flow and save in output file.

In order to compile the file, please add the `optical_flow` folder to the 
LD_LIBRAY_PATH and then run:

$ export LD_LIBRAY_PATH=$LD_LIBRAY_PATH:<path_to_this_folder>
$ g++ main.cpp -I. -L. -lof -o main
$./main <input_file>

*/

using namespace std;

void genOpticalFlow(string fin1, string fin2, string fout){
    CTensor<float> img1, img2;
    img1.readFromPPM(fin1.c_str());
    img2.readFromPPM(fin2.c_str());
    CTensor<float> flow;
    opticalFlow(img1, img2, flow);

    CTensor<float> flow_img;
    COpticFlow::flowToImage(flow, flow_img);
    flow_img.writeToPPM(fout.c_str());
}

int main(int argc, char **argv){
    if(argc != 2){
        cout << "ERROR: Too few arguments. Input file is missing" << endl;
        return 1;
    }

    ifstream infile;
    infile.open (argv[1]);

    string fin1, fin2, fout;
    while(!infile.eof()){
        infile >> fin1 >> fin2 >> fout;
        genOpticalFlow(fin1, fin2, fout);
    }
    infile.close();
    
    return 0;
}
