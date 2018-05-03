# Optical Flow generator

This library is freely available by [University of Freiburg](https://lmb.informatik.uni-freiburg.de/resources/software.php) and can be downladed directly in this [link](https://lmb.informatik.uni-freiburg.de/resources/binaries/eccv2004Linux64.zip). 

In order to run you have to create a file containing the path of the first and second images and the path to the output file as:

```
/home/user/Car/Car_Hime_2_6450_6590_frame_1.ppm /home/user/Car/Car_Hime_2_6450_6590_frame_3.ppm /home/user/OPF/Car/Car_Hime_2_6450_6590_frame_1.ppm
```

All files must be in PPM format. If you have JPG files, first run `convert_ppm.py` file. In order to compile the `main.cpp` file, first add the path to the this folder to `LD_LIBRARY_PATH` using:

```
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_this_folder>
```

Then compile the file by running the command:

```
$ g++ main.cpp -I. -L. -lof -o main
```

To run the program, call the `main` file passing the path to the file with all paths as argument, as:

```
$./main <file_with_paths>
```


---
64bit Linux library for ECCV 2004 optical flow method

Copyright (c) 2009 Thomas Brox

------------------------------
Terms of use
------------------------------

This program is provided for research purposes only. Any commercial
use is prohibited. If you are interested in a commercial use, please 
contact the copyright holder. 

This program is distributed WITHOUT ANY WARRANTY. 

If you use this program in your research work, you must cite the 
following publication:

T. Brox, A. Bruhn, N. Papenberg, J. Weickert: 
High accuracy optical flow estimation based on a theory for warping, 
European Conference on Computer Vision (ECCV), Springer, LNCS, Vol. 3024, 
25-36, May 2004. 

