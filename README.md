# OpenGLSC
An implementation of "Superpixel Segmentation Using Linear Spectral Clustering" on GPU

We provide an executable file in ./bin, source code in ./src and Makefile in ./build
This code support only 64-bit Windows for now but you can compile it on Linux with c++11 support. 

to execute the app, you need to use the following command.
``` batch
GLSC.exe 118035.jpg 16
```

If you want to compile the code by your self, you have to install OpenCV 2.4.X and 
Visual Studio. And modify the Makefile in build folder the corresponding pathes. and then you 
can execute the following command in your **cmd**.

``` batch
%comspec% /k ""Your\VisualStudio\Path\VC\vcvarsall.bat"" amd64
cd .\build
nmake 
```

We are very happy if you can cite our paper,

``` bibtex
@article{GLSC,
  author="Ban, Zhihua and Liu, Jianguo and Fouriaux, Jeremy",
  title="{GLSC}: {LSC} superpixels at over 130 {FPS}",
  journal="Journal of Real-Time Image Processing",
  year="2016",
  issn="1861-8219",
  doi="10.1007/s11554-016-0652-5"
}
```

The original paper is:
``` bibtex
@inproceedings{li2015superpixel,
  title={Superpixel segmentation using linear spectral clustering},
  author={Li, Zhengqin and Chen, Jiansheng},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1356--1363},
  year={2015},
  organization={IEEE}
}
```



**Note: this source code only can be used in noncommercial research.**
