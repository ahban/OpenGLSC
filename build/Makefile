TARGET=..\bin\GLSC.exe

OPENCV_INC=K:\OpenCV\2.4.10\include
OPENCV_LIB=K:\OpenCV\2.4.10\x64\vc12.0\lib

CUDA_INC=%CUDA_PATH%\include
CUDA_LIB=%CUDA_PATH%\lib\x64


all:
	cl /c /O2 /EHsc /I$(OPENCV_INC) /I$(CUDA_INC) ..\src\main.cpp
	nvcc -O2 -c ..\src\DReductor.cu ..\src\gpu_lsc.cu -I$(CUDA_INC) -I$(OPENCV_INC) -w
	link /LIBPATH:$(CUDA_LIB) cudart.lib /LIBPATH:$(OPENCV_LIB) opencv_core2410.lib opencv_highgui2410.lib *.obj /OUT:$(TARGET)
