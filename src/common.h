// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#ifndef __COMMON_H__
#define __COMMON_H__

#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <iostream>

//extern int g_iter_num;

#define ERROR_OUT__ std::cerr<<"[ERROR][File:"<<__FILE__<<"][Line:"<<__LINE__<<"]"


#define normalizer__ .003921568627450980f
//actual CIE standard
#define epsilon__  0.008856f
//actual CIE standard
#define kappa__  903.3f

//reference white
#define iXr__  1.052126558199432693f
//reference white
#define iYr__  1.0f
//reference white
#define iZr__  .918481126131339127f

#define LSC_PI 3.1415926535897932384626433832795f



#ifndef _DEBUG
#define TB__(A) int64 A; A = cv::getTickCount()
#define TE__(A) std::cout << #A << " : " << 1.E3 * double(cv::getTickCount() - A)/double(cv::getTickFrequency()) << "ms" << std::endl
#define TEE_(A,E) std::cout << #A << " : " << (E=1.E3 * double(cv::getTickCount() - A)/double(cv::getTickFrequency())) << "ms" << std::endl
#else
#define TB__(A)
#define TE__(A)
#endif

#ifndef _DEBUG
#define GTB__(A) int64 A;cudaDeviceSynchronize();A = cv::getTickCount()
#define GTE__(A) cudaDeviceSynchronize();std::cout << #A << " : " << 1.E3 * double(cv::getTickCount() - A)/double(cv::getTickFrequency()) << "ms" << std::endl
#define GTEE_(A,E) cudaDeviceSynchronize();std::cout << #A << " : " << (E = 1.E3 * double(cv::getTickCount() - A)/double(cv::getTickFrequency())) << "ms" << std::endl
#else
#define GTB__(A)
#define GTE__(A)
#endif

#define NN_FEATURES__ 10

#define REDUCTOR_GS__ 64
#define REDUCTOR_BS__ 128

#endif