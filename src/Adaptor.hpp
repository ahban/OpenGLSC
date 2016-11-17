// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#ifndef __ADAPTOR__HPP__
#define __ADAPTOR__HPP__

#include "AMatrix.hpp"
#include <opencv2/core/mat.hpp>

class Adaptor
{
public:
  
  // Labels to vectors
  static void AMatLabels2Vectors(vector<int> &labels, const HMati &h_labels){
    int width = h_labels.m_width;
    int height = h_labels.m_height;
    int steps = h_labels.m_steps;

    labels.resize(width*height);

    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        labels[y*width + x] = h_labels[y*steps+x];
      }
    }

  }

  // AMat to Mat
  template<unsigned int S, class T>
  static void AMat2Mat(cv::Mat &dst, const AMat<S,T> &src, const int type){
    int width = src.m_width;
    int height = src.m_height;

    switch (type)
    {
    case CV_8UC1:
      dst.create(height, width, CV_8UC1);
      _copy_1<uchar>(dst, src);
      break;

    case CV_8UC3:
      dst.create(height, width, CV_8UC3);
      _copy_3<Vec3b>(dst, src);
      break;

    case CV_32FC1:
      dst.create(height, width, CV_32FC1);
      _copy_1<float>(dst, src);
      break;

    case CV_32FC3:
      dst.create(height, width, CV_32FC3);
      _copy_3<Vec3f>(dst, src);
      break;
    case CV_32SC1:      
      dst.create(height, width, CV_32SC1);
      _copy_1<int>(dst, src);
      break;
    case CV_32SC3:
      dst.create(height, width, CV_32SC3);
      _copy_3<Vec3i>(dst, src);
      break;
    default:
      ERROR_OUT__ << "unsupported type." << endl;
      exit(EXIT_FAILURE);
      break;
    }    
  }
  
  // cv::Mat to HMatf  
  static void Mat2HMatc(HMatc &dst, const Mat &src){
    int width = src.cols; 
    int height = src.rows;
    if (src.type() != CV_8UC3){
      ERROR_OUT__ << "invalid type!" << endl;
      exit(EXIT_FAILURE);
    }

    dst.create(width, height, 3);

    uchar *pdst = dst.m_data;
    int steps = dst.m_steps;

    for (int c = 0; c < 3; c++){
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          pdst[y*steps + x] = uchar(src.at<Vec3b>(y, x)[c]);
        }
      }
      pdst += steps*height;
    }
  }



private:
  template<class CVT, unsigned int AS, class AT>
  static void _copy_3(cv::Mat &dst, const AMat<AS, AT> &src){
    int width = src.m_width;
    int height = src.m_height;
    int steps = src.m_steps;
    if (src.m_channels!=3){
      ERROR_OUT__ << "wrong channel number" << endl;
      exit(EXIT_FAILURE);
    }
    AT *sptr = src.m_data;
    for (int c = 0; c < 3; c++){
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          dst.at<CVT>(y, x)[c] = sptr[y*steps+x];
        }
      }
      sptr += height*steps;
    }
  }

  template<class CVT, unsigned int AS, class AT>
  static void _copy_1(cv::Mat &dst, const AMat<AS, AT> &src){
    int width  = src.m_width;
    int height = src.m_height;
    int steps  = src.m_steps;
    if (src.m_channels != 1){
      ERROR_OUT__ << "wrong channel number" << endl;
      exit(EXIT_FAILURE);
    }
    AT *sptr = src.m_data;
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        dst.at<CVT>(y, x) = sptr[y*steps + x];
      }
    }
  }
};


#endif
