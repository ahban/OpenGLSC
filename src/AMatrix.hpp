// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#ifndef __ALIGNED_MATRIX__
#define __ALIGNED_MATRIX__
#include "common.h"
#include <cuda_runtime_api.h>

#include <cassert>
#include <iostream>
#include <exception>
#include <cmath>
#include <vector>
#include <ctime>
#include <iomanip>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
using namespace cv;

#define MALLOC_HOST 

// memory allocation

// memory allocation
class Allocate
{
public:
#ifdef _WIN32
#include <stdlib.h>
  static void *align_allocate(size_t sz, size_t aligned_size){
    return _aligned_malloc(sz, aligned_size);
  }

  static void align_release(void *data){
    _aligned_free(data);
  }

#else
#include <stdlib.h>
  static void *align_allocate(size_t sz, size_t aligned_size){
    return aligned_alloc(aligned_size, sz);
  }

  static void align_release(void *data){
    free(data);
  }
#endif

	static void *host_allocate(size_t sz){
		char *data;
		cudaMallocHost((void**)(&data), sz);
		return data;
	}

	static void host_release(void *data){
		cudaFreeHost(data);
	}
};


// aligned Mat with SIZE_ALIGN Bytes (e.g. 16 Bytes)
// sizeof(T) must smaller than 16 and 16%sizeof(T)=0.
template <int SIZE_ALIGN, typename T>
class AMat
{
#define AMAT_ASSERT_VALID_ARGUMENTS(r,str) if (!(r)){ERROR_OUT__<<":"<<str<<std::endl;throw std::invalid_argument(str);}
public:
  // initialize with nothing.
  // AMat A; 
  // AMat *A = new AMat[N];
  AMat(){
    AMAT_ASSERT_VALID_ARGUMENTS(sizeof(T) <= SIZE_ALIGN, "invalid type");
    AMAT_ASSERT_VALID_ARGUMENTS((SIZE_ALIGN%sizeof(T) == 0), "invalid type");
    m_data = NULL;
    m_width = 0; m_height = 0;
    m_steps = 0; m_channels = 0;
  }

  // initialize with another object.
  // it is the user's responsibility to ensure obj has been initialized properly.
  // Note : the dynamically allocated AMat is not been initialized.
  // AMat A = B;
  // AMat A(B);
  AMat(const AMat &obj){
    m_width = obj.m_width;
    m_height = obj.m_height;
    m_channels = obj.m_channels;
    m_steps = obj.m_steps;
    if (obj.m_data){
#ifdef MALLOC_HOST
			m_data = (T*)Allocate::host_allocate(m_channels*m_height*m_steps*sizeof(T));
#else
			m_data = (T*)Allocate::align_allocate(m_channels*m_height*m_steps*sizeof(T), SIZE_ALIGN);
#endif
      if (NULL == m_data){
        std::cerr << "failed to allocate memory." << std::endl;
        exit(EXIT_FAILURE);
      }
      std::memcpy(m_data, obj.m_data, m_channels*m_height*m_steps*sizeof(T));
    }
    else
      m_data = NULL;
  }

  // initialize with arguments
  AMat(int _width, int _height, int _channels = 1){
    AMAT_ASSERT_VALID_ARGUMENTS(sizeof(T) <= SIZE_ALIGN, "invalid arguments.");
    AMAT_ASSERT_VALID_ARGUMENTS(SIZE_ALIGN%sizeof(T) == 0, "invalid arguments.");
    AMAT_ASSERT_VALID_ARGUMENTS(_width > 0 && _height > 0 && _channels > 0, "invalid arguments.");
		memset((void*)(this), 0, sizeof(AMat<SIZE_ALIGN,T>));
    this->create(_width, _height, _channels);
  }

  // release resources
  ~AMat(){
    if (m_data){      
#ifdef MALLOC_HOST
			Allocate::host_release(m_data);
#else
			Allocate::align_release(m_data);
#endif
      m_data = NULL;
      //cout << "deleted memory." << endl;
    }
    //cout << "deleting nothing." << endl;
  }

  // MatA = MatB;
  // MatA = MatB = MatC;
  // it is the user's responsibility to ensure obj has been initialized properly.
  // also MatA is not empty.
  const AMat& operator=(const AMat &obj){
    // MatA = MatA; #< nothing happend
    if (&obj == this){
      return *this;
    }

    //// MatA = AMat(); #< header copy
    //if (0==obj.m_width && 0==obj.m_height && 0==obj.m_channels && NULL==obj.m_data){
    //  m_width = m_height = m_channels = m_steps = 0;
    //  m_data = NULL;
    //  return *this;
    //}


    // MatA is not empty.
    // MatA = initialized B; #< only valid when they share the same header and non-empty.
    AMAT_ASSERT_VALID_ARGUMENTS(NULL != obj.m_data, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(obj.m_channels == m_channels, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(obj.m_width == m_width, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(obj.m_height == m_height, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(obj.m_steps == m_steps, "invalid argument.");

    std::memcpy(m_data, obj.m_data, m_channels*m_height*m_steps*sizeof(T));

    return *this;
  }

  // MatA = Val;
  // MatB = MatA = val;
  const AMat& operator=(const T &val){
    int len = m_channels*m_steps*m_height;
    AMAT_ASSERT_VALID_ARGUMENTS(len != 0, "invalid argument.");
    for (int i = 0; i < len; i++){
      this->m_data[i] = val;
    }
    return *this;
  }

  const AMat operator-(const AMat &B) const{
    AMAT_ASSERT_VALID_ARGUMENTS(m_height == B.m_height, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(m_width == B.m_width, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(m_channels == B.m_channels, "invalid argument.");
    AMAT_ASSERT_VALID_ARGUMENTS(m_channels*m_width*m_height != 0, "invalid argument.");

    AMat<SIZE_ALIGN, T> res;
    res.create(m_width, m_height, m_channels);

    int len = m_steps*m_height*m_channels;

    for (int i = 0; i < len; i++){
      res[i] = m_data[i] - B[i];
    }
    return res;
  }

  void create(int _width, int _height, int _channels = 1){

    // use the current memory or current memory is empty.
    if (m_width == _width && m_height == _height && _channels == m_channels){
      return;
    }

    m_width = _width;  m_height = _height; m_channels = _channels;
    // align x axis
    m_steps = (((m_width*sizeof(T)+SIZE_ALIGN - 1) / SIZE_ALIGN)*SIZE_ALIGN) / sizeof(T);

    // re-creating 
    if (m_data){ // m_data not null
#ifdef MALLOC_HOST
			Allocate::host_release(m_data);
#else
			Allocate::align_release(m_data);
#endif
      m_data = NULL;
    }
    if (m_channels*m_height*m_steps == 0){
      m_data = NULL;
      return;
    }
#ifdef MALLOC_HOST
		m_data = (T*)Allocate::host_allocate(m_channels*m_height*m_steps*sizeof(T));
#else
		m_data = (T*)Allocate::align_allocate(m_channels*m_height*m_steps*sizeof(T), SIZE_ALIGN);
#endif
    if (NULL == m_data){
      std::cerr << "failed to allocate memory." << std::endl;
      exit(EXIT_FAILURE);
    }
  }


  T& operator[](const int &i){
    return m_data[i];
  }
  const T& operator[](const int &i) const{
    return m_data[i];
  }

  // ele_type must be the same with T.
  // e.g. if T = float, then ele_type = CV_32FC1.
  // if ch >= 0 : only support one channel.
  // if ch <  0 : only support CV_8UC3.
  void copy_to_mat(cv::Mat &dst, const int ch, const int ele_type){
    if (ch >= 0){
      if (!m_data || ch >= m_channels || ch < 0){
        dst = Mat();
        return;
      }
      dst.create(m_height, m_width, ele_type);

      T* ptr = m_data + (m_height*m_steps)*ch;

      for (int y = 0; y < m_height; y++){
        for (int x = 0; x < m_width; x++){
          dst.at<T>(y, x) = ptr[x + y*m_steps];
        }
      }
    }
    else{
      AMAT_ASSERT_VALID_ARGUMENTS(ele_type == CV_8UC3, "invalid type.");
      AMAT_ASSERT_VALID_ARGUMENTS(1 == sizeof(T), "invalid type.");
      AMAT_ASSERT_VALID_ARGUMENTS(3 == m_channels, "invalid type.");
      dst.create(m_height, m_width, CV_8UC3);
      int steps = (int)dst.step;
      int layer = m_steps*m_height;
      uchar *dst_ptr = dst.data;

      for (int c = 0; c < 3; c++){
        uchar *src_ptr = m_data + c*layer;
        for (int y = 0; y < m_height; y++){
          uchar *y_src = src_ptr + y*m_steps;
          uchar *y_dst = dst_ptr + y*steps;
          for (int x = 0; x < m_width; x++){
            y_dst[x * 3 + c] = y_src[x];
          }
        }
      }
    }

  }

  // copy mat to ch-th AMat channel.
  // src must be one channel and size of element are equal.
  // if ch >= 0 : only support one channel.
  // if ch <  0 : only support CV_8UC3.
  void copy_from_mat(const cv::Mat &src, const int ch){
    if (ch >= 0){
      AMAT_ASSERT_VALID_ARGUMENTS(src.channels() == 1, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(src.rows == m_height, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(src.cols == m_width, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(src.elemSize() == sizeof(T), "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(ch >= 0 && ch < m_channels, "invalid channel destination.");
      int layer_size = m_steps*m_height;
      T* ptr = m_data + ch*layer_size;
      for (int y = 0; y < m_height; y++){
        memcpy(ptr + y*m_steps, src.row(y).data, m_width*sizeof(T));
      }
    }
    else{
      AMAT_ASSERT_VALID_ARGUMENTS(src.type() == CV_8UC3, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(m_channels == 3, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(m_width == src.cols, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(m_height == src.rows, "invalid mat.");
      AMAT_ASSERT_VALID_ARGUMENTS(1 == sizeof(T), "invalid AMat.");

      int steps = (int)(src.step);
      int layer = m_steps*m_height;
      uchar *src_ptr = src.data;

      for (int c = 0; c < 3; c++){
        uchar *dst_ptr = m_data + c*layer;
        for (int y = 0; y < m_height; y++){
          uchar *y_src = src_ptr + y*steps;
          uchar *y_dst = dst_ptr + y*m_steps;
          for (int x = 0; x < m_width; x++){
            y_dst[x] = y_src[3 * x + c];
          }
        }
      }
    }
  }

  // cout <<
  friend ostream& operator<<(ostream &output, AMat<SIZE_ALIGN, T> &am){
    if (am.m_channels == 0){
      return output;
    }
    output << endl;
    output << "width    = " << am.m_width << endl;
    output << "height   = " << am.m_height << endl;
    output << "channels = " << am.m_channels << endl;
    output << "steps    = " << am.m_steps << endl;
    output << "data = " << endl;
    output << "{" << std::endl;
    for (int c = 0; c < am.m_channels; c++){
      T* c_ptr = am.m_data + (am.m_steps*am.m_height)*c;
      output << " [";
      for (int y = 0; y < am.m_height; y++){
        T* y_ptr = c_ptr + y*am.m_steps;
        if (y != 0){
          output << "  ";
        }
        for (int x = 0; x < am.m_width - 1; x++){
          output << y_ptr[x] << ", ";
        }
        if (y == am.m_height - 1){
          output << y_ptr[am.m_width - 1] << "]" << std::endl;
        }
        else{
          output << y_ptr[am.m_width - 1] << ";" << std::endl;
        }
      }
      if (c < am.m_channels - 1){
        output << std::endl;
      }
      else{
        output << "}";// << std::endl;
      }
    }
    return output;
  }

public:
  T* m_data;
  int m_width;
  int m_height;
  int m_channels;
  int m_steps;
#undef AMAT_ASSERT_VALID_ARGUMENTS
};

typedef AMat<16, float> HMatf;
typedef AMat<16, int>   HMati;
typedef AMat<16, uchar> HMatc;

// aligned Mat with SIZE_ALIGN Bytes (e.g. 16 Bytes)
// sizeof(T) must smaller than 16 and 16%sizeof(T)=0.
template <int SIZE_ALIGN, typename T>
class GpuAMat
{
public:
#define AMAT_ASSERT_VALID_ARGUMENTS(r,str) if (!(r)){ERROR_OUT__<<":"<<str<<std::endl;throw std::invalid_argument(str);}
  // initialize with nothing.
  // AMat A; 
  // AMat *A = new AMat[N];
  GpuAMat(){
    AMAT_ASSERT_VALID_ARGUMENTS(sizeof(T) <= SIZE_ALIGN, "invalid type");
    AMAT_ASSERT_VALID_ARGUMENTS((SIZE_ALIGN%sizeof(T) == 0), "invalid type");
    m_data = NULL;
    m_width = 0; m_height = 0;
    m_steps = 0; m_channels = 0;
  }

  // initialize with another object.
  // it is the user's responsibility to ensure obj has been initialized properly.
  // Note : the dynamically allocated AMat is not been initialized.
  // AMat A = B;
  // AMat A(B);
  GpuAMat(const GpuAMat &obj){
    m_width = obj.m_width;
    m_height = obj.m_height;
    m_channels = obj.m_channels;
    m_steps = obj.m_steps;
    if (obj.m_data){
      AMAT_ASSERT_VALID_ARGUMENTS(cudaSuccess == cudaMalloc((void**)&m_data, m_channels*m_height*m_steps*sizeof(T)), "failed to create data.");
      if (NULL == m_data){
        std::cerr << "failed to allocate memory." << std::endl;
        exit(EXIT_FAILURE);
      }
      AMAT_ASSERT_VALID_ARGUMENTS(cudaSuccess == cudaMemcpy(m_data, obj.m_data, m_channels*m_height*m_steps*sizeof(T), cudaMemcpyDeviceToDevice), "copy fialed.");
    }
    else
      m_data = NULL;
  }

  // initialize with arguments
  GpuAMat(int _width, int _height, int _channels = 1){
    AMAT_ASSERT_VALID_ARGUMENTS(sizeof(T) <= SIZE_ALIGN, "invalid arguments.");
    AMAT_ASSERT_VALID_ARGUMENTS(SIZE_ALIGN%sizeof(T) == 0, "invalid arguments.");
    AMAT_ASSERT_VALID_ARGUMENTS(_width > 0 && _height > 0 && _channels > 0, "invalid arguments.");
    this->create(_width, _height, _channels);
  }

  // release resources
  ~GpuAMat(){
    if (m_data){
      cudaFree(m_data);
      m_data = NULL;
      //cout << "deleted memory." << endl;
    }
    //cout << "deleting nothing." << endl;
  }

  void create(int _width, int _height, int _channels = 1){

    // use the current memory or current memory is empty.
    if (m_width == _width && m_height == _height && _channels == m_channels){
      return;
    }

    m_width = _width;  m_height = _height; m_channels = _channels;
    // align x axis
    m_steps = (((m_width*sizeof(T)+SIZE_ALIGN - 1) / SIZE_ALIGN)*SIZE_ALIGN) / sizeof(T);

    // re-creating 
    if (m_data){ // m_data not null
      cudaFree(m_data);
      m_data = NULL;
    }
    if (m_channels*m_height*m_steps == 0){
      m_data = NULL;
      return;
    }
    AMAT_ASSERT_VALID_ARGUMENTS(cudaSuccess == cudaMalloc((void**)&m_data, m_channels*m_height*m_steps*sizeof(T)), "failed to allocate memory.");
    if (NULL == m_data){
      std::cerr << "failed to allocate memory." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  void upload(const AMat<SIZE_ALIGN, T> &src){
    AMAT_ASSERT_VALID_ARGUMENTS(src.m_height == m_height, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(src.m_width == m_width, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(src.m_steps == m_steps, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(src.m_channels == m_channels, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(m_channels*m_height*m_width != 0, "empty GpuAMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(cudaSuccess == cudaMemcpy(m_data, src.m_data, m_steps*m_height*m_channels*sizeof(T), cudaMemcpyHostToDevice), "failed to copy data.");
  }

  void download(AMat<SIZE_ALIGN, T> &dst){
    AMAT_ASSERT_VALID_ARGUMENTS(dst.m_height == m_height, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(dst.m_width == m_width, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(dst.m_steps == m_steps, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(dst.m_channels == m_channels, "invalid AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(m_channels*m_height*m_width != 0, "empty AMat.");
    AMAT_ASSERT_VALID_ARGUMENTS(cudaSuccess == cudaMemcpy(dst.m_data, m_data, m_steps*m_height*m_channels*sizeof(T), cudaMemcpyDeviceToHost), "failed to copy data.");
  }


public:
  T* m_data;
  int m_width;
  int m_height;
  int m_channels;
  int m_steps;

#undef AMAT_ASSERT_VALID_ARGUMENTS
};

// float array 16-aligned
typedef GpuAMat<16, float> DMatf;
// int array 16-aligned
typedef GpuAMat<16, int  > DMati;
// uchar array 16-aligned
typedef GpuAMat<16, uchar> DMatc;

class Init
{
public:
  static void rand_labels(HMati &labels, int min_label, int max_label){
    int len = labels.m_steps * labels.m_height * labels.m_channels;
    for (int i = 0; i < len; i++){
      labels[i] = min_label + (rand() % (max_label - min_label));
    }
  }

  static void rand_val(HMatf &val, float min_val, float max_val){
    int len = val.m_steps*val.m_height*val.m_channels;
    for (int i = 0; i < len; i++){
      val[i] = (1.f*rand() / RAND_MAX)*(max_val - min_val) + min_val;
    }
  }
protected:
private:
};

class Utility{
public:
  template<int N, typename T>
  static T abs_diff_sum(const AMat<N,T> &A, const AMat<N,T> &B){
    T abs_sum = T(0);
    int width = A.m_width;
    int height = A.m_height;
    int channels = A.m_channels;
    int steps = A.m_steps;
    int layer = steps*height;
    for (int c = 0; c < channels; c++){
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          abs_sum += abs(A[c*layer+y*steps+x]-B[c*layer+y*steps+x]);
        }
      }
    }
    return abs_sum;
  }

  // C = A - B;
  template<int N, typename T>
  static T abs_diff(AMat<N, T> &C, const AMat<N, T> &A, const AMat<N, T> &B){
    T abs_sum = T(0);
    int width = A.m_width;
    int height = A.m_height;
    int channels = A.m_channels;
    
    C.create(width, height, channels);

    int steps = A.m_steps;
    int layer = steps*height;

    for (int c = 0; c < channels; c++){
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          C[c*layer + y*steps + x] = abs(A[c*layer + y*steps + x] - B[c*layer + y*steps + x]);
        }
      }
    }
    return abs_sum;
  }

  // max(A - B);
  template<int N, typename T>
  static T max_diff(const AMat<N, T> &A, const AMat<N, T> &B){
    int width = A.m_width;
    int height = A.m_height;
    int channels = A.m_channels;
    int steps = A.m_steps;
    int layer = steps*height;
    T max_val = 0;

    for (int c = 0; c < channels; c++){
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          T val = abs(A[c*layer + y*steps + x] - B[c*layer + y*steps + x]);
          if (max_val < val){
            max_val = val;
          }
        }
      }
    }
    return max_val;
  }

  // max(A - B)/A;
  template<int N, typename T>
  static T max_relative_diff(const AMat<N, T> &A, const AMat<N, T> &B){
    int width = A.m_width;
    int height = A.m_height;
    int channels = A.m_channels;
    int steps = A.m_steps;
    int layer = steps*height;
    T max_val = 0;

    for (int c = 0; c < channels; c++){
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          T val = abs((A[c*layer + y*steps + x] - B[c*layer + y*steps + x]) / A[c*layer + y*steps + x]);
          if (max_val < val){
            max_val = val;
          }
        }
      }
    }
    return max_val;
  }

  static void to_vector(vector<int> &olabels, const HMati &ilabels){
    int width = ilabels.m_width;
    int height = ilabels.m_height;
    olabels.resize(width*height);
    int steps = ilabels.m_steps;
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        olabels[y*width + x] = ilabels[y*steps+x];
      }
    }
  }
};

#endif
