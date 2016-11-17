// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#ifndef __REDUCTOR_HPP__
#define __REDUCTOR_HPP__

#include "AMatrix.hpp"
#include "common.h"


template<unsigned int GS, unsigned int BS, unsigned int N, class T> void
run_sum_(T *odata, T *idata, const unsigned int width, const unsigned int height, const unsigned int steps);

template<unsigned int GS, unsigned int BS, unsigned int N, class T> void
run_mean_(T *odata, T *idata, const unsigned int width, const unsigned int height, const unsigned int steps);


template<unsigned int GS, unsigned int BS, unsigned int NN, class T>
class DReductor
{
#define REDUCTOR_ASSERT(r) if(!(r)){cerr<<"ERROR:"<<__FILE__<<":"<<__LINE__<<endl;exit(EXIT_FAILURE);}
public:
  DReductor(){
    REDUCTOR_ASSERT(cudaSuccess == cudaMallocHost((void**)&h_o, (NN + 1)*GS*sizeof(T)));
    REDUCTOR_ASSERT(cudaSuccess == cudaMalloc((void**)&d_o, (NN + 1)*GS*sizeof(T)));
  }
  ~DReductor(){
    if (h_o){
      REDUCTOR_ASSERT(cudaSuccess == cudaFreeHost(h_o)); h_o = NULL;
      REDUCTOR_ASSERT(cudaSuccess == cudaFree(d_o)); d_o = NULL;
    }
  }

  void run_sum(T* idata, const unsigned int width, const unsigned int height, const unsigned int steps, const unsigned int num, const bool needs_transfer = true){
    if (num != NN){
      cerr << "[ERROR] : numer of layer not be the same." << endl;
      exit(EXIT_FAILURE);
    }
    run_sum_<GS, BS, NN, T>(d_o, idata, width, height, steps);
    if (needs_transfer){
      REDUCTOR_ASSERT(cudaSuccess == cudaMemcpy(h_o, d_o, sizeof(T)*NN, cudaMemcpyDeviceToHost));
    }
  }

  void run_mean(T* idata, const unsigned int width, const unsigned int height, const unsigned int steps, const unsigned int num, const bool needs_transfer = true){
    if (num != NN){
      cerr << "[ERROR] : numer of layer not be the same." << endl;
      exit(EXIT_FAILURE);
    }
    run_mean_<GS, BS, NN, T>(d_o, idata, width, height, steps);
    
    if (needs_transfer){
      REDUCTOR_ASSERT(cudaSuccess == cudaMemcpy(h_o, d_o, sizeof(T)*NN, cudaMemcpyDeviceToHost));
    }
  }
  
public:
  T* h_o;
  T* d_o;
#undef REDUCTOR_ASSERT
};

typedef DReductor<REDUCTOR_GS__, REDUCTOR_BS__, NN_FEATURES__, float> DReductorf;
typedef DReductor<REDUCTOR_GS__, REDUCTOR_BS__, NN_FEATURES__, int  > DReductori;



template<unsigned int NN, unsigned int ALIGN_SIZE, class T>
class HReductor
{
public:
  HReductor(){
    h_o = (T*)Allocate::align_allocate(NN*sizeof(T), ALIGN_SIZE);
    if (h_o == NULL){
      cerr << "Error : " << __FILE__ << ":" << __LINE__ << ":failed to allocate memory." << endl;
      exit(EXIT_FAILURE);
    }
  }
  ~HReductor(){
    if (h_o){      
      Allocate::align_release(h_o); h_o = NULL;
    }
  }

  void run_sum(T* idata, const unsigned int width, const unsigned int height, const unsigned int steps, const unsigned int num, bool doomp){
    if (NN != num){
      cerr << "[ERROR] : numer of layer not be the same." << endl;
      exit(EXIT_FAILURE);
    }    
    int layer = height*steps;
#pragma omp parallel for schedule(dynamic) if(doomp)
    for (int i = 0; i < NN; i++){
      h_o[i] = 0;
      T* pi = idata + i*layer;
      for (unsigned int y = 0; y < height; y++){
        for (unsigned int x = 0; x < width; x++){
          h_o[i] += pi[y*steps + x];
        }
      }      
    }
  }

  void run_mean(T* idata, const unsigned int width, const unsigned int height, const unsigned int steps, const unsigned int num, bool doomp){
    if (NN != num){
      cerr << "[ERROR] : numer of layer not be the same." << endl;
      exit(EXIT_FAILURE);
    }
    int layer = height*steps;

#pragma omp parallel for schedule(dynamic) if(doomp)
    for (int i = 0; i < NN; i++){
      h_o[i] = 0;
      T* pi = idata + i*layer;

      for (unsigned int y = 0; y < height; y++){
        for (unsigned int x = 0; x < width; x++){
          h_o[i] += pi[y*steps + x];
        }
      }

      h_o[i] = T(float(h_o[i]) / float(width*height));
    }
  }

public:
  T* h_o;
};


typedef HReductor<NN_FEATURES__, 16, float> HReductorf;
typedef HReductor<NN_FEATURES__, 16, int  > HReductori;

#endif