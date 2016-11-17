// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#include "AMatrix.hpp"
#include <cuda.h>
#include "common.h"

// dim3(GS)
// dim3(BS)
// /*number of layers*/
template<unsigned int GS, unsigned int BS, unsigned int N, class T>
__global__ void kernel_reduction(
  T *odata, T *idata,
  const unsigned int width,
  const unsigned int height,
  const unsigned int steps, 
  const unsigned int layer /*layer size = steps*height*/
){
  __shared__ T smem[BS];
  const unsigned int tid = threadIdx.x;
  const unsigned int ty  = blockIdx.x;
  unsigned int i = 0;
  unsigned int y = 0;
  unsigned int x = 0;
  T sumv = 0;

  // for each layer
  for (; i < N; i++){
    sumv = 0;
    for (y = ty; y < height; y+=GS){
      for (x = tid; x < width; x+=BS){
        sumv += idata[y*steps + x];
      }
    }

    smem[tid] = sumv;
    __syncthreads();

    if (BS >= 512){ if (tid < 256) smem[tid] = sumv = sumv + smem[tid + 256]; __syncthreads(); }
    if (BS >= 256){ if (tid < 128) smem[tid] = sumv = sumv + smem[tid + 128]; __syncthreads(); }
    if (BS >= 128){ if (tid <  64) smem[tid] = sumv = sumv + smem[tid +  64]; __syncthreads(); }
    if (BS >=  64){ if (tid <  32) smem[tid] = sumv = sumv + smem[tid +  32]; __syncthreads(); }
    if (BS >=  32){ if (tid <  16) smem[tid] = sumv = sumv + smem[tid +  16]; __syncthreads(); }
    if (BS >=  16){ if (tid <   8) smem[tid] = sumv = sumv + smem[tid +   8]; __syncthreads(); }
    if (BS >=   8){ if (tid <   4) smem[tid] = sumv = sumv + smem[tid +   4]; __syncthreads(); }
    if (BS >=   4){ if (tid <   2) smem[tid] = sumv = sumv + smem[tid +   2]; __syncthreads(); }
    if (BS >= 2){ if (tid < 1){ odata[i*GS + ty] = sumv + smem[1]; } }
    idata += layer;
  }
}
// dim3(N)  /*number of layers*/
// dim3(BS)  
template<unsigned int BS, unsigned int N, class T>
__global__ void kernel_reduction(T* odata, const T* idata){

  __shared__ T smem[BS];
  T sumv = 0;
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x;
  smem[tid] = sumv = idata[i*BS + tid];
  __syncthreads();
  if (BS >= 512){ if (tid < 256) smem[tid] = sumv = sumv + smem[tid + 256]; __syncthreads(); }
  if (BS >= 256){ if (tid < 128) smem[tid] = sumv = sumv + smem[tid + 128]; __syncthreads(); }
  if (BS >= 128){ if (tid <  64) smem[tid] = sumv = sumv + smem[tid +  64]; __syncthreads(); }
  if (BS >=  64){ if (tid <  32) smem[tid] = sumv = sumv + smem[tid +  32]; __syncthreads(); }
  if (BS >=  32){ if (tid <  16) smem[tid] = sumv = sumv + smem[tid +  16]; __syncthreads(); }
  if (BS >=  16){ if (tid <   8) smem[tid] = sumv = sumv + smem[tid +   8]; __syncthreads(); }
  if (BS >=   8){ if (tid <   4) smem[tid] = sumv = sumv + smem[tid +   4]; __syncthreads(); }
  if (BS >=   4){ if (tid <   2) smem[tid] = sumv = sumv + smem[tid +   2]; __syncthreads(); }
  if (BS >= 2){ if (tid < 1) { odata[i] = sumv + smem[1]; } }
}

// dim3(N)  /*number of layers*/
// dim3(BS)  
template<unsigned int BS, unsigned int N, class T>
__global__ void kernel_reduction_factor(T* odata, const T* idata, const float factor){

  __shared__ T smem[BS];
  T sumv = 0;
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x;
  smem[tid] = sumv = idata[i*BS + tid];
  __syncthreads();
  if (BS >= 512){ if (tid < 256) smem[tid] = sumv = sumv + smem[tid + 256]; __syncthreads(); }
  if (BS >= 256){ if (tid < 128) smem[tid] = sumv = sumv + smem[tid + 128]; __syncthreads(); }
  if (BS >= 128){ if (tid <  64) smem[tid] = sumv = sumv + smem[tid +  64]; __syncthreads(); }
  if (BS >=  64){ if (tid <  32) smem[tid] = sumv = sumv + smem[tid +  32]; __syncthreads(); }
  if (BS >=  32){ if (tid <  16) smem[tid] = sumv = sumv + smem[tid +  16]; __syncthreads(); }
  if (BS >=  16){ if (tid <   8) smem[tid] = sumv = sumv + smem[tid +   8]; __syncthreads(); }
  if (BS >=   8){ if (tid <   4) smem[tid] = sumv = sumv + smem[tid +   4]; __syncthreads(); }
  if (BS >=   4){ if (tid <   2) smem[tid] = sumv = sumv + smem[tid +   2]; __syncthreads(); }
  if (BS >= 2){ if (tid < 1) { odata[i] = (sumv + smem[1])*factor; } }
}

template<unsigned int GS, unsigned int BS, unsigned int N, class T> void 
run_sum_(T *odata, T *idata, const unsigned int width, const unsigned int height, const unsigned int steps)
{
  if (GS > N){
    kernel_reduction<GS, BS, N, T> <<<GS, BS>>>(odata + GS, idata, width, height, steps, steps*height);
    kernel_reduction<GS, N, T> <<<N, GS>>>(odata, odata + GS);
  }
  else{
    cerr << "not support yet." << endl;
    exit(EXIT_FAILURE);
  }
}

template void run_sum_<REDUCTOR_GS__, REDUCTOR_BS__, NN_FEATURES__, float>
(float *odata, float *idata, const unsigned int width, const unsigned int height, const unsigned int steps);
template void run_sum_<REDUCTOR_GS__, REDUCTOR_BS__, NN_FEATURES__, int  >
(int   *odata, int   *idata, const unsigned int width, const unsigned int height, const unsigned int steps);


template<unsigned int GS, unsigned int BS, unsigned int N, class T> void
run_mean_(T *odata, T *idata, const unsigned int width, const unsigned int height, const unsigned int steps)
{
  if (GS > N){
    kernel_reduction<GS, BS, N, T> << <GS, BS >> >(odata + GS, idata, width, height, steps, steps*height);
    kernel_reduction_factor<GS, N, T> << <N, GS >> >(odata, odata + GS, 1.f/float(width*height));
  }
  else{
    cerr << "not support yet." << endl;
    exit(EXIT_FAILURE);
  }
}

template void run_mean_<REDUCTOR_GS__, REDUCTOR_BS__, NN_FEATURES__, float>
(float *odata, float *idata, const unsigned int width, const unsigned int height, const unsigned int steps);
template void run_mean_<REDUCTOR_GS__, REDUCTOR_BS__, NN_FEATURES__, int  >
(int   *odata, int   *idata, const unsigned int width, const unsigned int height, const unsigned int steps);