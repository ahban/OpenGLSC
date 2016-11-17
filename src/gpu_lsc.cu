// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#include "common.h"
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////
// Local K-Means 
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_center;
texture<float, cudaTextureType1D, cudaReadModeElementType> tex_features;
texture<int  , cudaTextureType1D, cudaReadModeElementType> tex_nlab;

// BX = 32!
// BY = ...
template <unsigned int BX, unsigned int BY>
__global__ 
void kernel_update_seeds(int *nlab, float *fptr, float *wptr, float *sptr, int width, int height, int fsteps, int ssteps, int x_num, int y_num, int L){
  __shared__ float smem[BY][BX];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int sx = blockIdx.x;
  int sy = blockIdx.y * blockDim.y + threadIdx.y;

  int xx = 0;  int yy = 0;
  int XXB = 0;  int YYB = 0;  int XXE = 0;  int YYE = 0;
  int sid = 0;
  
  float fea0 = 0.f;  float fea1 = 0.f;  float fea2 = 0.f;  float fea3 = 0.f;
  float fea4 = 0.f;  float fea5 = 0.f;  float fea6 = 0.f;  float fea7 = 0.f;
  float fea8 = 0.f;  float fea9 = 0.f;  float sumw = 0.f;  float wval = 0.f;  

  int idx = 0;

  int flayer = height*fsteps;
  sid = sy*x_num + sx;

  if (sy > (y_num - 1))
    goto for_sync_1;

  XXB = sptr[sid*ssteps + NN_FEATURES__    ]; // broadcast // one transaction
  YYB = sptr[sid*ssteps + NN_FEATURES__ + 1]; // broadcast // one transaction
  XXE = ((XXB + L + 31) >> 5) << 5; // xx end
  YYE = ((YYB + L + 31) >> 5) << 5; // yy end
  if (XXE > width) XXE = width;
  XXB = (XXB - L);
  if (XXB <     0)
    XXB = 0;
  else
    XXB = (XXB >> 5) << 5;
  if (YYE > height)
    YYE = height;
  YYB = (YYB - L);
  if (YYB < 0) YYB = 0;
  else YYB = (YYB >> 5) << 5;

  for (yy = YYB; yy < YYE; yy++){
    for (xx = XXB + tx; xx < XXE; xx += 32){
      idx = xx + yy*fsteps;      
      if (tex1Dfetch(tex_nlab, idx) == sid){
        wval = wptr[idx];
        sumw += wval;
        fea0 += fptr[idx             ] * wval;
        fea1 += fptr[idx + flayer    ] * wval;
        fea2 += fptr[idx + flayer * 2] * wval;
        fea3 += fptr[idx + flayer * 3] * wval;
        fea4 += fptr[idx + flayer * 4] * wval;
        fea5 += fptr[idx + flayer * 5] * wval;
        fea6 += fptr[idx + flayer * 6] * wval;
        fea7 += fptr[idx + flayer * 7] * wval;
        fea8 += fptr[idx + flayer * 8] * wval;
        fea9 += fptr[idx + flayer * 9] * wval;
      }
    }
  }

for_sync_1:

  // sum of weight
  smem[ty][tx] = sumw; __syncthreads();
  if (tx < 0x10) smem[ty][tx] = sumw = sumw + smem[ty][tx + 0x10]; __syncthreads();
  if (tx < 0x08) smem[ty][tx] = sumw = sumw + smem[ty][tx + 0x08]; __syncthreads();
  if (tx < 0x04) smem[ty][tx] = sumw = sumw + smem[ty][tx + 0x04]; __syncthreads();
  if (tx < 0x02) smem[ty][tx] = sumw = sumw + smem[ty][tx + 0x02]; __syncthreads();
  if (tx < 0x01){
    sumw = sumw + smem[ty][1];
    if (sumw == 0.f) sumw = 1.f; else sumw = 1.f / sumw;
  }

#define REDUC_____(N)   smem[ty][tx] = fea##N; __syncthreads(); \
  if (tx < 0x10) smem[ty][tx] = fea##N = fea##N + smem[ty][tx + 0x10]; __syncthreads(); \
  if (tx < 0x08) smem[ty][tx] = fea##N = fea##N + smem[ty][tx + 0x08]; __syncthreads(); \
  if (tx < 0x04) smem[ty][tx] = fea##N = fea##N + smem[ty][tx + 0x04]; __syncthreads(); \
  if (tx < 0x02) smem[ty][tx] = fea##N = fea##N + smem[ty][tx + 0x02]; __syncthreads(); \
  if (tx < 0x01){  \
    fea##N = fea##N + smem[ty][1]; \
    if (sy < y_num) \
      sptr[sid*ssteps + (N)] = fea##N *sumw; \
  }

  REDUC_____(0);
  REDUC_____(1);
  REDUC_____(2);
  REDUC_____(3);
  REDUC_____(4);
  REDUC_____(5);
  REDUC_____(6);
  REDUC_____(7);
  REDUC_____(8);
  REDUC_____(9);
  
#undef  REDUC_____
}

void gpu_update_seeds(int *nlab, float *fptr, float *wptr, float *sptr, int width, int height, int fsteps, int ssteps, int x_steps){
  //int x_steps = (int)sqrt(float(width*height) / float(K));
  int x_num = width  / x_steps;
  int y_num = height / x_steps;

  // compute a maximum fixed search region
  int aligned_height = y_num*x_steps;
  int aligned_width  = x_num*x_steps;
  int s_hx_0 = (width - aligned_width) >> 1;
  int s_hx_1 = (width - aligned_width) - s_hx_0;
  int s_hy_0 = (height - aligned_height) >> 1;
  int s_hy_1 = (height - aligned_height) - s_hy_0;
  int s_m = std::max(s_hy_1, s_hx_1) + x_steps;
  int L = x_steps - x_steps/2 + s_m;  

#define BX__ 32
#define BY__ 4
  int block_x = BX__;
  int block_y = BY__;
  int grid_x = x_num;
  int grid_y = (y_num + block_y - 1) / block_y;

  dim3 blocks(block_x, block_y);
  dim3 grids(grid_x, grid_y);

  cudaChannelFormatDesc channel_desc_int;
  channel_desc_int = cudaCreateChannelDesc<int>();
  cudaBindTexture(NULL, &tex_nlab, nlab, &channel_desc_int, fsteps*height*sizeof(int));
  kernel_update_seeds<BX__,BY__><<<grids,blocks>>>(nlab, fptr, wptr, sptr, width, height, fsteps, ssteps, x_num, y_num, L);
  cudaUnbindTexture(&tex_nlab);
#undef BX__
#undef BY__
}


template<unsigned int BX, unsigned int BY>
__global__
void kernel_update_labels(int *nlab, int *olab, float *fptr, int width, int height, int psteps, int ssteps, int layer, int x_num, int y_num){
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float smem[NN_FEATURES__][BY][BX];
  
  if (x >= width || y >= height)
    return;

  int  pid = y*psteps + x;

  float od = FLT_MAX;
  int   nl = 0; 

  float ff = 0.f; char dx = -1;
  float td = 0.f; char dy = -1;  

  int clab = olab[pid];
  int clab_y = clab / x_num;
  int clab_x = clab - clab_y*x_num;

  int ol = 0;
  int olab_x = 0;
  int olab_y = 0;

  smem[0][ty][tx] = fptr[pid            ];
  smem[1][ty][tx] = fptr[pid + layer    ];
  smem[2][ty][tx] = fptr[pid + layer * 2];
  smem[3][ty][tx] = fptr[pid + layer * 3];
  smem[4][ty][tx] = fptr[pid + layer * 4];
  smem[5][ty][tx] = fptr[pid + layer * 5];
  smem[6][ty][tx] = fptr[pid + layer * 6];
  smem[7][ty][tx] = fptr[pid + layer * 7];
  smem[8][ty][tx] = fptr[pid + layer * 8];
  smem[9][ty][tx] = fptr[pid + layer * 9];

  for (dy = -1; dy < 2; dy++){
    for (dx = -1; dx < 2; dx++){

      olab_x = clab_x + dx;
      olab_y = clab_y + dy;

      if (olab_x < 0 || olab_y < 0 || olab_x >(x_num - 1) || olab_y >(y_num - 1)){
        continue;
      }
      ol = olab_x + olab_y*x_num;

      ff = smem[0][ty][tx] - tex1Dfetch(tex_center, ol*ssteps    );  td  = ff*ff;// feature 0
      ff = smem[1][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 1);  td += ff*ff;// feature 1
      ff = smem[2][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 2);  td += ff*ff;// feature 2
      ff = smem[3][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 3);  td += ff*ff;// feature 3
      ff = smem[4][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 4);  td += ff*ff;// feature 4
      ff = smem[5][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 5);  td += ff*ff;// feature 5
      ff = smem[6][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 6);  td += ff*ff;// feature 6
      ff = smem[7][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 7);  td += ff*ff;// feature 7
      ff = smem[8][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 8);  td += ff*ff;// feature 8
      ff = smem[9][ty][tx] - tex1Dfetch(tex_center, ol*ssteps + 9);  td += ff*ff;// feature 9

      if (td < od){
        od = td;
        nl = ol;
      }
    }
  }
  nlab[pid] = nl;
}

void gpu_update_labels(int* nlab, int *olab, float* fptr, float *sptr, int width, int height, int psteps, int ssteps, int x_steps){
  int layer = height*psteps;
#define BX 32
#define BY 4
  int block_x = BX;
  int block_y = BY;

  int grid_x = (width  + block_x - 1) / block_x;
  int grid_y = (height + block_y - 1) / block_y;
  
  dim3 grids(grid_x, grid_y);
  dim3 blocks(block_x, block_y);
  
  //int x_steps = (int)sqrt(float(height*width) / (float(K)));
  int x_num = width / x_steps;
  int y_num = height / x_steps;
  //cudaFuncSetCacheConfig(kernel_update_labels_tex, cudaFuncCachePreferL1);
  // bind texture memory
  cudaChannelFormatDesc channel_desc_float;
  channel_desc_float = cudaCreateChannelDesc<float>();
  cudaBindTexture(NULL, &tex_center,   sptr, &channel_desc_float, x_num*y_num*ssteps*sizeof(float));
  cudaBindTexture(NULL, &tex_features, fptr, &channel_desc_float, NN_FEATURES__*psteps*height*sizeof(float));
  
  kernel_update_labels<BX,BY><<<grids, blocks>>>(nlab, olab, fptr, width, height, psteps, ssteps, layer, x_num, y_num);
  
  cudaUnbindTexture(&tex_center);
  cudaUnbindTexture(&tex_features);
#undef BX
#undef BY
}
// end Local K-Means 
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Color converter
__global__
void kernel_bgr2lab(float *plab, uchar* pbgr, int width, int height, int lab_steps, int lab_layer, int bgr_steps, int bgr_layer){
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  if (x >= width || y >= height){
    return;
  }
  float R = normalizer__*(float)(pbgr[y*bgr_steps + x + bgr_layer * 2]);
  float G = normalizer__*(float)(pbgr[y*bgr_steps + x + bgr_layer]);
  float B = normalizer__*(float)(pbgr[y*bgr_steps + x]);

  if (R <= 0.04045f)	R = R / 12.92f;
  else				R = pow((R + 0.055f) / 1.055f, 2.4f);
  if (G <= 0.04045f)	G = G / 12.92f;
  else				G = pow((G + 0.055f) / 1.055f, 2.4f);
  if (B <= 0.04045f)	B = B / 12.92f;
  else				B = pow((B + 0.055f) / 1.055f, 2.4f);

  float X = R*0.412453f + G*0.357580f + B*0.180423f;
  float Y = R*0.212671f + G*0.715160f + B*0.072169f;
  float Z = R*0.019334f + G*0.119193f + B*0.950227f;

  Y = Y * iYr__;
  Z = Z * iZr__;
  X = X * iXr__;

  if (X > epsilon__)	X = pow(X, 1.f / 3.f);
  else			 X = (kappa__*X + 16.f) / 116.f;
  if (Y > epsilon__)	Y = pow(Y, 1.f / 3.f);
  else			 Y = (kappa__*Y + 16.f) / 116.f;
  if (Z > epsilon__)	Z = pow(Z, 1.f / 3.f);
  else			 Z = (kappa__*Z + 16.f) / 116.f;

  plab[y*lab_steps + x] = (uchar)((116.f*Y - 16.f) / 100.f * 255.f + 0.5f);
  plab[y*lab_steps + x + lab_layer] = (uchar)(500.f*(X - Y) + 128.f + 0.5f);
  plab[y*lab_steps + x + lab_layer * 2] = (uchar)(200.f*(Y - Z) + 128.f + 0.5f);

}

void gpu_bgr2lab(float *plab, uchar* pbgr, int width, int height, int lab_steps, int bgr_steps){
  int bgr_layer = height*bgr_steps;
  int lab_layer = height*lab_steps;

  int block_x = 32;
  int block_y = 4;
  int grid_x = (width + block_x - 1) / block_x;
  int grid_y = (height + block_y - 1) / block_y;
  dim3 grids(grid_x, grid_y, 1);
  dim3 blocks(block_x, block_y, 1);
  kernel_bgr2lab <<<grids, blocks>>>(plab, pbgr, width, height, lab_steps, lab_layer, bgr_steps, bgr_layer);
}
// end Color converter
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// initialize features
__global__
void kernel_init_features_1(float *fptr, float *cptr, const int width, const int height, const int steps, const int layer, const float CC, const float CD, const float cn, const float sn){
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  if (x >= width || y >= height){
    return;
  }

  float theta = 0.f;
  int ind = y*steps + x;

  // L
  theta = cptr[ind] * cn;
  fptr[ind] = CC * cos(theta); // layer 0
  fptr[ind + layer] = CC * sin(theta); // layer 1

  // A 
  theta = cptr[ind + layer] * cn;
  fptr[ind + layer * 2] = 2.55f*CC*cos(theta); // layer 2
  fptr[ind + layer * 3] = 2.55f*CC*sin(theta); // layer 3

  // B 
  theta = cptr[ind + layer * 2] * cn;
  fptr[ind + layer * 4] = 2.55f*CC*cos(theta); // layer 4
  fptr[ind + layer * 5] = 2.55f*CC*sin(theta); // layer 5

  // X
  theta = ((float)(x)* sn);
  fptr[ind + layer * 6] = CD*cos(theta); // layer 6
  fptr[ind + layer * 7] = CD*sin(theta); // layer 7

  // Y
  theta = ((float)(y)* sn);
  fptr[ind + layer * 8] = CD*cos(theta); // layer 8
  fptr[ind + layer * 9] = CD*sin(theta); // layer 9 

}

void gpu_init_features_1(float *fptr, float *cptr, const int width, const int height, const int steps, const int x_steps, const float CC, const float CD, const int num){
  if (num != NN_FEATURES__){
    ERROR_OUT__ << "not support yet" << std::endl;
    exit(EXIT_FAILURE);
  }
  const float cn = (1.f / 255.f) * LSC_PI * 0.5f;
  //const float sn = LSC_PI * 0.5f * sqrt(float(K) / float(width*height));
  const float sn = LSC_PI * 0.5f / float(x_steps);

  const int layer = height*steps;

  int block_x = 32;
  int block_y = 4;
  int grid_x = (width + block_x - 1) / block_x;
  int grid_y = (height + block_y - 1) / block_y;
  dim3 grids(grid_x, grid_y, 1);
  dim3 blocks(block_x, block_y, 1);

  kernel_init_features_1 <<<grids, blocks>>>(fptr, cptr, width, height, steps, layer, CC, CD, cn, sn);
  if (cudaSuccess != cudaGetLastError()){ ERROR_OUT__ << std::endl; exit(EXIT_FAILURE); };
}

__global__
void kernel_init_features_2(float *fptr, float *wptr, const float *sigmas, int width, int height, int steps, int layer){
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int x = threadIdx.x + blockIdx.x*blockDim.x;

  int indf = y*steps + x;
  float ww = 0.f;

  if (x >= width || y >= height){
    return;
  }

  ww  = fptr[indf            ] * sigmas[0];
  ww += fptr[indf + layer    ] * sigmas[1];
  ww += fptr[indf + layer * 2] * sigmas[2];
  ww += fptr[indf + layer * 3] * sigmas[3];
  ww += fptr[indf + layer * 4] * sigmas[4];
  ww += fptr[indf + layer * 5] * sigmas[5];
  ww += fptr[indf + layer * 6] * sigmas[6];
  ww += fptr[indf + layer * 7] * sigmas[7];
  ww += fptr[indf + layer * 8] * sigmas[8];
  ww += fptr[indf + layer * 9] * sigmas[9];

  wptr[indf] = ww;

  ww = 1.f / ww;

  fptr[indf            ] *= ww;
  fptr[indf + layer    ] *= ww;
  fptr[indf + layer * 2] *= ww;
  fptr[indf + layer * 3] *= ww;
  fptr[indf + layer * 4] *= ww;
  fptr[indf + layer * 5] *= ww;
  fptr[indf + layer * 6] *= ww;
  fptr[indf + layer * 7] *= ww;
  fptr[indf + layer * 8] *= ww;
  fptr[indf + layer * 9] *= ww;

}
void gpu_init_features_2(float *fptr, float *wptr, const float *sigmas, const int width, const int height, const int steps){

  int layer = height*steps;

  int block_x = 32;
  int block_y = 4;

  int grid_x = (width + block_x - 1) / block_x;
  int grid_y = (height + block_y - 1) / block_y;

  dim3 grids(grid_x, grid_y);
  dim3 blocks(block_x, block_y);
  kernel_init_features_2<<<grids, blocks>>>(fptr, wptr, sigmas, width, height, steps, layer);
  if (cudaSuccess != cudaGetLastError()){ ERROR_OUT__ << std::endl; exit(EXIT_FAILURE); };
}
// end initialize features
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// initialize labels
__global__
void kernel_init_labels(int *plabs, int width, int height, int steps, int x_steps, int rhl, int rhu, int wn, int hn){
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  if (x > (width-1) || y > (height-1)){
    return;
  }
  int nx = (x - rhl) / x_steps;
  int ny = (y - rhu) / x_steps;
  if (nx < 0)  nx = 0;
  else if (nx >(wn - 1)) nx = wn - 1;
  if (ny < 0)  ny = 0;
  else if (ny >(hn - 1)) ny = hn - 1;
  plabs[y*steps + x] = nx + ny*wn;
}

void gpu_init_labels(int *plabs, int width, int height, int steps, int x_steps){
  int width_num  = width / x_steps;
  int height_num = height / x_steps;

  int aligned_height = height_num*x_steps;
  int aligned_width  = width_num *x_steps;

  int reminder_width = width - aligned_width;
  int reminder_height = height - aligned_height;

  int reminder_half_width_left  = reminder_width  >> 1;
  int reminder_half_width_right = reminder_width  - reminder_half_width_left;
  int reminder_half_height_up   = reminder_height >> 1;
  int reminder_half_height_down = reminder_height - reminder_half_height_up;

  int block_x = 32;
  int block_y = 4;
  int grid_x = (width + block_x - 1) / block_x;
  int grid_y = (height + block_y - 1) / block_y;
  dim3 grids(grid_x, grid_y, 1);
  dim3 blocks(block_x, block_y, 1);
  kernel_init_labels<<<grids, blocks>>>(plabs, width, height, steps, x_steps, reminder_half_width_left, reminder_half_height_up, width_num, height_num);
  if (cudaSuccess != cudaGetLastError()){ ERROR_OUT__ << std::endl; exit(EXIT_FAILURE); };
}

// end initialize labels
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// initialize seeds
__global__
void kernel_init_seeds(float *sptr, float *fptr, int x_num, int y_num, int x_steps, int half_steps, int x_half_rest, int y_half_rest, int fsteps, int ssteps, int flayer){
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int x = threadIdx.x + blockIdx.x*blockDim.x;

  if (y >= y_num || x >= x_num){
    return;
  }

  int fid = (y*x_steps + y_half_rest + half_steps)*fsteps + x_half_rest + x*x_steps + half_steps;
  int cid = y*x_num + x;
  sptr[cid*ssteps     ] = fptr[fid             ];
  sptr[cid*ssteps +  1] = fptr[fid + flayer    ];
  sptr[cid*ssteps +  2] = fptr[fid + flayer * 2];
  sptr[cid*ssteps +  3] = fptr[fid + flayer * 3];
  sptr[cid*ssteps +  4] = fptr[fid + flayer * 4];
  sptr[cid*ssteps +  5] = fptr[fid + flayer * 5];
  sptr[cid*ssteps +  6] = fptr[fid + flayer * 6];
  sptr[cid*ssteps +  7] = fptr[fid + flayer * 7];
  sptr[cid*ssteps +  8] = fptr[fid + flayer * 8];
  sptr[cid*ssteps +  9] = fptr[fid + flayer * 9];
  sptr[cid*ssteps + 10] = float(x_half_rest + x*x_steps + half_steps);
  sptr[cid*ssteps + 11] = float(y*x_steps + y_half_rest + half_steps);
}

void gpu_init_seeds(float *sptr, float *fptr, int ssteps, int width, int height, int fsteps, int x_steps){

  //int x_steps = (int)sqrt(float(width*height) / float(K));
  int x_num = width / x_steps;
  int y_num = height / x_steps;

  int x_half_rest = (width - x_num*x_steps) >> 1;
  int y_half_rest = (height - y_num*x_steps) >> 1;
  int half_steps = x_steps >> 1;

  int flayer = fsteps*height;

  int block_x = 32;
  int block_y = 4;
  int grid_x = (x_num + block_x - 1) / block_x;
  int grid_y = (y_num + block_y - 1) / block_y;
  dim3 grids(grid_x, grid_y, 1);
  dim3 blocks(block_x, block_y, 1);

  kernel_init_seeds<<<grids, blocks>>>(sptr, fptr, x_num, y_num, x_steps, half_steps, x_half_rest, y_half_rest, fsteps, ssteps, flayer);
  if (cudaSuccess != cudaGetLastError()){ ERROR_OUT__ << std::endl; exit(EXIT_FAILURE); };
}
// end initialize seeds
//////////////////////////////////////////////////////////////////////////



