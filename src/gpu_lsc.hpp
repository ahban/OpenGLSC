// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#ifndef __GPU_LSC_HPP__
#define __GPU_LSC_HPP__

#include "common.h"
#include "AMatrix.hpp"
#include "Adaptor.hpp"
#include "Reductor.hpp"

void gpu_bgr2lab(float *plab, uchar* pbgr, int width, int height, int lab_steps, int bgr_steps);

void gpu_init_features_1(float *fptr, float *cptr, const int width, const int height, const int steps, const int x_steps, const float CC, const float CD, const int num);

void gpu_init_features_2(float *fptr, float *wptr, const float *sigmas, const int width, const int height, const int steps);

void gpu_init_seeds(float *sptr, float *fptr, int ssteps, int width, int height, int fsteps, int x_steps);

void gpu_init_labels(int *plabs, int width, int height, int steps, int x_steps);

void gpu_update_labels(int* nlab, int *olab, float* fptr, float *sptr, int width, int height, int psteps, int ssteps, int x_steps);
void gpu_update_seeds(int *nlab, float *fptr, float *wptr, float *sptr, int width, int height, int fsteps, int ssteps, int x_steps);


class GPU_LSC
{
public:
  // note src = BGR and in [0,255]
  void segmentation_K(const HMatc &src, const int K, const float CC = 20.f, const float ratio = 0.15f, int iter_num = 10){

    //int iter_num = g_iter_num;

    const int width = src.m_width;
    const int height = src.m_height;

    const int x_steps = (int)sqrt(float(width*height) / float(K));
    const int x_num = width / x_steps;
    const int y_num = height / x_steps;

    float CD = CC*ratio;

    // allocate memory
    d_BGR.create(width, height, 3);
    d_LAB.create(width, height, 3);
    d_F.create(width, height, NN_FEATURES__);
    d_W.create(width, height, 1);
    d_L0.create(width, height, 1);
    d_L1.create(width, height, 1);
    h_LF.create(width, height, 1);
    d_S.create(NN_FEATURES__ + 2, x_num*y_num, 1); // each line : f0,f1,...,f9,x,y

    // upload to GPU memory
    d_BGR.upload(src);

    // LSC pipeline

    // 1. -> 0
    // color converter
    gpu_bgr2lab(d_LAB.m_data, d_BGR.m_data, width, height, d_LAB.m_steps, d_BGR.m_steps);

    // 2. -> 1
    // feature init
    gpu_init_features_1(d_F.m_data, d_LAB.m_data, width, height, d_F.m_steps, x_steps, CC, CD, d_F.m_channels);
    d_reductorf.run_mean(d_F.m_data, width, height, d_F.m_steps, d_F.m_channels, false);
    gpu_init_features_2(d_F.m_data, d_W.m_data, d_reductorf.d_o, width, height, d_F.m_steps);

    // 3. -> 2
    // initialize cluster seeds
    gpu_init_seeds(d_S.m_data, d_F.m_data, d_S.m_steps, width, height, d_F.m_steps, x_steps);

    // 4. -> 0
    // initialize labels
    gpu_init_labels(d_L0.m_data, width, height, d_L0.m_steps, x_steps);
		gpu_update_labels(d_L1.m_data, d_L0.m_data, d_F.m_data, d_S.m_data, width, height, d_F.m_steps, d_S.m_steps, x_steps);

    for (int it = 1; it < iter_num; it++){
			gpu_update_seeds(d_L1.m_data, d_F.m_data, d_W.m_data, d_S.m_data, width, height, d_F.m_steps, d_S.m_steps, x_steps);
			gpu_update_labels(d_L1.m_data, d_L0.m_data, d_F.m_data, d_S.m_data, width, height, d_F.m_steps, d_S.m_steps, x_steps);
    }
    d_L1.download(h_LF);
    //return h_LF;
  }

  void segmentation_S(const HMatc &src, const int S, const float CC = 20.f, const float ratio = 0.15f, int iter_num = 10){
#define GGTB__(A) int64 A;cudaDeviceSynchronize();A = cv::getTickCount()
#define GGTEE_(A,E) cudaDeviceSynchronize(); (E = 1.E3 * double(cv::getTickCount() - A)/double(cv::getTickFrequency()))
    //int iter_num = g_iter_num;

		double time_temp;

    const int width = src.m_width;
    const int height = src.m_height;

    const int x_steps = S;
    const int x_num = width / x_steps;
    const int y_num = height / x_steps;

    float CD = CC*ratio;

    // allocate memory
    d_BGR.create(width, height, 3);
    d_LAB.create(width, height, 3);
    d_F.create(width, height, NN_FEATURES__);
    d_W.create(width, height, 1);
    d_L0.create(width, height, 1);
    d_L1.create(width, height, 1);
    h_LF.create(width, height, 1);
    d_S.create(NN_FEATURES__ + 2, x_num*y_num, 1); // each line : f0,f1,...,f9,x,y

    // upload to GPU memory
		GGTB__(bgr_upload);
    d_BGR.upload(src);
		GGTEE_(bgr_upload, time_data_transfer);


    // LSC pipeline
		GGTB__(glsc_main);
    // 1. -> 0
    // color converter
    gpu_bgr2lab(d_LAB.m_data, d_BGR.m_data, width, height, d_LAB.m_steps, d_BGR.m_steps);
    // 2. -> 1
    // feature init
    gpu_init_features_1(d_F.m_data, d_LAB.m_data, width, height, d_F.m_steps, x_steps, CC, CD, d_F.m_channels);
    d_reductorf.run_mean(d_F.m_data, width, height, d_F.m_steps, d_F.m_channels, false);
    gpu_init_features_2(d_F.m_data, d_W.m_data, d_reductorf.d_o, width, height, d_F.m_steps);

    // 3. -> 2
    // initialize cluster seeds
    gpu_init_seeds(d_S.m_data, d_F.m_data, d_S.m_steps, width, height, d_F.m_steps, x_steps);

    // 4. -> 0
    // initialize labels
    gpu_init_labels(d_L0.m_data, width, height, d_L0.m_steps, x_steps);
		gpu_update_labels(d_L1.m_data, d_L0.m_data, d_F.m_data, d_S.m_data, width, height, d_F.m_steps, d_S.m_steps, x_steps);

    for (int it = 1; it < iter_num; it++){
			gpu_update_seeds(d_L1.m_data, d_F.m_data, d_W.m_data, d_S.m_data, width, height, d_F.m_steps, d_S.m_steps, x_steps);
			gpu_update_labels(d_L1.m_data, d_L0.m_data, d_F.m_data, d_S.m_data, width, height, d_F.m_steps, d_S.m_steps, x_steps);
    }
		GGTEE_(glsc_main, time_mod_lsc);

		GGTB__(lable_download);
    d_L1.download(h_LF);
		GGTEE_(lable_download, time_temp);
		time_data_transfer += time_temp;
		//return h_LF;
  }
public:
	double time_data_transfer;
	double time_mod_lsc;

  DMatc d_BGR;
  DMatf d_LAB;

  DMatf d_F;
  DMatf d_W;

  DMatf d_S;

  DMati d_L0; // labels 0, needs to be initialized firstly.
  DMati d_L1; // labels 1, served as a temporal label memory.
  HMati h_LF; // final labels transfered back from GPU.

  DReductorf d_reductorf;
};




#endif