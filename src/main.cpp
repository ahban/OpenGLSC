// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#include "common.h"
#include "AMatrix.hpp"
#include "Adaptor.hpp"
#include "Reductor.hpp"

#include "gpu_lsc.hpp"
#include "connectivity.hpp"

#include <opencv2/opencv.hpp>
using namespace cv;

inline int get_num_sm_cores(int major, int minor){
  switch ((major<<4)+minor){
    case 0x20: return 32;  // Fermi   (SM 2.0) GF100 class
    case 0x21: return 48;  // Fermi   (SM 2.1) GF10x class
    case 0x30: return 192; // Kepler  (SM 3.0) GK10x class
    case 0x32: return 192; // Kepler  (SM 3.2) GK10x class
    case 0x35: return 192; // Kepler  (SM 3.5) GK11x class
    case 0x37: return 192; // Kepler  (SM 3.7) GK21x class
    case 0x50: return 128; // Maxwell (SM 5.0) GM10x class
    case 0x52: return 128; // Maxwell (SM 5.2) GM20x class
    case 0x53: return 128; // Maxwell (SM 5.3) GM20x class
    case 0x60: return 64;  // Pascal  (SM 6.0) GP100 class
    case 0x61: return 128; // Pascal  (SM 6.1) GP10x class
    case 0x62: return 128; // Pascal  (SM 6.2) GP10x class
    default: return -1;
  }
}

#include "tools.hpp"

class Evaluate
{
public:
	void run(string image_name, int xsteps=16){

		vector<int> m_labels;
		
		// some fixed parameters stated in our paper.
		float CC = 20.f;
		float ratio = 0.1f;
		int iter_num = 5;		
		
		
		// some other variables.
		
		const int total_run = 5;
		Mat src;	
		
		int64 TT;
		double ETT;
		double time_glsc_kernels;
		double time_glsc_datatrans;
		double time_glsc_post;
		double time_glsc;
    #define __TT TT = cv::getTickCount()
    #define __ET ETT = 1.E3 * double(cv::getTickCount() - TT) / double(cv::getTickFrequency())
		
		
		int gpu_counts;
		cudaGetDeviceCount(&gpu_counts);
		if (gpu_counts < 1){
			cout << "no GPU has been found." << endl;
			exit(EXIT_FAILURE);
		}
		
		
		// for each GPU
		Connectivity_equal ctl;
		for (int g = 0; g < gpu_counts; g++){
			src = imread(image_name, CV_LOAD_IMAGE_COLOR);
			if (src.empty()){
				cerr << "Failed to read the image you specified." << endl;
				exit(EXIT_FAILURE);
			}
			Adaptor::Mat2HMatc(m_src_bgr, src);
			int width  = src.cols;
			int height = src.rows;

			cudaSetDevice(g);
			GPU_LSC gpu_lsc;
			cudaDeviceProp cudaprop;
			cudaGetDeviceProperties(&cudaprop, g);
			
			cout << "##################################################################" << endl;
			cout << "GPU "<< g << " Configuration: " << endl;
			cout << "number of cores : " << cudaprop.multiProcessorCount << "x" << get_num_sm_cores(cudaprop.major, cudaprop.minor) << "=" << cudaprop.multiProcessorCount * get_num_sm_cores(cudaprop.major, cudaprop.minor) << endl;
			cout << "GPU name        : " << cudaprop.name << endl;
			cout << "Clock Rate      : " << double(cudaprop.clockRate) / 1024. << "MHz" << endl;
			cout << "Global Memory   : " << double(cudaprop.totalGlobalMem) / 1024. / 1024. / 1024. << "GB" << endl;
			cout << "Bandwidth       : " << cudaprop.memoryBusWidth << "*" << 2 << "*" << cudaprop.memoryClockRate << "/" << "(1024.*1024.*8.)" << " = " << (cudaprop.memoryBusWidth * 2. * cudaprop.memoryClockRate) / (1024.*1024.*8.) << "GB/s, ";			
			cout << endl;
					
			
			// warming up
			gpu_lsc.segmentation_S(m_src_bgr, xsteps, CC, ratio, iter_num);
			gpu_lsc.segmentation_S(m_src_bgr, xsteps, CC, ratio, iter_num);

			// measure time for kernels and data transfer.
			time_glsc_kernels   = 0;
			time_glsc_datatrans = 0;
			for (int t = 0; t < total_run; t++){
				gpu_lsc.segmentation_S(m_src_bgr, xsteps, CC, ratio, iter_num);
				time_glsc_kernels   += gpu_lsc.time_mod_lsc;
				time_glsc_datatrans += gpu_lsc.time_data_transfer;
			}
			time_glsc_kernels   = time_glsc_kernels   / double(total_run);
			time_glsc_datatrans = time_glsc_datatrans / double(total_run);

			// measure time for the post-processing.
			time_glsc_post = 0;
			for (int t = 0; t < total_run; t++){
				gpu_lsc.segmentation_S(m_src_bgr, xsteps, CC, ratio, iter_num);
				__TT;
				ctl.merge_small_regions(gpu_lsc.h_LF, (xsteps * xsteps) >> 2);
				__ET;
				time_glsc_post += ETT;
			}
			time_glsc_post = time_glsc_post / double(total_run);
			

			time_glsc = time_glsc_kernels + time_glsc_datatrans + time_glsc_post;
			// display segmentation results.
			{
				vector<int> labels;
				Adaptor::AMatLabels2Vectors(labels, gpu_lsc.h_LF);
				string window_name = string("") + to_string(1000./time_glsc) + "FPS";
				imshow(window_name, Draw::contour(&(labels[0]), src));
				waitKey(10);
			}
			
			cout << "image size      : " << width << "x" << height << endl;			
			cout << "v_x             : " << xsteps << endl;
			cout << "kernel time     : " << time_glsc_kernels << "ms" << endl;
			cout << "data transfer   : " << time_glsc_datatrans << "ms" << endl;
			cout << "post-processing : " << time_glsc_post << "ms" << endl;
			cout << "total time      : " << time_glsc << "ms = " << (1000./time_glsc) << "FPS" << endl;
			
			cout << "##################################################################" << endl;
		} // end gpus
		
		waitKey(0);		
	}
private:
	HMatc m_src_bgr;
};

int main(int argc, char** argv){
	if (argc != 3){
		cout << "Usage : " << argv[0] << " <image name> <v_x>" << endl;
		cout << "Press Enter to exist!" << endl;
		getchar();
		return EXIT_FAILURE;
	}
	
	int vx = atoi(argv[2]);
	
	Evaluate eter;
	eter.run(argv[1], vx);
	return EXIT_SUCCESS;
}
