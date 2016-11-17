
// copyright reserved by Zhihua Ban
/*!
* \file tools.hpp
* \date 2015/05/10 13:54
*
* \author Ban Zhihua
* Contact: sawpara@126.com
*
* \version 0.21
*
* \brief tools to export labeled result
*
* TODO: long description
*
* 0.21
*  : rename the class names
* \note
*/


#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__


#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <inttypes.h>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

//#include <Eigen/Eigen>

//////////////////////////////////////////////////////////////////////////
// better the label
class Better_Tool{
public:
	Better_Tool(){}
	~Better_Tool(){}
	// return number of superpixels
	// labels counted from new_start
	template<class T>
	int relabel(T *labels, const size_t len, const T new_start){

		// find min and max
		std::pair<T*, T*> lab_mm = std::minmax_element(labels, labels + len);
		T lab_min = *(lab_mm.first);
		T lab_max = *(lab_mm.second);

		// construct map
		int *p_map = NULL;
		if (lab_max < 0){
			lab_max = 0;
		}

		if (lab_min < 0){
			m_map.resize(lab_max + 1 - lab_min);
			p_map = &(m_map[-lab_min]);
		}
		else{
			m_map.resize(lab_max + 1);
			p_map = &(m_map[0]);
		}
		m_map.assign(m_map.size(), -1);

		// relabel
		int new_lab = new_start;
		for (size_t i = 0; i < len; i++){
			if (p_map[labels[i]] < 0){
				p_map[labels[i]] = new_lab;
				new_lab++;
			}
			labels[i] = p_map[labels[i]];
		}
		return new_lab - new_start;
	}
private:
	vector<int32_t> m_map;
};

class Better{
public:
	// return number of superpixels
	// labels counted from new_start
	template<class T>
	static int relabel(T *labels, const size_t len, const T new_start){
		Better_Tool bt;
		return bt.relabel(labels, len, new_start);
	}
};

//class Map{
//public:
//	template<class T>
//	static void maping(T *labels, const size_t len, const Eigen::ArrayXi &mi){
//		for (size_t i = 0; i < len; i++){
//			labels[i] = mi(labels[i]);
//		}
//	}
//};



//////////////////////////////////////////////////////////////////////////
// export drawing out

class Draw_Tool{
public:
	// type = 0 : major & minor; 1 : only major; 2 : minor
	// ND   = 4 : 4-connected neighbor; 8 : 8-connected neighbor.
	template<int ND = 4, class TL>
	Mat contour(const TL* labels, const Mat &img, const int type = 1, Vec3b major = Vec3b(0, 0, 250), Vec3b minor = Vec3b(0, 100, 180)){
		int width = img.cols;
		int height = img.rows;
		Mat new_image = img.clone();
		_detect_boundary<ND>(labels, width, height);
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				if (m_al_bound[y*width + x] == ms_major_marker && 2 != type){
					new_image.at<Vec3b>(y, x) = major;
				}
				else if (m_al_bound[y*width + x] == ms_minor_marker && 1 != type){
					new_image.at<Vec3b>(y, x) = minor;
				}
			}
		}
		return new_image;
	}

	template<class TL>
	Mat rdcolor(const TL *labels, const int &width, const int &height)
	{
		cv::Mat new_image(height, width, CV_8UC3);
		vector<TL> new_labels;
		new_labels.resize(width*height);
		for (size_t i = 0; i < new_labels.size(); i++){
			new_labels[i] = labels[i];
		}
		int num_sps = m_bt.relabel(&(new_labels[0]), new_labels.size(), 0);

		vector<vector<uchar> > colors(3, vector<uchar>(num_sps));
		for (int k = 0; k < num_sps; ++k) {
			colors[0][k] = rand() & 255;
			colors[1][k] = rand() & 255;
			colors[2][k] = rand() & 255;
		}

		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				new_image.at<Vec3b>(y, x) = Vec3b(
					colors[0][new_labels[y*width + x]],
					colors[1][new_labels[y*width + x]],
					colors[2][new_labels[y*width + x]]);
			}
		}

		return new_image;
	}

	template<class TL>
	Mat meanval(const TL *labels, const Mat &img){
		int height = img.rows;
		int width = img.cols;
		vector<TL> new_labels(height*width);
		memcpy(&(new_labels[0]), labels, sizeof(TL)*height*width);

		cv::Mat new_image = img.clone();

		int num_sp = m_bt.relabel(&(new_labels[0]), new_labels.size(), 0);

		vector<int> count_nums(num_sp);
		count_nums.assign(num_sp, 0);

		int num_samples = height*width;

		for (int i = 0; i < num_samples; i++){
			count_nums[new_labels[i]]++;
		}

		vector<Vec3i> mean_sample(num_sp);
		mean_sample.assign(num_sp, Vec3i(0, 0, 0));

		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				Vec3b pix = img.at<Vec3b>(y, x);
				mean_sample[new_labels[y*width + x]] += pix;
			}
		}

		for (int i = 0; i < num_sp; i++){
			mean_sample[i] = mean_sample[i] / count_nums[i];
		}

		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				new_image.at<Vec3b>(y, x) = mean_sample[new_labels[y*width + x]];
			}
		}

		return new_image;
	}


protected:
	template<int ND = 4, class TL>
	void _detect_boundary(const TL *labels, const int &width, const int &height){
		m_al_bound.resize(width*height);
		m_al_bound.assign(width*height, 0); // start with no boundary
		const int dx[] = { -1, 0, 1, 1, -1, 0, 1, -1 };
		const int dy[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
		const int nd = 8;
		const int bg = (ND == 4 ? 1 : 0);
		const int st = (ND == 4 ? 2 : 1);
		// search for 1-st boundary
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				int cur_label = labels[y*width + x];
				for (int dd = bg; dd < nd; dd += st){
					int xx = x + dx[dd];
					int yy = y + dy[dd];
					if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
					if (cur_label != labels[xx + yy*width]){
						m_al_bound[y*width + x] = ms_major_marker;
						break;
					}
				}
			}
		}
		// search for 2-nd boundary
		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++){
				if (m_al_bound[y*width + x] == ms_major_marker){
					for (int dd = bg; dd < nd; dd += st){
						int xx = x + dx[dd];
						int yy = y + dy[dd];
						if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
						if (m_al_bound[xx + yy*width] == 0){
							m_al_bound[xx + yy*width] = ms_minor_marker;
						}
					}
				}
			}
		}
	}
private:
	// boundary of output of algorithm
	vector<uint8_t> m_al_bound;
	static const uint8_t ms_major_marker = 255;
	static const uint8_t ms_minor_marker = 180;
	Better_Tool m_bt;
};


class Draw{
public:
	// type = 0 : main_mark & sub_mark; 1 : main_mark; 2 : sub_mark
	// ND = 4, 8
	template<int ND = 4, class TL>
	static Mat contour(const TL *labels, const Mat &img, const int type = 0, Vec3b major = Vec3b(0, 0, 250), Vec3b minor = Vec3b(0, 100, 180)){
		Draw_Tool dt;
		return dt.contour<ND>(labels, img, type, major, minor);
	}

	template<class TL>
	static Mat rdcolor(const TL *labels, const int &width, const int &height){
		Draw_Tool dt;
		return dt.rdcolor(labels, width, height);
	}

	template<class TL>
	static Mat meanval(const TL *labels, const Mat &img){
		Draw_Tool dt;
		return dt.meanval(labels, img);
	}
};

#endif
