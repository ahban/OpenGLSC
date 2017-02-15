// Zhihua Ban all rights reserved.
// If you have questions, please contact me at sawpara@126.com

#ifndef __CONNECTIVITY_HPP__
#define __CONNECTIVITY_HPP__

#include "AMatrix.hpp"
#include <vector>
using namespace std;

struct ConnectedCell{
  ConnectedCell(int _new_count, int _old_id) : new_count(_new_count), old_id(_old_id){}
  int new_count;
  int old_id;
};

class Connectivity_equal
{
public:
  vector<int> best_newlab;
  vector<int> best_count;
  vector<ConnectedCell> connected_cells;
  HMati m_filnal;
public:

  int merge_small_regions(HMati &labels, int min_size){

    //const int ddx[] = { +1, -1, 0, 0 };
    //const int ddy[] = { 0, 0, +1, -1 };
    //const int ndd = 4;
    
    const int ddx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    const int ddy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    const int ndd = 8;

    const int width  = labels.m_width;
    const int height = labels.m_height;
    const int steps  = labels.m_steps;
    
    m_labels.create(width, height, 1);
    m_cx.create(width, height, 1);
    m_cy.create(width, height, 1);

    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        m_labels[y*steps + x] = -1; // no label
      }
    }

    int old_lab = 0;
    int new_lab = 0;
    int adj_lab = 0;
    int index;
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        
        index = y*steps + x;

        // if has new label, then continue.
        if (m_labels[index]>-1) continue;

        m_labels[index] = new_lab;
        old_lab = labels[index];
        m_cx[0] = x;
        m_cy[0] = y;


        // find a adjacent label
        for (int d = 0; d < ndd; d++){
          int xx = m_cx[0] + ddx[d];
          int yy = m_cy[0] + ddy[d];
          // boundary check
          if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
          int tpid = yy*steps + xx;
          if (m_labels[tpid]>-1) adj_lab = m_labels[tpid];

        }

        // if has no new label, count the number 
        int count = 1;
        for (int c = 0; c < count; c++){
          for (int d = 0; d < ndd; d++){
            int xx = m_cx[c] + ddx[d];
            int yy = m_cy[c] + ddy[d];
            // boundary check
            if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
            int tpid = yy*steps + xx;
            if (m_labels[tpid] < 0 && labels[tpid]==old_lab){
              m_cx[count] = xx;
              m_cy[count] = yy;
              m_labels[tpid] = new_lab;
              count++;
            }
          }
        }
        // if the superpixel size is less than a limit, assign an adjacent label
        if (count < min_size){
          for (int c = 0; c < count; c++){
            int tpid = m_cy[c] * steps + m_cx[c];
            m_labels[tpid] = adj_lab;
          }
          new_lab--;
        }
        new_lab++;
      }
    }

    std::swap(m_labels.m_data, labels.m_data);
    return new_lab;
  }
private:
  HMati m_labels;
  HMati m_cx;
  HMati m_cy;
};


#endif