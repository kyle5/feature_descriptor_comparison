#ifndef CALL_FREAK_LIBRARY__
#define CALL_FREAK_LIBRARY__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
using cv::Mat;
using cv::KeyPoint;
#include <opencv2/highgui/highgui.hpp>
#include <vector>
using std::vector;
#include <cstdio>

vector<KeyPoint> get_FREAK_keypoints( Mat img_input );
Mat get_FREAK_descriptors( Mat img_input, vector<KeyPoint> keypoints_input );

#endif
