// this will be a C++ library code to call freak on an input image

#include "call_freak_on_an_input_image.h"

using namespace std;
using namespace cv;

vector<KeyPoint> get_FREAK_keypoints( Mat img_input ) {
	img_input.convertTo(img_input, CV_8U);
/*
	image format CV_U8
	Mat img_input_uc_8u;
	img_input.convertTo( img_input_uc_8u, CV_8U );
*/
	vector<KeyPoint> keypoints_detected;
	FAST( img_input, keypoints_detected, 50 );
	return keypoints_detected;
}

Mat get_FREAK_descriptors( Mat img_input, vector<KeyPoint> keypoints_input ) {
	img_input.convertTo(img_input, CV_8U);
	FREAK feature_extractor( true, true );
	Mat features;
	feature_extractor.compute( img_input, keypoints_input, features );
	return features;
}
