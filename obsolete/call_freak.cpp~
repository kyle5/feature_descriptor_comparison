#include "Python.h"
#include <stdexcept>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

//#include "call_freak_on_an_input_image.h"

using namespace std;
using namespace cv;

static PyObject *process_multiple_images_cpp( PyObject *self, PyObject *args );

static vector<vector<KeyPoint> > get_keypoints_from_images( vector<Mat> image_variables, string detector_string );
static vector<Mat> get_features_from_images( vector<Mat> image_variables, vector<vector<KeyPoint> > &all_keypoints_detected, string descriptor_string );
static vector<vector<DMatch> > match_features_from_images( vector<Mat> all_image_features );

static vector<KeyPoint> get_keypoints( Mat img_input, string detector_string );
static Mat get_descriptors( Mat img_input, vector<KeyPoint> &keypoints_input, string descriptor_string );

static Mat convert_pylist_to_mat( PyObject *listObj, int img_h );
static PyObject *convert_vector_of_keypoints_to_python_list( vector<vector<KeyPoint> > all_image_keypoints );
static PyObject *convert_vector_of_matches_to_python_list( vector< vector<DMatch> > matched_feature_indices );

static void write_descriptors_to_file( Mat freak_descriptors_mat, string descriptors_cpp_file_path );


static PyMethodDef passing_list_methods[] = {
	{"process_multiple_images_python", process_multiple_images_cpp, METH_VARARGS, "call freak method on multiple images"},
	{NULL, NULL, 0, NULL}
};

extern "C" void initcall_freak(void) {
	Py_InitModule( "call_freak", passing_list_methods );
}

static Mat convert_pylist_to_mat( PyObject *listObj, int img_h ) {
	PyObject *numObj, *temp_p2;
	int sizeoutput = PyList_Size( listObj );
	int estimated_width = sizeoutput / img_h;
	fprintf( stderr, "sizeoutput: %d\testimated_width: %d\timg_h: %d\n", sizeoutput, estimated_width, img_h );
	Mat img_variable = Mat::zeros( img_h, estimated_width, CV_8U );
	for (int j = 0; j < sizeoutput; j++ ) {
		numObj = PyList_GetItem( listObj, j );
		temp_p2 = PyNumber_Float( numObj );
	  double cur_double = PyFloat_AsDouble( temp_p2 );
		int cur_x = j / img_h;
		int cur_y = j % img_h;
		img_variable.at<uchar>( cur_y, cur_x ) = (uchar) cur_double;
	}
	fprintf( stderr, "img_variable\trows: %d\tcols: %d\n", (int) img_variable.rows, (int) img_variable.cols );	
	return img_variable;
}

static void write_descriptors_to_file( Mat freak_descriptors_mat, string descriptors_cpp_file_path ) {
	FileStorage file( descriptors_cpp_file_path.c_str(), FileStorage::WRITE );
	file << "freak_descriptors_mat" << freak_descriptors_mat;
	file.release();
}

static vector<Mat> read_images_as_mat_variables( PyObject *listObj ) {
	long temp_long = (long) PyList_Size(listObj);
	if ( temp_long != 5 ) {
		throw runtime_error( "number of images is inconsistent: or if in python len(input_images) != 5, then take this out..." );
	}
	vector<Mat> image_variables;
	// for each element in the list:
	for ( long i = 0; i < temp_long; i++ ) {
		//		check that the element is a tuple and has length 3
		PyObject *element = PyList_GetItem( listObj, i );
		int valid_tuple_input = PyTuple_Check( element );
		long py_tuple_size = (long) PyTuple_Size( element );
		if ( valid_tuple_input != 1 || py_tuple_size != 3 ) {
			PyErr_SetString(PyExc_TypeError,"Invalid input (tuple image objects)!");
			throw runtime_error( "Invalid input (tuple image objects)!" );
		}
		// 		get the third argumnet
		PyObject *third_element_of_tuple = PyTuple_GetItem( element, 2 );
		int valid_float_input = PyFloat_Check(third_element_of_tuple);
		if ( valid_float_input != 1 ) {
			PyErr_SetString(PyExc_TypeError,"Invalid input (float image data height)!");
			throw runtime_error( "Invalid input (float image data height)!" );
		}
		PyObject *temp_py_float = PyNumber_Float( third_element_of_tuple );
		double img_h = PyFloat_AsDouble( temp_py_float );
		Mat img_variable;
		if ( img_h == -1 ) {
			fprintf(stderr, "image read from file!\n");
			// read the image by filepath
			const char* filepath = PyString_AsString( PyTuple_GetItem( element, 0 ) );
			Mat img_variable_bgr = imread( filepath );
			// convert image to grayscale
			cvtColor( img_variable_bgr, img_variable, CV_BGR2GRAY );
			// check that the img_variable is uchar and grayscale
			if ( img_variable.type() != 0 ) {
				throw runtime_error("Type is not uchar?!");
			}
		} else {
			fprintf(stderr, "image read from data from Python!\n");
			PyObject *img_data_as_list = PyTuple_GetItem( element, 1 );
			double img_data_length = (double) PyList_Size( img_data_as_list );
			fprintf( stderr, "img_data_length python format: %.2f\nimg_h: %.2f\n", img_data_length, img_h );
			img_variable = convert_pylist_to_mat( img_data_as_list, img_h );
		}
		image_variables.push_back( img_variable );
	}
	return image_variables;
}

static vector<vector<KeyPoint> > get_keypoints_from_images( vector<Mat> image_variables, string detector_string ) {
	vector<vector<KeyPoint> > all_keypoints_detected;
	for( int i = 0; i < (int) image_variables.size(); i++ ) {
		Mat img_variable = image_variables[i];
		// get freak features from the current image variable
		vector<KeyPoint> keypoints_detected = get_keypoints( img_variable, detector_string );
		all_keypoints_detected.push_back(keypoints_detected);
	}
	return all_keypoints_detected;
}

static vector<Mat> get_features_from_images( vector<Mat> image_variables, vector<vector<KeyPoint> > &all_keypoints_detected, string descriptor_string ) {
	fprintf(stderr, "start of get_features_from_images\n");
	vector<Mat> freak_feature_descriptors;
	for( int i = 0; i < (int) image_variables.size(); i++ ) {
		Mat img_variable = image_variables[i];
		Mat freak_descriptors_mat = get_descriptors( img_variable, all_keypoints_detected[i], descriptor_string );
		freak_feature_descriptors.push_back( freak_descriptors_mat );
	}
	fprintf(stderr, "end of get_features_from_images\n");
	return freak_feature_descriptors;
}

static vector<vector<DMatch> > match_features_from_images( vector<Mat> all_image_features ) {
	BFMatcher matcher(NORM_L2);
	vector<vector<DMatch> > all_matches;
	int all_but_one = ((int) all_image_features.size()) - 1;
	for(int i = 0; i < all_but_one; i++ ) {
		Mat descriptors_1, descriptors_2;
		descriptors_1 = all_image_features[i];
		descriptors_2 = all_image_features[i+1];
		vector< DMatch > matches;
		matcher.match( descriptors_1, descriptors_2, matches );
		fprintf( stderr, "Matching iteration: %d\n", i );
		if ( (int) matches.size() == 0 ) throw runtime_error( "No matches!" ); 
		all_matches.push_back( matches );
	}
	return all_matches;
}

vector<vector<DMatch> > find_matches_fitting_homography( vector<vector<KeyPoint> > all_image_keypoints, vector<vector<DMatch> > all_initial_matches ) {
	vector<vector<DMatch> > all_ransac_matches;
	for ( int i = 0; i < ((int)all_image_keypoints.size())-1; i++ ) {
		// call the premade method to view the matches
		vector<KeyPoint> image_keypoints_1 = all_image_keypoints[i];
		vector<KeyPoint> image_keypoints_2 = all_image_keypoints[i+1];
		vector<DMatch> initial_matches = all_initial_matches[i];
		// visualize (and perhaps prune) the matches
		vector<Point2f> matched_points_1, matched_points_2;
		for( int j = 0; j < (int) initial_matches.size(); j++ ) {
			DMatch cur_match = initial_matches[j];
			KeyPoint matched_keypoint_1 = image_keypoints_1[ cur_match.queryIdx ];
			KeyPoint matched_keypoint_2 = image_keypoints_2[ cur_match.trainIdx ];
			matched_points_1.push_back( Point2f(matched_keypoint_1.pt.x, matched_keypoint_1.pt.y) );
			matched_points_2.push_back( Point2f(matched_keypoint_2.pt.x, matched_keypoint_2.pt.y) );
		}
		Mat valid_match_indices;
		fprintf( stderr, "matched_points_1: %d\tmatched_points_2: %d\n", (int) matched_points_1.size(), (int) matched_points_2.size() );
		Mat H = findHomography( matched_points_1, matched_points_2, CV_RANSAC, 10, valid_match_indices );
		vector<KeyPoint> ransac_keypoints_1, ransac_keypoints_2;
		vector<DMatch> ransac_matches;
		uchar *valid_match_indices_data = (uchar*) valid_match_indices.data;
		for( int j = 0; j < (int) matched_points_1.size(); j++ ) {
			if ( valid_match_indices_data[j] < 1 ) continue;
			DMatch cur_match = initial_matches[j];
			ransac_keypoints_1.push_back( image_keypoints_1[cur_match.queryIdx] );
			ransac_keypoints_2.push_back( image_keypoints_2[cur_match.trainIdx] );
			ransac_matches.push_back( cur_match );
		}
		all_ransac_matches.push_back( ransac_matches );
	}
	return all_ransac_matches;
}

static void visualize_image_matches( vector<Mat> all_image_variables, vector<vector<KeyPoint> > all_image_keypoints, vector<vector<DMatch> > all_ransac_matches ) {
	for ( int i = 0; i < ((int) all_image_variables.size())-1; i++ ) {
		// for each set of 2 images
		Mat image_variable_1 = all_image_variables[i];
		Mat image_variable_2 = all_image_variables[i+1];
		vector<KeyPoint> image_keypoints_1 = all_image_keypoints[i];
		vector<KeyPoint> image_keypoints_2 = all_image_keypoints[i+1];
		vector<DMatch> cur_ransac_matches = all_ransac_matches[i];
		Mat matches_img;
		drawMatches( image_variable_1, image_keypoints_1, image_variable_2, image_keypoints_2, cur_ransac_matches, matches_img );
		imshow( "Matches", matches_img );
		cv::waitKey(0);
	}
}

static PyObject *convert_vector_of_keypoints_to_python_list( vector<vector<KeyPoint> > all_image_keypoints ) {
	PyObject *returned_keypoints = PyList_New( (int) all_image_keypoints.size() );
	for( int i = 0; i < (int) all_image_keypoints.size(); i++ ) {
		vector<KeyPoint> cur_keypoints = all_image_keypoints[i];
		PyObject *cur_keypoints_python = PyList_New( (int) cur_keypoints.size() );
		for ( int j = 0; j < (int) cur_keypoints.size(); j++ ) {
			KeyPoint cur_keypoint = cur_keypoints[j];
			PyObject *cur_keypoint_python_list = PyList_New( 2 );
			PyObject *keypoint_x_python = PyFloat_FromDouble((double)cur_keypoint.pt.x);
			PyObject *keypoint_y_python = PyFloat_FromDouble((double)cur_keypoint.pt.y);
			PyList_SetItem( cur_keypoint_python_list, 0, keypoint_x_python );
			PyList_SetItem( cur_keypoint_python_list, 1, keypoint_y_python );
			PyList_SetItem( cur_keypoints_python, j, cur_keypoint_python_list );
		}
		PyList_SetItem( returned_keypoints, i, cur_keypoints_python );
	}
	return returned_keypoints;
}

static PyObject *convert_vector_of_matches_to_python_list( vector< vector<DMatch> > matched_feature_indices ) {
	PyObject *returned_matches = PyList_New( (int) matched_feature_indices.size() );
	for ( int i = 0; i < (int) matched_feature_indices.size(); i++ ) {
		vector<DMatch> final_matches = matched_feature_indices[i];
		PyObject *cur_img_pair_matches_python = PyList_New( (int) final_matches.size() );
		for ( int j = 0; j < (int) final_matches.size(); j++ ) {
			DMatch cur_match = final_matches[j];
			int query_idx = cur_match.queryIdx;
			int train_idx = cur_match.trainIdx;
			PyObject *cur_match_python = PyList_New( 2 );
			PyObject *query_idx_python = PyFloat_FromDouble((double)query_idx);
			PyObject *train_idx_python = PyFloat_FromDouble((double)train_idx);
			PyList_SetItem( cur_match_python, 0, query_idx_python );
			PyList_SetItem( cur_match_python, 1, train_idx_python );
			PyList_SetItem( cur_img_pair_matches_python, j, cur_match_python );
		}
		PyList_SetItem( returned_matches, i, cur_img_pair_matches_python );
	}
	return returned_matches;
}

static PyObject *process_multiple_images_cpp( PyObject *self, PyObject *args ) {
	double num_args = (double) PyTuple_Size( args );
	int num_args_int = (int) num_args;
	if (num_args_int != 3) runtime_error("Three args needed as input!");
	PyObject *listObj, *detector_string_obj, *descriptor_string_obj;
	// check that the first argument is a list of image paths and image data
	//PyArg_ParseTuple(args, "O|O|O:ref", &listObj, &detector_string_obj, &descriptor_string_obj );
	listObj = PyTuple_GetItem(args, 0);
	int valid_list_input = PyList_Check( listObj );
	if ( valid_list_input == 1 ) {
		fprintf( stderr, "valid_list_input\n" );
		// get detector type
		detector_string_obj = PyTuple_GetItem(args, 1);
		int valid_detector_string_obj = PyString_Check( detector_string_obj );
		if ( valid_detector_string_obj == 1 ) {
			fprintf( stderr, "valid_detector_string_input\n" );
			// get descriptor type
			descriptor_string_obj = PyTuple_GetItem(args, 2);
			int valid_descriptor_string_input = PyString_Check( descriptor_string_obj );
			if ( valid_descriptor_string_input == 1 ) {
				fprintf( stderr, "valid_descriptor_string_input\n" );
				char *detector_string_char_ptr = PyString_AsString( detector_string_obj );
				string detector_string( detector_string_char_ptr );
				transform(detector_string.begin(), detector_string.end(), detector_string.begin(), ::tolower);
				char *descriptor_string_char_ptr = PyString_AsString( descriptor_string_obj );
				string descriptor_string( descriptor_string_char_ptr );
				transform(descriptor_string.begin(), descriptor_string.end(), descriptor_string.begin(), ::tolower);

				vector<Mat> all_image_variables = read_images_as_mat_variables( listObj );
				vector<vector<KeyPoint> > all_image_keypoints = get_keypoints_from_images( all_image_variables, detector_string );
				vector<Mat> all_image_features = get_features_from_images( all_image_variables, all_image_keypoints, descriptor_string );
				vector<vector<DMatch> > all_initial_matches = match_features_from_images( all_image_features );
				vector<vector<DMatch> > ransac_matches = find_matches_fitting_homography( all_image_keypoints, all_initial_matches );
				// visualize the image matches
				// visualize_image_matches( all_image_variables, all_image_keypoints, ransac_matches );
				// return the keypoints and the matches
				PyObject *returned_keypoints = convert_vector_of_keypoints_to_python_list( all_image_keypoints );
				PyObject *returned_matches = convert_vector_of_matches_to_python_list( ransac_matches );
				PyObject *returned_python_object = PyList_New( 2 );
				PyList_SetItem( returned_python_object, 0, returned_keypoints );
				PyList_SetItem( returned_python_object, 1, returned_matches );
				return returned_python_object;
			}
		}
	}
  PyErr_SetString(PyExc_TypeError,"Invalid input!");
  return NULL;
}

static vector<KeyPoint> get_keypoints( Mat img_input, string detector_string ) {
	img_input.convertTo(img_input, CV_8U);
	vector<KeyPoint> keypoints_detected;
	if ( strcmp( detector_string.c_str(), "fast" ) == 0 ) { // fast
		FAST( img_input, keypoints_detected, 50 );
	} else if ( strcmp( detector_string.c_str(), "dog" ) == 0 ) { // SIFT: difference of gaussians
		SIFT sift;
		sift.detect( img_input, keypoints_detected );
	} else if ( strcmp( detector_string.c_str(), "box_filters" ) == 0 ) { // SURF: box filters
		SURF surf;
		surf.detect( img_input, keypoints_detected );
	}
	return keypoints_detected;
}

static Mat get_descriptors( Mat img_input, vector<KeyPoint> &keypoints_input, string descriptor_string ) {
	img_input.convertTo(img_input, CV_8U);
	Mat features;
	if ( descriptor_string.compare( string( "freak" ) ) == 0 ) {
		FREAK feature_extractor( true, true );
		feature_extractor.compute( img_input, keypoints_input, features );
	} else if ( descriptor_string.compare( string( "surf" ) ) == 0 ) {
		SURF surf( 400, 4, 2, true );
		surf( img_input, cv::Mat(), keypoints_input, features, true );
	} else if ( descriptor_string.compare( string( "orb" ) ) == 0 ) {
		ORB orb;
	  orb(img_input, cv::Mat(), keypoints_input, features, true);
	}
	return features;
}
