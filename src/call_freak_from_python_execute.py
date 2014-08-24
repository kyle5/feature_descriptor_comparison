import os, sys, Image, math, random
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy as sp
sys.path = ['lib/python'] + sys.path
import call_feature_descriptor_methods

def convert_list_to_mat( lst_input, mat_height_input ):
	mat_height = int( mat_height_input )
	s_lst_input = len( lst_input )
	mat_width = s_lst_input / mat_height
	mat_width = int( mat_width )
	s = ( mat_height, mat_width )
	mat_of_features = np.empty(s, int)
	counter = 0
	widths = range(0, mat_width)
	heights = range(0, mat_height)
	j = 0
	feature_values_mat = np.zeros( (mat_height, mat_width) )
	for i in widths:
		for j in heights:
			cur_idx = i*mat_height+j
			cur_value = lst_input[cur_idx]
			feature_values_mat[j, i] = cur_value
	return feature_values_mat

def convert_img_to_lst( input_img ):
	input_img_data = input_img.load()
	s_img = input_img.size
	img_h = float( s_img[1] )
	lst_img = []
	for i in range( s_img[0] ):
		for j in range( s_img[1] ):
			cur_pixel = input_img_data[ i, j ]
			cur_pixel = float( cur_pixel )
			lst_img.append( cur_pixel )
	return lst_img, img_h

def setup_image_data_and_paths( lst_of_image_paths, pass_image_data ):
	image_paths_and_data = []
	for i in range(len(lst_of_image_paths)):
		cur_img_path = lst_of_image_paths[i]
		if pass_image_data == 1:
			img = Image.open(cur_img_path)
			img_gray = img.convert('L')
			lst_img, img_h = convert_img_to_lst( img_gray )
		else:
			lst_img = []
			img_h = -1.0
		cur_img_path_and_data = tuple( ( cur_img_path, lst_img, img_h ) )
		image_paths_and_data.append( cur_img_path_and_data )
	return image_paths_and_data

def read_images( image_paths_and_data ):
	computed_images = [None]*len( image_paths_and_data )
	for i in range( len(image_paths_and_data) ):
		cur_img_path = image_paths_and_data[i][0]
		img = cv2.imread( cur_img_path, 0 )
		computed_images[i] = img
	return computed_images

if __name__ == "__main__":
	pass_image_data = 1;
	detector_type = 'fast'; # choose from: fast, dog, box_filters
	descriptor_type = 'freak'; # choose from: freak, surf, orb
	visualize_results = 1

	root_directory_path = os.path.dirname(os.path.realpath(__file__))
	pictures_directory = root_directory_path + '/../sample_images/';
	image_paths = [ "{0}2014-08-22-223833.jpg".format(pictures_directory), "{0}2014-08-22-223845.jpg".format(pictures_directory), '{0}2014-08-23-160154.jpg'.format(pictures_directory), '{0}2014-08-23-160204.jpg'.format(pictures_directory), '{0}2014-08-23-160435.jpg'.format(pictures_directory) ];
	image_paths_and_data = setup_image_data_and_paths( image_paths, pass_image_data )
	output = call_feature_descriptor_methods.process_multiple_images_python( image_paths_and_data, detector_type, descriptor_type )
	keypoints_list = output[0]
	matches_list = output[1]
	if len( keypoints_list ) != (len( matches_list ) + 1):
		raise Exception( "Returned keypoints and matches are not of the correct length" )
	if visualize_results == 1:
		computed_images = read_images( image_paths_and_data )
		for i in range( len(matches_list) ):
			kp1 = keypoints_list[i]
			kp2 = keypoints_list[i+1]
			cur_matches = matches_list[i]
			img_path_and_data_1 = image_paths_and_data[i]
			img_path_and_data_2 = image_paths_and_data[i+1]
			img1 = computed_images[i]
			img2 = computed_images[i+1]
			h1, w1 = img1.shape[:2]
			h2, w2 = img2.shape[:2]
			view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
			view[:h1, :w1, 0] = img1
			view[:h2, w1:, 0] = img2
			view[:, :, 1] = view[:, :, 0]
			view[:, :, 2] = view[:, :, 0]
			for m in cur_matches:
				match_query_idx = int(m[0]);
				match_train_idx = int(m[1]);
				color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
				cv2.line(view, (int(kp1[match_query_idx][0]), int(kp1[match_query_idx][1])) , (int(kp2[match_train_idx][0] + w1), int(kp2[match_train_idx][1])), color)
			cv2.imshow("view", view)
			cv2.waitKey()
