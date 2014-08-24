# this will be a python method that will setup the call to FREAK in C++ code
# this will be a python script that calls sets up libraries for a simple C++ method

from distutils.core import setup, Extension

call_feature_descriptor_methods = Extension('call_feature_descriptor_methods',
sources = ['src/call_feature_descriptor_methods.cpp'],
include_dirs = ['/usr/local/include'],
libraries = ['opencv_core', 'opencv_highgui', 'opencv_features2d',  'opencv_imgproc', 'opencv_calib3d', 'opencv_nonfree' ],
library_dirs = ['/usr/local/lib']
)

setup(name='PackageName', version='1.0', description='List passing module', ext_modules=[call_feature_descriptor_methods])
