build_dir="build"
lib_dir="lib"

if [ -d ${build_dir} ]; then
	rm -r ${build_dir}
fi

if [ -d ${lib_dir} ]; then
	rm -r ${lib_dir}
fi

python src/call_freak_from_python_setup.py ${build_dir}
python src/call_freak_from_python_setup.py install --home .
