# CMake generated Testfile for 
# Source directory: /home/uladzislau/Desktop/aip/project_cpp/bin
# Build directory: /home/uladzislau/Desktop/aip/project_cpp/bin/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(model_test "/home/uladzislau/Desktop/aip/project_cpp/bin/build/model_tests" "--force-colors" "-d")
set_tests_properties(model_test PROPERTIES  _BACKTRACE_TRIPLES "/home/uladzislau/Desktop/aip/project_cpp/bin/CMakeLists.txt;76;add_test;/home/uladzislau/Desktop/aip/project_cpp/bin/CMakeLists.txt;0;")
add_test(app_test "/home/uladzislau/Desktop/aip/project_cpp/bin/build/app_tests" "--force-colors" "-d")
set_tests_properties(app_test PROPERTIES  _BACKTRACE_TRIPLES "/home/uladzislau/Desktop/aip/project_cpp/bin/CMakeLists.txt;77;add_test;/home/uladzislau/Desktop/aip/project_cpp/bin/CMakeLists.txt;0;")
subdirs("_deps/armadillo-build")
subdirs("_deps/doctest-build")
