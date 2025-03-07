cmake_minimum_required(VERSION 3.16) 
project(model_try) 
set(CMAKE_CXX_STANDARD 17) 

link_directories(../libraries)
add_executable(model_try model_try.cpp model.cpp model.h app.h app.cpp)
target_link_libraries(model_try libarmadillo.so)
