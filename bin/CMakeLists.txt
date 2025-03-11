cmake_minimum_required(VERSION 3.16) 
project(model_try) 
set(CMAKE_CXX_STANDARD 17) 

include_directories(${CMAKE_SOURCE_DIR}/../libraries)
link_directories(${CMAKE_SOURCE_DIR}/../libraries)

add_executable(model_try model_try.cpp model.cpp model.h app.h app.cpp)
target_link_libraries(model_try libarmadillo.so)

add_executable(model_tests model_tests.cpp model.cpp model.h)
target_link_libraries(model_tests libarmadillo.so doctest)

add_executable(app_tests app_tests.cpp app.cpp app.h model.cpp model.h)
target_link_libraries(app_tests libarmadillo.so doctest)

enable_testing()
add_subdirectory(external)

add_test(NAME model_test COMMAND tests --force-colors -d)
add_test(NAME app_test COMMAND tests --force-colors -d)


