cmake_minimum_required(VERSION 3.16)
project(model_try LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG        8ab2cf95897fbead004e1bfbd41b62efdef7bf38
)

set(ARMADILLO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ARMADILLO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(armadillo)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(doctest)

FetchContent_Declare(
    stb_image
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)

FetchContent_Populate(stb_image)
set(STB_INCLUDE_DIR "${stb_image_SOURCE_DIR}")

add_executable(model_try
    model_try.cpp
    model.cpp
    model.h
    app.cpp
    app.h
)

target_link_libraries(model_try PRIVATE armadillo
)

target_include_directories(model_try PRIVATE ${STB_INCLUDE_DIR}
)

add_executable(model_tests
    model_tests.cpp
    model.cpp
    model.h
)

target_link_libraries(model_tests armadillo doctest)

target_include_directories(model_tests PRIVATE ${STB_INCLUDE_DIR} 
)

add_executable(app_tests
    app_tests.cpp
    app.cpp
    app.h
    model.cpp
    model.h
)

target_link_libraries(app_tests armadillo doctest)

target_include_directories(app_tests PRIVATE ${STB_INCLUDE_DIR}
)

enable_testing()
add_test(NAME model_test COMMAND model_tests --force-colors -d)
add_test(NAME app_test   COMMAND app_tests   --force-colors -d)
