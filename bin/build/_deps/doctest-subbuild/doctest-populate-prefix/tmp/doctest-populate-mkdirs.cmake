# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-src"
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-build"
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix"
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix/tmp"
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp"
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix/src"
  "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/uladzislau/Desktop/aip/project_cpp/bin/build/_deps/doctest-subbuild/doctest-populate-prefix/src/doctest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
