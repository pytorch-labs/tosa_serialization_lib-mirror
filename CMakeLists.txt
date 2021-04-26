#TOSA serialization library

# Copyright (c) 2020-2021, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Contains TOSA flatbuffer serialization library content.

cmake_minimum_required(VERSION 3.13.4)
project(TosaSerialization)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_VERBOSE_MAKEFILE ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include)

add_library(tosa_serialization_lib STATIC
  src/tosa_serialization_handler.cpp
  src/numpy_utils.cpp
  )

add_subdirectory(third_party)

add_executable(serialization_read_write
  test/src/serialization_read_write.cpp
)

target_link_libraries(serialization_read_write
  tosa_serialization_lib
  flatbuffers
)

add_executable(serialization_npy_test
  test/src/serialization_npy_test.cpp
)

target_link_libraries(serialization_npy_test
  tosa_serialization_lib
  flatbuffers
)
