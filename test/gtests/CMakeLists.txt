# Copyright (c) 2024, ARM Limited.
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

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571 # v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(serialization_tests)

target_sources(serialization_tests PRIVATE test_serialization_read_write.cpp)
target_sources(serialization_tests PRIVATE test_serialization_npy.cpp)
target_sources(serialization_tests PRIVATE test_u8_conversion.cpp)
target_sources(serialization_tests PRIVATE test_single_attr.cpp)
target_sources(serialization_tests PRIVATE test_serialization_unit.cpp)

target_link_libraries(serialization_tests PRIVATE GTest::gtest_main tosa_serialization_lib)
target_compile_definitions(serialization_tests PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(serialization_tests)