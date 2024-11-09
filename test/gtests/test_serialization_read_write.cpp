// Copyright (c) 2024, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

// Tests that TOSA files can be read and written without erroring.
// Does NOT test if the serialization/de-serialization is actually correct.
#include <filesystem>
#include <gtest/gtest.h>
#include <tosa_serialization_handler.h>

using namespace tosa;

// comment out this until we can update to a frame work test tosa file
#if false
TEST(SerializationCpp, ReadWrite)
{
    std::string source_dir           = CMAKE_SOURCE_DIR;
    std::filesystem::path read_path  = source_dir + "/test/examples/test_add_1x4x4x4_f32.tosa";
    std::filesystem::path write_path = source_dir + "/test/tmp/Serialization.ReadWrite.tosa";

    // Creating /tmp directory or removing our writing file if necessary
    std::filesystem::create_directories(write_path.parent_path());
    std::filesystem::remove(write_path);

    TosaSerializationHandler handler;

    tosa_err_t err = handler.LoadFileTosaFlatbuffer(read_path.c_str());
    EXPECT_EQ(err, TOSA_OK);

    err = handler.SaveFileTosaFlatbuffer(write_path.c_str());
    EXPECT_EQ(err, TOSA_OK);

    // Cleaning up the written file
    std::filesystem::remove(write_path);
}
#endif
