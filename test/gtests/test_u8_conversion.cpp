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
#include "test_serialization_utils.h"
#include <gtest/gtest.h>
#include <tosa_serialization_handler.h>

using namespace tosa;

class U8Conversion : public testing::TestWithParam<std::string>
{
public:
    std::default_random_engine gen;
    std::uniform_int_distribution<int64_t> gen_data;
    void SetUp()
    {
        gen      = std::default_random_engine(RANDOM_SEED);
        gen_data = std::uniform_int_distribution<int64_t>(std::numeric_limits<int64_t>::min(),
                                                          std::numeric_limits<int64_t>::max());
    }
};

TEST_P(U8Conversion, )
{

    std::vector<uint8_t> encoded;

    // Testing on various vector sizes, since odd lengths require padding which can cause unwanted behavior
    for (size_t out_size : { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40 })
    {
        // The random data for most datatypes will be static-casted from this vector of random I64's.
        // This works because I64 is the largest datatype in the standard right now.
        std::vector<int64_t> enough_random_bytes(out_size);
        std::generate(enough_random_bytes.begin(), enough_random_bytes.end(),
                      [this]() mutable { return gen_data(gen); });

        // Most TOSA datatypes have a numeric C++ equivalent, either by default, or from numpy_utils.h
        // or cfloat.h. The exceptions (Bool, I4, F16, and I48) require some extra effort
        if (GetParam() == "I48")
        {
            // I48 is represented with the 48 least-significant bits of an int64_t, so we take
            // the randomly generated int64_t's and replace the 16 most significant bits
            // with copies of the sign bit to imitate sign-extension
            std::vector<int64_t> in_I48(out_size), out_I48;
            std::memcpy(in_I48.data(), enough_random_bytes.data(), out_size * sizeof(int64_t));
            for (size_t i = 0; i < out_size; ++i)
            {
                in_I48[i] &= 0x0000'ffff'ffff'ffff;
                if (in_I48[i] >> 47)
                    in_I48[i] |= 0xffff'0000'0000'0000;
            }
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertI48toU8(in_I48, encoded));
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertU8toI48(encoded, out_size, out_I48));
            EXPECT_EQ(in_I48, out_I48) << "I48";
        }
        else if (GetParam() == "I4")
        {
            // To create random I4's, we generate random I8's and constrain them to the correct range.
            std::vector<int8_t> in_I4(out_size), out_I4;
            std::memcpy(in_I4.data(), enough_random_bytes.data(), out_size * sizeof(int8_t));
            for (size_t i = 0; i < out_size; ++i)
                in_I4[i] = (in_I4[i] % 15 + 15) % 15 - 7;    // The TOSA standard requires I4's to be from -7 to 7.
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertI4toU8(in_I4, encoded));
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertU8toI4(encoded, out_size, out_I4));
            EXPECT_EQ(in_I4, out_I4) << "I4";
        }
        else if (GetParam() == "F16")
        {
            // ConvertF16toU8 accepts F32's and internally converts them to F16. To test U8 conversion,
            // we generate random F16's and then cast them to F32's to be serialized.
            std::vector<half_float::half> in_F16(out_size), out_F16;
            std::memcpy(in_F16.data(), enough_random_bytes.data(), out_size * sizeof(half_float::half));
            std::vector<_Float32> in_F16_as_F32(out_size);
            for (size_t i = 0; i < out_size; ++i)
                in_F16_as_F32[i] = half_float::half_cast<float, half_float::half>(in_F16[i]);
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertF16toU8(in_F16_as_F32, encoded));
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertU8toF16(encoded, out_size, out_F16));
            EXPECT_TRUE(nan_tolerant_equals(in_F16, out_F16)) << "F16";
        }
        else if (GetParam() == "Bool")
        {
            // No memcpy since vector<bool> is unusual
            std::vector<bool> in_Bool(out_size), out_Bool;
            for (size_t i = 0; i < out_size; ++i)
                in_Bool[i] = (enough_random_bytes[i] & 1) == 0;
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertBooltoU8(in_Bool, encoded));
            EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertU8toBool(encoded, out_size, out_Bool));
            EXPECT_EQ(in_Bool, out_Bool) << "Bool";
        }

#define U8_TEST_CASE(func_name, type_name)                                                                             \
    else if (GetParam() == #func_name)                                                                                 \
    {                                                                                                                  \
        std::vector<type_name> in_##func_name(out_size), out_##func_name;                                              \
        std::memcpy(in_##func_name.data(), enough_random_bytes.data(), out_size * sizeof(type_name));                  \
        EXPECT_EQ(TOSA_OK, TosaSerializationHandler::Convert##func_name##toU8(in_##func_name, encoded));               \
        EXPECT_EQ(TOSA_OK, TosaSerializationHandler::ConvertU8to##func_name(encoded, out_size, out_##func_name));      \
        EXPECT_TRUE(nan_tolerant_equals(in_##func_name, out_##func_name)) << #func_name;                               \
    }

        U8_TEST_CASE(I64, int64_t)
        U8_TEST_CASE(I32, int32_t)
        U8_TEST_CASE(I16, int16_t)
        U8_TEST_CASE(I8, int8_t)

        // We "generate random floats" by using the random bytes from int64_t and static-casting them to floats.
        // This creates some NaN's/infinities if the format supports them, but we want those for testing anyway.
        // Note that we don't test whether NaN's mantissa is preserved (in formats where there are multiple possibilities)
        U8_TEST_CASE(BF16, bf16)
        U8_TEST_CASE(F32, _Float32)
        U8_TEST_CASE(FP8E4M3, fp8e4m3)
        U8_TEST_CASE(FP8E5M2, fp8e5m2)

        else
        {
            FAIL() << "No test for this type";
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    SerializationCpp,
    U8Conversion,
    testing::Values("I64", "I48", "I32", "I16", "I8", "I4", "BF16", "FP8E4M3", "FP8E5M2", "F32", "F16", "Bool"),
    [](auto info) { return info.param; });