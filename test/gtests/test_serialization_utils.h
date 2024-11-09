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

#ifndef _TEST_SERIALIZATION_UTILS_H
#define _TEST_SERIALIZATION_UTILS_H

#include <random>
#include <tosa_serialization_handler.h>

using namespace tosa;

// See test/gtests/CMakeLists.txt for where CMAKE_SOURCE_DIR is defined.
// This assumes that you've run cmake starting from the repository's root folder.
inline std::string source_dir = CMAKE_SOURCE_DIR;

// Macros for testing write/read to flatbuffers and json.
#define WRITE_READ_TOSA_TEST(write_tsh, read_tsh, err, path, msg)                                                      \
    err = write_tsh.SaveFileTosaFlatbuffer(path);                                                                      \
    ASSERT_EQ(err, TOSA_OK) << "Failed to write to flatbuffer: " << msg;                                               \
    err = read_tsh.LoadFileTosaFlatbuffer(path);                                                                       \
    ASSERT_EQ(err, TOSA_OK) << "Failed to read from flatbuffer: " << msg;                                              \
    ASSERT_TRUE(write_tsh == read_tsh) << "Flatbuffer write/read failed: " << msg;                                     \
    std::remove(path);
#define WRITE_READ_JSON_TEST(write_tsh, read_tsh, err, schema_path, path, msg)                                         \
    err = write_tsh.LoadFileSchema(schema_path);                                                                       \
    ASSERT_EQ(err, TOSA_OK) << "Failed to read schema: " << msg;                                                       \
    err = read_tsh.LoadFileSchema(schema_path);                                                                        \
    ASSERT_EQ(err, TOSA_OK) << "Failed to read schema: " << msg;                                                       \
    err = write_tsh.SaveFileJson(path);                                                                                \
    ASSERT_EQ(err, TOSA_OK) << "Failed to write json: " << msg;                                                        \
    err = read_tsh.LoadFileJson(path);                                                                                 \
    ASSERT_EQ(err, TOSA_OK) << "Failed to read json: " << msg;                                                         \
    ASSERT_TRUE(write_tsh == read_tsh) << "JSON write/read failed: " << msg;                                           \
    std::remove(path);

// Function and helper macros to check that all the member variables of two attributes are equal.
#define EQ_ARGS_0(lhs, rhs, ...) true
#define EQ_ARGS_1(lhs, rhs, T, F, V) (lhs->V() == rhs->V())
#define EQ_ARGS_2(lhs, rhs, T, F, V, ...) (lhs->V() == rhs->V() && EQ_ARGS_1(lhs, rhs, __VA_ARGS__))
#define EQ_ARGS_3(lhs, rhs, T, F, V, ...) (lhs->V() == rhs->V() && EQ_ARGS_2(lhs, rhs, __VA_ARGS__))
#define EQ_ARGS_4(lhs, rhs, T, F, V, ...) (lhs->V() == rhs->V() && EQ_ARGS_3(lhs, rhs, __VA_ARGS__))
#define EQ_ARGS_5(lhs, rhs, T, F, V, ...) (lhs->V() == rhs->V() && EQ_ARGS_4(lhs, rhs, __VA_ARGS__))
#define EQ_ARGS_6(lhs, rhs, T, F, V, ...) (lhs->V() == rhs->V() && EQ_ARGS_5(lhs, rhs, __VA_ARGS__))
#define EQ_ARGS_7(lhs, rhs, T, F, V, ...) (lhs->V() == rhs->V() && EQ_ARGS_6(lhs, rhs, __VA_ARGS__))

inline bool attribute_match(Attribute attribute_type, TosaAttributeBase* lhs, TosaAttributeBase* rhs)
{
    switch (attribute_type)
    {
        case Attribute_NONE:
            return true;
#define DEF_ATTRIBUTE(NAME, NUM_ARGS, ...)                                                                             \
    case Attribute_##NAME##_Attribute:                                                                                 \
        return EQ_ARGS_##NUM_ARGS(static_cast<Tosa##NAME##Attribute*>(lhs), static_cast<Tosa##NAME##Attribute*>(rhs),  \
                                  __VA_ARGS__);
#include "attribute.def"
#undef DEF_ATTRIBUTE
        default:
            printf("Invalid attribute type!\n");
            return false;
    }
}

#undef EQ_ARGS_1
#undef EQ_ARGS_2
#undef EQ_ARGS_3
#undef EQ_ARGS_4
#undef EQ_ARGS_5
#undef EQ_ARGS_6
#undef EQ_ARGS_7

// Check deep equality of vectors of pointers. Assumes that the vectors' elements have their own operator== defined.
#define DEEP_EQUALS(A, B)                                                                                              \
    std::equal(A.begin(), A.end(), B.begin(), B.end(), [](const auto& a, const auto& b) { return *a == *b; })
// Equality operators for serialization objects.
inline bool operator==(TosaSerializationTensor& lhs, TosaSerializationTensor& rhs)
{
    return (lhs.GetDtype() == rhs.GetDtype()) && (lhs.GetShape() == rhs.GetShape()) &&
           (lhs.GetName() == rhs.GetName()) && (lhs.GetVariable() == rhs.GetVariable()) &&
           (lhs.GetData() == rhs.GetData()) && (lhs.GetIsUnranked() == rhs.GetIsUnranked()) &&
           (lhs.GetVariableName() == rhs.GetVariableName());
}
inline bool operator==(TosaSerializationOperator& lhs, TosaSerializationOperator& rhs)
{
    return (lhs.GetOp() == rhs.GetOp()) && (lhs.GetInputTensorNames() == rhs.GetInputTensorNames()) &&
           (lhs.GetOutputTensorNames() == rhs.GetOutputTensorNames()) &&
           (lhs.GetAttributeType() == rhs.GetAttributeType()) &&
           attribute_match(lhs.GetAttributeType(), lhs.GetAttribute(), rhs.GetAttribute());
}
inline bool operator==(TosaSerializationBasicBlock& lhs, TosaSerializationBasicBlock& rhs)
{
    return (lhs.GetName() == rhs.GetName()) && (lhs.GetRegionName() == rhs.GetRegionName()) &&
           (lhs.GetInputs() == rhs.GetInputs()) && (lhs.GetOutputs() == rhs.GetOutputs()) &&
           DEEP_EQUALS(lhs.GetOperators(), rhs.GetOperators()) && DEEP_EQUALS(lhs.GetTensors(), rhs.GetTensors());
}
inline bool operator==(TosaSerializationRegion& lhs, TosaSerializationRegion& rhs)
{
    return (lhs.GetName() == rhs.GetName()) && DEEP_EQUALS(lhs.GetBlocks(), rhs.GetBlocks());
}
inline bool operator==(TosaSerializationHandler& lhs, TosaSerializationHandler& rhs)
{
    return DEEP_EQUALS(lhs.GetRegions(), rhs.GetRegions());
}
#undef DEEP_EQUALS

// Helpers for random attribute generation.
#define LIST_GENERATED_ARGS_0(...)
#define LIST_GENERATED_ARGS_1(T, F, V) generate_value_##T##_##F()
#define LIST_GENERATED_ARGS_2(T, F, V, ...) generate_value_##T##_##F(), LIST_GENERATED_ARGS_1(__VA_ARGS__)
#define LIST_GENERATED_ARGS_3(T, F, V, ...) generate_value_##T##_##F(), LIST_GENERATED_ARGS_2(__VA_ARGS__)
#define LIST_GENERATED_ARGS_4(T, F, V, ...) generate_value_##T##_##F(), LIST_GENERATED_ARGS_3(__VA_ARGS__)
#define LIST_GENERATED_ARGS_5(T, F, V, ...) generate_value_##T##_##F(), LIST_GENERATED_ARGS_4(__VA_ARGS__)
#define LIST_GENERATED_ARGS_6(T, F, V, ...) generate_value_##T##_##F(), LIST_GENERATED_ARGS_5(__VA_ARGS__)
#define LIST_GENERATED_ARGS_7(T, F, V, ...) generate_value_##T##_##F(), LIST_GENERATED_ARGS_6(__VA_ARGS__)

#define RANDOM_SEED 22
#define RANDOM_VEC_MAX_LENGTH 100

inline std::default_random_engine gen(RANDOM_SEED);

inline std::uniform_int_distribution<int32_t> rand_vec_length(0, RANDOM_VEC_MAX_LENGTH);
inline std::uniform_int_distribution<int32_t> rand_i32(std::numeric_limits<int32_t>::min(),
                                                       std::numeric_limits<int32_t>::max());
inline std::uniform_int_distribution<char>
    rand_char('a', '~');    // Generating ASCII character that isn't a control character or DELETE
inline std::uniform_int_distribution<uint32_t> rand_dtype(DType_MIN, DType_MAX);
inline std::uniform_int_distribution<uint32_t> rand_resize_mode(ResizeMode_MIN, ResizeMode_MAX);
inline std::uniform_int_distribution<uint32_t> rand_nan_propagation_mode(NanPropagationMode_MIN,
                                                                         NanPropagationMode_MAX);
inline std::uniform_int_distribution<int8_t> rand_bit(0, 1);

inline int32_t generate_value_int32_t_S()
{
    return rand_i32(gen);
}
inline int16_t generate_value_int16_t_S()
{
    return static_cast<int16_t>(rand_i32(gen));
}
inline std::vector<int32_t> generate_value_int32_t_V()
{
    int length = rand_vec_length(gen);
    std::vector<int32_t> vec;
    for (size_t i = 0; i < length; ++i)
        vec.push_back(rand_i32(gen));
    return vec;
}
inline std::vector<int16_t> generate_value_int16_t_V()
{
    int length = rand_vec_length(gen);
    std::vector<int16_t> vec;
    for (size_t i = 0; i < length; ++i)
        vec.push_back(static_cast<int16_t>(rand_i32(gen)));
    return vec;
}
inline std::vector<uint8_t> generate_value_uint8_t_V()
{
    int length = rand_vec_length(gen);
    std::vector<uint8_t> vec;
    for (size_t i = 0; i < length; ++i)
        vec.push_back(static_cast<uint8_t>(rand_i32(gen)));
    return vec;
}
inline std::string generate_value_string_S()
{
    int length = rand_vec_length(gen);
    std::string str;
    for (size_t i = 0; i < length; ++i)
        str.push_back(rand_char(gen));
    return str;
}
inline DType generate_value_DType_S()
{
    return (DType)rand_dtype(gen);
}
inline ResizeMode generate_value_ResizeMode_S()
{
    return (ResizeMode)rand_resize_mode(gen);
}
inline NanPropagationMode generate_value_NanPropagationMode_S()
{
    return (NanPropagationMode)rand_nan_propagation_mode(gen);
}
inline bool generate_value_bool_S()
{
    return rand_bit(gen) == 0;
}

// // NaN-sensitive equality for vectors of floats.
template <typename T>
bool nan_tolerant_equals(std::vector<T> a, std::vector<T> b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if ((a[i] != b[i]) && !(std::isnan(a[i])) && !(std::isnan(b[i])))
            return false;
    return true;
}

///
/// Testing graph construction utility.
///

inline void pushBackEmptyRegion(TosaSerializationHandler& handler, const std::string& region_name)
{
    handler.GetRegions().emplace_back(std::make_unique<TosaSerializationRegion>(region_name));
}

inline void pushBackEmptyBasicBlock(TosaSerializationRegion* region,
                                    const std::string& block_name,
                                    const std::string& region_name)
{
    region->GetBlocks().emplace_back(std::make_unique<TosaSerializationBasicBlock>(block_name, region_name));
}

inline void pushBackOperator(TosaSerializationBasicBlock* block, std::unique_ptr<TosaSerializationOperator> op)
{
    block->GetOperators().emplace_back(std::move(op));
}

inline std::unique_ptr<TosaSerializationTensor> createTensor(const std::string& name,
                                                             const std::vector<int32_t>& shape = {},
                                                             DType dtype                       = DType_UNKNOWN,
                                                             const std::vector<uint8_t>& data  = {},
                                                             const bool variable               = false,
                                                             const bool is_unranked            = false,
                                                             const std::string& variable_name  = "")
{
    return std::make_unique<TosaSerializationTensor>(name, shape, dtype, data, variable, is_unranked, variable_name);
}

inline std::vector<std::string> SetupDummyInput(TosaSerializationBasicBlock* block, int input_nums)
{
    std::vector<std::string> input_names;
    const std::string input_prefix = "in_";
    for (int i = 0; i < input_nums; i++)
    {
        auto input_str = input_prefix + std::to_string(i + 1);
        input_names.push_back(input_str);
        // Create and insert a dummy tensor for testing.
        block->GetTensors().emplace_back(createTensor(input_str));
        block->GetInputs().push_back(input_str);
    }
    return input_names;
}

inline std::vector<std::string> SetupDummyOutput(TosaSerializationBasicBlock* block, int output_nums)
{
    std::vector<std::string> output_names;
    const std::string output_prefix = "out_";
    for (int i = 0; i < output_nums; i++)
    {
        auto output_str = output_prefix + std::to_string(i + 1);
    }
    return output_names;
}

#endif    // _TEST_SERIALIZATION_UTILS_H
