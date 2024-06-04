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

TEST(SerializationCpp, EmptyTensor)
{
    // Testing that a region with one empty tensor and nothing else can be losslessly
    // serialized and deserialized into a TOSA flatbuffer or json file.

    tosa_err_t err;

    for (int i = DType_MIN; i <= DType_MAX; ++i)
    {
        std::string name           = generate_value_string_S();
        std::vector<int32_t> shape = generate_value_int32_t_V();
        bool variable              = generate_value_bool_S();
        bool is_unranked           = generate_value_bool_S();
        std::string variable_name  = variable ? generate_value_string_S() : "";

        auto region = new TosaSerializationRegion(
            "main_region",
            { new TosaSerializationBasicBlock(
                "main_block", "main_region", {},
                { new TosaSerializationTensor(name, shape, (DType)i, {}, variable, is_unranked, variable_name) },
                { name }, { name }) });

        TosaSerializationHandler handler1, handler2, handler3;
        handler1.GetRegions().push_back(region);
        WRITE_READ_TOSA_TEST(handler1, handler2, err, (source_dir + "/test/tmp/Serialization.EmptyTensor.tosa").c_str(),
                             EnumNameDType((DType)i));

        WRITE_READ_JSON_TEST(handler2, handler3, err, (source_dir + "/schema/tosa.fbs").c_str(),
                             (source_dir + "/test/tmp/Serialization.EmptyTensor.json").c_str(),
                             EnumNameDType((DType)i));
    }
}

TEST(SerializationCpp, FullTensor)
{
    // Serializing a tensor with u8 data of arbitrary length. In this case the tensor is
    // the input for an identity operator, with an output tensor that is the same
    // as the input but without the data

    std::default_random_engine gen(RANDOM_SEED);
    std::uniform_int_distribution<uint8_t> gen_data(std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());

    std::string input_name    = generate_value_string_S();
    std::string output_name   = generate_value_string_S();
    bool variable             = true;
    bool is_unranked          = generate_value_bool_S();
    std::string variable_name = generate_value_string_S();

    TosaSerializationTensor *input_tensor, *output_tensor;
    tosa_err_t err;

    // Testing for tensors with various lengths
    for (int data_bytes : { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40 })
    {

        input_tensor = new TosaSerializationTensor(input_name, { 1, data_bytes }, DType_UINT8, {}, variable,
                                                   is_unranked, variable_name);
        output_tensor =
            new TosaSerializationTensor(output_name, { 1, data_bytes }, DType_UINT8, {}, false, is_unranked, "");

        std::vector<uint8_t> data(data_bytes);
        ASSERT_TRUE(data.size() == data_bytes);

        std::generate(data.begin(), data.end(), [&gen_data, &gen]() mutable { return gen_data(gen); });
        input_tensor->SetData(data);

        auto region = new TosaSerializationRegion(
            "main_region", { new TosaSerializationBasicBlock(
                               "main_block", "main_region",
                               { new TosaSerializationOperator(Op_IDENTITY, Attribute_NONE, new TosaNoneAttribute(),
                                                               { input_name }, { output_name }) },
                               { input_tensor, output_tensor }, { input_name }, { output_name }) });

        TosaSerializationHandler handler1, handler2, handler3;

        handler1.GetRegions().push_back(region);
        WRITE_READ_TOSA_TEST(handler1, handler2, err, (source_dir + "/test/tmp/Serialization.FullTensor.tosa").c_str(),
                             data_bytes << " bytes");
        WRITE_READ_JSON_TEST(handler2, handler3, err, (source_dir + "/schema/tosa.fbs").c_str(),
                             (source_dir + "/test/tmp/Serialization.FullTensor.json").c_str(), data_bytes << " bytes");
    }
}

TEST(SerializationCpp, SingleOp)
{
    // Serializing a region with one operator and empty input/output tensors. The operators don't have their attributes.

    tosa_err_t err;
    for (int i = Op_MIN; i <= Op_MAX; ++i)
    {
        auto region = new TosaSerializationRegion(
            "main_region",
            { new TosaSerializationBasicBlock(
                "main_block", "main_region",
                { new TosaSerializationOperator((Op)i, Attribute_NONE, new TosaNoneAttribute(), { "t1" }, { "t2" }) },
                { new TosaSerializationTensor("t1", {}, DType_UNKNOWN, {}, false, false),
                  new TosaSerializationTensor("t2", {}, DType_UNKNOWN, {}, false, false) },
                { "t1" }, { "t2" }) });

        TosaSerializationHandler handler1, handler2, handler3;
        handler1.GetRegions().push_back(region);
        WRITE_READ_TOSA_TEST(handler1, handler2, err, (source_dir + "/test/tmp/Serialization.SingleOp.tosa").c_str(),
                             EnumNameOp((Op)i));

        WRITE_READ_JSON_TEST(handler2, handler3, err, (source_dir + "/schema/tosa.fbs").c_str(),
                             (source_dir + "/test/tmp/Serialization.SingleOp.json").c_str(), EnumNameOp((Op)i));
    }
}
