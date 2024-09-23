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

        TosaSerializationHandler handler1, handler2, handler3;
        handler1.GetRegions().emplace_back(std::make_unique<TosaSerializationRegion>("main_region"));
        auto region = handler1.GetRegions().back().get();
        region->GetBlocks().emplace_back(std::make_unique<TosaSerializationBasicBlock>("main_block", "main_region"));
        auto block = region->GetBlocks().back().get();
        std::vector<uint8_t> empty_data;
        block->GetTensors().emplace_back(std::make_unique<TosaSerializationTensor>(
            name, shape, (DType)i, empty_data, variable, is_unranked, variable_name));
        block->GetInputs().push_back(name);
        block->GetOutputs().push_back(name);

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

    std::vector<std::string> input_names{ input_name };
    std::vector<std::string> output_names{ output_name };

    tosa_err_t err;

    // Testing for tensors with various lengths
    for (int data_bytes : { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40 })
    {
        TosaSerializationHandler handler1, handler2, handler3;
        handler1.GetRegions().emplace_back(std::make_unique<TosaSerializationRegion>("main_region"));
        auto region = handler1.GetRegions().back().get();

        region->GetBlocks().emplace_back(std::make_unique<TosaSerializationBasicBlock>("main_block", "main_region"));
        auto block = region->GetBlocks().back().get();

        std::vector<uint8_t> empty_data;
        std::vector<int32_t> shape = { 1, data_bytes };

        block->GetOperators().emplace_back(std::make_unique<TosaSerializationOperator>(
            Op_IDENTITY, Attribute_NONE, new TosaNoneAttribute(), input_names, output_names));

        block->GetTensors().emplace_back(std::make_unique<TosaSerializationTensor>(
            input_name, shape, DType_UINT8, empty_data, variable, is_unranked, variable_name));
        auto input_tensor = block->GetTensors().back().get();
        block->GetTensors().emplace_back(std::make_unique<TosaSerializationTensor>(output_name, shape, DType_UINT8,
                                                                                   empty_data, false, is_unranked, ""));
        auto output_tensor = block->GetTensors().back().get();

        std::vector<uint8_t> data(data_bytes);
        ASSERT_TRUE(data.size() == data_bytes);

        std::generate(data.begin(), data.end(), [&gen_data, &gen]() mutable { return gen_data(gen); });
        input_tensor->SetData(data);

        block->GetInputs().push_back(input_name);
        block->GetOutputs().push_back(output_name);
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
    for (int op = Op_MIN; op <= Op_MAX; ++op)
    {
        TosaSerializationHandler handler1, handler2, handler3;
        handler1.GetRegions().emplace_back(std::make_unique<TosaSerializationRegion>("main_region"));
        auto region = handler1.GetRegions().back().get();
        region->GetBlocks().emplace_back(std::make_unique<TosaSerializationBasicBlock>("main_block", "main_region"));
        auto block = region->GetBlocks().back().get();
        std::vector<std::string> input_names{ "t1" };
        std::vector<std::string> output_names{ "t2" };
        block->GetOperators().emplace_back(std::make_unique<TosaSerializationOperator>(
            static_cast<Op>(op), Attribute_NONE, new TosaNoneAttribute(), input_names, output_names));
        std::vector<int32_t> shape;
        std::vector<uint8_t> empty_data;
        block->GetTensors().emplace_back(
            std::make_unique<TosaSerializationTensor>("t1", shape, DType_UNKNOWN, empty_data, false, false));
        block->GetTensors().emplace_back(
            std::make_unique<TosaSerializationTensor>("t2", shape, DType_UNKNOWN, empty_data, false, false));
        block->GetInputs().push_back("t1");
        block->GetOutputs().push_back("t2");

        WRITE_READ_TOSA_TEST(handler1, handler2, err, (source_dir + "/test/tmp/Serialization.SingleOp.tosa").c_str(),
                             EnumNameOp(static_cast<Op>(op)));

        WRITE_READ_JSON_TEST(handler2, handler3, err, (source_dir + "/schema/tosa.fbs").c_str(),
                             (source_dir + "/test/tmp/Serialization.SingleOp.json").c_str(),
                             EnumNameOp(static_cast<Op>(op)));
    }
}
