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
#include <list>
#include <tosa_serialization_handler.h>

using namespace tosa;

class SingleAttr : public testing::TestWithParam<Attribute>
{
public:
    std::map<Attribute, TosaAttributeBase*> attrs;
    void SetUp()
    {
        // Randomly generating an attribute of each type. The LIST_GENERATED_ARGS_... macros use attribute.def to
        // call the correct generate_value_... functions in test_serialization_utils.h. For example,
        // attrs[Attribute_ConvAttribute] = new TosaConvAttribute(generate_value_int32_t_V(), ...);
#define DEF_ATTRIBUTE(NAME, NUM_ARGS, ...)                                                                             \
    attrs[Attribute_##NAME##Attribute] = new Tosa##NAME##Attribute(LIST_GENERATED_ARGS_##NUM_ARGS(__VA_ARGS__));
#include "attribute.def"
#undef DEF_ATTRIBUTE
        attrs[Attribute_NONE] = new TosaNoneAttribute();
    }
};

TEST_P(SingleAttr, )
{
    if (!attrs.count(GetParam()))
    {
        // skip attributes that are not in attribute.def
        GTEST_SKIP();
    }
    tosa_err_t err;
    TosaSerializationHandler handler1, handler2, handler3;
    handler1.GetRegions().emplace_back(std::make_unique<TosaSerializationRegion>("main_region"));
    auto region = handler1.GetRegions().back().get();
    region->GetBlocks().emplace_back(std::make_unique<TosaSerializationBasicBlock>("main_block", "main_region"));
    auto block = region->GetBlocks().back().get();
    std::vector<std::string> input_names{ "t1" };
    std::vector<std::string> output_names{ "t2" };
    block->GetOperators().emplace_back(std::make_unique<TosaSerializationOperator>(
        Op_UNKNOWN, GetParam(), attrs[GetParam()], input_names, output_names));
    std::vector<int32_t> shape;
    std::vector<uint8_t> empty_data;
    block->GetTensors().emplace_back(
        std::make_unique<TosaSerializationTensor>("t1", shape, DType_UNKNOWN, empty_data, false, false));
    block->GetTensors().emplace_back(
        std::make_unique<TosaSerializationTensor>("t2", shape, DType_UNKNOWN, empty_data, false, false));
    block->GetInputs().push_back("t1");
    block->GetOutputs().push_back("t2");
    WRITE_READ_TOSA_TEST(handler1, handler2, err, (source_dir + "/test/tmp/Serialization.SingleAttr.tosa").c_str(),
                         EnumNameAttribute(GetParam()));

    WRITE_READ_JSON_TEST(handler2, handler3, err, (source_dir + "/schema/tosa.fbs").c_str(),
                         (source_dir + "/test/tmp/Serialization.SingleAttr.json").c_str(),
                         EnumNameAttribute(GetParam()));
}

TEST(SingleAttr, NanPropagation)
{
    std::list<Op> op_list = { Op_MAX_POOL2D, Op_CLAMP, Op_MAXIMUM, Op_MINIMUM, Op_REDUCE_MAX, Op_REDUCE_MIN };

    auto generate_NanPropagationMode = [&] {
        std::uniform_int_distribution<uint32_t> valid_nan_propagation_mode(NanPropagationMode_PROPAGATE,
                                                                           NanPropagationMode_IGNORE);
        return static_cast<NanPropagationMode>(valid_nan_propagation_mode(gen));
    };

    std::map<Attribute, std::unique_ptr<TosaAttributeBase>> attrs;
    attrs[Attribute_AxisAttribute] =
        std::make_unique<TosaAxisAttribute>(generate_value_int32_t_S(), generate_NanPropagationMode());
    attrs[Attribute_ClampAttribute] = std::make_unique<TosaClampAttribute>(
        generate_value_uint8_t_V(), generate_value_uint8_t_V(), generate_NanPropagationMode());
    attrs[Attribute_PoolAttribute] = std::make_unique<TosaPoolAttribute>(
        generate_value_int32_t_V(), generate_value_int32_t_V(), generate_value_int32_t_V(), generate_value_int32_t_S(),
        generate_value_int32_t_S(), generate_value_DType_S(), generate_NanPropagationMode());
    attrs[Attribute_NanPropagationAttribute] =
        std::make_unique<TosaNanPropagationAttribute>(generate_NanPropagationMode());

    for (Op op : op_list)
    {
        Attribute attr_enum;
        std::vector<OP_INPUT_TYPE> operand_types;
        switch (op)
        {
            case Op_ARGMAX: {
                attr_enum     = Attribute_AxisAttribute;
                operand_types = { TENSOR };
                break;
            }
            case Op_MAX_POOL2D: {
                attr_enum     = Attribute_PoolAttribute;
                operand_types = { TENSOR };
                break;
            }
            case Op_CLAMP: {
                attr_enum     = Attribute_ClampAttribute;
                operand_types = { TENSOR };
                break;
            }
            case Op_MAXIMUM:
            case Op_MINIMUM: {
                attr_enum     = Attribute_NanPropagationAttribute;
                operand_types = { TENSOR, TENSOR };
                break;
            }
            case Op_REDUCE_MAX:
            case Op_REDUCE_MIN: {
                attr_enum     = Attribute_AxisAttribute;
                operand_types = { TENSOR };
                break;
            }
            default:
                FAIL() << "Operator " << EnumNamesOp()[op] << " does not support NaN propagation mode";
        }

        TosaSerializationHandler handler1, handler2, handler3;
        pushBackEmptyRegion(handler1, "main_region");

        auto region = handler1.GetRegions().back().get();
        pushBackEmptyBasicBlock(region, "main_block", "main_region");

        auto block = region->GetBlocks().back().get();

        std::vector<std::string> input_names  = SetupDummyInput(block, operand_types.size());
        std::vector<std::string> output_names = SetupDummyOutput(block, 1);

        auto ser_op = std::make_unique<TosaSerializationOperator>(static_cast<Op>(op), attr_enum,
                                                                  attrs[attr_enum].get(), input_names, output_names);
        pushBackOperator(block, std::move(ser_op));

        const auto schema_path = source_dir + "/schema/tosa.fbs";
        const auto tosa_path   = source_dir + "/test/tmp/Serialization.SingleAttr.NanPropagation.tosa";
        const auto json_path   = source_dir + "/test/tmp/Serialization.SingleAttr.NanPropagation.json";

        tosa_err_t err;
        WRITE_READ_TOSA_TEST(handler1, handler2, err, tosa_path.c_str(),
                             EnumNameAttribute(Attribute_NanPropagationAttribute));

        WRITE_READ_JSON_TEST(handler2, handler3, err, schema_path.c_str(), json_path.c_str(),
                             EnumNameAttribute(Attribute_NanPropagationAttribute));
    }
}

INSTANTIATE_TEST_SUITE_P(SerializationCpp, SingleAttr, testing::ValuesIn(EnumValuesAttribute()), [](auto info) {
    return EnumNameAttribute(info.param);
});
