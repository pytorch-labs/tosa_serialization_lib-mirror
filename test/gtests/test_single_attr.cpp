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
    tosa_err_t err;
    auto region = new TosaSerializationRegion(
        "main_region",
        { new TosaSerializationBasicBlock(
            "main_block", "main_region",
            { new TosaSerializationOperator(Op_UNKNOWN, GetParam(), attrs[GetParam()], { "t1" }, { "t2" }) },
            { new TosaSerializationTensor("t1", {}, DType_UNKNOWN, {}, false, false),
              new TosaSerializationTensor("t2", {}, DType_UNKNOWN, {}, false, false) },
            { "t1" }, { "t2" }) });

    TosaSerializationHandler handler1, handler2, handler3;
    handler1.GetRegions().push_back(region);
    WRITE_READ_TOSA_TEST(handler1, handler2, err, (source_dir + "/test/tmp/Serialization.SingleAttr.tosa").c_str(),
                         EnumNameAttribute(GetParam()));

    WRITE_READ_JSON_TEST(handler2, handler3, err, (source_dir + "/schema/tosa.fbs").c_str(),
                         (source_dir + "/test/tmp/Serialization.SingleAttr.json").c_str(),
                         EnumNameAttribute(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(SerializationCpp, SingleAttr, testing::ValuesIn(EnumValuesAttribute()), [](auto info) {
    return EnumNameAttribute(info.param);
});