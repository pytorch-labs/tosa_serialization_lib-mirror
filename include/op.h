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

#ifndef _TOSA_SERIALIZATION_OP_H
#define _TOSA_SERIALIZATION_OP_H

#include "tosa_generated.h"

using std::string;

namespace tosa
{

#define Attribute_NoneAttribute Attribute_NONE

enum OP_INPUT_TYPE
{
    TENSOR,
    SHAPE,
    TENSOR_LIST,
    SHAPE_LIST,
};

template <Op op>
struct TosaSerializationOpInfo;

#define DEF_OP(OP_NAME, ATTR_NAME, ...)                                                                                \
    template <>                                                                                                        \
    struct TosaSerializationOpInfo<Op_##OP_NAME>                                                                       \
    {                                                                                                                  \
        using attr_type = Tosa##ATTR_NAME##Attribute;                                                                  \
        static const Attribute attr()                                                                                  \
        {                                                                                                              \
            return Attribute_##ATTR_NAME##Attribute;                                                                   \
        };                                                                                                             \
                                                                                                                       \
        static const std::vector<OP_INPUT_TYPE>& operandTypes()                                                        \
        {                                                                                                              \
            operand_types = { __VA_ARGS__ };                                                                           \
            return operand_types;                                                                                      \
        }                                                                                                              \
                                                                                                                       \
        static std::vector<OP_INPUT_TYPE> operand_types;                                                               \
    };

#include "op.def"
#undef DEF_OP

}    // namespace tosa

#endif    // _TOSA_SERIALIZATION_OP_H