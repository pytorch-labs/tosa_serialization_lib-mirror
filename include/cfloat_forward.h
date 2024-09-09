// Copyright (c) 2022-2024, ARM Limited.
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

#ifndef CT_CFLOAT_FORWARD_H
#define CT_CFLOAT_FORWARD_H
#include <cstddef>

namespace ct
{
/// \brief Bitfield specification of the features provided of a specified
/// floating point type.
enum class FloatFeatures
{
    None       = 0x0,
    HasNaN     = 0x1,    ///< The type can represent NaN values
    HasInf     = 0x2,    ///< The type can represent Infinity
    HasDenorms = 0x4,    ///< The type can represent denormal/subnormal values
};

template <size_t ContainerBits, size_t ExponentBits, FloatFeatures>
class cfloat_advanced;
}    // namespace ct

#endif    // CT_CFLOAT_FORWARD_H
