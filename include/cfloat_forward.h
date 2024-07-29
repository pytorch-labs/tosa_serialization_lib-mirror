/******************************************************************************
 *
 *  This confidential and proprietary software may be used only as
 *  authorised by a licensing agreement from Arm Limited
 *  (C) COPYRIGHT 2022-2024 ARM Limited and/or its affiliates
 *  ALL RIGHTS RESERVED
 *
 *  The entire notice above must be reproduced on all authorised
 *  copies and copies may only be made to the extent permitted
 *  by a licensing agreement from Arm Limited.
 *
 *****************************************************************************/
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
