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

#ifndef CT_CMICROSCALING_H
#define CT_CMICROSCALING_H
#include "cfloat.h"
#include <cassert>
#include <cstdint>

namespace ct
{
/// \brief E8M0 scale type
class fp8_e8m0
{
public:
    constexpr static bool has_nan               = true;
    constexpr static bool has_inf               = false;
    constexpr static ct::FloatFeatures features = ct::FloatFeatures::HasNaN;
    constexpr static size_t n_bits              = 8;
    constexpr static size_t n_exponent_bits     = 8;
    constexpr static size_t n_significand_bits  = 0;
    constexpr static int64_t exponent_bias      = 127;

    constexpr inline fp8_e8m0()
        : m_data(0)
    {}

    inline fp8_e8m0(const float& f)
        : fp8_e8m0(ct::cfloat_cast<binary32, fp8_e8m0>()(f))
    {}

    constexpr inline fp8_e8m0(const fp8_e8m0& f)
        : m_data(f.m_data)
    {}

    constexpr inline fp8_e8m0(const ct::float_support::hidden&, const uint8_t& bits)
        : m_data(bits)
    {}

    static constexpr inline fp8_e8m0 from_bits(const uint8_t& bits)
    {
        return fp8_e8m0(ct::float_support::hidden(), bits);
    }

    static constexpr inline fp8_e8m0 from_bits(bool pm, const int8_t exp_bits, const uint8_t mantissa_bits)
    {
        assert(mantissa_bits == 0);
        return fp8_e8m0(ct::float_support::hidden(), static_cast<uint8_t>(exp_bits));
    }

    static constexpr inline fp8_e8m0 NaN()
    {
        return from_bits(0b11111111);
    }

    inline operator float() const
    {
        return cfloat_cast<fp8_e8m0, binary32>()(*this);
    }

    constexpr inline bool is_nan() const
    {
        return m_data == 0xff;
    }
    constexpr inline bool is_infinity() const
    {
        return false;
    }
    constexpr inline bool is_zero() const
    {
        return false;
    }
    constexpr inline bool sign() const
    {
        return false;
    }
    constexpr inline uint8_t exponent_bits() const
    {
        return m_data;
    }
    constexpr inline int16_t exponent() const
    {
        return static_cast<int16_t>(m_data) - 127;
    }
    constexpr inline uint8_t significand() const
    {
        return 0;
    }

    constexpr inline fp8_e8m0& operator=(const fp8_e8m0& other)
    {
        m_data = other.m_data;
        return *this;
    }

    constexpr inline fp8_e8m0& operator=(fp8_e8m0&& other)
    {
        m_data = other.m_data;
        return *this;
    }

    constexpr inline bool operator==(const fp8_e8m0& o) const
    {
        return m_data == o.m_data;
    }

    constexpr inline bool operator!=(const fp8_e8m0& o) const
    {
        return m_data != o.m_data;
    }

    constexpr inline uint8_t bits() const
    {
        return m_data;
    }

private:
    uint8_t m_data{ 0 };
};

// MXFP types element types
using fp6_e3m2 = ct::cfloat_advanced<6, 3, ct::FloatFeatures::HasDenorms>;
using fp6_e2m3 = ct::cfloat_advanced<6, 2, ct::FloatFeatures::HasDenorms>;
using fp4_e2m1 = ct::cfloat_advanced<4, 2, ct::FloatFeatures::HasDenorms>;

/// \brief MXINT8 element type
//
// This is strictly more like Q1.6, but by pretending that this is a
// floating point number with a fixed exponent of 0, and which can only
// represent denormals, we can make it interoperate with `cfloat_cast`.
class mxint8
{
public:
    constexpr static bool has_nan               = false;
    constexpr static bool has_inf               = false;
    constexpr static ct::FloatFeatures features = ct::FloatFeatures::HasDenorms;
    constexpr static size_t n_bits              = 8;
    constexpr static size_t n_exponent_bits     = 0;
    constexpr static size_t n_significand_bits  = 7;
    constexpr static int64_t exponent_bias      = 0;

    constexpr inline mxint8()
        : m_data(0)
    {}

    inline mxint8(const float& f)
        : mxint8(ct::cfloat_cast<binary32, mxint8>()(f))
    {}

    constexpr inline mxint8(const mxint8& f)
        : m_data(f.m_data)
    {}

    constexpr inline mxint8(const ct::float_support::hidden&, const int8_t& bits)
        : m_data(bits)
    {}

    /// \brief Construct an mxint8 from the given bit pattern.
    ///
    /// This is expected to match the bit pattern from the OCP spec
    /// (negative numbers represented using two's complement).
    static constexpr inline mxint8 from_bits(const int8_t& bits)
    {
        return mxint8(ct::float_support::hidden(), bits);
    }

    /// \brief Construct an mxint8 from floating point like fields.
    ///
    /// Note that the mantissa/significand is expected in unsigned format.
    /// Internally this is negated when `pm` is set to `true` (for negative)
    /// - so the underlying storage matches `mxint8` in having negative
    /// numbers stored in two's complement.
    static constexpr inline mxint8 from_bits(bool pm, const int8_t exp_bits, const uint8_t mantissa_bits)
    {
        assert(exp_bits == 0);
        assert(mantissa_bits >> 7 == 0);
        return mxint8(ct::float_support::hidden(),
                      pm ? -static_cast<int8_t>(mantissa_bits) : static_cast<int8_t>(mantissa_bits));
    }

    /// \brief Return a NaN representation for mxint8
    ///
    /// mxint8 does not have a NaN representation, per the C++ spec we
    /// return 0 when asked for a NaN.
    static constexpr inline mxint8 NaN()
    {
        return from_bits(0);
    }

    /// \brief Return the largest positive or negative number.
    ///
    /// The OCP spec permits us to reduce the negative range of mxint8 to be
    /// symmetrical with the positive range, however this implementation
    /// just exposes the full range of the underlying type.
    static constexpr inline mxint8 max(bool pm)
    {
        return from_bits(pm ? 0x80 : 0x1f);
    }

    inline constexpr int64_t exponent() const
    {
        return exponent_bias;
    }
    inline constexpr uint64_t exponent_bits() const
    {
        return 0;
    }
    inline constexpr bool sign() const
    {
        return m_data < 0;
    }
    inline constexpr bool is_nan() const
    {
        return false;
    }
    inline constexpr bool is_infinity() const
    {
        return false;
    }
    inline constexpr bool is_zero() const
    {
        return m_data == 0;
    }

    inline constexpr uint64_t significand() const
    {
        // NOTE We have to convert back from two's complement to be
        // compatible with `cfloat_cast`.
        return sign() ? -m_data : m_data;
    }

    inline operator float() const
    {
        return ct::cfloat_cast<mxint8, binary32>()(*this);
    }

    constexpr inline mxint8& operator=(const mxint8& other)
    {
        m_data = other.m_data;
        return *this;
    }

    constexpr inline mxint8& operator=(mxint8&& other)
    {
        m_data = other.m_data;
        return *this;
    }

    constexpr inline bool operator==(const mxint8& o) const
    {
        return m_data == o.m_data;
    }

    constexpr inline bool operator!=(const mxint8& o) const
    {
        return m_data != o.m_data;
    }

    constexpr inline uint8_t bits() const
    {
        return static_cast<uint8_t>(m_data);
    }

private:
    int8_t m_data{ 0 };
};

}    // namespace ct

#endif    // CT_CMICROSCALING_H
