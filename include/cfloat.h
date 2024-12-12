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

#ifndef CT_CFLOAT_H
#define CT_CFLOAT_H
#include "cfloat_forward.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(__cpp_lib_bit_cast)
#include <bit>
#endif    // defined(__cpp_lib_bit_cast)

namespace ct
{
constexpr FloatFeatures operator&(const FloatFeatures& a, const FloatFeatures& b)
{
    using T = std::underlying_type_t<FloatFeatures>;
    return static_cast<FloatFeatures>(static_cast<T>(a) & static_cast<T>(b));
}

constexpr FloatFeatures operator|(const FloatFeatures& a, const FloatFeatures& b)
{
    using T = std::underlying_type_t<FloatFeatures>;
    return static_cast<FloatFeatures>(static_cast<T>(a) | static_cast<T>(b));
}

constexpr FloatFeatures& operator|=(FloatFeatures& a, const FloatFeatures& b)
{
    a = a | b;
    return a;
}

namespace float_support
{
struct hidden
{};

/// \brief Get the number of bytes required to store the given number of
/// bits.
///
/// NOTE This is distinct from the number of bytes required to represent
/// the number of bits - a power of two number of bytes will always be
/// returned by this method.
constexpr size_t get_storage_bytes(const size_t n_bits)
{
    const size_t n_bytes = (n_bits + 7) / 8;
    size_t storage_bytes = 1;
    for (; storage_bytes < n_bytes; storage_bytes <<= 1)
        ;
    return storage_bytes;
}

/// \brief Utility method to convert from an older representation of the
/// floating-point features to the FloatFeatures bitfield.
constexpr FloatFeatures get_float_flags(bool has_nan, bool has_denorm, bool has_inf)
{
    FloatFeatures r = FloatFeatures::None;

    if (has_nan)
        r |= FloatFeatures::HasNaN;

    if (has_denorm)
        r |= FloatFeatures::HasDenorms;

    if (has_inf)
        r |= FloatFeatures::HasInf;

    return r;
}

/// \brief Shorthand for all support features
static constexpr FloatFeatures AllFeats = get_float_flags(true, true, true);

// Map from a number of storage bytes to a suitable storage type
template <size_t n_bytes>
struct storage_type;

#define STORAGE_TYPE(T)                                                                                                \
    template <>                                                                                                        \
    struct storage_type<sizeof(T)>                                                                                     \
    {                                                                                                                  \
        using type = T;                                                                                                \
    }
STORAGE_TYPE(uint8_t);
STORAGE_TYPE(uint16_t);
STORAGE_TYPE(uint32_t);
STORAGE_TYPE(uint64_t);
#undef STORAGE_TYPE

template <size_t n_storage_bytes>
using storage_type_t = typename storage_type<n_storage_bytes>::type;

#if defined(__cpp_lib_bit_cast)
#define BITCAST_CONSTEXPR constexpr inline

// If bit_cast is available then use it

constexpr inline uint32_t get_bits(const float& f)
{
    return std::bit_cast<uint32_t>(f);
}
constexpr inline float from_bits(const uint32_t& i)
{
    return std::bit_cast<float>(i);
}

#else
#define BITCAST_CONSTEXPR inline

// Otherwise `memcpy` is the safe (non-UB) of achieving the same result

inline uint32_t get_bits(const float& f)
{
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return i;
}

inline float from_bits(const uint32_t& i)
{
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}
#endif

}    // namespace float_support

/// \brief Rounding mode for floating-point casts
enum class RoundMode
{
    // Roundings to nearest

    /// \brief Round to the nearest value, with ties going to the nearest
    /// value with an even least significant digit.
    TiesToEven,

    /// \brief Round to the nearest value, with ties going to the nearest
    /// value with an odd least significant digit.
    TiesToOdd,

    ///\brief Round to the nearest value, with ties to the nearest value
    /// with larger absolute magnitude.
    TiesToAway,

    // Directed roundings

    TowardZero,                ///< Truncate
    TowardPositiveInfinity,    ///< Round up/ceiling
    TowardNegativeInfinity,    ///< Round down/floor
};

/// \brief Overflow mode for narrowing floating-point casts.
///
/// Determine the behaviour for values which cannot be represented by the
/// destination type.
enum class OverflowMode
{
    Saturate,    ///< Map to the largest representable value
    Overflow     ///< Map to infinity (if available) or NaN
};

/// \brief Subnormal/denormal handing
///
/// Determine the behaviour for values which are subnormals/denormals in the
/// destination type.
enum class SubnormalMode
{
    Retain,
    FlushToZero
};

namespace float_support
{
constexpr inline bool is_round_to_nearest(const RoundMode& rm)
{
    return (rm == RoundMode::TiesToEven || rm == RoundMode::TiesToOdd || rm == RoundMode::TiesToAway);
}

}    // namespace float_support

/// Functor for casting cfloat_advanced
///
/// Specific casting behavior can be specified when constructing the
/// functor.
///
/// By default, OVERFLOW mode is used when the destination type has either
/// infinity or NaN representations. Otherwise SATURATE mode is used. It is
/// illegal to specify OVERFLOW mode for a type which has neither infinity
/// or NaN representations - this will result in a compilation error.
template <class in_type,
          class out_type,
          RoundMode round_mode = RoundMode::TiesToEven,
          OverflowMode overflow_mode =
              (out_type::has_nan || out_type::has_inf) ? OverflowMode::Overflow : OverflowMode::Saturate,
          SubnormalMode subnormal_mode = SubnormalMode::Retain>
class cfloat_cast
{
    constexpr static FloatFeatures in_feats  = in_type::features;
    constexpr static FloatFeatures out_feats = out_type::features;
    constexpr static size_t in_bits          = in_type::n_bits;
    constexpr static size_t in_exp_bits      = in_type::n_exponent_bits;
    constexpr static size_t out_bits         = out_type::n_bits;
    constexpr static size_t out_exp_bits     = out_type::n_exponent_bits;

public:
    constexpr cfloat_cast()
    {
        // SATURATE mode MUST be specified if the destination type does not
        // have either NaN or infinity representations.
        static_assert(overflow_mode == OverflowMode::Saturate || out_type::has_nan || out_type::has_inf);
    }

    /// \brief Cast from `in` to the given `out_type`
    //
    // This code relies on an understanding of the storage format used by
    // `cfloat_advanced`. See the documentation of that class for further
    // details.
    constexpr out_type operator()(const in_type& in) const
    {
        // Shortcut for types which differ only in the number of significand
        // bits, and where the output type is wider than the input type. For
        // example, bfloat16 and binary32.
        if constexpr (in_exp_bits == out_exp_bits && out_bits >= in_bits && in_feats == out_feats)
        {
            return out_type::from_bits(static_cast<typename out_type::storage_t>(in.bits()) << (out_bits - in_bits));
        }

        // Get initial values for the new floating point type
        const bool sign_bit       = in.sign();
        int64_t new_exponent_bits = 0;
        uint64_t new_significand  = 0;

        if (in.is_nan() || in.is_infinity())
        {
            // The mapping of infinity to the destination type depends upon
            // the overflow mode and the features of the destination type.
            // OVERFLOW mode is the "expected" behaviour, in which exception
            // values (NaN and infinity) map to themselves in the
            // destination type (assuming they exist). In SATURATION mode,
            // infinity maps to the largest absolute value of the
            // destination type _even if_ an infinity encoding is available.
            // See the FP8 specification document.
            //
            // By default, exceptional values are encoded with an all-1
            // exponent field.
            new_exponent_bits = (UINT64_C(1) << out_exp_bits) - 1;

            if (in.is_nan())
            {
                // NaN always maps to NaN if it's available.
                //
                // NB: if the type has both NaN AND Infinity support, then
                // the entirety of the significand can be used to encode
                // different values of NaN (excepting significand = 0,
                // which is reserved for infinity). This makes it possible
                // to encode both quiet and signalling varieties.
                // Generally, the MSB of the significand represents "not
                // quiet".  However, when there is only 1 NaN encoding
                // (which is generally the case when infinity is not
                // supported), then there cannot be separate quiet and
                // signalling varieties of NaN.
                if constexpr (out_type::has_inf)
                {
                    // Set the `not_quiet` bit.
                    new_significand = UINT64_C(1) << (out_type::n_significand_bits - 1);
                    if constexpr (in_type::n_significand_bits > 0)
                    {
                        // Copy across the `not_quiet` bit from the other
                        // type; but not the payload.
                        new_significand &=
                            (static_cast<uint64_t>(in.significand()) >> (in_type::n_significand_bits - 1))
                            << (out_type::n_significand_bits - 1);
                    }
                    // Also set the LSB to ensure that we've encoded a NaN
                    // of some variety (and not infinity). This could be
                    // conditional on the `not_quiet` bit, but
                    // unconditionally setting the LSB is fine.
                    new_significand |= 0x1;
                }
                else
                {
                    // If there is no representation of infinity then we
                    // assume a single encoding of NaN, with all bits set.
                    new_significand = (UINT64_C(1) << out_type::n_significand_bits) - 1;
                }
            }
            else if constexpr (overflow_mode == OverflowMode::Saturate)
            {
                // In SATURATE mode, infinity in the input maps to the
                // largest absolute value in the output type; even if
                // infinity is available. This is in compliance with Table 3
                // of the FP8 specification.
                return out_type::max(sign_bit);
            }
            else if constexpr (!out_type::has_inf && overflow_mode == OverflowMode::Overflow)
            {
                // In OVERFLOW mode, infinities in the input type map to NaN
                // in the output type, if infinity is not available.
                new_significand = (UINT64_C(1) << out_type::n_significand_bits) - 1;
            }
        }
        else if (!in.is_zero())
        {
            const int64_t this_exponent_bits = in.exponent_bits();
            {
                constexpr int64_t exponent_rebias = out_type::exponent_bias - in_type::exponent_bias;
                new_exponent_bits                 = std::max(this_exponent_bits + exponent_rebias, exponent_rebias + 1);
            }
            if constexpr (in_type::n_significand_bits > 0)
                new_significand = in.significand() << (64 - in_type::n_significand_bits);
            // Normalise subnormals
            if (this_exponent_bits == 0)
            {
                // Shift the most-significant 1 out of the magnitude to
                // convert it to a significand. Modify the exponent
                // accordingly.
                // NOTE: We know that there's a 1 somewhere in the
                // significand, because `in.is_zero()` was not true.
                while (~new_significand & (UINT64_C(1) << 63))
                {
                    new_exponent_bits--;
                    new_significand <<= 1;
                }
                new_exponent_bits--;
                new_significand <<= 1;
            }

            // Apply overflow to out-of-range values; this must occur before
            // rounding, as out-of-range values could be rounded down to the
            // largest representable value.
            if constexpr (overflow_mode == OverflowMode::Overflow)
            {
                // Determine the maximum value of exponent, and unrounded
                // significand.
                constexpr bool inf_and_nan     = out_type::has_nan && out_type::has_inf;
                constexpr int64_t max_exp_bits = (INT64_C(1) << out_exp_bits) - (inf_and_nan ? 2 : 1);
                constexpr uint64_t max_significand =
                    (out_type::n_significand_bits > 0)
                        ? ((UINT64_C(1) << out_type::n_significand_bits) - (inf_and_nan ? 1 : 2))
                              << (64 - out_type::n_significand_bits)
                        : 0;

                // If the exponent is strictly larger than the largest
                // possible, or the exponent is equal to the largest
                // possible AND the (unrounded) significand is strictly
                // larger than the largest possible then return an
                // appropriate overflow value.
                if (new_exponent_bits > max_exp_bits ||
                    (new_exponent_bits == max_exp_bits && new_significand > max_significand))
                {
                    if constexpr (out_type::has_inf)
                        return out_type::infinity(sign_bit);
                    else
                        return out_type::NaN();
                }
            }

            // Handle output subnormals; either reinsert the leading `1` and
            // align the significand correctly; or flush-to-zero.
            if (new_exponent_bits <= 0)
            {
                if constexpr (subnormal_mode == SubnormalMode::FlushToZero)
                {
                    // Flush to zero
                    new_significand = 0;
                }
                else if (new_exponent_bits < -63)
                {
                    new_significand = 0;
                }
                else
                {
                    // Shift to handle the non-positive exponent, and insert
                    // a leading one in the appropriate bit.
                    new_significand =
                        (UINT64_C(1) << (63 + new_exponent_bits)) | (new_significand >> (1 - new_exponent_bits));
                }

                // Set the new exponent bits to zero to represent a
                // subnormal.
                new_exponent_bits = 0;
            }

            // Align the significand for the output type
            // Switch to a new representation of the significand:
            //  * new_significand [aligned to LSB])
            //  * rest_of_significand [aligned to MSB]
            constexpr uint32_t realign_shift   = 64 - out_type::n_significand_bits;
            const uint64_t rest_of_significand = new_significand << (64 - realign_shift);
            if constexpr (realign_shift >= 64)
            {
                // This can occur for FP8_E8M0 types
                new_significand = 0;
            }
            else
            {
                new_significand = new_significand >> realign_shift;
            }

            // Apply rounding based on values shifted out of the significand
            if (rest_of_significand && (round_mode != RoundMode::TowardZero))
            {
                if constexpr (float_support::is_round_to_nearest(round_mode))
                {
                    // Increment the significand if:
                    //  * the shifted out bits are greater than half-way
                    //    between two representable numbers
                    //  * the shifted out bits are exactly half-way AND
                    //    either:
                    //    * the rounding mode is ties away from zero
                    //    * the rounding mode is ties to even AND the
                    //      significand is odd
                    //    * the rounding mode is ties to odd AND the
                    //      significand is even
                    constexpr uint64_t tie = UINT64_C(1) << 63;
                    if (rest_of_significand > tie || (rest_of_significand == tie &&
                                                      (round_mode == RoundMode::TiesToAway ||
                                                       ((new_significand & 1) && round_mode == RoundMode::TiesToEven) ||
                                                       (!(new_significand & 1) && round_mode == RoundMode::TiesToOdd))))
                    {
                        new_significand += 1;
                    }
                }
                else if constexpr (round_mode == RoundMode::TowardPositiveInfinity)
                {
                    // Truncate negative values, round up positive values
                    if (!sign_bit)
                    {
                        new_significand += 1;
                    }
                }
                else if constexpr (round_mode == RoundMode::TowardNegativeInfinity)
                {
                    // Truncate positive values, round up negative values
                    if (sign_bit)
                    {
                        new_significand += 1;
                    }
                }

                // Check if rounding caused the significand to overflow
                constexpr uint64_t max_significand = (UINT64_C(1) << out_type::n_significand_bits) - 1;
                if (new_significand > max_significand)
                {
                    new_significand = 0;
                    new_exponent_bits++;
                }
            }

            // Saturate or overflow if the value is larger than can be
            // represented in the output type. This can only occur if
            // the size of the exponent of the new type is not greater
            // than the exponent of the old type.
            if constexpr (out_exp_bits <= in_exp_bits)
            {
                constexpr int64_t inf_exp_bits = (INT64_C(1) << out_exp_bits) - 1;
                if (new_exponent_bits >= inf_exp_bits)
                {
                    if constexpr (out_type::has_inf && overflow_mode == OverflowMode::Overflow)
                    {
                        // If the output type has a representation of
                        // infinity, and we are in OVERFLOW Mode, then
                        // return infinity.
                        new_exponent_bits = inf_exp_bits;
                        new_significand   = 0;
                    }
                    else if constexpr (out_type::has_inf)
                    {
                        // If the output type has a representation of
                        // infinity, and we are in SATURATE mode, then
                        // return the largest representable real number.
                        new_exponent_bits = inf_exp_bits - 1;
                        new_significand   = (UINT64_C(1) << out_type::n_significand_bits) - 1;
                    }
                    else if (new_exponent_bits > inf_exp_bits)
                    {
                        if constexpr (overflow_mode == OverflowMode::Overflow)
                            return out_type::NaN();
                        else
                            return out_type::max(sign_bit);
                    }
                    else
                    {
                        constexpr uint64_t max_significand =
                            (UINT64_C(1) << out_type::n_significand_bits) - (out_type::has_nan ? 2 : 1);
                        if (new_significand > max_significand)
                        {
                            if constexpr (overflow_mode == OverflowMode::Saturate)
                                new_significand = max_significand;
                            else
                                return out_type::NaN();
                        }
                    }
                }
            }
        }

        return out_type::from_bits(sign_bit, new_exponent_bits, new_significand);
    }
};

/// \brief Bit-accurate representation storage of IEEE754 compliant and
///        derived floating point types.
///
/// Template parameters allow for specification of the number of bits, the
/// number of exponent bits, and the features of the floating point types.
/// The number of significand bits is `n_bits - n_exponent_bits - 1`. It is
/// not possible to represent a signless type, such as FP8 E8M0.
///
/// For an imaginary 7-bit type, FP7 E4M2; the storage for various values
/// given different floating point features is given below:
///
/// Value                      All features   No infinity  No features
/// -------------------------- ------------   -----------  -----------
/// Positive zero +0            00 0000 00    As before    As before
/// Negative zero -0            11 0000 00    As before    As before
/// Positive/negative infinity  SS 1111 00    N/A          N/A
/// Signalling NaN              SS 1111 01    SS 1111 11   N/A
/// Quiet NaN                   SS 1111 11    N/A          N/A
/// Largest normal              SS 1110 11    SS 1111 10   SS 1111 11
/// Smallest normal             SS 0001 00    As before    SS 0000 01
/// Largest denormal            SS 0000 11    SS 0000 11   N/A
///
/// Note that the sign bit is extended to fill the storage type.
template <size_t _n_bits, size_t n_exp_bits, FloatFeatures Feats = float_support::AllFeats>
class cfloat_advanced
{
public:
    using storage_t = float_support::storage_type_t<float_support::get_storage_bytes(_n_bits)>;

    static constexpr size_t n_bits             = _n_bits;
    static constexpr size_t n_exponent_bits    = n_exp_bits;
    static constexpr size_t n_significand_bits = n_bits - (1 + n_exp_bits);
    static constexpr int64_t exponent_bias     = (INT64_C(1) << (n_exp_bits - 1)) - 1;

    static constexpr FloatFeatures features = Feats;
    static constexpr bool has_nan           = (Feats & FloatFeatures::HasNaN) != FloatFeatures::None;
    static constexpr bool has_inf           = (Feats & FloatFeatures::HasInf) != FloatFeatures::None;
    static constexpr bool has_denorms       = (Feats & FloatFeatures::HasDenorms) != FloatFeatures::None;

    /// \brief Construct a floating point type with the given bit
    /// representation.
    static constexpr cfloat_advanced from_bits(storage_t bits)
    {
        return cfloat_advanced(float_support::hidden(), bits);
    }

    /// \brief Construct a float from the given sign, exponent and
    /// significand bits.
    static constexpr cfloat_advanced from_bits(bool pm, storage_t e, storage_t s)
    {
        storage_t bits = pm ? -1 : 0;

        bits <<= n_exp_bits;
        bits |= e;

        bits <<= n_significand_bits;
        if (has_denorms || e)
            bits |= s;

        return cfloat_advanced(float_support::hidden(), bits);
    }

    /// \brief (Hidden) Construct a float type from a given bit pattern
    constexpr cfloat_advanced(const float_support::hidden&, storage_t bits)
        : m_data(bits)
    {}

    constexpr cfloat_advanced()
        : m_data(0)
    {}
    constexpr cfloat_advanced(const cfloat_advanced& other)
        : m_data(other.m_data)
    {}

    constexpr cfloat_advanced& operator=(const cfloat_advanced& other)
    {
        this->m_data = other.m_data;
        return *this;
    }

    constexpr cfloat_advanced& operator=(cfloat_advanced&& other)
    {
        this->m_data = other.m_data;
        return *this;
    }

    /// \brief Get a NaN representation
    static constexpr cfloat_advanced NaN()
    {
        static_assert(has_nan);

        // NaN is always encoded with all 1s in the exponent.
        // If Inf exists, then NaN is encoded as a non-zero significand; if
        // Inf doesn't exist then NaN is encoded as all ones in the
        // significand.
        constexpr uint64_t exp_bits = (UINT64_C(1) << n_exponent_bits) - 1;
        constexpr uint64_t sig_bits = has_inf ? 1 : (UINT64_C(1) << n_significand_bits) - 1;
        return cfloat_advanced::from_bits(false, exp_bits, sig_bits);
    }

    /// \brief Get a representation of infinity
    static constexpr cfloat_advanced infinity(const bool& sign)
    {
        static_assert(has_inf);

        // Inf is always encoded with all 1s in the exponent, and all zeros
        // in the significand.
        return cfloat_advanced::from_bits(sign, (UINT64_C(1) << n_exponent_bits) - 1, 0);
    }

    /// \brief Get the largest representable value
    static constexpr cfloat_advanced max(const bool& sign)
    {
        if constexpr (has_nan && has_inf)
        {
            // Where we have NaN and Infinity, exponents all `1` corresponds
            // to some of these values.
            return from_bits(sign, (UINT64_C(1) << n_exponent_bits) - 2, (UINT64_C(1) << n_significand_bits) - 1);
        }
        else if constexpr (has_nan || has_inf)
        {
            // Where we have either NaN or infinity (but not both),
            // exponents all `1` AND significand all `1` corresponds to the
            // special value.
            return from_bits(sign, (UINT64_C(1) << n_exponent_bits) - 1, (UINT64_C(1) << n_significand_bits) - 2);
        }
        else
        {
            // With no special values to encode, the maximum value is
            // encoded as all `1`s.
            return from_bits(sign, (UINT64_C(1) << n_exponent_bits) - 1, (UINT64_C(1) << n_significand_bits) - 1);
        }
    }

    /// \brief Cast to a different floating point representation.
    template <size_t out_n_bits, size_t out_n_exp_bits, FloatFeatures OutFeats>
    constexpr inline operator cfloat_advanced<out_n_bits, out_n_exp_bits, OutFeats>() const
    {
        using out_type = cfloat_advanced<out_n_bits, out_n_exp_bits, OutFeats>;
        return cfloat_cast<cfloat_advanced, out_type>().operator()(*this);
    }

    /// \brief Convert from a 32-bit floating point value
    BITCAST_CONSTEXPR
    cfloat_advanced(const float& f)
    {
        // If this format exactly represents the binary32 format then get
        // the bits from the provided float; otherwise get a binary32
        // representation and then convert to this format.
        if constexpr (represents_binary32())
            m_data = float_support::get_bits(f);
        else
            m_data =
                static_cast<cfloat_advanced<n_bits, n_exp_bits, Feats>>(static_cast<cfloat_advanced<32, 8>>(f)).m_data;
    }

    /// \brief Cast to a 32-bit floating point value
    BITCAST_CONSTEXPR operator float() const
    {
        // If this format exactly represents the binary32 format then return
        // a float; otherwise get a binary32 representation and then return
        // a float.

        // clang-format off
        if constexpr (represents_binary32())
            return float_support::from_bits(m_data);
        else
            return static_cast<float>(this->operator cfloat_advanced<32, 8>());
        // clang-format on
    }

    /// \brief Return whether this type represents the IEEE754 binary32
    /// format
    constexpr static inline bool represents_binary32()
    {
        return std::is_same_v<storage_t, uint32_t> && n_exp_bits == 8 && Feats == float_support::AllFeats;
    }

    constexpr auto operator-() const
    {
        constexpr storage_t sign_bits =
            static_cast<storage_t>(std::numeric_limits<std::make_unsigned_t<storage_t>>::max() << (n_bits - 1));
        return from_bits(m_data ^ sign_bits);
    }

    constexpr bool is_subnormal() const
    {
        return exponent_bits() == 0 && significand() != 0;
    }

    constexpr bool is_zero() const
    {
        // Zero is represented by everything but the sign bit(s) being zero
        constexpr storage_t sign_bit = (storage_t(1) << (n_bits - 1));
        return (m_data & ~sign_bit) == 0;
    }

    constexpr bool is_nan() const
    {
        return has_nan && (exponent_bits() == (UINT64_C(1) << n_exponent_bits) - 1) &&
               ((has_inf && significand()) || (!has_inf && significand() == (UINT64_C(1) << n_significand_bits) - 1));
    }

    constexpr bool is_infinity() const
    {
        return has_inf && ((exponent_bits() == (UINT64_C(1) << n_exponent_bits) - 1) && (significand() == 0));
    }

    constexpr inline const storage_t& bits() const
    {
        return m_data;
    }

    /// \brief Get the exponent
    constexpr inline int64_t exponent() const
    {
        return std::max<int64_t>(exponent_bits(), INT64_C(1)) - exponent_bias;
    }

    /// \brief Get the sign bit
    constexpr inline bool sign() const
    {
        return (m_data >> (n_bits - 1)) & 0x1;
    }

    /// \brief Get the bits from the exponent
    constexpr inline uint64_t exponent_bits() const
    {
        constexpr uint64_t mask = (UINT64_C(1) << n_exp_bits) - 1;
        return (m_data >> n_significand_bits) & mask;
    }

    constexpr inline uint64_t significand() const
    {
        return m_data & ((UINT64_C(1) << n_significand_bits) - 1);
    }

    constexpr inline bool operator==(const cfloat_advanced& other) const
    {
        return !is_nan() && !other.is_nan() &&    // Neither operand is NaN
               ((is_zero() && other.is_zero()) || (m_data == other.m_data));
    }

    constexpr inline bool operator!=(const cfloat_advanced& other) const
    {
        return !(*this == other);
    }

    constexpr inline cfloat_advanced& operator+=(const cfloat_advanced& rhs)
    {
        this->m_data = static_cast<cfloat_advanced>(static_cast<float>(*this) + static_cast<float>(rhs)).bits();
        return *this;
    }

private:
    storage_t m_data = 0;
};

// This should probably be exported so we can use it elsewhere
#undef BITCAST_CONSTEXPR

/// \brief Wrapper to maintain API compatibility with older code, which was
/// limited to power-of-two sizes of floats.
template <typename storage_t,
          size_t n_exp_bits,
          bool has_nan,
          bool with_denorm,
          bool with_infinity,
          std::enable_if_t<(n_exp_bits + 1 < sizeof(storage_t) * 8), bool> = true>
using cfloat = cfloat_advanced<sizeof(storage_t) * 8,
                               n_exp_bits,
                               float_support::get_float_flags(has_nan, with_denorm, with_infinity)>;

namespace float_support
{
// Pre-C++23 these can't be computed as constexpr, so have to hardcode
// them

template <int>
struct digits10;    // floor(log10(2) * (digits - 1)
template <int>
struct max_digits10;    // ceil(log10(2) * digits + 1)
template <int>
struct min_exponent10;    // floor(log10(2) * min_exponent)
template <int>
struct max_exponent10;    // floor(log10(2) * max_exponent)

template <>
struct digits10<8>
{
    constexpr static inline int value = 2;
};

template <>
struct max_digits10<8>
{
    constexpr static inline int value = 4;
};

template <>
struct digits10<10>
{
    constexpr static inline int value = 2;
};

template <>
struct max_digits10<10>
{
    constexpr static inline int value = 5;
};

template <>
struct digits10<24>
{
    constexpr static inline int value = 6;
};

template <>
struct max_digits10<24>
{
    constexpr static inline int value = 9;
};

template <>
struct min_exponent10<-13>
{
    constexpr static inline int value = -3;
};

template <>
struct max_exponent10<16>
{
    constexpr static inline int value = 4;
};

template <>
struct min_exponent10<-125>
{
    constexpr static inline int value = -37;
};

template <>
struct max_exponent10<128>
{
    constexpr static inline int value = 38;
};

template <int d>
inline constexpr int digits10_v = digits10<d>::value;
template <int d>
inline constexpr int max_digits10_v = max_digits10<d>::value;

template <int e>
inline constexpr int min_exponent10_v = min_exponent10<e>::value;

template <int e>
inline constexpr int max_exponent10_v = max_exponent10<e>::value;

}    // namespace float_support

// Typedef some commonly used floating point types
using binary32 = ct::cfloat_advanced<32, 8, ct::float_support::AllFeats>;
using float32  = binary32;
using bfloat16 = ct::cfloat_advanced<16, 8, ct::float_support::AllFeats>;
using binary16 = ct::cfloat_advanced<16, 5, ct::float_support::AllFeats>;
using float16  = binary16;
using fp8_e4m3 = ct::cfloat_advanced<8, 4, ct::FloatFeatures::HasNaN | ct::FloatFeatures::HasDenorms>;
using fp8_e5m2 = ct::cfloat_advanced<8, 5, ct::float_support::AllFeats>;

}    // namespace ct

namespace std
{

template <size_t n_bits, size_t n_exp_bits, ct::FloatFeatures Feats>
struct is_floating_point<ct::cfloat_advanced<n_bits, n_exp_bits, Feats>> : std::integral_constant<bool, true>
{};

template <size_t n_bits, size_t n_exp_bits, ct::FloatFeatures Feats>
class numeric_limits<ct::cfloat_advanced<n_bits, n_exp_bits, Feats>>
{
    using this_cfloat = ct::cfloat_advanced<n_bits, n_exp_bits, Feats>;

public:
    static constexpr bool is_specialized = true;

    static constexpr auto min() noexcept
    {
        return this_cfloat::from_bits(false, 1, 0);
    }

    static constexpr auto max() noexcept
    {
        return this_cfloat::max(false);
    }
    static constexpr auto lowest() noexcept
    {
        return -max();
    }

    static constexpr int digits       = this_cfloat::n_significand_bits + 1;
    static constexpr int digits10     = ct::float_support::digits10_v<digits>;
    static constexpr int max_digits10 = ct::float_support::max_digits10_v<digits>;

    static constexpr bool is_signed  = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact   = false;
    static constexpr int radix       = 2;

    static constexpr auto epsilon() noexcept
    {
        return this_cfloat::from_bits(false, this_cfloat::exponent_bias - this_cfloat::n_significand_bits, 0);
    }

    static constexpr auto round_error() noexcept
    {
        return this_cfloat::from_bits(0, this_cfloat::exponent_bias - 1, 0);
    }

    static constexpr int min_exponent   = (1 - this_cfloat::exponent_bias) + 1;
    static constexpr int min_exponent10 = ct::float_support::min_exponent10_v<min_exponent>;
    static constexpr int max_exponent   = this_cfloat::exponent_bias + 1;
    static constexpr int max_exponent10 = ct::float_support::max_exponent10_v<max_exponent>;

    static constexpr bool has_infinity             = this_cfloat::has_inf;
    static constexpr bool has_quiet_NaN            = this_cfloat::has_nan && this_cfloat::has_inf;
    static constexpr bool has_signaling_NaN        = this_cfloat::has_nan;
    static constexpr float_denorm_style has_denorm = this_cfloat::has_denorms ? denorm_present : denorm_absent;
    static constexpr bool has_denorm_loss          = false;

    static constexpr auto infinity() noexcept
    {
        if constexpr (this_cfloat::has_inf)
        {
            return this_cfloat::infinity(false);
        }
        else
        {
            return this_cfloat::from_bits(false, 0, 0);
        }
    }

    static constexpr auto quiet_NaN() noexcept
    {
        const uint64_t exp_bits = (UINT64_C(1) << this_cfloat::n_exponent_bits) - 1;
        const uint64_t sig_bits = this_cfloat::has_inf ? (UINT64_C(1) << (this_cfloat::n_significand_bits - 1)) | 1
                                                       : (UINT64_C(1) << this_cfloat::n_significand_bits) - 1;
        return this_cfloat::from_bits(false, exp_bits, sig_bits);
    }

    static constexpr auto signaling_NaN() noexcept
    {
        const uint64_t exp_bits = (UINT64_C(1) << this_cfloat::n_exponent_bits) - 1;
        const uint64_t sig_bits = this_cfloat::has_inf ? 1 : (UINT64_C(1) << this_cfloat::n_significand_bits) - 1;
        return this_cfloat::from_bits(false, exp_bits, sig_bits);
    }

    static constexpr auto denorm_min() noexcept
    {
        return this_cfloat::from_bits(false, 0, 1);
    }

    static constexpr bool is_iec559  = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo  = false;

    static constexpr bool traps                    = false;
    static constexpr bool tinyness_before          = false;
    static constexpr float_round_style round_style = round_to_nearest;
};

}    // namespace std

#endif    //  CT_CFLOAT_H
