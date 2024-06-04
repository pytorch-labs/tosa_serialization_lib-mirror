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

#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <tosa_serialization_handler.h>

using namespace tosa;

template <typename T>
using uniform_any_distribution = std::conditional_t<std::is_same<T, bool>::value,
                                                    std::bernoulli_distribution,
                                                    std::conditional_t<std::is_floating_point<T>::value,
                                                                       std::uniform_real_distribution<T>,
                                                                       std::uniform_int_distribution<T>>>;
template <typename T>
uniform_any_distribution<T> get_any_distribution()
{
    return uniform_any_distribution<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}
template <>
uniform_any_distribution<bool> get_any_distribution<bool>()
{
    return std::bernoulli_distribution(0.5);
}

template <class T>
int test_numpy_type(std::vector<int32_t> shape, std::default_random_engine& gen, std::string& filename)
{
    size_t total_size = 1;

    uniform_any_distribution<T> gen_data = get_any_distribution<T>();

    for (auto i : shape)
    {
        total_size *= i;
    }

    auto buffer = std::make_unique<T[]>(total_size);
    for (size_t i = 0; i < total_size; i++)
    {
        buffer[i] = gen_data(gen);
    }

    NumpyUtilities::NPError err = NumpyUtilities::writeToNpyFile(filename.c_str(), shape, buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error writing file, code " << err << std::endl;
        return 1;
    }

    auto read_buffer = std::make_unique<T[]>(total_size);
    err              = NumpyUtilities::readFromNpyFile(filename.c_str(), total_size, read_buffer.get());
    if (err != NumpyUtilities::NO_ERROR)
    {
        std::cout << "Error reading file, code " << err << std::endl;
        return 1;
    }
    if (memcmp(buffer.get(), read_buffer.get(), total_size * sizeof(T)))
    {
        std::cout << "Miscompare" << std::endl;
        return 1;
    }
    return 0;
}

template <typename T>
class NumpyTest : public testing::Test
{};

TYPED_TEST_SUITE_P(NumpyTest);

TYPED_TEST_P(NumpyTest, WriteRead)
{
    std::string source_dir = CMAKE_SOURCE_DIR;
    int32_t seed           = 23;
    std::default_random_engine gen(seed);
    std::string filename       = source_dir + "/test/tmp/Serialization.NumpyTest.npy";
    std::vector<int32_t> shape = { 3, 1, 5 };
    EXPECT_EQ(0, test_numpy_type<TypeParam>(shape, gen, filename));
    std::remove(filename.c_str());
}

REGISTER_TYPED_TEST_SUITE_P(NumpyTest, WriteRead);
using MyTypes = testing::Types<int32_t, int64_t, float, double, bool>;
INSTANTIATE_TYPED_TEST_SUITE_P(SerializationCpp, NumpyTest, MyTypes);
