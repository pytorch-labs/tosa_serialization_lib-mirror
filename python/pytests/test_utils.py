# Copyright (c) 2025, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import random
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2, int4, finfo, iinfo


def generate_random_data(dtype_str):
    # Creating the random data.

    shape = random.sample(range(1, 16), random.randint(1, 3))

    FLOAT_TYPES = {
        "FP32": np.float32,
        "FP16": np.float16,
        "BF16": bfloat16,
        "FP8E4M3": float8_e4m3fn,
        "FP8E5M2": float8_e5m2,
    }
    INT_TYPES = {
        "INT4": int4,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "UINT16": np.uint16,
        "UINT8": np.uint8,
    }

    if dtype_str in FLOAT_TYPES:
        py_dtype = FLOAT_TYPES[dtype_str]
        data = np.random.uniform(
            finfo(py_dtype).min, finfo(py_dtype).max, shape
        ).astype(py_dtype)

        # Generating -inf, inf, -nan, nan with a 10% chance each.
        # Note that fp8e4m3 doesn't have infinities so they become NaN
        mask = np.random.rand(*shape)
        data = np.select(
            [mask < 0.1, mask < 0.2, mask < 0.3, mask < 0.4],
            np.array([-np.inf, np.inf, -np.nan, np.nan]).astype(py_dtype),
            data,
        )
    elif dtype_str in INT_TYPES:
        py_dtype = INT_TYPES[dtype_str]
        data = np.random.uniform(
            iinfo(py_dtype).min, iinfo(py_dtype).max, shape
        ).astype(py_dtype)
    elif dtype_str == "BOOL":
        py_dtype = bool
        data = (np.random.rand(*shape) >= 0.5).astype(bool)
    elif dtype_str == "INT48":
        py_dtype = np.int64
        data = np.random.uniform(-(2**47), 2**47 - 1, shape).astype(py_dtype)
    elif dtype_str == "SHAPE":
        py_dtype = np.int64
        data = np.random.uniform(
            iinfo(py_dtype).min, iinfo(py_dtype).max, shape
        ).astype(py_dtype)
    else:
        raise NotImplementedError(
            f"Random tensor generation for type {dtype_str} not implemented. \
Consider adding to SKIPPED_DTYPES."
        )

    return data, shape, py_dtype
