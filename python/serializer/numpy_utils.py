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
from tosa.DType import DType
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2, int4


def load_npy(fileName, dtype):
    vals = np.load(fileName, allow_pickle=False)

    if vals is not None:
        # Numpy does not support deserialising the dtypes in ml_dtypes
        if dtype == DType.FP8E5M2:
            vals = vals.view(float8_e5m2)
        elif dtype == DType.FP8E4M3:
            vals = vals.view(float8_e4m3fn)
        elif dtype == DType.BF16:
            vals = vals.view(bfloat16)
        elif dtype == DType.INT4:
            vals = vals.view(int4)

    return vals


def save_npy(fileName, vals, dtype):
    if vals is not None:
        # Numpy does not support serialising fp8e5m2 values, so
        # FP8E5M2 arrays should be received bitcasted as uint8 arrays.
        if dtype == DType.FP8E5M2 and vals.dtype != np.uint8:
            vals = vals.view(np.uint8)
        np.save(fileName, vals, allow_pickle=False)
