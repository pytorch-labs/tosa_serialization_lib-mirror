#!/usr/bin/env python3

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

import serializer.tosa_serializer as ts
import numpy as np
import pytest
from pytests.test_utils import generate_random_data
from serializer.numpy_utils import save_npy, load_npy

TESTED_DTYPES = set(ts.DTypeNames)


def dtype_str_cases():
    for dtype_str in TESTED_DTYPES:
        if dtype_str in ["UNKNOWN"]:
            continue
        yield dtype_str


@pytest.mark.parametrize("dtype_str", dtype_str_cases())
def test_load_save(tmp_path, request, dtype_str):
    dtype = ts.dtype_str_to_val(dtype_str)
    original_data, shape, py_dtype = generate_random_data(dtype_str)

    testname: str = request.node.name
    test_dir = tmp_path / testname
    test_dir.mkdir(exist_ok=True)
    file_path = test_dir / f"test_{dtype_str}.npy"

    save_npy(file_path, original_data, dtype)

    loaded_data = load_npy(file_path, dtype)

    assert np.array_equal(loaded_data, original_data, equal_nan=True)
