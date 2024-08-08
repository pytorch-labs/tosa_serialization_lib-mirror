#!/usr/bin/env python3

# Copyright (c) 2024, ARM Limited.
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

import json
import pathlib
import random
import subprocess
import serializer.tosa_serializer as ts
import pytest
import numpy as np
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2, int4, finfo, iinfo


# These datatypes are skipped during testing, presumably since they are new
# and tests have not yet been implemented. This should be emptied frequently.
SKIPPED_DTYPES = []

TESTED_DTYPES = set(ts.DTypeNames) - set(SKIPPED_DTYPES)


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
        if dtype_str == "FP8E5M2":
            # FP8E5M2 arrays should be received bitcasted as uint8 arrays
            data = data.view(np.uint8)

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


def serialize_and_load_json(ser: ts.TosaSerializer, tosa_filename) -> dict:
    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    flatc = base_dir / "third_party/flatbuffers/flatc"
    tosa_fbs = base_dir / "schema/tosa.fbs"

    # Serializing to flatbuffer and writing to a temporary file
    with open(tosa_filename, "wb") as f:
        f.write(ser.serialize())

    # Using flatc to convert the flatbuffer to strict json
    _ = subprocess.run(
        [
            flatc,
            "--json",
            "--strict-json",
            "--defaults-json",
            "-o",
            tosa_filename.parent,
            tosa_fbs,
            "--",
            tosa_filename,
        ],
        check=True,
    )

    assert str(tosa_filename).endswith(".tosa")
    json_filename = str(tosa_filename).removesuffix(".tosa") + ".json"

    with open(json_filename, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("dtype_str", TESTED_DTYPES)
def test_single_intermediate(request, dtype_str):
    """
    Creating an intermediate tensor of each dtype
    """

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname = request.node.name

    # Creating a new folder for each test case to dump numpy files
    ser = ts.TosaSerializer(tmp_dir / testname)
    (tmp_dir / testname).mkdir(exist_ok=True)

    shape = random.sample(range(1, 2**31), random.randint(1, 16))
    tensor = ser.addIntermediate(shape, ts.dtype_str_to_val(dtype_str))

    assert tensor.dtype == ts.dtype_str_to_val(dtype_str)
    assert tensor.shape == shape

    serialized = serialize_and_load_json(ser, tmp_dir / testname / f"{testname}.tosa")

    tensor_serialized = serialized["regions"][0]["blocks"][0]["tensors"][0]

    assert (
        tensor_serialized.items()
        >= {
            "is_unranked": False,
            "shape": shape,
            "type": dtype_str,
            "variable": False,
        }.items()
    )


def placeholder_cases():
    for dtype_str in TESTED_DTYPES:
        # The ml_dtypes library has issues with serializing FP8E5M2 to .npy
        # files, so we don't test it.
        if dtype_str in ["UNKNOWN", "FP8E5M2"]:
            continue
        yield dtype_str


@pytest.mark.parametrize("dtype_str", placeholder_cases())
def test_single_placeholder(request, dtype_str):
    """
    Creating a placeholder tensor of each dtype. The data of these placeholder
    tensors is saved in .npy files.
    """

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname: str = request.node.name

    data, shape, py_dtype = generate_random_data(dtype_str)

    # Creating a new folder for each test case to dump numpy files
    ser = ts.TosaSerializer(tmp_dir / testname)
    (tmp_dir / testname).mkdir(exist_ok=True)

    tensor = ser.addPlaceholder(shape, ts.dtype_str_to_val(dtype_str), data)

    serialized = serialize_and_load_json(ser, tmp_dir / testname / f"{testname}.tosa")

    tensor_serialized = serialized["regions"][0]["blocks"][0]["tensors"][0]

    assert (
        tensor_serialized.items()
        >= {
            "is_unranked": False,
            "shape": shape,
            "type": dtype_str,
            "variable": False,
        }.items()
    )

    npy_data = np.load(
        ser.currRegion.pathPrefix / tensor.placeholderFilename,
    ).view(py_dtype)

    assert np.array_equal(npy_data, data, equal_nan=True)


def const_cases():
    for dtype_str in TESTED_DTYPES:
        for const_mode in ts.ConstMode.__members__.values():
            # We don't support uint8 or uint16 serialization to flatbuffer;
            # see convertDataToUint8Vec
            if dtype_str in ["UNKNOWN", "UINT8", "UINT16"]:
                continue
            # The ml_dtypes library has issues with serializing FP8E5M2 to
            # .npy files, so we don't test it.
            if dtype_str == "FP8E5M2" and const_mode != ts.ConstMode.EMBED:
                continue
            yield dtype_str, const_mode


@pytest.mark.parametrize("dtype_str,const_mode", const_cases())
def test_single_const(request, dtype_str, const_mode):
    """
    Creating a const tensor of each dtype. The data of these placeholder
    tensors is saved in .npy files and/or the flatbuffer itself, depending
    on the const mode.
    """

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname = request.node.name

    data, shape, py_dtype = generate_random_data(dtype_str)

    # Creating a new folder for each test case to dump numpy files
    ser = ts.TosaSerializer(tmp_dir / testname, constMode=const_mode)
    (tmp_dir / testname).mkdir(exist_ok=True)

    tensor = ser.addConst(shape, ts.dtype_str_to_val(dtype_str), data)

    serialized = serialize_and_load_json(ser, tmp_dir / testname / f"{testname}.tosa")

    tensor_serialized = serialized["regions"][0]["blocks"][0]["tensors"][0]

    assert (
        tensor_serialized.items()
        >= {
            "is_unranked": False,
            "shape": shape,
            "type": dtype_str,
            "variable": False,
        }.items()
    )

    # Testing if data is correctly serialized to .npy
    if const_mode in [ts.ConstMode.INPUTS, ts.ConstMode.EMBED_DUMP]:
        npy_data = np.load(
            ser.currRegion.pathPrefix / f"{tensor.name}.npy",
        ).view(py_dtype)
        assert np.array_equal(npy_data, data, equal_nan=True)

    # Testing if data is correctly serialized as bytes to flatbuffer.
    if const_mode in [ts.ConstMode.EMBED, ts.ConstMode.EMBED_DUMP]:
        u8_data = np.array(tensor_serialized["data"], dtype=np.uint8)

        # Note that TOSA flatbuffer INT/SHAPE serialization is ALWAYS
        # little-endian regardless of the system byteorder; see
        # TosaSerializer.convertDataToUint8Vec. So all
        # uses of .view() here are forced little-endian.

        if dtype_str == "INT48":
            assert np.array_equal(
                np.bitwise_and(data, 0x0000_FFFF_FFFF_FFFF),
                np.pad(u8_data.reshape(-1, 6), ((0, 0), (0, 2)))
                .view(np.dtype("<i8"))  # int64, forced little-endian
                .reshape(shape),
            )
        elif dtype_str == "INT4":
            # Unpacking each uint8 into two int4's
            first = u8_data.astype(int4)
            second = (u8_data >> 4).astype(int4)
            alternating = np.ravel((first, second), order="F").copy()

            # There could be an extra int4 added for padding, so we check
            # that the flatbuffer array's size is correct and then force
            # it to the shape we want
            assert alternating.size == (np.prod(shape) + 1) // 2 * 2
            assert np.array_equal(data, np.resize(alternating, shape))
        elif dtype_str == "FP8E5M2":
            # data is uint8 already
            assert np.array_equal(data.flatten(), u8_data.flatten())
        else:
            assert np.array_equal(
                data,
                u8_data.view(
                    np.dtype(py_dtype).newbyteorder("<")
                    # forced little-endian
                ).reshape(shape),
                equal_nan=True,
            )
