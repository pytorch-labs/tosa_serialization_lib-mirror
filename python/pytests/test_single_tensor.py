#!/usr/bin/env python3

# Copyright (c) 2024-2025, ARM Limited.
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
from ml_dtypes import int4
from pytests.test_utils import generate_random_data


TESTED_DTYPES = set(ts.DTypeNames)


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
        if dtype_str in ["UNKNOWN"]:
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
        else:
            assert np.array_equal(
                data,
                u8_data.view(
                    np.dtype(py_dtype).newbyteorder("<")
                    # forced little-endian
                ).reshape(shape),
                equal_nan=True,
            )
