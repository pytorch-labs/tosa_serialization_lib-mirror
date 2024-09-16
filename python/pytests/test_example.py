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
import subprocess
import serializer.tosa_serializer as ts
import numpy as np


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


def test_example_select(request):

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname = request.node.name

    ser = ts.TosaSerializer(tmp_dir / testname)
    (tmp_dir / testname).mkdir(exist_ok=True)

    input_0 = ts.TosaSerializerTensor("input-0", [2048, 2048, 1], ts.DType.BOOL)
    input_1 = ts.TosaSerializerTensor("input-1", [2048, 2048, 3], ts.DType.INT8)
    input_2 = ts.TosaSerializerTensor("input-2", [2048, 2048, 3], ts.DType.INT8)

    ser.addInputTensor(input_0)
    ser.addInputTensor(input_1)
    ser.addInputTensor(input_2)

    result_0 = ser.addOutput([2048, 2048, 3], ts.DType.INT8)

    ser.addOperator(
        ts.TosaOp.Op().SELECT, ["input-0", "input-1", "input-2"], result_0.name
    )

    serialized = serialize_and_load_json(ser, tmp_dir / testname / f"{testname}.tosa")

    with open(
        base_dir / "python/pytests/examples/test_select_2048x2048x3_i8.json"
    ) as f:
        expected = json.load(f)

    assert serialized["regions"] == expected["regions"]


def test_example_conv2d(request):
    """Testing that pytest and the Python serialization library work"""

    # Defining filepaths
    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname = request.node.name

    ser = ts.TosaSerializer(tmp_dir / testname)
    (tmp_dir / testname).mkdir(exist_ok=True)

    # Creating an example TOSA region
    ser.addConst([5, 5], ts.DType.FP32, np.eye(5, 5), "const-weight")
    ser.addConst([3], ts.DType.FP32, np.ones(3), "const-bias")
    ser.addInputTensor(ts.TosaSerializerTensor("input-0", [256, 256, 3], ts.DType.FP32))
    ser.addInputTensor(ts.TosaSerializerTensor("input-zp", [1], ts.DType.FP32))
    ser.addInputTensor(ts.TosaSerializerTensor("weight-zp", [1], ts.DType.FP32))
    ser.addOutput([256, 256, 3], ts.DType.FP32)

    attr = ts.TosaSerializerAttribute()
    attr.ConvAttribute([2, 2, 2, 2], [1, 1], [1, 1], False, ts.DType.FP32)
    ser.addOperator(
        ts.TosaOp.Op().CONV2D,
        ["input-0", "const-weight", "const-bias", "input-zp", "weight-zp"],
        ["result-0"],
        attr,
    )

    serialized = serialize_and_load_json(ser, tmp_dir / testname / f"{testname}.tosa")

    with open(base_dir / "python/pytests/examples/test_conv2d_256x256x3_f32.json") as f:
        expected = json.load(f)

    assert serialized["regions"] == expected["regions"]
