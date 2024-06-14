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
import pytest


def get_ops():
    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"

    # Helper function for querying lists of dictionaries for a value
    def select(data: list[dict], key, value) -> dict:
        return next(filter(lambda item: item[key] == value, data), None)

    with open(tmp_dir / "tosa.json", encoding="utf-8") as f:
        tosa_schema = json.load(f)

    op_info = select(tosa_schema["enums"], "name", "tosa.Op")["values"]

    for i in op_info:
        yield i["name"]


@pytest.mark.parametrize("op_name", get_ops())
def test_single_op(request, op_name):
    """
    Creating an operator of each type with empty input and output tensors
    and an empty attribute, serializing, deserializing, and checking that
    arguments are preserved.
    """

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname = request.node.name
    (tmp_dir / testname).mkdir(exist_ok=True)

    flatc = base_dir / "third_party/flatbuffers/flatc"
    tosa_fbs = base_dir / "schema/tosa.fbs"

    # Defining temp filepaths
    tosa_file = tmp_dir / testname / f"{testname}.tosa"
    json_file = tmp_dir / testname / f"{testname}.json"

    # Creating example tensors to reference in the operator
    ser = ts.TosaSerializer(tmp_dir)
    ser.currRegion.currBasicBlock.addTensor("t1", [1], ts.DType.INT32)
    ser.currRegion.currBasicBlock.addTensor("t2", [1], ts.DType.INT32)
    ser.currRegion.currBasicBlock.addInput("t1")
    ser.currRegion.currBasicBlock.addOutput("t2")

    # Adding an operator of the given op_name.
    ser.currRegion.currBasicBlock.addOperator(
        getattr(ts.TosaOp.Op(), op_name), ["t1"], ["t2"], None
    )

    # Serializing to flatbuffer and writing to a temporary file
    with open(tosa_file, "wb") as f:
        f.write(ser.serialize())

    # Using flatc to convert the flatbuffer to strict json
    _ = subprocess.run(
        [
            flatc,
            "--json",
            "--strict-json",
            "--defaults-json",
            "-o",
            tosa_file.parent,
            tosa_fbs,
            "--",
            tosa_file,
        ],
        check=True,
    )

    with open(json_file, encoding="utf-8") as f:
        serialized = json.load(f)

    # Getting the arguments of the operator that we serialized
    new_op = serialized["regions"][0]["blocks"][0]["operators"][0]

    assert new_op == {
        "attribute_type": "NONE",
        "inputs": ["t1"],
        "outputs": ["t2"],
        "op": op_name,
    }
