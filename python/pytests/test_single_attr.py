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
import string
import serializer.tosa_serializer as ts
import pytest


# In some instances, the Python parameter names in TosaSerializerAttribute
# differ from the argument names in the schema. Here are the discrepancies.
# ('schema': 'python')
FIELD_NAME_REPLACEMENTS = {
    # spelling differences
    ("TransposeConvAttribute", "out_pad"): "outpad",
    # these are for a reason; PadAttribute and ClampAttribute have
    # inputs that are byte arrays, and the param names reflect this
    ("PadAttribute", "pad_const"): "pad_const_val_as_bytes",
    ("ClampAttribute", "min_val"): "min_val_as_bytes",
    ("ClampAttribute", "max_val"): "max_val_as_bytes",
}

# When converting the tosa schema to json, the enums are lost and
# replaced with UInt, so the enum names are hard-coded here.
ENUM_FIELDS = {
    ("ConvAttribute", "acc_type"): "DType",
    ("PoolAttribute", "acc_type"): "DType",
    ("PoolAttribute", "nan_mode"): "NanPropagationMode",
    ("TransposeConvAttribute", "acc_type"): "DType",
    ("ResizeAttribute", "mode"): "ResizeMode",
    ("ClampAttribute", "nan_mode"): "NanPropagationMode",
    ("NanPropagationAttribute", "nan_mode"): "NanPropagationMode",
    ("AxisAttribute", "nan_mode"): "NanPropagationMode",
}


def get_attributes():
    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"

    # Helper function for querying lists of dictionaries for a value
    def select(data: list[dict], key, value) -> dict:
        return next(filter(lambda item: item[key] == value, data), None)

    with open(tmp_dir / "tosa.json", encoding="utf-8") as f:
        tosa_schema = json.load(f)

    attribute_info = select(
        tosa_schema["enums"],
        "name",
        "tosa.Attribute",
    )["values"]

    for i in attribute_info:
        # The library doesn't support custom or none attributes.
        # CustomAttribute and MatMulAttribute are deprecated
        if i["name"] not in ["NONE", "CustomAttribute", "MatMulAttribute"]:
            yield i["name"]


@pytest.mark.parametrize("attribute_name", get_attributes())
def test_single_attr(request, attribute_name):
    """
    Creating an attribute of each type, serializing, deserializing, and
    checking that arguments are preserved.
    """

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"
    testname = request.node.name
    (tmp_dir / testname).mkdir(exist_ok=True)

    flatc = base_dir / "third_party/flatbuffers/flatc"
    tosa_fbs = base_dir / "schema/tosa.fbs"

    with open(tmp_dir / "tosa.json", encoding="utf-8") as f:
        tosa_schema = json.load(f)

    # Defining temp filepaths
    testname = request.node.name
    tosa_file = tmp_dir / testname / f"{testname}.tosa"
    json_file = tmp_dir / testname / f"{testname}.json"

    # Helper function for querying lists of dictionaries for a value
    def select(data: list[dict], key, value) -> dict:
        return next(filter(lambda item: item[key] == value, data), None)

    # Creating example tensors to reference in the operator
    ser = ts.TosaSerializer(tmp_dir / testname)
    ser.currRegion.currBasicBlock.addTensor("t1", [1], ts.DType.INT32)
    ser.currRegion.currBasicBlock.addTensor("t2", [1], ts.DType.INT32)
    ser.currRegion.currBasicBlock.addInput("t1")
    ser.currRegion.currBasicBlock.addOutput("t2")

    # 'py_kwargs' is what we will pass to the Python function to
    # create the attribute, while 'expected' is what we expect
    # to see from the graph serialized as json.
    # So 'py_kwargs' needs to worry about the FIELD_NAME_REPLACEMENTS,
    # but we use the unaltered field names from the schema for 'expected'.
    expected = {}
    py_kwargs = {}

    if attribute_name in [
        "PadAttribute",
        "ClampAttribute",
        "ClampAndNanPropagationAttribute",
    ]:
        py_kwargs["serializer_builder"] = ser.builder

    # Getting the fields of the attribute from the schema
    fields = select(
        tosa_schema["objects"],
        "name",
        f"tosa.{attribute_name}",
    )["fields"]
    for field in fields:
        if field.get("deprecated", False):
            continue

        field_name = field["name"]
        kwarg = FIELD_NAME_REPLACEMENTS.get(
            (attribute_name, field_name),
            field_name,
        )

        # Randomly generating the field based on type

        if (attribute_name, field_name) in ENUM_FIELDS:
            enum_name = ENUM_FIELDS[(attribute_name, field_name)]
            enum = select(
                tosa_schema["enums"],
                "name",
                f"tosa.{enum_name}",
            )["values"]
            choice = random.choice(enum)

            py_kwargs[kwarg] = choice["value"]
            expected[field_name] = choice["name"]
            continue

        field_type = field["type"]

        if field_type["base_type"] == "Vector" and field_type["element"] == "UByte":
            py_kwargs[kwarg] = random.randbytes(random.randint(1, 16))
            # json stores bytes as list[uint8]
            expected[field_name] = list(py_kwargs[kwarg])
        elif field_type["base_type"] == "Vector" and field_type["element"] == "Int":
            expected[field_name] = py_kwargs[kwarg] = random.sample(
                range(-(2**31), 2**31), random.randint(1, 16)
            )
        elif field_type["base_type"] == "Vector" and field_type["element"] == "Short":
            expected[field_name] = py_kwargs[kwarg] = random.sample(
                range(-(2**15), 2**15), random.randint(1, 16)
            )
        elif field_type["base_type"] == "Int":
            expected[field_name] = py_kwargs[kwarg] = random.randint(
                -(2**31), 2**31 - 1
            )
        elif field_type["base_type"] == "Bool":
            expected[field_name] = py_kwargs[kwarg] = random.choice(
                [True, False],
            )
        elif field_type["base_type"] == "String":
            expected[field_name] = py_kwargs[kwarg] = "".join(
                random.choices(
                    string.ascii_uppercase + string.digits,
                    k=random.randint(1, 16),
                )
            )
        else:
            raise NotImplementedError(
                f"{attribute_name}.{field_name} is of an unknown type and random "
                "arguments couldn't be generated for testing. If it uses an enum, "
                f"consider adding to ENUM_FIELDS. {field_type}"
            )

    # Creating the attribute and adding it to the serializer
    attr = ts.TosaSerializerAttribute()

    # This line calls the attribute function,
    # e.g. attr.ConvAttribute(pad=[...], ...)
    getattr(attr, attribute_name)(**py_kwargs)

    ser.currRegion.currBasicBlock.addOperator(
        ts.TosaOp.Op().UNKNOWN, ["t1"], ["t2"], attr
    )
    # TODO: we use Op.UNKNOWN since there's no easy mapping
    # for attribute <-> operator. Op is just a uint so we're
    # not losing much coverage, but this would be useful

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

    # Getting the arguments of the attribute that we serialized
    new_attr = serialized["regions"][0]["blocks"][0]["operators"][0]["attribute"]
    assert expected == new_attr
