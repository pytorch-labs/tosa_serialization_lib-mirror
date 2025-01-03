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

import pathlib
import shutil
import subprocess
import pytest


def pytest_sessionstart():
    """Initializes temporary directory."""

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    # Using flatc and flatbuffers' reflection feature to convert tosa.fbs to
    # json for easier reading
    flatbuffers_dir = base_dir / "third_party/flatbuffers"
    flatc = flatbuffers_dir / "flatc"
    reflection_fbs = flatbuffers_dir / "reflection/reflection.fbs"
    tosa_fbs = base_dir / "schema/tosa.fbs"

    # Using flatbuffers reflection to serialize the TOSA flatbuffers schema
    # into binary
    _ = subprocess.run(
        [flatc, "--binary", "--schema", "-o", tmp_dir, tosa_fbs], check=True
    )

    # This file is generated by the previous command
    tosa_bfbs = tmp_dir / "tosa.bfbs"

    # Deserializing the binary into JSON using the reflection schema
    _ = subprocess.run(
        [
            flatc,
            "--json",
            "--strict-json",
            "-o",
            tmp_dir,
            reflection_fbs,
            "--",
            tosa_bfbs,
        ],
        check=True,
    )


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--leave-tmp", dest="leave_tmp", action="store_true")


def pytest_sessionfinish(session: pytest.Session):
    """Cleaning up temporary files, unless the --leave-tmp flag is set"""

    if not session.config.option.leave_tmp:
        base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
        tmp_dir = base_dir / "python/pytests/tmp"
        shutil.rmtree(tmp_dir)
