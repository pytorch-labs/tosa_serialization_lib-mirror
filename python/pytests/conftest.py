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


def pytest_sessionstart():
    """Initializes temporary directory."""

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"

    tmp_dir.mkdir(exist_ok=True)


def pytest_sessionfinish():
    """Cleaning up temporary files."""

    base_dir = (pathlib.Path(__file__).parent / "../..").resolve()
    tmp_dir = base_dir / "python/pytests/tmp"

    shutil.rmtree(tmp_dir)
