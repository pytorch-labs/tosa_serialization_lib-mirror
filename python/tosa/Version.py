# automatically generated by the FlatBuffers compiler, do not modify

# Copyright (c) 2020-2021, ARM Limited.
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


# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Version(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsVersion(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Version()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def VersionBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # Version
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Version
    def _major(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Version
    def _minor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 22

    # Version
    def _patch(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Version
    def _experimental(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def VersionStart(builder): builder.StartObject(4)
def VersionAdd_major(builder, Major): builder.PrependInt32Slot(0, Major, 0)
def VersionAdd_minor(builder, Minor): builder.PrependInt32Slot(1, Minor, 22)
def VersionAdd_patch(builder, Patch): builder.PrependInt32Slot(2, Patch, 0)
def VersionAdd_experimental(builder, Experimental): builder.PrependBoolSlot(3, Experimental, 0)
def VersionEnd(builder): return builder.EndObject()
