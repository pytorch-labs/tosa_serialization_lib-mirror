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

class PadQuantInfo(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsPadQuantInfo(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PadQuantInfo()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def PadQuantInfoBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # PadQuantInfo
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # PadQuantInfo
    def InputZp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def PadQuantInfoStart(builder): builder.StartObject(1)
def PadQuantInfoAddInputZp(builder, inputZp): builder.PrependInt32Slot(0, inputZp, 0)
def PadQuantInfoEnd(builder): return builder.EndObject()