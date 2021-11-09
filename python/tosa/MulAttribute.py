# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class MulAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMulAttribute(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MulAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def MulAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # MulAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MulAttribute
    def Shift(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def MulAttributeStart(builder): builder.StartObject(1)
def MulAttributeAddShift(builder, shift): builder.PrependInt32Slot(0, shift, 0)
def MulAttributeEnd(builder): return builder.EndObject()