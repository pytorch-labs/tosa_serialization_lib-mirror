# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReduceSumAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReduceSumAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReduceSumAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ReduceSumAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # ReduceSumAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReduceSumAttribute
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def ReduceSumAttributeStart(builder):
    builder.StartObject(1)

def Start(builder):
    ReduceSumAttributeStart(builder)

def ReduceSumAttributeAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)

def AddAxis(builder, axis):
    ReduceSumAttributeAddAxis(builder, axis)

def ReduceSumAttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return ReduceSumAttributeEnd(builder)
