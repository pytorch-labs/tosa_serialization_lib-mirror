# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class REVERSE_Attribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = REVERSE_Attribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsREVERSE_Attribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def REVERSE_AttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # REVERSE_Attribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # REVERSE_Attribute
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def REVERSE_AttributeStart(builder):
    builder.StartObject(1)

def Start(builder):
    REVERSE_AttributeStart(builder)

def REVERSE_AttributeAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)

def AddAxis(builder, axis):
    REVERSE_AttributeAddAxis(builder, axis)

def REVERSE_AttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return REVERSE_AttributeEnd(builder)
