# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SLICE_Attribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SLICE_Attribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSLICE_Attribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SLICE_AttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # SLICE_Attribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def SLICE_AttributeStart(builder):
    builder.StartObject(0)

def Start(builder):
    SLICE_AttributeStart(builder)

def SLICE_AttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return SLICE_AttributeEnd(builder)
