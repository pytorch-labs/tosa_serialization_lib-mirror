# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FullyConnectedAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FullyConnectedAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFullyConnectedAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def FullyConnectedAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # FullyConnectedAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def FullyConnectedAttributeStart(builder):
    builder.StartObject(2)

def Start(builder):
    FullyConnectedAttributeStart(builder)

def FullyConnectedAttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return FullyConnectedAttributeEnd(builder)