# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class REDUCE_PRODUCT_Attribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = REDUCE_PRODUCT_Attribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsREDUCE_PRODUCT_Attribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def REDUCE_PRODUCT_AttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # REDUCE_PRODUCT_Attribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # REDUCE_PRODUCT_Attribute
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def REDUCE_PRODUCT_AttributeStart(builder):
    builder.StartObject(1)

def Start(builder):
    REDUCE_PRODUCT_AttributeStart(builder)

def REDUCE_PRODUCT_AttributeAddAxis(builder, axis):
    builder.PrependInt32Slot(0, axis, 0)

def AddAxis(builder, axis):
    REDUCE_PRODUCT_AttributeAddAxis(builder, axis)

def REDUCE_PRODUCT_AttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return REDUCE_PRODUCT_AttributeEnd(builder)
