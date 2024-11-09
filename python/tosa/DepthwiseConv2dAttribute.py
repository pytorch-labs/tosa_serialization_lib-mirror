# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DepthwiseConv2dAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DepthwiseConv2dAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDepthwiseConv2dAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DepthwiseConv2dAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # DepthwiseConv2dAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DepthwiseConv2dAttribute
    def Pad(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # DepthwiseConv2dAttribute
    def PadAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # DepthwiseConv2dAttribute
    def PadLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DepthwiseConv2dAttribute
    def PadIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # DepthwiseConv2dAttribute
    def Stride(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # DepthwiseConv2dAttribute
    def StrideAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # DepthwiseConv2dAttribute
    def StrideLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DepthwiseConv2dAttribute
    def StrideIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # DepthwiseConv2dAttribute
    def Dilation(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # DepthwiseConv2dAttribute
    def DilationAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # DepthwiseConv2dAttribute
    def DilationLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DepthwiseConv2dAttribute
    def DilationIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # DepthwiseConv2dAttribute
    def LocalBound(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # DepthwiseConv2dAttribute
    def AccType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def DepthwiseConv2dAttributeStart(builder):
    builder.StartObject(5)

def Start(builder):
    DepthwiseConv2dAttributeStart(builder)

def DepthwiseConv2dAttributeAddPad(builder, pad):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(pad), 0)

def AddPad(builder, pad):
    DepthwiseConv2dAttributeAddPad(builder, pad)

def DepthwiseConv2dAttributeStartPadVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartPadVector(builder, numElems):
    return DepthwiseConv2dAttributeStartPadVector(builder, numElems)

def DepthwiseConv2dAttributeAddStride(builder, stride):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(stride), 0)

def AddStride(builder, stride):
    DepthwiseConv2dAttributeAddStride(builder, stride)

def DepthwiseConv2dAttributeStartStrideVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStrideVector(builder, numElems):
    return DepthwiseConv2dAttributeStartStrideVector(builder, numElems)

def DepthwiseConv2dAttributeAddDilation(builder, dilation):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dilation), 0)

def AddDilation(builder, dilation):
    DepthwiseConv2dAttributeAddDilation(builder, dilation)

def DepthwiseConv2dAttributeStartDilationVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartDilationVector(builder, numElems):
    return DepthwiseConv2dAttributeStartDilationVector(builder, numElems)

def DepthwiseConv2dAttributeAddLocalBound(builder, localBound):
    builder.PrependBoolSlot(3, localBound, 0)

def AddLocalBound(builder, localBound):
    DepthwiseConv2dAttributeAddLocalBound(builder, localBound)

def DepthwiseConv2dAttributeAddAccType(builder, accType):
    builder.PrependUint32Slot(4, accType, 0)

def AddAccType(builder, accType):
    DepthwiseConv2dAttributeAddAccType(builder, accType)

def DepthwiseConv2dAttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return DepthwiseConv2dAttributeEnd(builder)
