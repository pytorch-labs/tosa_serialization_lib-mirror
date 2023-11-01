# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ConvAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ConvAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsConvAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ConvAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # ConvAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ConvAttribute
    def Pad(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConvAttribute
    def PadAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConvAttribute
    def PadLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConvAttribute
    def PadIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # ConvAttribute
    def Stride(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConvAttribute
    def StrideAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConvAttribute
    def StrideLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConvAttribute
    def StrideIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # ConvAttribute
    def Dilation(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConvAttribute
    def DilationAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConvAttribute
    def DilationLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConvAttribute
    def DilationIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # ConvAttribute
    def InputZp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConvAttribute
    def WeightZp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ConvAttribute
    def LocalBound(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def ConvAttributeStart(builder):
    builder.StartObject(6)

def Start(builder):
    ConvAttributeStart(builder)

def ConvAttributeAddPad(builder, pad):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(pad), 0)

def AddPad(builder, pad):
    ConvAttributeAddPad(builder, pad)

def ConvAttributeStartPadVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartPadVector(builder, numElems: int) -> int:
    return ConvAttributeStartPadVector(builder, numElems)

def ConvAttributeAddStride(builder, stride):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(stride), 0)

def AddStride(builder, stride):
    ConvAttributeAddStride(builder, stride)

def ConvAttributeStartStrideVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStrideVector(builder, numElems: int) -> int:
    return ConvAttributeStartStrideVector(builder, numElems)

def ConvAttributeAddDilation(builder, dilation):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dilation), 0)

def AddDilation(builder, dilation):
    ConvAttributeAddDilation(builder, dilation)

def ConvAttributeStartDilationVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartDilationVector(builder, numElems: int) -> int:
    return ConvAttributeStartDilationVector(builder, numElems)

def ConvAttributeAddInputZp(builder, inputZp):
    builder.PrependInt32Slot(3, inputZp, 0)

def AddInputZp(builder, inputZp):
    ConvAttributeAddInputZp(builder, inputZp)

def ConvAttributeAddWeightZp(builder, weightZp):
    builder.PrependInt32Slot(4, weightZp, 0)

def AddWeightZp(builder, weightZp):
    ConvAttributeAddWeightZp(builder, weightZp)

def ConvAttributeAddLocalBound(builder, localBound):
    builder.PrependBoolSlot(5, localBound, 0)

def AddLocalBound(builder, localBound):
    ConvAttributeAddLocalBound(builder, localBound)

def ConvAttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return ConvAttributeEnd(builder)
