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
    def Padding(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ConvAttribute
    def PaddingAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ConvAttribute
    def PaddingLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ConvAttribute
    def PaddingIsNone(self):
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

def Start(builder): builder.StartObject(3)
def ConvAttributeStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddPadding(builder, padding): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(padding), 0)
def ConvAttributeAddPadding(builder, padding):
    """This method is deprecated. Please switch to AddPadding."""
    return AddPadding(builder, padding)
def StartPaddingVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConvAttributeStartPaddingVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartPaddingVector(builder, numElems)
def AddStride(builder, stride): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(stride), 0)
def ConvAttributeAddStride(builder, stride):
    """This method is deprecated. Please switch to AddStride."""
    return AddStride(builder, stride)
def StartStrideVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConvAttributeStartStrideVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartStrideVector(builder, numElems)
def AddDilation(builder, dilation): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dilation), 0)
def ConvAttributeAddDilation(builder, dilation):
    """This method is deprecated. Please switch to AddDilation."""
    return AddDilation(builder, dilation)
def StartDilationVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ConvAttributeStartDilationVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartDilationVector(builder, numElems)
def End(builder): return builder.EndObject()
def ConvAttributeEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)