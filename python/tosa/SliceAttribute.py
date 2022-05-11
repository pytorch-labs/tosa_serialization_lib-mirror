# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SliceAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SliceAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSliceAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SliceAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # SliceAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SliceAttribute
    def Begin(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SliceAttribute
    def BeginAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SliceAttribute
    def BeginLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SliceAttribute
    def BeginIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SliceAttribute
    def Size(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SliceAttribute
    def SizeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SliceAttribute
    def SizeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SliceAttribute
    def SizeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def Start(builder): builder.StartObject(2)
def SliceAttributeStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddBegin(builder, begin): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(begin), 0)
def SliceAttributeAddBegin(builder, begin):
    """This method is deprecated. Please switch to AddBegin."""
    return AddBegin(builder, begin)
def StartBeginVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SliceAttributeStartBeginVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartBeginVector(builder, numElems)
def AddSize(builder, size): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(size), 0)
def SliceAttributeAddSize(builder, size):
    """This method is deprecated. Please switch to AddSize."""
    return AddSize(builder, size)
def StartSizeVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SliceAttributeStartSizeVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartSizeVector(builder, numElems)
def End(builder): return builder.EndObject()
def SliceAttributeEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)