# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tosa

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class WhileLoopAttribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WhileLoopAttribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsWhileLoopAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def WhileLoopAttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x4F\x53\x41", size_prefixed=size_prefixed)

    # WhileLoopAttribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # WhileLoopAttribute
    def CondBranch(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # WhileLoopAttribute
    def BodyBranch(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def Start(builder): builder.StartObject(2)
def WhileLoopAttributeStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddCondBranch(builder, condBranch): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(condBranch), 0)
def WhileLoopAttributeAddCondBranch(builder, condBranch):
    """This method is deprecated. Please switch to AddCondBranch."""
    return AddCondBranch(builder, condBranch)
def AddBodyBranch(builder, bodyBranch): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(bodyBranch), 0)
def WhileLoopAttributeAddBodyBranch(builder, bodyBranch):
    """This method is deprecated. Please switch to AddBodyBranch."""
    return AddBodyBranch(builder, bodyBranch)
def End(builder): return builder.EndObject()
def WhileLoopAttributeEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)