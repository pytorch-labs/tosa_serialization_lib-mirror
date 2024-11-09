# Copyright (c) 2020-2025, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import serializer.tosa_serializer as ts
import json
import flatbuffers
import numpy as np
from ml_dtypes import bfloat16, float8_e4m3fn
from serializer.numpy_utils import save_npy
from enum import IntEnum, unique
from tosa import (
    TosaGraph,
    TosaRegion,
    TosaBasicBlock,
    TosaTensor,
    TosaOperator,
    Version,
)
import tosa.DType as TosaDType
import tosa.Op as TosaOp
from tosa.Op import Op

# Keep version number in sync with the version default value with schema/tosa.fbs
TOSA_VERSION_MAJOR = 1
TOSA_VERSION_MINOR = 1
TOSA_VERSION_PATCH = 0
TOSA_VERSION_DRAFT = True
TOSA_VERSION = [
    TOSA_VERSION_MAJOR,
    TOSA_VERSION_MINOR,
    TOSA_VERSION_PATCH,
    TOSA_VERSION_DRAFT,
]

# File identifier needs to be kept in sync with schema/tosa.fbs
TOSA_GRAPH_IDENTIFIER = b"\x54\x4F\x53\x41"

# With the way flatc generates its python types, there is no programatic way
# to get string names for the integer types.  Manually maintain a string table
# here.
DType = TosaDType.DType()
DTypeNames = [
    "UNKNOWN",
    "BOOL",
    "UINT8",
    "INT4",
    "INT8",
    "INT16",
    "INT32",
    "INT48",
    "FP32",
    "UINT16",
    "FP16",
    "BF16",
    "SHAPE",
    "FP8E4M3",
    "FP8E5M2",
]

ByteMask = np.uint64(0xFF)


def dtype_str_to_val(name):
    for i in range(len(DTypeNames)):
        if name.casefold() == DTypeNames[i].casefold():
            return i
    raise Exception("Unable to parse DType name {}".format(name))


class TosaSerializerUnion:
    """This class handles encapsulating and serializing union types into flatbuffers"""

    def __init__(self):
        # A tuple of the start and end functions.
        # Set by the options constructors below
        self.optFcns = None

        # The type from the tosa.Options enumeration.
        # Set by the options constructors below.
        self.utype = None

        # Each of these lists is a tuple of the add function and the
        # value being added.  Set by the options constructors below.
        self.ints = []
        self.bools = []
        self.floats = []
        self.strings = []
        self.int16vecs = []
        self.intvecs = []
        self.fpvecs = []

    def serialize(self, builder):
        # We have to build strings and vectors first
        strList = []
        intVecList = []
        fpVecList = []

        for fcn, val in self.strings:
            strList.append((fcn, builder.CreateString(val)))

        for fcn, val in self.intvecs:
            intVecList.append((fcn, TosaSerializer.serializeInt32Vec(builder, val)))

        for fcn, val in self.int16vecs:
            intVecList.append((fcn, TosaSerializer.serializeInt16Vec(builder, val)))

        for fcn, val in self.fpvecs:
            fpVecList.append((fcn, TosaSerializer.serializeFpVec(builder, val)))

        startFcn, endFcn = self.optFcns

        # Then serialize the options object from the list of primitives and
        # other serialized values
        startFcn(builder)
        for fcn, val in self.ints:
            fcn(builder, val)

        for fcn, val in self.bools:
            fcn(builder, val)

        for fcn, val in self.floats:
            fcn(builder, val)

        for fcn, val in strList:
            fcn(builder, val)

        for fcn, val in intVecList:
            fcn(builder, val)

        for fcn, val in fpVecList:
            fcn(builder, val)

        return endFcn(builder)


class TosaSerializerAttribute(TosaSerializerUnion):
    """This class handles encapsulating all of the enumerated types for attributes"""

    def __init__(self):
        super().__init__()

    def ArgMaxAttribute(self, axis, nan_mode):
        from tosa import ArgMaxAttribute as a, Attribute

        self.utype = Attribute.Attribute().ArgMaxAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))
        self.ints.append((a.AddNanMode, nan_mode))

    def AvgPool2dAttribute(
        self,
        kernel,
        stride,
        pad,
        input_zp,
        output_zp,
        acc_type,
    ):
        from tosa import AvgPool2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().AvgPool2dAttribute

        self.optFcns = (a.Start, a.End)
        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddKernel, kernel))
        self.intvecs.append((a.AddStride, stride))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddOutputZp, output_zp))
        self.ints.append((a.AddAccType, acc_type))

    def Conv2dAttribute(self, pad, stride, dilation, local_bound, acc_type):
        from tosa import Conv2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().Conv2dAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def Conv3dAttribute(self, pad, stride, dilation, local_bound, acc_type):
        from tosa import Conv3dAttribute as a, Attribute

        self.utype = Attribute.Attribute().Conv3dAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def DepthwiseConv2dAttribute(self, pad, stride, dilation, local_bound, acc_type):
        from tosa import DepthwiseConv2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().DepthwiseConv2dAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def FFT2dAttribute(self, inverse, local_bound):
        from tosa import FFT2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().FFT2dAttribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddInverse, inverse))
        self.bools.append((a.AddLocalBound, local_bound))

    def MatMulAttribute(self):
        from tosa import MatMulAttribute as a, Attribute

        self.utype = Attribute.Attribute().MatMulAttribute
        self.optFcns = (a.Start, a.End)

    def MaxPool2dAttribute(self, kernel, stride, pad, nan_mode):
        from tosa import MaxPool2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().MaxPool2dAttribute

        self.optFcns = (a.Start, a.End)
        self.intvecs.append((a.AddKernel, kernel))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddPad, pad))
        self.ints.append((a.AddNanMode, nan_mode))

    def RFFT2dAttribute(self, local_bound):
        from tosa import RFFT2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().RFFT2dAttribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddLocalBound, local_bound))

    def TransposeConv2dAttribute(self, out_pad, stride, local_bound, acc_type):
        from tosa import TransposeConv2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().TransposeConv2dAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddOutPad, out_pad))
        self.intvecs.append((a.AddStride, stride))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def ClampAttribute(
        self, serializer_builder, min_val_as_bytes, max_val_as_bytes, nan_mode
    ):
        from tosa import ClampAttribute as a, Attribute

        self.utype = Attribute.Attribute().ClampAttribute
        self.optFcns = (a.Start, a.End)

        # min/max float attributes serialized as uint8 vectors
        serialized_min_val = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, min_val_as_bytes
        )
        serialized_max_val = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, max_val_as_bytes
        )

        self.floats.append((a.AddMinVal, serialized_min_val))
        self.floats.append((a.AddMaxVal, serialized_max_val))
        self.ints.append((a.AddNanMode, nan_mode))

    def ErfAttribute(self):
        from tosa import ErfAttribute as a, Attribute

        self.utype = Attribute.Attribute().ErfAttribute
        self.optFcns = (a.Start, a.End)

    def SigmoidAttribute(self):
        from tosa import SigmoidAttribute as a, Attribute

        self.utype = Attribute.Attribute().SigmoidAttribute
        self.optFcns = (a.Start, a.End)

    def TanhAttribute(self):
        from tosa import TanhAttribute as a, Attribute

        self.utype = Attribute.Attribute().TanhAttribute
        self.optFcns = (a.Start, a.End)

    def AddAttribute(self):
        from tosa import AddAttribute as a, Attribute

        self.utype = Attribute.Attribute().AddAttribute
        self.optFcns = (a.Start, a.End)

    def ArithmeticRightShiftAttribute(self, round):
        from tosa import ArithmeticRightShiftAttribute as a, Attribute

        self.utype = Attribute.Attribute().ArithmeticRightShiftAttribute
        self.optFcns = (
            a.Start,
            a.End,
        )

        self.bools.append((a.AddRound, round))

    def BitwiseAndAttribute(self):
        from tosa import BitwiseAndAttribute as a, Attribute

        self.utype = Attribute.Attribute().BitwiseAndAttribute
        self.optFcns = (a.Start, a.End)

    def BitwiseOrAttribute(self):
        from tosa import BitwiseOrAttribute as a, Attribute

        self.utype = Attribute.Attribute().BitwiseOrAttribute
        self.optFcns = (a.Start, a.End)

    def BitwiseXorAttribute(self):
        from tosa import BitwiseXorAttribute as a, Attribute

        self.utype = Attribute.Attribute().BitwiseXorAttribute
        self.optFcns = (a.Start, a.End)

    def IntDivAttribute(self):
        from tosa import IntDivAttribute as a, Attribute

        self.utype = Attribute.Attribute().IntDivAttribute
        self.optFcns = (a.Start, a.End)

    def LogicalAndAttribute(self):
        from tosa import LogicalAndAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogicalAndAttribute
        self.optFcns = (a.Start, a.End)

    def LogicalLeftShiftAttribute(self):
        from tosa import LogicalLeftShiftAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogicalLeftShiftAttribute
        self.optFcns = (a.Start, a.End)

    def LogicalRightShiftAttribute(self):
        from tosa import LogicalRightShiftAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogicalRightShiftAttribute
        self.optFcns = (a.Start, a.End)

    def LogicalOrAttribute(self):
        from tosa import LogicalOrAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogicalOrAttribute
        self.optFcns = (a.Start, a.End)

    def LogicalXorAttribute(self):
        from tosa import LogicalXorAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogicalXorAttribute
        self.optFcns = (a.Start, a.End)

    def MaximumAttribute(self, nan_mode):
        from tosa import MaximumAttribute as a, Attribute

        self.utype = Attribute.Attribute().MaximumAttribute
        self.optFcns = (a.Start, a.End)
        self.ints.append((a.AddNanMode, nan_mode))

    def MinimumAttribute(self, nan_mode):
        from tosa import MinimumAttribute as a, Attribute

        self.utype = Attribute.Attribute().MinimumAttribute
        self.optFcns = (a.Start, a.End)
        self.ints.append((a.AddNanMode, nan_mode))

    def MulAttribute(self):
        from tosa import MulAttribute as a, Attribute

        self.utype = Attribute.Attribute().MulAttribute
        self.optFcns = (a.Start, a.End)

    def PowAttribute(self):
        from tosa import PowAttribute as a, Attribute

        self.utype = Attribute.Attribute().PowAttribute
        self.optFcns = (a.Start, a.End)

    def SubAttribute(self):
        from tosa import SubAttribute as a, Attribute

        self.utype = Attribute.Attribute().SubAttribute
        self.optFcns = (a.Start, a.End)

    def TableAttribute(self):
        from tosa import TableAttribute as a, Attribute

        self.utype = Attribute.Attribute().TableAttribute
        self.optFcns = (a.Start, a.End)

    def AbsAttribute(self):
        from tosa import AbsAttribute as a, Attribute

        self.utype = Attribute.Attribute().AbsAttribute
        self.optFcns = (a.Start, a.End)

    def BitwiseNotAttribute(self):
        from tosa import BitwiseNotAttribute as a, Attribute

        self.utype = Attribute.Attribute().BitwiseNotAttribute
        self.optFcns = (a.Start, a.End)

    def CeilAttribute(self):
        from tosa import CeilAttribute as a, Attribute

        self.utype = Attribute.Attribute().CeilAttribute
        self.optFcns = (a.Start, a.End)

    def ClzAttribute(self):
        from tosa import ClzAttribute as a, Attribute

        self.utype = Attribute.Attribute().ClzAttribute
        self.optFcns = (a.Start, a.End)

    def CosAttribute(self):
        from tosa import CosAttribute as a, Attribute

        self.utype = Attribute.Attribute().CosAttribute
        self.optFcns = (a.Start, a.End)

    def ExpAttribute(self):
        from tosa import ExpAttribute as a, Attribute

        self.utype = Attribute.Attribute().ExpAttribute
        self.optFcns = (a.Start, a.End)

    def FloorAttribute(self):
        from tosa import FloorAttribute as a, Attribute

        self.utype = Attribute.Attribute().FloorAttribute
        self.optFcns = (a.Start, a.End)

    def LogAttribute(self):
        from tosa import LogAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogAttribute
        self.optFcns = (a.Start, a.End)

    def LogicalNotAttribute(self):
        from tosa import LogicalNotAttribute as a, Attribute

        self.utype = Attribute.Attribute().LogicalNotAttribute
        self.optFcns = (a.Start, a.End)

    def NegateAttribute(self, input1_zp, output_zp):
        from tosa import NegateAttribute as a, Attribute

        self.utype = Attribute.Attribute().NegateAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInput1Zp, input1_zp))
        self.ints.append((a.AddOutputZp, output_zp))

    def ReciprocalAttribute(self):
        from tosa import ReciprocalAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReciprocalAttribute
        self.optFcns = (a.Start, a.End)

    def RsqrtAttribute(self):
        from tosa import RsqrtAttribute as a, Attribute

        self.utype = Attribute.Attribute().RsqrtAttribute
        self.optFcns = (a.Start, a.End)

    def SinAttribute(self):
        from tosa import SinAttribute as a, Attribute

        self.utype = Attribute.Attribute().SinAttribute
        self.optFcns = (a.Start, a.End)

    def SelectAttribute(self):
        from tosa import SelectAttribute as a, Attribute

        self.utype = Attribute.Attribute().SelectAttribute
        self.optFcns = (a.Start, a.End)

    def EqualAttribute(self):
        from tosa import EqualAttribute as a, Attribute

        self.utype = Attribute.Attribute().EqualAttribute
        self.optFcns = (a.Start, a.End)

    def GreaterAttribute(self):
        from tosa import GreaterAttribute as a, Attribute

        self.utype = Attribute.Attribute().GreaterAttribute
        self.optFcns = (a.Start, a.End)

    def GreaterEqualAttribute(self):
        from tosa import GreaterEqualAttribute as a, Attribute

        self.utype = Attribute.Attribute().GreaterEqualAttribute
        self.optFcns = (a.Start, a.End)

    def ReduceAllAttribute(self, axis):
        from tosa import ReduceAllAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReduceAllAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReduceAnyAttribute(self, axis):
        from tosa import ReduceAnyAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReduceAnyAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReduceMaxAttribute(self, axis, nan_mode):
        from tosa import ReduceMaxAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReduceMaxAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))
        self.ints.append((a.AddNanMode, nan_mode))

    def ReduceMinAttribute(self, axis, nan_mode):
        from tosa import ReduceMinAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReduceMinAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))
        self.ints.append((a.AddNanMode, nan_mode))

    def ReduceProductAttribute(self, axis):
        from tosa import ReduceProductAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReduceProductAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReduceSumAttribute(self, axis):
        from tosa import ReduceSumAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReduceSumAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ConcatAttribute(self, axis):
        from tosa import ConcatAttribute as a, Attribute

        self.utype = Attribute.Attribute().ConcatAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def PadAttribute(self, serializer_builder, pad_const_val_as_bytes):
        from tosa import PadAttribute as a, Attribute

        self.utype = Attribute.Attribute().PadAttribute
        self.optFcns = (a.Start, a.End)

        # serialize pad_const_val_as_bytes as uint8 vector
        serialized_pad_const_val = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, pad_const_val_as_bytes
        )

        self.floats.append((a.AddPadConst, serialized_pad_const_val))

    def ReshapeAttribute(self):
        from tosa import ReshapeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReshapeAttribute
        self.optFcns = (a.Start, a.End)

    def ReverseAttribute(self, axis):
        from tosa import ReverseAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReverseAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def SliceAttribute(self):
        from tosa import SliceAttribute as a, Attribute

        self.utype = Attribute.Attribute().SliceAttribute
        self.optFcns = (a.Start, a.End)

    def TileAttribute(self):
        from tosa import TileAttribute as a, Attribute

        self.utype = Attribute.Attribute().TileAttribute
        self.optFcns = (a.Start, a.End)

    def TransposeAttribute(self, perms):
        from tosa import TransposeAttribute as a, Attribute

        self.utype = Attribute.Attribute().TransposeAttribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPerms, perms))

    def GatherAttribute(self):
        from tosa import GatherAttribute as a, Attribute

        self.utype = Attribute.Attribute().GatherAttribute
        self.optFcns = (a.Start, a.End)

    def ScatterAttribute(self):
        from tosa import ScatterAttribute as a, Attribute

        self.utype = Attribute.Attribute().ScatterAttribute
        self.optFcns = (a.Start, a.End)

    def ResizeAttribute(self, mode):
        from tosa import ResizeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ResizeAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddMode, mode))

    def CastAttribute(self):
        from tosa import CastAttribute as a, Attribute

        self.utype = Attribute.Attribute().CastAttribute
        self.optFcns = (a.Start, a.End)

    def RescaleAttribute(
        self,
        input_zp,
        output_zp,
        scale32,
        double_round,
        per_channel,
        input_unsigned,
        output_unsigned,
    ):
        from tosa import RescaleAttribute as a, Attribute

        self.utype = Attribute.Attribute().RescaleAttribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddOutputZp, output_zp))
        self.bools.append((a.AddScale32, scale32))
        self.bools.append((a.AddDoubleRound, double_round))
        self.bools.append((a.AddPerChannel, per_channel))
        self.bools.append((a.AddInputUnsigned, input_unsigned))
        self.bools.append((a.AddOutputUnsigned, output_unsigned))

    def ConstAttribute(self):
        from tosa import ConstAttribute as a, Attribute

        self.utype = Attribute.Attribute().ConstAttribute
        self.optFcns = (a.Start, a.End)

    def IdentityAttribute(self):
        from tosa import IdentityAttribute as a, Attribute

        self.utype = Attribute.Attribute().IdentityAttribute
        self.optFcns = (a.Start, a.End)

    def CustomAttribute(
        self,
        serializer_builder,
        operator_name,
        domain_name,
        implementation_attrs_as_bytes,
    ):
        from tosa import CustomAttribute as a, Attribute

        self.utype = Attribute.Attribute().CustomAttribute
        self.optFcns = (a.Start, a.End)

        implementation_attrs = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, implementation_attrs_as_bytes
        )

        self.strings.append((a.AddOperatorName, operator_name))
        self.strings.append((a.AddDomainName, domain_name))
        self.floats.append((a.AddImplementationAttrs, implementation_attrs))

    def CondIfAttribute(self, then_graph, else_graph):
        from tosa import CondIfAttribute as a, Attribute

        self.utype = Attribute.Attribute().CondIfAttribute
        self.optFcns = (a.Start, a.End)

        self.strings.append((a.AddThenGraph, then_graph))
        self.strings.append((a.AddElseGraph, else_graph))

    def WhileLoopAttribute(self, cond_graph, body_graph):
        from tosa import WhileLoopAttribute as a, Attribute

        self.utype = Attribute.Attribute().WhileLoopAttribute
        self.optFcns = (a.Start, a.End)

        self.strings.append((a.AddCondGraph, cond_graph))
        self.strings.append((a.AddBodyGraph, body_graph))

    def YieldAttribute(self):
        from tosa import YieldAttribute as a, Attribute

        self.utype = Attribute.Attribute().YieldAttribute
        self.optFcns = (a.Start, a.End)

    def VariableAttribute(self):
        from tosa import VariableAttribute as a, Attribute

        self.utype = Attribute.Attribute().VariableAttribute
        self.optFcns = (a.Start, a.End)

    def VariableReadAttribute(self):
        from tosa import VariableReadAttribute as a, Attribute

        self.utype = Attribute.Attribute().VariableReadAttribute
        self.optFcns = (a.Start, a.End)

    def VariableWriteAttribute(self):
        from tosa import VariableWriteAttribute as a, Attribute

        self.utype = Attribute.Attribute().VariableWriteAttribute
        self.optFcns = (a.Start, a.End)

    def ConstShapeAttribute(self):
        from tosa import ConstShapeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ConstShapeAttribute
        self.optFcns = (a.Start, a.End)

    def setAttribute(self, op: Op, *args):
        ATTRIBUTE_MAP = {
            Op.ARGMAX: self.ArgMaxAttribute,
            Op.AVG_POOL2D: self.AvgPool2dAttribute,
            Op.CONV2D: self.Conv2dAttribute,
            Op.CONV3D: self.Conv3dAttribute,
            Op.DEPTHWISE_CONV2D: self.DepthwiseConv2dAttribute,
            Op.FFT2D: self.FFT2dAttribute,
            Op.MATMUL: self.MatMulAttribute,
            Op.MAX_POOL2D: self.MaxPool2dAttribute,
            Op.RFFT2D: self.RFFT2dAttribute,
            Op.TRANSPOSE_CONV2D: self.TransposeConv2dAttribute,
            Op.CLAMP: self.ClampAttribute,
            Op.ERF: self.ErfAttribute,
            Op.SIGMOID: self.SigmoidAttribute,
            Op.TANH: self.TanhAttribute,
            Op.ADD: self.AddAttribute,
            Op.ARITHMETIC_RIGHT_SHIFT: self.ArithmeticRightShiftAttribute,
            Op.BITWISE_AND: self.BitwiseAndAttribute,
            Op.BITWISE_OR: self.BitwiseOrAttribute,
            Op.BITWISE_XOR: self.BitwiseXorAttribute,
            Op.INTDIV: self.IntDivAttribute,
            Op.LOGICAL_AND: self.LogicalAndAttribute,
            Op.LOGICAL_LEFT_SHIFT: self.LogicalLeftShiftAttribute,
            Op.LOGICAL_RIGHT_SHIFT: self.LogicalRightShiftAttribute,
            Op.LOGICAL_OR: self.LogicalOrAttribute,
            Op.LOGICAL_XOR: self.LogicalXorAttribute,
            Op.MAXIMUM: self.MaximumAttribute,
            Op.MINIMUM: self.MinimumAttribute,
            Op.MUL: self.MulAttribute,
            Op.POW: self.PowAttribute,
            Op.SUB: self.SubAttribute,
            Op.TABLE: self.TableAttribute,
            Op.ABS: self.AbsAttribute,
            Op.BITWISE_NOT: self.BitwiseNotAttribute,
            Op.CEIL: self.CeilAttribute,
            Op.CLZ: self.ClzAttribute,
            Op.COS: self.CosAttribute,
            Op.EXP: self.ExpAttribute,
            Op.FLOOR: self.FloorAttribute,
            Op.LOG: self.LogAttribute,
            Op.LOGICAL_NOT: self.LogicalNotAttribute,
            Op.NEGATE: self.NegateAttribute,
            Op.RECIPROCAL: self.ReciprocalAttribute,
            Op.RSQRT: self.RsqrtAttribute,
            Op.SIN: self.SinAttribute,
            Op.SELECT: self.SelectAttribute,
            Op.EQUAL: self.EqualAttribute,
            Op.GREATER: self.GreaterAttribute,
            Op.GREATER_EQUAL: self.GreaterEqualAttribute,
            Op.REDUCE_ALL: self.ReduceAllAttribute,
            Op.REDUCE_ANY: self.ReduceAnyAttribute,
            Op.REDUCE_MAX: self.ReduceMaxAttribute,
            Op.REDUCE_MIN: self.ReduceMinAttribute,
            Op.REDUCE_PRODUCT: self.ReduceProductAttribute,
            Op.REDUCE_SUM: self.ReduceSumAttribute,
            Op.CONCAT: self.ConcatAttribute,
            Op.PAD: self.PadAttribute,
            Op.RESHAPE: self.ReshapeAttribute,
            Op.REVERSE: self.ReverseAttribute,
            Op.SLICE: self.SliceAttribute,
            Op.TILE: self.TileAttribute,
            Op.TRANSPOSE: self.TransposeAttribute,
            Op.GATHER: self.GatherAttribute,
            Op.SCATTER: self.ScatterAttribute,
            Op.RESIZE: self.ResizeAttribute,
            Op.CAST: self.CastAttribute,
            Op.RESCALE: self.RescaleAttribute,
            Op.CONST: self.ConstAttribute,
            Op.IDENTITY: self.IdentityAttribute,
            Op.CUSTOM: self.CustomAttribute,
            Op.COND_IF: self.CondIfAttribute,
            Op.WHILE_LOOP: self.WhileLoopAttribute,
            Op.YIELD: self.YieldAttribute,
            Op.VARIABLE: self.VariableAttribute,
            Op.VARIABLE_WRITE: self.VariableWriteAttribute,
            Op.VARIABLE_READ: self.VariableReadAttribute,
            Op.CONST_SHAPE: self.ConstShapeAttribute,
        }
        ATTRIBUTE_MAP[op](*args)


class TosaSerializerTensor:
    def __init__(
        self,
        name,
        shape,
        dtype,
        data=None,
        placeholderFilename=None,
    ):
        self.name = name

        if isinstance(shape, np.ndarray):
            shape = shape.astype(int).tolist()
        shape = list(map(int, shape))

        self.shape = shape
        self.dtype = dtype

        if dtype == DType.FP32:
            fntype = np.float32
        elif dtype == DType.BF16:
            fntype = bfloat16
        elif dtype == DType.FP8E4M3:
            fntype = float8_e4m3fn
        elif dtype == DType.FP8E5M2:
            # We should receive FP8E5M2 arrays as uint8 to mitigate a bug
            # in ml_dtypes serialisation.
            fntype = np.uint8
        elif dtype == DType.FP16:
            fntype = np.float16
        else:
            fntype = int

        if isinstance(data, np.ndarray):
            data = data.flatten().astype(fntype).tolist()
            data = list(map(fntype, data))
        elif isinstance(data, list):
            data = list(map(fntype, data))
        elif data is not None:
            # Assume data is rank 0 data type
            data = list(map(fntype, [data]))
        else:
            data = None

        self.data = data

        # Filename for placeholder tensors.  These get generated by the test generation
        # process and are written to disk, but are considered input tensors by the
        # network so they do not appear in the TOSA serialiazation.  However, if we
        # want to form a unit test around these input tensors, we can get the filename
        # from here.
        self.placeholderFilename = placeholderFilename

    def __str__(self):
        concatString = "TosaSerializerTensor name: {} shape: {} dtype: {}".format(
            self.name,
            self.shape,
            DTypeNames[self.dtype],
        )
        return concatString

    def setDtype(self, dtype):
        self.dtype = dtype

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fb_shapes = TosaSerializer.serializeInt32Vec(builder, self.shape)
        if self.data:
            u8_data = TosaSerializer.convertDataToUint8Vec(self.dtype, self.data)
            fb_data = TosaSerializer.serializeUint8Vec(builder, u8_data)

        TosaTensor.Start(builder)
        TosaTensor.AddName(builder, fb_name)
        TosaTensor.AddShape(builder, fb_shapes)
        TosaTensor.AddType(builder, self.dtype)
        if self.data:
            TosaTensor.AddData(builder, fb_data)

        return TosaTensor.End(builder)


class TosaSerializerOperator:
    def __init__(self, op, inputs, outputs, attributes=None):
        if attributes is None:
            attributes = TosaSerializerAttribute()
            attributes.setAttribute(op)
        self.op = op
        self.attributes = attributes
        self.inputs = TosaSerializer.toList(inputs)
        self.outputs = TosaSerializer.toList(outputs)

    def __str__(self):
        concatString = "Op {}\n----\n".format(self.op)

        for i in self.inputs:
            concatString = concatString + "  Input:  {}\n".format(i)
        for o in self.outputs:
            concatString = concatString + "  Output: {}\n".format(o)

        return concatString

    def serialize(self, builder):
        fb_inputs = TosaSerializer.serializeStrVec(
            builder, self.inputs, TosaOperator.StartInputsVector
        )
        fb_outputs = TosaSerializer.serializeStrVec(
            builder, self.outputs, TosaOperator.StartOutputsVector
        )
        # Need to serialize attributes enums still
        if self.attributes is not None:
            fb_attributes = self.attributes.serialize(builder)

        TosaOperator.Start(builder)
        TosaOperator.AddOp(builder, self.op)
        TosaOperator.AddInputs(builder, fb_inputs)
        TosaOperator.AddOutputs(builder, fb_outputs)
        if self.attributes is not None:
            TosaOperator.AddAttributeType(builder, self.attributes.utype)
            TosaOperator.AddAttribute(builder, fb_attributes)

        return TosaOperator.End(builder)


class TosaSerializerBasicBlock:
    def __init__(self, name):
        self.name = name
        self.operators = []

        # Dict assures uniqueness, but allows us to look up by name
        self.tensors = dict()

        self.inputs = []
        self.outputs = []

    def addTensor(
        self,
        name,
        shape,
        dtype,
        data=None,
        placeholderFilename=None,
    ):
        if name not in self.tensors:
            self.tensors[name] = TosaSerializerTensor(
                name, shape, dtype, data, placeholderFilename
            )

        return self.tensors[name]

    def addInput(self, name):
        self.inputs.append(name)

    def addOutput(self, name):
        self.outputs.append(name)

    def addOperator(self, op, inputs, outputs, attributes=None):
        self.operators.append(TosaSerializerOperator(op, inputs, outputs, attributes))

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fbv_inputs = TosaSerializer.serializeStrVec(
            builder, list(self.inputs), TosaBasicBlock.StartInputsVector
        )
        fbv_outputs = TosaSerializer.serializeStrVec(
            builder, list(self.outputs), TosaBasicBlock.StartOutputsVector
        )
        fbv_tensors = TosaSerializer.serializeObjVec(
            builder,
            list(self.tensors.values()),
            TosaBasicBlock.StartTensorsVector,
        )
        fbv_operators = TosaSerializer.serializeObjVec(
            builder, self.operators, TosaBasicBlock.StartOperatorsVector
        )

        TosaBasicBlock.Start(builder)
        TosaBasicBlock.AddName(builder, fb_name)
        TosaBasicBlock.AddInputs(builder, fbv_inputs)
        TosaBasicBlock.AddOutputs(builder, fbv_outputs)
        TosaBasicBlock.AddTensors(builder, fbv_tensors)
        TosaBasicBlock.AddOperators(builder, fbv_operators)
        return TosaBasicBlock.End(builder)


# How CONSTs are treated in the flatbuffer
@unique
class ConstMode(IntEnum):
    EMBED = 0
    EMBED_DUMP = 1
    INPUTS = 2


class TosaSerializerRegion:
    def __init__(self, name, pathPrefix, constMode=ConstMode.EMBED):
        self.name = name
        self.basicBlocks = []
        self.currInputIdx = 0
        self.currConstIdx = 0
        self.currLayerIdx = 1
        self.currResultIdx = 0
        self.pathPrefix = pathPrefix
        self.constMode = constMode

    def addBasicBlock(self, name):
        self.currBasicBlock = TosaSerializerBasicBlock(name)
        self.basicBlocks.append(self.currBasicBlock)

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fbv_basicBlocks = TosaSerializer.serializeObjVec(
            builder, self.basicBlocks, TosaRegion.StartBlocksVector
        )

        TosaRegion.Start(builder)
        TosaRegion.AddName(builder, fb_name)
        TosaRegion.AddBlocks(builder, fbv_basicBlocks)
        return TosaRegion.End(builder)

    def addPlaceholder(self, shape, dtype, vals):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        name = "input-{}".format(self.currInputIdx)
        filename = "{}.npy".format(name)
        self.currInputIdx = self.currInputIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, None, filename)
        # This is always an input to the block
        self.currBasicBlock.addInput(name)

        save_npy(os.path.join(self.pathPrefix, filename), vals, dtype)

        return tens

    def addConst(self, shape, dtype, vals, name=None):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        if name is None:
            name = "const-{}".format(self.currInputIdx)
            self.currInputIdx = self.currInputIdx + 1

        if vals is not None:
            # Numpy does not support serialising fp8e5m2 values, so
            # FP8E5M2 arrays should be received bitcasted as uint8 arrays
            # TODO: drop support for uint8 np.dtype in FP8E5M2 arrays
            if dtype == DType.FP8E5M2:
                vals = vals.view(np.uint8)

        if self.constMode == ConstMode.INPUTS:
            # Save const as input file
            filename = "{}.npy".format(name)
            tensor_vals = None
            self.currBasicBlock.addInput(name)
        else:
            # Embed const in flatbuffer
            filename = None
            tensor_vals = vals

        tens = self.currBasicBlock.addTensor(name, shape, dtype, tensor_vals, filename)
        # Add the operator now
        if dtype == DType.SHAPE:
            self.currBasicBlock.addOperator(TosaOp.Op().CONST_SHAPE, [], [name])
        else:
            self.currBasicBlock.addOperator(TosaOp.Op().CONST, [], [name])

        # Save the const data to file for debug or as input files
        if vals is not None and self.constMode in [
            ConstMode.EMBED_DUMP,
            ConstMode.INPUTS,
        ]:
            filename = "{}.npy".format(name)
            save_npy(os.path.join(self.pathPrefix, filename), vals, dtype)

        return tens

    def addIntermediate(self, shape, dtype):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        name = "layer-{}".format(self.currLayerIdx)
        self.currLayerIdx = self.currLayerIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, None)

        return tens

    def addInputTensor(self, tensor):
        self.currBasicBlock.addTensor(
            tensor.name,
            tensor.shape,
            tensor.dtype,
            tensor.data,
            tensor.placeholderFilename,
        )
        self.currBasicBlock.addInput(tensor.name)

    def addOutputTensor(self, tensor):
        self.currBasicBlock.addOutput(tensor.name)

    def addOutput(self, shape, dtype):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        name = "result-{}".format(self.currResultIdx)
        self.currResultIdx = self.currResultIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, None)
        self.currBasicBlock.addOutput(name)
        return tens

    def addOperator(self, op, inputs, outputs, attributes=None):
        if op in [TosaOp.Op().CONST, TosaOp.Op().CONST_SHAPE]:
            raise Exception("Use addConst() to add CONST or CONST_SHAPE ops")

        return self.currBasicBlock.addOperator(
            op,
            inputs,
            outputs,
            attributes,
        )


@unique
class TensorDir(IntEnum):
    PLACEHOLDER = 0
    CONST = 1
    INTERMEDIATE = 2
    RESULT = 3


class TosaSerializer:
    def __init__(self, pathPrefix, constMode=ConstMode.EMBED):
        self.builder = flatbuffers.Builder(0)

        # Enables inspection of constant data outside of graph
        self.constMode = constMode

        self.regions = []
        self.startRegion("main", pathPrefix)

        self.currRegion.addBasicBlock("main")

        # Is this an illegal test that is expected to fail?
        self.expectedReturnCode = 0
        self.expectedFailure = False
        self.expectedFailureDesc = ""

    def __str__(self):
        concatString = ""
        for region in self.regions:
            concatString = concatString + str(region)
        return concatString

    def addPlaceholder(self, shape, dtype, vals):
        return self.currRegion.addPlaceholder(shape, dtype, vals)

    def addConst(self, shape, dtype, vals, name=None):
        return self.currRegion.addConst(shape, dtype, vals, name)

    def addIntermediate(self, shape, dtype):
        return self.currRegion.addIntermediate(shape, dtype)

    def addInputTensor(self, tensor):
        self.currRegion.addInputTensor(tensor)

    def addOutputTensor(self, tensor):
        self.currRegion.addOutputTensor(tensor)

    def addOutput(self, shape, dtype):
        return self.currRegion.addOutput(shape, dtype)

    def addOperator(self, op, inputs, outputs, attributes=None):
        return self.currRegion.addOperator(op, inputs, outputs, attributes)

    def addBasicBlock(self, name):
        self.currRegion.addBasicBlock(name)

    def setExpectedReturnCode(self, val, fail, desc=""):
        self.expectedReturnCode = val
        self.expectedFailureDesc = desc
        self.expectedFailure = fail

    def serialize(self):
        builder = self.builder

        Version.Start(builder)
        Version.Add_Major(builder, TOSA_VERSION[0])
        Version.Add_Minor(builder, TOSA_VERSION[1])
        Version.Add_Patch(builder, TOSA_VERSION[2])
        Version.Add_Draft(builder, TOSA_VERSION[3])
        version = Version.End(builder)

        fbv_region = TosaSerializer.serializeObjVec(
            builder, self.regions, TosaGraph.StartRegionsVector
        )

        TosaGraph.Start(builder)
        TosaGraph.AddVersion(builder, version)
        TosaGraph.AddRegions(builder, fbv_region)
        graph = TosaGraph.End(builder)

        self.builder.Finish(graph, TOSA_GRAPH_IDENTIFIER)
        return self.builder.Output()

    def writeJson(self, tosa_filename):
        """Write a json test file so that it is fairly easy to pick up the test
        and generate commands for third party tool"""
        test_desc = dict()

        test_desc["tosa_file"] = tosa_filename
        ifm_name = []
        ifm_file = []
        ofm_name = []
        ofm_file = []

        for region in self.regions:
            for block in region.basicBlocks:
                if block and block.name == "main":
                    for i in block.inputs:
                        ifm_name.append(i)
                        ifm_file.append(block.tensors[i].placeholderFilename)
                    for o in block.outputs:
                        ofm_name.append(o)
                        # Make up an OFM filename here.  One isn't generated until the
                        # reference tool is run, so any name is a good name
                        ofm_file.append("ref-{}.npy".format(o))

        test_desc["ifm_name"] = ifm_name
        test_desc["ifm_file"] = ifm_file
        test_desc["ofm_name"] = ofm_name
        test_desc["ofm_file"] = ofm_file
        test_desc["expected_return_code"] = self.expectedReturnCode
        test_desc["expected_failure"] = self.expectedFailure
        if self.expectedFailureDesc:
            test_desc["expected_failure_desc"] = self.expectedFailureDesc

        return json.dumps(test_desc, indent="  ")

    def startRegion(self, name, pathPrefix):
        self.currRegion = TosaSerializerRegion(name, pathPrefix, self.constMode)
        self.regions.append(self.currRegion)

    @staticmethod
    def serializeStrVec(builder, vec, start_fcn):
        fb_strs = [builder.CreateString(i) for i in vec]
        start_fcn(builder, len(fb_strs))
        for s in fb_strs[::-1]:
            builder.PrependUOffsetTRelative(s)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeUint8Vec(builder, vec):
        builder.StartVector(1, len(vec), 8)
        for v in vec[::-1]:
            builder.PrependUint8(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeInt16Vec(builder, vec):
        builder.StartVector(2, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependInt16(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeInt32Vec(builder, vec):
        builder.StartVector(4, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependInt32(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeFpVec(builder, vec):
        builder.StartVector(4, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependFloat32(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def serializeObjVec(builder, vec, start_fcn):
        serialized_vec = []
        for v in vec[::-1]:
            serialized_vec.append(v.serialize(builder))

        start_fcn(builder, len(vec))
        for v in serialized_vec:
            builder.PrependUOffsetTRelative(v)
        try:
            return builder.EndVector()
        except TypeError:
            return builder.EndVector(len(vec))

    @staticmethod
    def toList(val):
        if isinstance(val, list):
            return val
        else:
            return [val]

    @staticmethod
    def convertDataToUint8Vec(dtype, data):
        u8_data = list()
        # little endianess
        if dtype == DType.BOOL:
            for val in data:
                val_u8 = np.uint8(val)
                u8_data.append(val_u8)
        elif dtype == DType.INT4:
            in_size = len(data)
            out_size = (in_size + 1) // 2
            for i in range(out_size):
                val_0 = np.array(data[2 * i]).astype(np.uint8)
                if (2 * i + 1) < in_size:
                    val_1 = np.array(data[2 * i + 1]).astype(np.uint8)
                else:
                    val_1 = 0
                mask = np.uint8(0xF)
                val_u8 = (val_0 & mask) | ((val_1 & mask) << 4)
                u8_data.append(val_u8)
        elif dtype == DType.INT8:
            for val in data:
                val_u8 = np.array(val).astype(dtype=np.uint8)
                u8_data.append(val_u8)
        elif dtype == DType.INT16:
            for val in data:
                val_u16 = np.array(val).astype(dtype=np.uint16)
                b0 = val_u16 & ByteMask
                b1 = (val_u16 >> np.uint16(8)) & ByteMask
                u8_data.extend([b0, b1])
        elif dtype == DType.INT32:
            for val in data:
                val_u32 = np.array(val).astype(dtype=np.uint32)
                b0 = val_u32 & ByteMask
                b1 = (val_u32 >> np.uint32(8)) & ByteMask
                b2 = (val_u32 >> np.uint32(16)) & ByteMask
                b3 = (val_u32 >> np.uint32(24)) & ByteMask
                u8_data.extend([b0, b1, b2, b3])
        elif dtype == DType.INT48:
            for val in data:
                val_u64 = np.array(val).astype(np.uint64)
                b0 = val_u64 & ByteMask
                b1 = (val_u64 >> np.uint64(8)) & ByteMask
                b2 = (val_u64 >> np.uint64(16)) & ByteMask
                b3 = (val_u64 >> np.uint64(24)) & ByteMask
                b4 = (val_u64 >> np.uint64(32)) & ByteMask
                b5 = (val_u64 >> np.uint64(40)) & ByteMask
                u8_data.extend([b0, b1, b2, b3, b4, b5])
        elif dtype == DType.SHAPE:
            for val in data:
                val_u64 = np.array(val).astype(np.uint64)
                b0 = val_u64 & ByteMask
                b1 = (val_u64 >> np.uint64(8)) & ByteMask
                b2 = (val_u64 >> np.uint64(16)) & ByteMask
                b3 = (val_u64 >> np.uint64(24)) & ByteMask
                b4 = (val_u64 >> np.uint64(32)) & ByteMask
                b5 = (val_u64 >> np.uint64(40)) & ByteMask
                b6 = (val_u64 >> np.uint64(48)) & ByteMask
                b7 = (val_u64 >> np.uint64(56)) & ByteMask
                u8_data.extend([b0, b1, b2, b3, b4, b5, b6, b7])
        elif dtype == DType.FP16:
            np_arr = np.array(data, dtype=np.float16)
            u8_data.extend(np_arr.view(np.uint8))
        elif dtype == DType.FP32:
            np_arr = np.array(data, dtype=np.float32)
            u8_data.extend(np_arr.view(np.uint8))
        elif dtype == DType.BF16:
            np_arr = np.array(data, dtype=bfloat16)
            u8_data.extend(np_arr.view(np.uint8))
        elif dtype == DType.FP8E4M3:
            for val in data:
                val_f8 = np.array(val).astype(float8_e4m3fn).view(np.uint8)
                u8_data.append(val_f8)
        elif dtype == DType.FP8E5M2:
            # Numpy does not support serialising fp8e5m2 values, so
            # the array we get for serialisation is a uint8 array
            np_arr = np.array(data, dtype=np.uint8)
            u8_data.extend(np_arr.view(np.uint8))
        elif dtype == TosaDType.DType:
            # Serialize DType enum data as uint8 bytes
            for val in data:
                np_arr = np.array(data, dtype=np.uint32)
                u8_data.extend(np_arr.view(np.uint8))
        else:
            raise Exception("unsupported data type {}".format(DTypeNames[dtype]))
        return u8_data
