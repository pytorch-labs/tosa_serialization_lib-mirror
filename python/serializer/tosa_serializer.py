# Copyright (c) 2020-2024, ARM Limited.
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
        from tosa import ARGMAX_Attribute as a, Attribute

        self.utype = Attribute.Attribute().ARGMAX_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))
        self.ints.append((a.AddNanMode, nan_mode))

    def AvgPool2DAttribute(
        self,
        kernel,
        stride,
        pad,
        input_zp,
        output_zp,
        acc_type,
    ):
        from tosa import AVG_POOL2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().AVG_POOL2D_Attribute

        self.optFcns = (a.Start, a.End)
        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddKernel, kernel))
        self.intvecs.append((a.AddStride, stride))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddOutputZp, output_zp))
        self.ints.append((a.AddAccType, acc_type))

    def Conv2DAttribute(
        self, pad, stride, dilation, input_zp, weight_zp, local_bound, acc_type
    ):
        from tosa import CONV2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CONV2D_Attribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def Conv3DAttribute(
        self, pad, stride, dilation, input_zp, weight_zp, local_bound, acc_type
    ):
        from tosa import CONV3D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CONV3D_Attribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def DepthwiseConv2DAttribute(
        self, pad, stride, dilation, input_zp, weight_zp, local_bound, acc_type
    ):
        from tosa import DEPTHWISE_CONV2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().DEPTHWISE_CONV2D_Attribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPad, pad))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddDilation, dilation))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def FFT2DAttribute(self, inverse, local_bound):
        from tosa import FFT2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().FFT2D_Attribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddInverse, inverse))
        self.bools.append((a.AddLocalBound, local_bound))

    def MatMulAttribute(self, a_zp, b_zp):
        from tosa import MATMUL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().MATMUL_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAZp, a_zp))
        self.ints.append((a.AddBZp, b_zp))

    def MaxPool2DAttribute(self, kernel, stride, pad, nan_mode):
        from tosa import MAX_POOL2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().MAX_POOL2D_Attribute

        self.optFcns = (a.Start, a.End)
        self.intvecs.append((a.AddKernel, kernel))
        self.intvecs.append((a.AddStride, stride))
        self.intvecs.append((a.AddPad, pad))
        self.ints.append((a.AddNanMode, nan_mode))

    def RFFT2DAttribute(self, local_bound):
        from tosa import RFFT2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RFFT2D_Attribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddLocalBound, local_bound))

    def TransposeConv2DAttribute(
        self, out_pad, stride, input_zp, weight_zp, local_bound, acc_type
    ):
        from tosa import TRANSPOSE_CONV2D_Attribute as a, Attribute

        self.utype = Attribute.Attribute().TRANSPOSE_CONV2D_Attribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddOutPad, out_pad))
        self.intvecs.append((a.AddStride, stride))
        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddWeightZp, weight_zp))
        self.bools.append((a.AddLocalBound, local_bound))
        self.ints.append((a.AddAccType, acc_type))

    def ClampAttribute(
        self, serializer_builder, min_val_as_bytes, max_val_as_bytes, nan_mode
    ):
        from tosa import CLAMP_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CLAMP_Attribute
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

    def ERFAttribute(self):
        from tosa import ERF_Attribute as a, Attribute

        self.utype = Attribute.Attribute().ERF_Attribute
        self.optFcns = (a.Start, a.End)

    def SigmoidAttribute(self):
        from tosa import SIGMOID_Attribute as a, Attribute

        self.utype = Attribute.Attribute().SIGMOID_Attribute
        self.optFcns = (a.Start, a.End)

    def TanhAttribute(self):
        from tosa import TANH_Attribute as a, Attribute

        self.utype = Attribute.Attribute().TANH_Attribute
        self.optFcns = (a.Start, a.End)

    def AddAttribute(self):
        from tosa import ADD_Attribute as a, Attribute

        self.utype = Attribute.Attribute().ADD_Attribute
        self.optFcns = (a.Start, a.End)

    def ArithmeticRightShiftAttribute(self, round):
        from tosa import ARITHMETIC_RIGHT_SHIFT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().ARITHMETIC_RIGHT_SHIFT_Attribute
        self.optFcns = (
            a.Start,
            a.End,
        )

        self.bools.append((a.AddRound, round))

    def BitwiseAndAttribute(self):
        from tosa import BITWISE_AND_Attribute as a, Attribute

        self.utype = Attribute.Attribute().BITWISE_AND_Attribute
        self.optFcns = (a.Start, a.End)

    def BitwiseOrAttribute(self):
        from tosa import BITWISE_OR_Attribute as a, Attribute

        self.utype = Attribute.Attribute().BITWISE_OR_Attribute
        self.optFcns = (a.Start, a.End)

    def BitwiseXorAttribute(self):
        from tosa import BITWISE_XOR_Attribute as a, Attribute

        self.utype = Attribute.Attribute().BITWISE_XOR_Attribute
        self.optFcns = (a.Start, a.End)

    def IntDivAttribute(self):
        from tosa import INTDIV_Attribute as a, Attribute

        self.utype = Attribute.Attribute().INTDIV_Attribute
        self.optFcns = (a.Start, a.End)

    def LogicalAndAttribute(self):
        from tosa import LOGICAL_AND_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOGICAL_AND_Attribute
        self.optFcns = (a.Start, a.End)

    def LogicalLeftShiftAttribute(self):
        from tosa import LOGICAL_LEFT_SHIFT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOGICAL_LEFT_SHIFT_Attribute
        self.optFcns = (a.Start, a.End)

    def LogicalRightShiftAttribute(self):
        from tosa import LOGICAL_RIGHT_SHIFT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOGICAL_RIGHT_SHIFT_Attribute
        self.optFcns = (a.Start, a.End)

    def LogicalOrAttribute(self):
        from tosa import LOGICAL_OR_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOGICAL_OR_Attribute
        self.optFcns = (a.Start, a.End)

    def LogicalXorAttribute(self):
        from tosa import LOGICAL_XOR_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOGICAL_XOR_Attribute
        self.optFcns = (a.Start, a.End)

    def MaximumAttribute(self, nan_mode):
        from tosa import MAXIMUM_Attribute as a, Attribute

        self.utype = Attribute.Attribute().MAXIMUM_Attribute
        self.optFcns = (a.Start, a.End)
        self.ints.append((a.AddNanMode, nan_mode))

    def MinimumAttribute(self, nan_mode):
        from tosa import MINIMUM_Attribute as a, Attribute

        self.utype = Attribute.Attribute().MINIMUM_Attribute
        self.optFcns = (a.Start, a.End)
        self.ints.append((a.AddNanMode, nan_mode))

    def MulAttribute(self, shift):
        from tosa import MUL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().MUL_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddShift, shift))

    def PowAttribute(self):
        from tosa import POW_Attribute as a, Attribute

        self.utype = Attribute.Attribute().POW_Attribute
        self.optFcns = (a.Start, a.End)

    def SubAttribute(self):
        from tosa import SUB_Attribute as a, Attribute

        self.utype = Attribute.Attribute().SUB_Attribute
        self.optFcns = (a.Start, a.End)

    def TableAttribute(self):
        from tosa import TABLE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().TABLE_Attribute
        self.optFcns = (a.Start, a.End)

    def AbsAttribute(self):
        from tosa import ABS_Attribute as a, Attribute

        self.utype = Attribute.Attribute().ABS_Attribute
        self.optFcns = (a.Start, a.End)

    def BitwiseNotAttribute(self):
        from tosa import BITWISE_NOT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().BITWISE_NOT_Attribute
        self.optFcns = (a.Start, a.End)

    def CeilAttribute(self):
        from tosa import CEIL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CEIL_Attribute
        self.optFcns = (a.Start, a.End)

    def ClzAttribute(self):
        from tosa import CLZ_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CLZ_Attribute
        self.optFcns = (a.Start, a.End)

    def CosAttribute(self):
        from tosa import COS_Attribute as a, Attribute

        self.utype = Attribute.Attribute().COS_Attribute
        self.optFcns = (a.Start, a.End)

    def ExpAttribute(self):
        from tosa import EXP_Attribute as a, Attribute

        self.utype = Attribute.Attribute().EXP_Attribute
        self.optFcns = (a.Start, a.End)

    def FloorAttribute(self):
        from tosa import FLOOR_Attribute as a, Attribute

        self.utype = Attribute.Attribute().FLOOR_Attribute
        self.optFcns = (a.Start, a.End)

    def LogAttribute(self):
        from tosa import LOG_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOG_Attribute
        self.optFcns = (a.Start, a.End)

    def LogicalNotAttribute(self):
        from tosa import LOGICAL_NOT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().LOGICAL_NOT_Attribute
        self.optFcns = (a.Start, a.End)

    def NegateAttribute(self, input1_zp, output_zp):
        from tosa import NEGATE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().NEGATE_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInput1Zp, input1_zp))
        self.ints.append((a.AddOutputZp, output_zp))

    def ReciprocalAttribute(self):
        from tosa import RECIPROCAL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RECIPROCAL_Attribute
        self.optFcns = (a.Start, a.End)

    def RsqrtAttribute(self):
        from tosa import RSQRT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RSQRT_Attribute
        self.optFcns = (a.Start, a.End)

    def SinAttribute(self):
        from tosa import SIN_Attribute as a, Attribute

        self.utype = Attribute.Attribute().SIN_Attribute
        self.optFcns = (a.Start, a.End)

    def SelectAttribute(self):
        from tosa import SELECT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().SELECT_Attribute
        self.optFcns = (a.Start, a.End)

    def EqualAttribute(self):
        from tosa import EQUAL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().EQUAL_Attribute
        self.optFcns = (a.Start, a.End)

    def GreaterAttribute(self):
        from tosa import GREATER_Attribute as a, Attribute

        self.utype = Attribute.Attribute().GREATER_Attribute
        self.optFcns = (a.Start, a.End)

    def GreaterEqualAttribute(self):
        from tosa import GREATER_EQUAL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().GREATER_EQUAL_Attribute
        self.optFcns = (a.Start, a.End)

    def ReduceAllAttribute(self, axis):
        from tosa import REDUCE_ALL_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REDUCE_ALL_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReduceAnyAttribute(self, axis):
        from tosa import REDUCE_ANY_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REDUCE_ANY_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReduceMaxAttribute(self, axis, nan_mode):
        from tosa import REDUCE_MAX_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REDUCE_MAX_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))
        self.ints.append((a.AddNanMode, nan_mode))

    def ReduceMinAttribute(self, axis, nan_mode):
        from tosa import REDUCE_MIN_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REDUCE_MIN_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))
        self.ints.append((a.AddNanMode, nan_mode))

    def ReduceProductAttribute(self, axis):
        from tosa import REDUCE_PRODUCT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REDUCE_PRODUCT_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ReduceSumAttribute(self, axis):
        from tosa import REDUCE_SUM_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REDUCE_SUM_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def ConcatAttribute(self, axis):
        from tosa import CONCAT_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CONCAT_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def PadAttribute(self, serializer_builder, pad_const_val_as_bytes):
        from tosa import PAD_Attribute as a, Attribute

        self.utype = Attribute.Attribute().PAD_Attribute
        self.optFcns = (a.Start, a.End)

        # serialize pad_const_val_as_bytes as uint8 vector
        serialized_pad_const_val = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, pad_const_val_as_bytes
        )

        self.floats.append((a.AddPadConst, serialized_pad_const_val))

    def ReshapeAttribute(self):
        from tosa import RESHAPE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RESHAPE_Attribute
        self.optFcns = (a.Start, a.End)

    def ReverseAttribute(self, axis):
        from tosa import REVERSE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().REVERSE_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddAxis, axis))

    def SliceAttribute(self):
        from tosa import SLICE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().SLICE_Attribute
        self.optFcns = (a.Start, a.End)

    def TileAttribute(self):
        from tosa import TILE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().TILE_Attribute
        self.optFcns = (a.Start, a.End)

    def TransposeAttribute(self, perms):
        from tosa import TRANSPOSE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().TRANSPOSE_Attribute
        self.optFcns = (a.Start, a.End)

        self.intvecs.append((a.AddPerms, perms))

    def GatherAttribute(self):
        from tosa import GATHER_Attribute as a, Attribute

        self.utype = Attribute.Attribute().GATHER_Attribute
        self.optFcns = (a.Start, a.End)

    def ScatterAttribute(self):
        from tosa import SCATTER_Attribute as a, Attribute

        self.utype = Attribute.Attribute().SCATTER_Attribute
        self.optFcns = (a.Start, a.End)

    def ResizeAttribute(self, mode):
        from tosa import RESIZE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RESIZE_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddMode, mode))

    def CastAttribute(self):
        from tosa import CAST_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CAST_Attribute
        self.optFcns = (a.Start, a.End)

    def CastStochasticAttribute(self):
        from tosa import CAST_STOCHASTIC_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CAST_STOCHASTIC_Attribute
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
        from tosa import RESCALE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RESCALE_Attribute
        self.optFcns = (a.Start, a.End)

        self.ints.append((a.AddInputZp, input_zp))
        self.ints.append((a.AddOutputZp, output_zp))
        self.bools.append((a.AddScale32, scale32))
        self.bools.append((a.AddDoubleRound, double_round))
        self.bools.append((a.AddPerChannel, per_channel))
        self.bools.append((a.AddInputUnsigned, input_unsigned))
        self.bools.append((a.AddOutputUnsigned, output_unsigned))

    def ConstAttribute(self):
        from tosa import CONST_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CONST_Attribute
        self.optFcns = (a.Start, a.End)

    def RandSeedAttribute(self):
        from tosa import RAND_SEED_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RAND_SEED_Attribute
        self.optFcns = (a.Start, a.End)

    def RandUniformAttribute(self, use_seed):
        from tosa import RAND_UNIFORM_Attribute as a, Attribute

        self.utype = Attribute.Attribute().RAND_UNIFORM_Attribute
        self.optFcns = (a.Start, a.End)

        self.bools.append((a.AddUseSeed, use_seed))

    def IdentityAttribute(self):
        from tosa import IDENTITY_Attribute as a, Attribute

        self.utype = Attribute.Attribute().IDENTITY_Attribute
        self.optFcns = (a.Start, a.End)

    def CustomAttribute(
        self,
        serializer_builder,
        operator_name,
        domain_name,
        implementation_attrs_as_bytes,
    ):
        from tosa import CUSTOM_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CUSTOM_Attribute
        self.optFcns = (a.Start, a.End)

        implementation_attrs = ts.TosaSerializer.serializeUint8Vec(
            serializer_builder, implementation_attrs_as_bytes
        )

        self.strings.append((a.AddOperatorName, operator_name))
        self.strings.append((a.AddDomainName, domain_name))
        self.floats.append((a.AddImplementationAttrs, implementation_attrs))

    def CondIfAttribute(self, then_graph, else_graph):
        from tosa import COND_IF_Attribute as a, Attribute

        self.utype = Attribute.Attribute().COND_IF_Attribute
        self.optFcns = (a.Start, a.End)

        self.strings.append((a.AddThenGraph, then_graph))
        self.strings.append((a.AddElseGraph, else_graph))

    def WhileLoopAttribute(self, cond_graph, body_graph):
        from tosa import WHILE_LOOP_Attribute as a, Attribute

        self.utype = Attribute.Attribute().WHILE_LOOP_Attribute
        self.optFcns = (a.Start, a.End)

        self.strings.append((a.AddCondGraph, cond_graph))
        self.strings.append((a.AddBodyGraph, body_graph))

    def YieldAttribute(self):
        from tosa import YIELD_Attribute as a, Attribute

        self.utype = Attribute.Attribute().YIELD_Attribute
        self.optFcns = (a.Start, a.End)

    def VariableAttribute(self):
        from tosa import VARIABLE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().VARIABLE_Attribute
        self.optFcns = (a.Start, a.End)

    def VariableReadAttribute(self):
        from tosa import VARIABLE_READ_Attribute as a, Attribute

        self.utype = Attribute.Attribute().VARIABLE_READ_Attribute
        self.optFcns = (a.Start, a.End)

    def VariableWriteAttribute(self):
        from tosa import VARIABLE_WRITE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().VARIABLE_WRITE_Attribute
        self.optFcns = (a.Start, a.End)

    def ConstShapeAttribute(self):
        from tosa import CONST_SHAPE_Attribute as a, Attribute

        self.utype = Attribute.Attribute().CONST_SHAPE_Attribute
        self.optFcns = (a.Start, a.End)


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

    def addOperator(self, op, inputs, outputs, attributes):
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

        if vals is not None:
            np.save(os.path.join(self.pathPrefix, filename), vals, False)

        return tens

    def addConst(self, shape, dtype, vals, name=None):
        if not self.currBasicBlock:
            raise Exception("addTensor called without valid basic block")

        if name is None:
            name = "const-{}".format(self.currInputIdx)
            self.currInputIdx = self.currInputIdx + 1

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
            attr = TosaSerializerAttribute()
            attr.ConstShapeAttribute()
            self.currBasicBlock.addOperator(TosaOp.Op().CONST_SHAPE, [], name, attr)
        else:
            attr = TosaSerializerAttribute()
            attr.ConstAttribute()
            self.currBasicBlock.addOperator(TosaOp.Op().CONST, [], name, attr)

        # Save the const data to file for debug or as input files
        if vals is not None and self.constMode in [
            ConstMode.EMBED_DUMP,
            ConstMode.INPUTS,
        ]:
            filename = "{}.npy".format(name)
            np.save(os.path.join(self.pathPrefix, filename), vals, False)

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

    def addOperator(self, op, inputs, outputs, attributes):
        if op == TosaOp.Op().CONST:
            raise Exception("Use addConstTensor() to add CONST ops")

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

    def addOperator(self, op, inputs, outputs, attributes):
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
            for val in data:
                # Numpy does not support serialising fp8e5m2 values, so usually
                # the array we get for serialisation is a uint8 array
                val_f8 = np.array(val).astype(np.uint8).view(np.uint8)
                u8_data.append(val_f8)
        elif dtype == TosaDType.DType:
            # Serialize DType enum data as uint8 bytes
            for val in data:
                np_arr = np.array(data, dtype=np.uint32)
                u8_data.extend(np_arr.view(np.uint8))
        else:
            raise Exception("unsupported data type {}".format(DTypeNames[dtype]))
        return u8_data
