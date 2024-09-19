TOSA Serialization Library
==========================

# Introduction

The *TOSA Serialization library* provides methods to read and write serialized
TOSA graphs (<https://developer.mlplatform.org/w/tosa/>).  The library includes
a FlatBuffers schema and a C++ API for reading and writing TOSA graphs.

# Usage

The section below describes serialization_lib API usage. For more
details, please refer to `include/tosa_serialization_handler.h`.

## TosaSerializationHandler

This is the top-level class that contains the entire TOSA graph.  In
particular, it contains a vector of `TosaSerializationRegion` objects,
and provides API for file IO, region access, and version checking.

    a. `LoadFileJson(filename)`:

        Loads json-formatted file "filename" from disk, and initialize the
        internal graph structure.

        Requires the schema file to be loaded via `LoadFileSchema()`.

    b. `SaveFileJson(filename)`:

        Snapshots the internal graph structure and saves out JSON-formatted file
        `filename` to disk.
        Requires the schema file to be loaded via `LoadFileSchema()`.

    c. `LoadFileTosaFlatbuffer(filename)`:

        Loads serialized flatbuffer file "filename" from disk, and initialize the
        internal graph structure.

    d. `SaveFileTosaFlatbuffer(filename)`:

        Snapshots the internal graph structure and saves out serialized
        flatbuffer file `filename` to disk.

    e. `GetVersion()`:

        Returns TOSA version implemented by the serialization library.

    f. `GetRegions()`:

        Returns vector of `TosaSerializationRegion`. A valid graph must have
        one `main` region as the first region being traversed.

    g. `GetMainRegion()`:

        Shortcut for accessing the first region.

    h.  `GetRegionByName(name)`

        Returns region whose name is 'name'. A valid graph must have one `main`
        region as the first region being traversed.

    i. `GetInputs()` / `GetOutputs()`:

        Shortcut for `main` region's input/output tensor name. Input tensors of
        the main block are usually treated as `tosa.PLACEHOLDER`. Output tensors
        are the output of the entire graph and should be evaluated when graph
        traversal has finished.

## TosaSerializationRegion

This is the region class. It contains vectors of `TosaSerializationBasicBlock` objects,
and provides API for block access.

    a. `GetName()`:

        Returns name of the region.

    b. `GetBlocks()`:

        Returns vector of TosaSerializationBasicBlock. A valid region must have
        at least one block.

    c. `GetBlockByName(name)`:

        Returns the `TosaSerializationBasicBlock` with name `name`. Returns `nullptr`
        if nothing matches.

## TosaSerializationBasicBlock

This is the basic-block class. It contains vectors of
`TosaSerializationOperator` and `TosaSerializationTensor`. Once entering
a basic block, all of the operators within the block will be evaluated
in order.

Upon reaching a TOSA control flow operator (`tosa.WHILE` and
`tosa.COND_IF`), the status of current unfinished block will be saved, and
the regions specified in control flow operator will be evaluated first. Once
the control flow regions finish its evaluation, the original unfinished
block status will be restored and evaluation continues.  This is more
analogous to a function call than a compiler basic block.

    a. `GetName()`:

        Returns name of the basic block.

    b. `GetRegionName()`:

        Returns name of the region containing the basic block.

    c. `GetOperators()`:

        Returns vector of `TosaSerializationOperator`

    d. `GetTensors()`:

        Returns vector of `TosaSerializationTensor`

    e. `GetTensorByName(name)`:

        Returns the `TosaSerializationTensor` with name `name`. Returns `nullptr`
        if nothing matches.

    f. `GetInputs()` / `GetOutputs()`:

        Returns input/output tensor name of the basic block.

## TosaSerializationOperator

The operator class contains (1) what TOSA Op, (2) attribute (compile-time-
known input) and (3) input/output tensor names.

    a. `GetOp()`:

        Returns TOSA Op. Defined in schema `tosa.fbs`.

    b. `GetAttribute()` / `GetAttributeType()`:

        `GetAttribute()` returns the base object of attribute.
        `GetAttributeType()` returns which type of attribute the base object
        needs to be casted to.  Type of attribute is defined in `tosa.fbs` and
        `include/attribute.def`.

    c. `GetInputTensorNames()` / `GetOutputTensorNames()`:

        Returns the input/output tensor names of the basic block.

## TosaSerializationTensor

The tensor class contains (1) data type, (2) shape, (3) properties and (4) data value.

    a. `GetName()` / `SetName(name)`:

        `GetName()` returns the name of the tensor. `SetName()` sets the name
        of the tensor.

    b. `GetShape()`:

        Returns the shape of the tensor as `vector<int32_t>`.

    c. `GetDtype()` / `SetDtype(dtype)`:

        `GetDtype()` returns the data type of the tensor. `SetDtype()` sets the
        data type of the tensor. DType is defined in `tosa.fbs`.

    d. `GetVariable()`:

        Returns whether tensor is a Tosa Variable.

    e. `GetIsUnranked()` / `SetIsUnranked(value)`:

        `GetIsUnranked()` returns whether tensor is an unranked tensor.
        `SetIsUnranked()` sets whether tensor is an unranked tensor.

    f. `GetData()` / `SetData(data)`:

        `GetData()` returns a vector of `uint8_t` values which stores the constant
        value for a constant tensor, or the initialization value for a variable tensor.
        `SetData()` sets the constant value for a constant tensor, or the initialization
        value for a variable tensor.

# License

The *TOSA Serialization Library* is licensed under Apache-2.0.

## Third Party Projects

- The [half](https://half.sourceforge.net/) library is licensed under MIT license.

Other third party projects are referenced as sub-modules and as such, are licensed under the licenses stated in their projects.
