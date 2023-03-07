
// Copyright (c) 2020-2021, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef _TOSA_SERIALIZATION_HANDLER_H
#define _TOSA_SERIALIZATION_HANDLER_H
#include "attribute.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "numpy_utils.h"
#include "tosa_generated.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Keep version number in sync with the version default value with schema/tosa.fbs
#define TOSA_VERSION_MAJOR 0
#define TOSA_VERSION_MINOR 51
#define TOSA_VERSION_PATCH 0
#define TOSA_VERSION_DRAFT true
#define TENSOR_BUFFER_FORCE_ALIGNMENT 8

namespace tosa
{

enum tosa_err_t
{
    TOSA_OK,
    TOSA_USER_ERROR,
    TOSA_FILE_ERROR,
    TOSA_MEMORY_ERROR,
    TOSA_SCHEMA_MISSING,
    TOSA_INTERNAL_ERROR,
    TOSA_VERSION_MISMATCH,
    NUM_TOSA_ERROR
};

struct TosaVersion
{
    int32_t _major;
    int32_t _minor;
    int32_t _patch;
    bool _draft;

    enum class compat_t
    {
        COMPLETELY_COMPATIBLE,
        PARTIALLY_COMPATIBLE,
        NOT_COMPATIBLE
    };

    TosaVersion() = default;
    TosaVersion(int32_t major, int32_t minor, int32_t patch, bool draft)
    {
        set_version(major, minor, patch, draft);
    }

    void set_version(int32_t major, int32_t minor, int32_t patch, bool draft)
    {
        _major = major;
        _minor = minor;
        _patch = patch;
        _draft = draft;
    }

    std::string to_string() const
    {
        std::string str;
        str += std::to_string(_major) + ".";
        str += std::to_string(_minor) + ".";
        str += std::to_string(_patch);
        if (_draft)
            str += "d";
        return str;
    }

    compat_t is_compatible(const TosaVersion& rhs) const
    {
        if (rhs._major == _major && rhs._minor == _minor)
        {
            if (rhs._patch == _patch && rhs._draft == _draft)
            {
                return TosaVersion::compat_t::COMPLETELY_COMPATIBLE;
            }
            else
            {
                return TosaVersion::compat_t::PARTIALLY_COMPATIBLE;
            }
        }
        return TosaVersion::compat_t::NOT_COMPATIBLE;
    }
};

class TosaSerializationHandler;

class TosaSerializationTensor
{
public:
    // constructor and destructor
    TosaSerializationTensor(const flatbuffers::String* name,
                            const flatbuffers::Vector<int32_t>* shape,
                            DType dtype,
                            const flatbuffers::Vector<uint8_t>* data);
    TosaSerializationTensor(const std::string& name,
                            const std::vector<int32_t>& shape,
                            DType dtype,
                            const std::vector<uint8_t>& data);
    TosaSerializationTensor();
    ~TosaSerializationTensor();

    // accessor
    std::string GetName() const
    {
        return _name;
    }
    const std::vector<int32_t>& GetShape() const
    {
        return _shape;
    }
    DType GetDtype()
    {
        return _dtype;
    }
    const std::vector<uint8_t>& GetData() const
    {
        return _data;
    }

    // modifier
    void SetDtype(DType dtype)
    {
        _dtype = dtype;
    }
    void SetName(std::string name)
    {
        _name = name;
    }
    void SetData(const std::vector<uint8_t>& data)
    {
        _data = data;
    }
    void SetData(std::vector<uint8_t>&& data)
    {
        _data = std::move(data);
    }

private:
    DType _dtype;                /* data type enumeration, see tosa_isa_generated.h */
    std::vector<int32_t> _shape; /* shape of the tensor */
    std::string _name;           /* name of the tensor, used for solving dependency */
    std::vector<uint8_t> _data;  /* data array */
};

class TosaSerializationOperator
{
public:
    // use default copy, void constructor
    // constructor and destructor
    TosaSerializationOperator(Op op,
                              Attribute attribute_type,
                              const TosaAttributeBase* attribute,
                              const std::vector<std::string>& input_tensor_names,
                              const std::vector<std::string>& output_tensor_names);
    TosaSerializationOperator(Op op,
                              Attribute attribute_type,
                              const TosaAttributeBase* attribute,
                              std::vector<std::string>&& input_tensor_names,
                              std::vector<std::string>&& output_tensor_names);
    ~TosaSerializationOperator();

    // accessor
    Op GetOp() const
    {
        return _op;
    }
    Attribute GetAttributeType() const
    {
        return _attribute_type;
    }
    TosaAttributeBase* GetAttribute() const
    {
        return _attribute;
    }
    std::vector<std::string>& GetInputTensorNames()
    {
        return _input_tensor_names;
    }
    std::vector<std::string>& GetOutputTensorNames()
    {
        return _output_tensor_names;
    }

private:
    void InitializeAttribute(Attribute attribute_type, const TosaAttributeBase* attribute);
    Op _op;                        /* operator enum, see tosa_isa_generated.h for enumeration table */
    Attribute _attribute_type;     /* operator attribute enum, used for dynamic casting TosaAttributeBase class */
    TosaAttributeBase* _attribute; /* real attribute class goes here */
    std::vector<std::string> _input_tensor_names;  /* array of input tensor names */
    std::vector<std::string> _output_tensor_names; /* array of output tensor names */
};

class TosaSerializationBasicBlock
{
public:
    // constructor and destructor
    TosaSerializationBasicBlock(const std::string& name,
                                const std::string& region_name,
                                const std::vector<TosaSerializationOperator*>& operators,
                                const std::vector<TosaSerializationTensor*>& tensors,
                                const std::vector<std::string>& inputs,
                                const std::vector<std::string>& outputs);
    TosaSerializationBasicBlock(std::string&& name,
                                std::string&& region_name,
                                std::vector<TosaSerializationOperator*>&& operators,
                                std::vector<TosaSerializationTensor*>&& tensors,
                                std::vector<std::string>&& inputs,
                                std::vector<std::string>&& outputs);
    ~TosaSerializationBasicBlock();

    // accessor
    std::string GetName() const
    {
        return _name;
    }
    std::string GetRegionName() const
    {
        return _region_name;
    }
    std::vector<TosaSerializationOperator*>& GetOperators()
    {
        return _operators;
    }

    std::vector<TosaSerializationTensor*>& GetTensors()
    {
        return _tensors;
    }

    TosaSerializationTensor* GetTensorByName(std::string name)
    {
        TosaSerializationTensor* result = nullptr;
        for (auto tensor : GetTensors())
        {
            if (tensor->GetName() == name)
            {
                result = tensor;
                break;
            }
        }
        return result;
    }

    std::vector<std::string>& GetInputs()
    {
        return _inputs;
    }

    std::vector<std::string>& GetOutputs()
    {
        return _outputs;
    }

private:
    std::string _name; /* name of basic block */
    std::string _region_name;
    std::vector<TosaSerializationOperator*> _operators; /* TosaSerializationOperator list */
    std::vector<TosaSerializationTensor*> _tensors;     /* TosaSerializationTensor list */
    std::vector<std::string> _inputs;                   /* array of string to specify block inputs */
    std::vector<std::string> _outputs;                  /* array of string to specify block outputs */
};

class TosaSerializationRegion
{
public:
    // constructor and desctructor
    TosaSerializationRegion(const std::string& name, const std::vector<TosaSerializationBasicBlock*>& blocks);
    TosaSerializationRegion(const std::string&& name, const std::vector<TosaSerializationBasicBlock*>&& blocks);
    ~TosaSerializationRegion();

    // accessors
    std::string GetName() const
    {
        return this->_name;
    }

    std::vector<TosaSerializationBasicBlock*>& GetBlocks()
    {
        return this->_blocks;
    }

    TosaSerializationBasicBlock* GetBlockByName(std::string name)
    {
        TosaSerializationBasicBlock* result = nullptr;
        for (auto block : GetBlocks())
        {
            if (block->GetName() == name)
            {
                result = block;
                break;
            }
        }
        return result;
    }

private:
    std::string _name;                                 /* name of basic block */
    std::vector<TosaSerializationBasicBlock*> _blocks; /* TosaSerializationBasicBlock list */
};

/*
 * this is a helper class for writing/reading Tosa ISA
 * supported format: .tosa (flatbuffer), .json
 * and provide high-level std::vector-like interface
 * to access internal data structure
 */
class TosaSerializationHandler
{
public:
    // constructor and destructor
    TosaSerializationHandler();
    ~TosaSerializationHandler();

    // file io
    tosa_err_t LoadFileJson(const char* filename);
    tosa_err_t LoadFileTosaFlatbuffer(const char* filename);
    tosa_err_t LoadFileTosaFlatbuffer(const void* input, int in_size);
    tosa_err_t SaveFileJson(const char* filename);
    tosa_err_t SaveFileTosaFlatbuffer(const char* filename);
    tosa_err_t LoadFileSchema(const char* schema_filename);

    // data format conversion. little-endian.
    static tosa_err_t ConvertF16toU8(const std::vector<float>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertF32toU8(const std::vector<float>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertI48toU8(const std::vector<int64_t>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertI32toU8(const std::vector<int32_t>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertI16toU8(const std::vector<int16_t>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertI8toU8(const std::vector<int8_t>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertI4toU8(const std::vector<int8_t>& in, std::vector<uint8_t>& out);
    static tosa_err_t ConvertBooltoU8(const std::vector<bool>& in, std::vector<uint8_t>& out);

    static tosa_err_t ConvertU8toF16(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<float>& out);
    static tosa_err_t ConvertU8toF32(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<float>& out);
    static tosa_err_t ConvertU8toI48(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int64_t>& out);
    static tosa_err_t ConvertU8toI32(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int32_t>& out);
    static tosa_err_t ConvertU8toI16(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int16_t>& out);
    static tosa_err_t ConvertU8toI8(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int8_t>& out);
    static tosa_err_t ConvertU8toI4(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<int8_t>& out);
    static tosa_err_t ConvertU8toBool(const std::vector<uint8_t>& in, uint32_t out_size, std::vector<bool>& out);

    // version
    const TosaVersion& GetVersion()
    {
        return _version;
    }

    // accessor
    std::vector<TosaSerializationRegion*>& GetRegions()
    {
        return _regions;
    }

    TosaSerializationRegion* GetMainRegion()
    {
        return _regions[0];
    }

    TosaSerializationRegion* GetRegionByName(std::string name)
    {
        TosaSerializationRegion* result = nullptr;
        for (auto region : GetRegions())
        {
            if (region->GetName() == name)
            {
                result = region;
                break;
            }
        }
        return result;
    }

    bool GetSchemaLoaded() const
    {
        return _schemaLoaded;
    }

protected:
    tosa_err_t Clear();
    tosa_err_t Deserialize(const uint8_t* buf);
    tosa_err_t Serialize();
    TosaVersion ParseTosaSchemaVersion(std::string schema);

private:
    TosaVersion _version;                           /* version struct */
    flatbuffers::FlatBufferBuilder _builder;        /* flatbuffer builder */
    flatbuffers::Parser _parser;                    /* flatbuffer parser, used for json parsing */
    std::vector<TosaSerializationRegion*> _regions; /* array structure to store all TosaSerializationRegion */
    bool _schemaLoaded;                             /* is the schema properly loaded? */
};

}    // namespace tosa

#endif    // _TOSA_SERIALIZATION_HANDLER_H