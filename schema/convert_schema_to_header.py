# Copyright (c) 2024, ARM Limited.
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

# This short python script converts the tosa flatbuffer schema file to a C/C++ header

with open("tosa.fbs", "rb") as schema_file:
    schema_string = schema_file.read().hex()

    # add trailing 00
    schema_string += "00"

    # split into bytes and add 0x for each byte
    bytes_list = [schema_string[i : i + 2] for i in range(0, len(schema_string), 2)]
    for i in range(len(bytes_list)):
        bytes_list[i] = " 0x" + bytes_list[i]

    # construct the string with , delimiter
    schema_string_splitted = ",".join(bytes_list)

    # wrap the long string into 16 bytes per new line
    # each byte is of length 6 including the leading white space. e.g, " 0x1a,"
    # 6*16 = 96
    byte_group_16 = [
        schema_string_splitted[i : i + 96]
        for i in range(0, len(schema_string_splitted), 96)
    ]
    for i in range(len(byte_group_16)):
        # remove leading whitespace for each 16 bytes group string
        byte_group_16[i] = byte_group_16[i][1:]
    # glue back to the long string with new lines
    schema_string_splitted_wrapped = "\n".join(byte_group_16)

    # construct the final long string
    var_name = "const char TOSA_SCHEMA[] = {\n"
    schema_string_splitted_wrapped = (
        var_name + schema_string_splitted_wrapped + " };\n\n"
    )
    schema_size_var = "const size_t TOSA_SCHEMA_SIZE = "
    schema_size_string = schema_size_var + str(len(bytes_list) - 1) + ";\n"
    schema_file_string = schema_string_splitted_wrapped + schema_size_string

    # save to a header file
    with open("../include/tosa_schema.h", "w") as schema_header_file:
        schema_header_file.write(schema_file_string)
