// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/abstract_syntax_list.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/object_pool.h"

namespace {
constexpr u32 OpCapability = 17;
constexpr u32 OpDecorate = 71;
constexpr u32 DecorationBuiltIn = 11;
constexpr u32 BuiltInPrimitiveId = 7;
constexpr u32 BuiltInBaseVertex = 4424;
constexpr u32 BuiltInBaseInstance = 4425;
constexpr u32 BuiltInDrawIndex = 4426;
constexpr u32 CapabilityGeometry = 2;
constexpr u32 CapabilityDrawParameters = 4427;

std::vector<u32> Emit(Shader::Stage stage, bool support_vertex_instance_id,
                      std::initializer_list<Shader::IR::Attribute> loads) {
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::IR::Block block{inst_pool};
    Shader::IR::Program program;
    program.stage = stage;
    program.blocks.push_back(&block);
    program.post_order_blocks.push_back(&block);
    Shader::IR::AbstractSyntaxNode block_node{};
    block_node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
    block_node.data.block = &block;
    Shader::IR::AbstractSyntaxNode return_node{};
    return_node.type = Shader::IR::AbstractSyntaxNode::Type::Return;
    program.syntax_list = {block_node, return_node};
    for (const Shader::IR::Attribute attribute : loads) {
        program.info.loads.Set(attribute);
    }
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010300;
    profile.support_vertex_instance_id = support_vertex_instance_id;
    return Shader::Backend::SPIRV::EmitSPIRV(profile, program);
}

bool HasCapability(const std::vector<u32>& code, u32 capability) {
    for (size_t offset = 5; offset < code.size();) {
        const u32 word_count = code[offset] >> 16;
        if (word_count == 0) {
            break;
        }
        if ((code[offset] & 0xffff) == OpCapability && code[offset + 1] == capability) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

bool HasBuiltIn(const std::vector<u32>& code, u32 builtin) {
    for (size_t offset = 5; offset < code.size();) {
        const u32 word_count = code[offset] >> 16;
        if (word_count == 0) {
            break;
        }
        if ((code[offset] & 0xffff) == OpDecorate && word_count >= 4 &&
            code[offset + 2] == DecorationBuiltIn && code[offset + 3] == builtin) {
            return true;
        }
        offset += word_count;
    }
    return false;
}
} // Anonymous namespace

TEST_CASE("SPIR-V draw parameter built-ins declare DrawParameters",
          "[video_core][spirv][draw_parameters]") {
    using Shader::IR::Attribute;
    struct Case {
        bool support_vertex_instance_id;
        Attribute load;
        u32 builtin;
    };
    for (const Case test : {
             Case{true, Attribute::BaseInstance, BuiltInBaseInstance},
             Case{true, Attribute::BaseVertex, BuiltInBaseVertex},
             Case{true, Attribute::DrawID, BuiltInDrawIndex},
             Case{false, Attribute::BaseInstance, BuiltInBaseInstance},
             Case{false, Attribute::BaseVertex, BuiltInBaseVertex},
             Case{false, Attribute::DrawID, BuiltInDrawIndex},
             Case{false, Attribute::InstanceId, BuiltInBaseInstance},
             Case{false, Attribute::VertexId, BuiltInBaseVertex},
         }) {
        const std::vector<u32> code =
            Emit(Shader::Stage::VertexB, test.support_vertex_instance_id, {test.load});
        REQUIRE(HasBuiltIn(code, test.builtin));
        REQUIRE(HasCapability(code, CapabilityDrawParameters));
    }
}

TEST_CASE("SPIR-V vertex and instance ids alone do not need DrawParameters",
          "[video_core][spirv][draw_parameters]") {
    using Shader::IR::Attribute;
    const std::vector<u32> code =
        Emit(Shader::Stage::VertexB, true, {Attribute::InstanceId, Attribute::VertexId});
    REQUIRE_FALSE(HasBuiltIn(code, BuiltInBaseInstance));
    REQUIRE_FALSE(HasBuiltIn(code, BuiltInBaseVertex));
    REQUIRE_FALSE(HasCapability(code, CapabilityDrawParameters));
}

TEST_CASE("SPIR-V fragment PrimitiveId declares Geometry", "[video_core][spirv][primitive_id]") {
    using Shader::IR::Attribute;
    const std::vector<u32> with_id =
        Emit(Shader::Stage::Fragment, false, {Attribute::PrimitiveId});
    REQUIRE(HasBuiltIn(with_id, BuiltInPrimitiveId));
    REQUIRE(HasCapability(with_id, CapabilityGeometry));

    const std::vector<u32> without_id = Emit(Shader::Stage::Fragment, false, {});
    REQUIRE_FALSE(HasBuiltIn(without_id, BuiltInPrimitiveId));
    REQUIRE_FALSE(HasCapability(without_id, CapabilityGeometry));
}

namespace {
std::vector<u32> EmitMeshCountFixture(Shader::Stage stage, bool guest_writes_count) {
    Shader::ObjectPool<Shader::IR::Inst> pool;
    Shader::IR::Block entry{pool}, body{pool};
    Shader::IR::Program program;
    program.stage = stage;
    program.workgroup_size = {32, 1, 1};
    program.mesh.payload_words = 32;
    program.mesh.max_vertices = 3;
    program.mesh.max_primitives = 1;
    program.info.is_task = stage == Shader::Stage::Task;
    program.info.is_mesh = stage == Shader::Stage::Mesh;
    program.blocks = {&entry, &body};
    program.post_order_blocks = {&body, &entry};
    program.syntax_list = {
        {{.block = &entry}, Shader::IR::AbstractSyntaxNode::Type::Block},
        {{.block = &body}, Shader::IR::AbstractSyntaxNode::Type::Block},
        {{}, Shader::IR::AbstractSyntaxNode::Type::Return}};
    Shader::IR::IREmitter prologue{entry};
    prologue.Prologue();
    Shader::IR::IREmitter ir{body};
    if (stage == Shader::Stage::Task) {
        if (guest_writes_count) {
            ir.IsbeWriteAttribute(ir.Imm32(4), ir.Imm32(7));
        }
        ir.Epilogue();
    }
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010500;
    auto code = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    if (const char* directory = std::getenv("EDEN_MESH_TEST_OUTPUT")) {
        std::filesystem::create_directories(directory);
        const auto name = std::to_string(static_cast<u32>(stage)) +
                          (guest_writes_count ? "-write.spv" : "-empty.spv");
        std::ofstream file{std::filesystem::path{directory} / name, std::ios::binary};
        file.write(reinterpret_cast<const char*>(code.data()),
                   static_cast<std::streamsize>(code.size() * sizeof(u32)));
        REQUIRE(file.good());
    }
    return code;
}
}

TEST_CASE("Task mesh count starts at zero before guest code and emission",
          "[video_core][spirv][mesh]") {
    for (const bool guest_writes : {false, true}) {
        const auto code = EmitMeshCountFixture(Shader::Stage::Task, guest_writes);
        std::map<u32, u32> constants;
        std::map<u32, std::pair<u32, u32>> pointers;
        std::map<u32, u32> loads;
        u32 entry{}, current{}, payload{}, zero_pointer{}, zero_offset{}, barrier_offset{};
        u32 branch_offset{}, guest_offset{}, emit_offset{}, load_offset{}, stores{}, barriers{};
        for (size_t i = 5; i < code.size();) {
            const u32 size = code[i] >> 16;
            REQUIRE(size != 0);
            REQUIRE(i + size <= code.size());
            const u32* w = code.data() + i;
            switch (w[0] & 0xffff) {
            case 43: // OpConstant
                if (size == 4) constants[w[2]] = w[3];
                break;
            case 59: // OpVariable: TaskPayloadWorkgroupEXT
                if (w[3] == 5402) payload = w[2];
                break;
            case 12: // OpExtInst GLSL UMin, used by PayloadWord
                if (size == 7 && w[4] == 38 && constants.contains(w[5]) && constants.contains(w[6])) {
                    constants[w[2]] = std::min(constants.at(w[5]), constants.at(w[6]));
                }
                break;
            case 194: // OpShiftRightLogical: guest byte address -> word
                if (constants.contains(w[3]) && constants.contains(w[4])) {
                    constants[w[2]] = constants.at(w[3]) >> constants.at(w[4]);
                }
                break;
            case 65: // OpAccessChain
                if (size == 5 && constants.contains(w[4])) {
                    pointers[w[2]] = {w[3], constants.at(w[4])};
                }
                break;
            case 248: // OpLabel
                current = w[1];
                if (!entry) entry = current;
                break;
            case 62: // OpStore
                ++stores;
                REQUIRE(pointers.contains(w[1]));
                REQUIRE(pointers.at(w[1]) == std::pair{payload, 1U});
                if (constants.at(w[2]) == 0) {
                    REQUIRE(current == entry);
                    zero_pointer = w[1];
                    zero_offset = static_cast<u32>(i);
                } else {
                    REQUIRE(guest_writes);
                    REQUIRE(constants.at(w[2]) == 7);
                    REQUIRE(current != entry);
                    guest_offset = static_cast<u32>(i);
                }
                break;
            case 224: // OpControlBarrier
                ++barriers;
                REQUIRE(constants.at(w[1]) == 2); // Workgroup execution
                REQUIRE(constants.at(w[2]) == 2); // Workgroup memory
                REQUIRE(constants.at(w[3]) == 0x108); // AcquireRelease | WorkgroupMemory
                if (current == entry) barrier_offset = static_cast<u32>(i);
                break;
            case 249: // Entry branch: sole predecessor of guest block
                REQUIRE(current == entry);
                REQUIRE(branch_offset == 0);
                branch_offset = static_cast<u32>(i);
                break;
            case 61: // OpLoad
                loads[w[2]] = w[3];
                load_offset = static_cast<u32>(i);
                break;
            case 5294: // OpEmitMeshTasksEXT
                REQUIRE(loads.contains(w[1]));
                REQUIRE(pointers.at(loads.at(w[1])) == pointers.at(zero_pointer));
                REQUIRE(constants.at(w[2]) == 1);
                REQUIRE(constants.at(w[3]) == 1);
                REQUIRE(w[4] == payload);
                emit_offset = static_cast<u32>(i);
                break;
            }
            i += size;
        }
        REQUIRE(payload != 0);
        REQUIRE(zero_offset != 0);
        REQUIRE(zero_offset < barrier_offset);
        REQUIRE(barrier_offset < branch_offset);
        REQUIRE(branch_offset < load_offset);
        REQUIRE(load_offset < emit_offset);
        REQUIRE(stores == (guest_writes ? 2 : 1));
        REQUIRE(barriers == 2);
        if (guest_writes) {
            REQUIRE(branch_offset < guest_offset);
            REQUIRE(guest_offset < load_offset);
        }
    }
}

TEST_CASE("Task count initialization leaves mesh and compute prologues unchanged",
          "[video_core][spirv][mesh]") {
    for (const auto stage : {Shader::Stage::Mesh, Shader::Stage::Compute}) {
        const auto code = EmitMeshCountFixture(stage, false);
        u32 stores{}, barriers{}, payload_accesses{};
        for (size_t i = 5; i < code.size();) {
            const u32 size = code[i] >> 16;
            REQUIRE(size != 0);
            REQUIRE(i + size <= code.size());
            const u32 opcode = code[i] & 0xffff;
            stores += opcode == 62;
            barriers += opcode == 224;
            payload_accesses += opcode == 65;
            i += size;
        }
        REQUIRE(stores == (stage == Shader::Stage::Mesh ? 1 : 0));
        REQUIRE(barriers == (stage == Shader::Stage::Mesh ? 1 : 0));
        REQUIRE(payload_accesses == 0);
    }
}
