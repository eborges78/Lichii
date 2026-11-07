// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace {
using namespace Shader;

struct TensorMode {
    std::string previous;
    bool existed{};

    explicit TensorMode(const char* mode) {
        if (const char* value = std::getenv("EDEN_NX2_TENSOR")) {
            previous = value;
            existed = true;
        }
        Set(mode);
    }

    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("EDEN_NX2_TENSOR", value ? value : "");
#else
        if (value) {
            setenv("EDEN_NX2_TENSOR", value, 1);
        } else {
            unsetenv("EDEN_NX2_TENSOR");
        }
#endif
    }

    ~TensorMode() {
        Set(existed ? previous.c_str() : nullptr);
    }
};

struct TensorProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile;

    explicit TensorProgram(u32 count = 1, u32 software_depth = 0, bool storage_output = true,
                           u32 rounding_case = 0) {
        program.stage = Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        program.info.uses_subgroup_shuffles = count != 0 && software_depth != 0;
        program.info.uses_subgroup_invocation_id = count != 0;
        profile.supported_spirv = 0x00010500;
        profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
        profile.max_subgroup_size = 32;
        IR::IREmitter ir{block};
        if (count == 0) {
            return;
        }
        const auto lane = ir.LaneId();
        std::array<IR::U32, 8> inputs;
        for (u32 i = 0; i < inputs.size(); ++i) {
            inputs[i] = software_depth ? ir.IAdd(ir.Imm32(0x3c003c00U + i * 0x01000100U),
                ir.ShiftLeftLogical(lane, ir.Imm32(3))) : ir.IAdd(ir.Imm32((i + 1) * 0x01000100U), lane);
        }
        if (rounding_case != 0) {
            // A, B, C, upper-K8 A. Only the first K component is nonzero, except
            // the cancellation case. All values are binary16 bit patterns.
            constexpr std::array<std::array<u32, 4>, 10> cases{{
                {0x1000, 0x3c00, 0x3c00, 0}, // even tie at 1
                {0x1000, 0x3c00, 0x3c01, 0}, // odd tie at 1 + ulp
                {0x0001, 0x3800, 0, 0},      // half of the minimum subnormal
                {0x0003, 0x3800, 0, 0},      // odd subnormal tie
                {0x8001, 0x3800, 0x8000, 0}, // negative underflow
                {0x7bff, 0x3c00, 0x4c00, 0}, // positive overflow tie
                {0xfbff, 0x3c00, 0xcc00, 0}, // negative overflow tie
                {0x0400, 0x3800, 0, 0},      // normal to subnormal
                {0x1000, 0x3c00, 0x3c00, 0x1000}, // K16 must round halfway
                {0xbc003c00, 0x3c00, 0x0001, 0},  // ordered cancellation
            }};
            const auto& values = cases.at(rounding_case - 1);
            const auto first = ir.IEqual(ir.BitwiseAnd(lane, ir.Imm32(3)), ir.Imm32(0));
            for (u32 i = 0; i < 4; ++i) {
                inputs[i] = IR::U32{ir.Select(first, ir.Imm32(values[i < 2 ? 0 : 3]), ir.Imm32(0))};
            }
            for (u32 i = 4; i < 8; ++i) {
                const u32 half = values[i < 6 ? 1 : 2];
                inputs[i] = ir.Imm32(half | (half << 16));
            }
        }
        for (u32 i = 0; i < count; ++i) {
            const auto result = ir.TensorMulAdd(
                                       ir.CompositeConstruct(inputs[0], inputs[1], inputs[2], inputs[3]),
                                       ir.CompositeConstruct(inputs[4], inputs[5]),
                                       ir.CompositeConstruct(inputs[6], inputs[7]), software_depth);
            inputs[6] = IR::U32{ir.CompositeExtract(result, 0)};
            inputs[7] = IR::U32{ir.CompositeExtract(result, 1)};
        }
        if (software_depth && storage_output) {
            program.info.used_storage_buffer_types = IR::Type::U32;
            program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
            program.info.uses_local_invocation_id = true;
            const auto offset = ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(3));
            block.AppendNewInst(IR::Opcode::WriteStorage32, {ir.Imm32(0), offset, inputs[6]});
            block.AppendNewInst(IR::Opcode::WriteStorage32,
                {ir.Imm32(0), ir.IAdd(offset, ir.Imm32(4)), inputs[7]});
        }
    }

    std::vector<u32> Emit(std::string_view name = {}) {
        auto code = Backend::SPIRV::EmitSPIRV(profile, program);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
            directory && !name.empty()) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
            REQUIRE(file.good());
        }
        return code;
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        REQUIRE(offset + size <= code.size());
        if ((code[offset] & 0xffffU) == opcode) {
            result.emplace_back(code.begin() + offset + 1, code.begin() + offset + size);
        }
        offset += size;
    }
    return result;
}

bool HasCapability(const std::vector<u32>& code, u32 capability) {
    const auto capabilities = Instructions(code, 17);
    return std::ranges::any_of(capabilities, [=](const auto& words) {
        return words[0] == capability;
    });
}

}

TEST_CASE("Software HMMA shares a bounded loop without tensor extensions", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram single{1, depth};
        const auto first = single.Emit(depth == 8 ? "software-k8" : "software-k16");
        REQUIRE_FALSE(HasCapability(first, 5357));
        REQUIRE(Instructions(first, 246).size() == 1); // OpLoopMerge
        REQUIRE(Instructions(first, 57).size() == 1); // OpFunctionCall
        TensorProgram repeated{64, depth};
        const auto many = repeated.Emit(depth == 8 ? "software-k8-many" : "software-k16-many");
        REQUIRE(Instructions(many, 246).size() == 1);
        REQUIRE(Instructions(many, 57).size() == 64);
        // Three fragment-vector constructions, the call and two result extracts
        // take at most 40 words per site; the matrix loop itself is never duplicated.
        REQUIRE(many.size() < first.size() + 64 * 40);
    }
}

TEST_CASE("Software HMMA rounds after each complete unrolled K8 group", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        for (const bool quad : {false, true}) {
            TensorProgram fixture{1, depth};
            fixture.profile.support_quad_shuffles = quad;
            const auto code = fixture.Emit();
            std::map<u32, u32> constants;
            for (const auto& words : Instructions(code, 43)) { // OpConstant
                constants.emplace(words[1], words[2]);
            }
            REQUIRE(Instructions(code, 247).empty()); // No per-pair rounding branch.
            const auto limits = Instructions(code, 176); // OpULessThan
            REQUIRE(limits.size() == 1);
            REQUIRE(constants.at(limits.front()[3]) == depth / 8);
            const auto branches = Instructions(code, 250); // OpBranchConditional
            REQUIRE(branches.size() == 1);
            REQUIRE(branches.front()[0] == limits.front()[1]);
            const u32 body = branches.front()[1];
            const auto loads = Instructions(code, 61); // OpLoad
            const auto counter_load = std::ranges::find_if(loads, [&](const auto& words) {
                return words[1] == limits.front()[2];
            });
            REQUIRE(counter_load != loads.end());
            const u32 counter = (*counter_load)[2];
            const auto additions = Instructions(code, 128); // OpIAdd
            u32 initializations = 0;
            u32 increments = 0;
            for (const auto& store : Instructions(code, 62)) {
                if (store[0] != counter) {
                    continue;
                }
                if (constants.contains(store[1])) {
                    REQUIRE(constants.at(store[1]) == 0);
                    ++initializations;
                } else {
                    const auto increment = std::ranges::find_if(additions, [&](const auto& words) {
                        return words[1] == store[1];
                    });
                    REQUIRE(increment != additions.end());
                    REQUIRE((*increment)[2] == limits.front()[2]);
                    REQUIRE(constants.at((*increment)[3]) == 1);
                    ++increments;
                }
            }
            REQUIRE(initializations == 1);
            REQUIRE(increments == 1);

            u32 block = 0;
            u32 fmas = 0;
            u32 rounded = 0;
            std::vector<u32> fma_ids;
            std::vector<u32> rounded_accumulators;
            for (size_t offset = 5; offset < code.size();) {
                const u32 length = code[offset] >> 16;
                const u32 opcode = code[offset] & 0xffffU;
                const auto words = std::span{code}.subspan(offset + 1, length - 1);
                if (opcode == 248) { // OpLabel
                    block = words[0];
                } else if (opcode == 12 && words[3] == 50) { // GLSL.std.450 Fma
                    REQUIRE(block == body);
                    REQUIRE(rounded == 0);
                    ++fmas;
                    fma_ids.push_back(words[1]);
                } else if (opcode == 12 && words[3] == 2) { // GLSL.std.450 RoundEven
                    REQUIRE(block == body);
                    REQUIRE(fmas == 32); // Eight ordered products per accumulator.
                    ++rounded;
                } else if (opcode == 62 && block == body) {
                    rounded_accumulators.push_back(words[0]);
                }
                offset += length;
            }
            REQUIRE(fmas == 32);
            REQUIRE(rounded == 4);
            REQUIRE(rounded_accumulators.size() == 4);
            std::ranges::sort(rounded_accumulators);
            REQUIRE(std::ranges::adjacent_find(rounded_accumulators) == rounded_accumulators.end());
            const auto decorations = Instructions(code, 71); // OpDecorate
            for (const u32 id : fma_ids) {
                REQUIRE(std::ranges::any_of(decorations, [=](const auto& words) {
                    return words[0] == id && words[1] == 42; // NoContraction
                }));
            }
        }
    }
}

TEST_CASE("Software HMMA emitted exchanges preserve matrix products across both warps",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        for (const u32 subgroup_size : {32U, 64U}) {
            for (const bool quad : {false, true}) {
                TensorProgram fixture{1, depth};
                fixture.profile.max_subgroup_size = subgroup_size;
                fixture.profile.support_quad_shuffles = quad;
                const auto code = fixture.Emit("software-k" + std::to_string(depth) +
                    "-wave" + std::to_string(subgroup_size) + (quad ? "-quad" : "-shuffle"));
                REQUIRE_FALSE(HasCapability(code, 5357));
                REQUIRE(HasCapability(code, 68) == quad); // GroupNonUniformQuad
                REQUIRE(Instructions(code, 246).size() == 1); // OpLoopMerge
                REQUIRE(Instructions(code, 345).size() == (quad ? 8 : 16));
                REQUIRE(Instructions(code, 365).size() == (quad ? 8 : 0));

                // Evaluate the emitted exchange-index expressions, not a duplicate of the
                // emitter's address calculation. Distinct values in each warp expose leaks.
                std::map<u32, std::pair<u32, std::vector<u32>>> definitions;
                for (const u32 opcode : {43U, 61U, 128U, 196U, 199U}) {
                    for (const auto& words : Instructions(code, opcode)) {
                        definitions.emplace(words[1], std::pair{opcode, words});
                    }
                }
                u32 lane_variable = 0;
                for (const auto& words : Instructions(code, 71)) { // BuiltIn SubgroupLocalInvocationId
                    if (words.size() >= 3 && words[1] == 11 && words[2] == 41) {
                        lane_variable = words[0];
                    }
                }
                REQUIRE(lane_variable != 0);
                std::function<u32(u32, u32)> evaluate = [&](u32 id, u32 lane) -> u32 {
                    const auto& [opcode, words] = definitions.at(id);
                    switch (opcode) {
                    case 43: return words[2];
                    case 61:
                        REQUIRE(words[2] == lane_variable);
                        return lane;
                    case 128: return evaluate(words[2], lane) + evaluate(words[3], lane);
                    case 196: return evaluate(words[2], lane) << evaluate(words[3], lane);
                    case 199: return evaluate(words[2], lane) & evaluate(words[3], lane);
                    default: FAIL("Unexpected exchange address opcode"); return 0;
                    }
                };
                std::vector<std::pair<u32, std::vector<u32>>> exchanges;
                for (size_t offset = 5; offset < code.size();) {
                    const u32 length = code[offset] >> 16;
                    const u32 opcode = code[offset] & 0xffffU;
                    if (opcode == 345 || opcode == 365) {
                        exchanges.emplace_back(opcode, std::vector<u32>{
                            code.begin() + offset + 1, code.begin() + offset + length});
                    }
                    offset += length;
                }
                REQUIRE(exchanges.size() == 16);
                const auto source_lane = [&](u32 exchange, u32 lane) {
                    const auto& [opcode, words] = exchanges[exchange];
                    REQUIRE(evaluate(words[2], lane) == 3); // Subgroup scope
                    const u32 index = evaluate(words[4], lane);
                    return opcode == 365 ? (lane & ~3U) + index : index;
                };
                const auto operand = [](u32 lane, u32 word, u32 half) {
                    return std::ldexp(static_cast<float>(static_cast<s32>(
                        (lane * 17 + word * 7 + half * 3) % 61) - 30), -5);
                };
                for (u32 lane = 0; lane < subgroup_size; ++lane) {
                    for (u32 row = 0; row < 2; ++row) {
                        for (u32 col = 0; col < 2; ++col) {
                            float actual = 0;
                            float expected = 0;
                            for (u32 k = 0; k < depth; ++k) {
                                const u32 component = (k % 8) / 2;
                                const u32 a_lane = source_lane(component * 4 + 2 + row, lane);
                                const u32 b_lane = source_lane(component * 4 + col, lane);
                                const u32 expected_a = (lane & ~31U) + (lane & 28U) + component;
                                const u32 expected_b = (lane & ~31U) + (lane & 3U) * 8 + col * 4 + component;
                                REQUIRE(a_lane == expected_a);
                                REQUIRE(b_lane == expected_b);
                                const u32 a_word = row + (k / 8) * 2;
                                const u32 b_word = 4 + k / 8;
                                actual = std::fma(operand(a_lane, a_word, k % 2),
                                                  operand(b_lane, b_word, k % 2), actual);
                                expected = std::fma(operand(expected_a, a_word, k % 2),
                                                    operand(expected_b, b_word, k % 2), expected);
                            }
                            REQUIRE(actual == expected);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Software HMMA uses quad broadcasts only in supported compute stages",
          "[video_core][tensor][ampere]") {
    TensorProgram fixture{1, 8, false};
    fixture.profile.support_quad_shuffles = true;
    fixture.program.stage = Stage::VertexB;
    fixture.profile.supported_subgroup_stages |= 1U << static_cast<u32>(Stage::VertexB);
    const auto code = fixture.Emit();
    REQUIRE(Instructions(code, 365).empty());
    REQUIRE(Instructions(code, 345).size() == 16);
}

TEST_CASE("Tensor f16 conversions request precise intermediate rounding and preserve fallback",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        // Case 0 varies A/B/C by lane; verify_tensor_fixture.py selects its reference
        // when --rounding-case is omitted. Cases 1..10 exercise rounding boundaries.
        for (u32 rounding_case = 0; rounding_case <= 10; ++rounding_case) {
            for (const bool native_round : {false, true}) {
                TensorProgram fixture{1, depth, true, rounding_case};
                fixture.profile.support_quad_shuffles = true;
                fixture.profile.support_float_controls = true;
                fixture.profile.support_tensor_float16_rounding = native_round;
                fixture.profile.support_fp16_signed_zero_nan_preserve = true;
                const auto code = fixture.Emit("round-case" + std::to_string(rounding_case) +
                    "-k" + std::to_string(depth) + (native_round ? "-fp16" : "-software"));
                const auto conversions = Instructions(code, 115); // OpFConvert
                REQUIRE(conversions.size() == (native_round ? 8 : 0));
                const auto decorations = Instructions(code, 71); // OpDecorate
                for (const auto& conversion : conversions) {
                    REQUIRE(std::ranges::any_of(decorations, [&](const auto& words) {
                        return words[0] == conversion[1] && words[1] == 42; // NoContraction
                    }));
                }
                for (size_t index = 0; index < conversions.size(); index += 2) {
                    const auto types = Instructions(code, 22); // OpTypeFloat
                    for (const auto& [offset, width] : {std::pair{0U, 16U}, std::pair{1U, 32U}}) {
                        REQUIRE(std::ranges::any_of(types, [&](const auto& words) {
                            return words[0] == conversions[index + offset][0] && words[1] == width;
                        }));
                    }
                    REQUIRE(conversions[index + 1][2] == conversions[index][1]);
                }
                if (native_round) {
                    u32 fmas_before_rounding = 0;
                    for (size_t offset = 5; offset < code.size();) {
                        const u32 length = code[offset] >> 16;
                        const u32 opcode = code[offset] & 0xffff;
                        if (opcode == 12 && code[offset + 4] == 50) { // GLSL.std.450 Fma
                            ++fmas_before_rounding;
                        } else if (opcode == 115) {
                            REQUIRE(fmas_before_rounding == 32);
                        }
                        offset += length;
                    }
                }
                REQUIRE(HasCapability(code, 9) == native_round); // Float16
                const auto modes = Instructions(code, 16); // OpExecutionMode
                for (const u32 mode : {4459U, 4461U, 4462U}) { // preserve denorm/sign/NaN, RTE
                    REQUIRE(std::ranges::any_of(modes, [=](const auto& words) {
                        return words.size() == 3 && words[1] == mode && words[2] == 16;
                    }) == native_round);
                }
                // The f32->f16->f32 pair cannot become an unrounded f32 accumulator.
                if (native_round) {
                    const auto types = Instructions(code, 22); // OpTypeFloat
                    const auto converts = Instructions(code, 115);
                    for (size_t i = 0; i < converts.size(); i += 2) {
                        const auto half_type = std::ranges::find_if(types, [&](const auto& words) {
                            return words[0] == converts[i][0];
                        });
                        REQUIRE(half_type != types.end());
                        REQUIRE((*half_type)[1] == 16);
                        REQUIRE(converts[i + 1][2] == converts[i][1]);
                    }
                }
            }
        }
    }
    for (const u32 blocker : {0U, 1U, 2U, 3U, 4U}) {
        TensorProgram fixture{1, 16};
        fixture.profile.support_tensor_float16_rounding = true;
        fixture.profile.support_float_controls = true;
        if (blocker == 0) fixture.profile.support_tensor_float16_rounding = false;
        if (blocker == 1) fixture.profile.support_float_controls = false;
        if (blocker == 2) fixture.profile.has_broken_fp16_float_controls = true;
        if (blocker == 3) fixture.program.info.uses_fp16_denorms_flush = true;
        if (blocker == 4) fixture.program.info.uses_fp64 = true;
        REQUIRE(Instructions(fixture.Emit(), 115).empty());
    }
}

TEST_CASE("Captured Bananza matrix shaders lower without an environment override",
          "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    for (const Ampere::Instruction instruction : {
             Ampere::Instruction{{0x102c723c, 0x30, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1020723c, 0x24, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1418723c, 0x1c, 0x8ff, 0x04aff000}}}) {
        const std::array code{instruction,
                              Ampere::Instruction{{0x794d, 0, 0x03800000, 0}}};
        const auto lowered = Ampere::LowerToMaxwell(code, false, true);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(Maxwell::Decode(lowered.instructions[0]) == Maxwell::Opcode::TENSOR_MUL_ADD);
    }
}

TEST_CASE("Native matrix lowering selects compact software helpers without legacy toggles", "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    std::array<Ampere::Instruction, 2> code{{
        {{0x0810723c, 12, 0x8ff, 0}}, {{0x0000794d, 0, 0x03800000, 0}}}};
    const auto supported = Ampere::LowerToMaxwell(code, false, false);
    REQUIRE(static_cast<bool>(supported));
    REQUIRE(supported.instructions.size() > 100);
    const auto compact = Ampere::LowerToMaxwell(code, false, true);
    REQUIRE(compact);
    REQUIRE(compact.instructions.size() == 2);
    REQUIRE(Maxwell::Decode(compact.instructions.front()) == Maxwell::Opcode::TENSOR_MUL_ADD);
    REQUIRE((compact.instructions.front() & (1ULL << 48)) != 0);
    for (const char* legacy : {"fast", "stub", "off"}) {
        TensorMode::Set(legacy);
        REQUIRE(Ampere::LowerToMaxwell(code, false, true).instructions == compact.instructions);
        REQUIRE(Ampere::LowerToMaxwell(code).instructions == supported.instructions);
    }
    code[0].words[0] &= ~0x7000U;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[0] |= 0x7000;
    code[0].words[3] = 1;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[3] = 0;
    code[0].words[0] += 1U << 24;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
}
