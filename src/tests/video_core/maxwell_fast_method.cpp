// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace VideoCommon {

struct GpuSplitTestAccess {
    static void Initialize(GpuSplit& split, Core::System& system, u64 position = 0) {
        split.system = &system;
        split.ring = std::make_unique<u64[]>(GpuSplit::RingWords);
        split.write_position = position;
        split.consumed_cache = position;
        split.published_local = position;
        split.publish_every = GpuSplit::RingWords;
    }

    static void EnableDmaSnapshots(GpuSplit& split, int mode = 10) {
        split.dma_snapshot_mode = mode;
        split.dma_snapshot_writes.Reset();
    }

    static u64 End(const GpuSplit& split) {
        return split.write_position;
    }

    static void Start(GpuSplit& split, Core::System& system) {
        Initialize(split, system);
        split.active = true;
        split.back_thread = std::jthread([&split](std::stop_token stop) { split.BackLoop(stop); });
    }

    static u64 NearWrap() {
        return GpuSplit::RingWords - 1;
    }

    static u64 Execute(GpuSplit& split, u64 position, u64 available) {
        const size_t index = position & GpuSplit::RingMask;
        const size_t size = split.Execute(
            &split.ring[index], std::min<u64>(available - position, GpuSplit::RingWords - index));
        return position + (size != 0 ? size : GpuSplit::RingWords - index);
    }
};

} // namespace VideoCommon

namespace {

using Tegra::Engines::Maxwell3D;

constexpr u32 CbSize = MAXWELL3D_REG_INDEX(const_buffer.size);
constexpr u32 CbAddressHigh = MAXWELL3D_REG_INDEX(const_buffer.address_high);
constexpr u32 CbAddressLow = MAXWELL3D_REG_INDEX(const_buffer.address_low);
constexpr u32 CbOffset = MAXWELL3D_REG_INDEX(const_buffer.offset);
constexpr u32 CbData = MAXWELL3D_REG_INDEX(const_buffer.buffer);
constexpr u32 DrawBegin = MAXWELL3D_REG_INDEX(draw.begin);
constexpr u32 DrawEnd = MAXWELL3D_REG_INDEX(draw.end);
constexpr u32 ShadowControl = MAXWELL3D_REG_INDEX(shadow_ram_control);

// Executable methods without memory or macro effects a null rasterizer cannot take.
constexpr std::array<u32, 16> executable_methods{
    DrawBegin,
    DrawEnd,
    MAXWELL3D_REG_INDEX(vertex_buffer.first),
    MAXWELL3D_REG_INDEX(vertex_buffer.count),
    MAXWELL3D_REG_INDEX(index_buffer.first),
    MAXWELL3D_REG_INDEX(index_buffer.count),
    MAXWELL3D_REG_INDEX(topology_override),
    MAXWELL3D_REG_INDEX(bind_groups[0].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[1].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[2].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[3].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[4].raw_config),
    MAXWELL3D_REG_INDEX(vertex_array_instance_first),
    MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent),
    Maxwell3D::RootTables::SelectMethod,
    Maxwell3D::RootTables::LoadMethod,
};

struct Fixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Maxwell3D slow{memory};
    Maxwell3D fast{memory};
    std::vector<u32> state_methods;

    Fixture() {
        slow.fast_maxwell3d = false;
        fast.fast_maxwell3d = true;
        for (auto* engine : {&slow, &fast}) {
            engine->BindRasterizer(&rasterizer);
            VideoCommon::Dirty::SetupDirtyFlags(engine->dirty.tables);
            // Give every register a flag so that a missed or extra write shows.
            for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
                if (engine->dirty.tables[1][method] == 0) {
                    engine->dirty.tables[1][method] =
                        static_cast<u8>(VideoCommon::Dirty::LastCommonEntry + method % 61);
                }
            }
            engine->dirty.flags.reset();
            // An unmapped buffer: the upload writes nothing, the offset still advances.
            engine->CallMethod(system, CbAddressHigh, 0x1, true);
            engine->CallMethod(system, CbAddressLow, 0x2000'0000, true);
            engine->CallMethod(system, CbSize, 0x10000, true);
        }
        for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
            if (!slow.execution_mask[method] && (method < CbSize || method > CbAddressLow)) {
                state_methods.push_back(method);
            }
        }
    }

    void Call(u32 method, u32 argument, bool is_last) {
        slow.CallMethod(system, method, argument, is_last);
        fast.CallMethod(system, method, argument, is_last);
    }

    void CallMulti(u32 method, const std::vector<u32>& words) {
        const u32 amount = static_cast<u32>(words.size());
        slow.CallMultiMethod(system, method, words.data(), amount, amount);
        fast.CallMultiMethod(system, method, words.data(), amount, amount);
    }

    void Compare() const {
        REQUIRE(fast.regs.reg_array == slow.regs.reg_array);
        REQUIRE(fast.shadow_state.reg_array == slow.shadow_state.reg_array);
        REQUIRE(fast.shadow_state.shadow_ram_control == slow.shadow_state.shadow_ram_control);
        REQUIRE(fast.dirty.flags == slow.dirty.flags);
        const auto& a = fast.draw_manager.draw_state;
        const auto& b = slow.draw_manager.draw_state;
        REQUIRE(a.topology == b.topology);
        REQUIRE(a.draw_mode == b.draw_mode);
        REQUIRE(a.draw_indexed == b.draw_indexed);
        REQUIRE(a.draw_element_count == b.draw_element_count);
        REQUIRE(a.base_index == b.base_index);
        REQUIRE(a.base_instance == b.base_instance);
        REQUIRE(a.instance_count == b.instance_count);
        REQUIRE(a.drawn_instances == b.drawn_instances);
        REQUIRE(std::memcmp(&a.vertex_buffer, &b.vertex_buffer, sizeof(a.vertex_buffer)) == 0);
        REQUIRE(std::memcmp(&a.index_buffer, &b.index_buffer, sizeof(a.index_buffer)) == 0);
        for (size_t stage = 0; stage < fast.state.shader_stages.size(); ++stage) {
            for (size_t slot = 0; slot < Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& x = fast.state.shader_stages[stage].const_buffers[slot];
                const auto& y = slow.state.shader_stages[stage].const_buffers[slot];
                REQUIRE(x.address == y.address);
                REQUIRE(x.size == y.size);
                REQUIRE(x.enabled == y.enabled);
            }
        }
        REQUIRE(fast.root_tables.tables == slow.root_tables.tables);
        REQUIRE(fast.root_tables.selected == slow.root_tables.selected);
        REQUIRE(fast.root_tables.offset == slow.root_tables.offset);
    }
};

u32 RandomArgument(std::mt19937& rng, u32 method, const Maxwell3D& engine) {
    // Mostly values the register already holds or small ones, as the macros write them.
    switch (rng() % 4) {
    case 0:
        return engine.regs.reg_array[method];
    case 1:
        return rng() % 4;
    default:
        break;
    }
    if (method == DrawBegin) {
        // Instance id in bits 26-27: first, subsequent (an instanced loop) or unchanged.
        return (static_cast<u32>(rng() % 3) << 26) | (rng() % 8);
    }
    if (method == Maxwell3D::RootTables::SelectMethod) {
        return (rng() % 8) | ((rng() % 64 * 4) << 8);
    }
    return rng() % 64;
}

} // Anonymous namespace

TEST_CASE("Maxwell3D fast method dispatch matches CallMethod on random method streams",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    for (const u32 seed : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        CAPTURE(seed);
        std::mt19937 rng{seed};
        for (u32 step = 0; step < 20000; ++step) {
            CAPTURE(step);
            const u32 kind = rng() % 100;
            if (kind < 45) {
                const u32 method = f.state_methods[rng() % f.state_methods.size()];
                f.Call(method, RandomArgument(rng, method, f.slow), rng() % 2 == 0);
            } else if (kind < 75) {
                const u32 method = executable_methods[rng() % executable_methods.size()];
                f.Call(method, RandomArgument(rng, method, f.slow), rng() % 2 == 0);
            } else if (kind < 85) {
                f.Call(CbData + rng() % 16, rng(), true);
            } else if (kind < 88) {
                f.Call(CbOffset, (rng() % 0x4000) * 4, true);
            } else if (kind < 92) {
                std::vector<u32> words(1 + rng() % 8);
                for (u32& word : words) {
                    word = rng() % 3 == 0 ? rng() : rng() % 4;
                }
                const u32 method = rng() % 2 == 0
                                       ? CbData + rng() % 16
                                       : f.state_methods[rng() % f.state_methods.size()];
                f.CallMulti(method, words);
            } else if (kind < 94) {
                // Track, track with filter, passthrough or replay.
                f.Call(ShadowControl, rng() % 4, true);
            } else if (kind < 97) {
                for (auto* engine : {&f.slow, &f.fast}) {
                    engine->dirty.flags.reset();
                }
            } else {
                for (auto* engine : {&f.slow, &f.fast}) {
                    engine->draw_manager.DrawDeferred(*engine);
                }
            }
            f.Compare();
        }
    }
}

TEST_CASE("Maxwell3D fast method dispatch issues deferred instanced draws as CallMethod does",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    constexpr u32 subsequent = 1u << 26;
    const auto draw = [&f](u32 begin) {
        f.Call(DrawBegin, begin, true);
        f.Call(MAXWELL3D_REG_INDEX(vertex_buffer.first), 0, true);
        f.Call(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3, true);
        f.Call(DrawEnd, 0, true);
        f.Compare();
    };
    draw(0);
    draw(subsequent);
    draw(subsequent);
    REQUIRE(f.fast.draw_manager.HasDeferredDraw());
    // A constant buffer upload, a bind and state changes each end the loop.
    f.Call(CbData, 7, true);
    f.Compare();
    REQUIRE_FALSE(f.fast.draw_manager.HasDeferredDraw());
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 1 | (3 << 4), true);
    f.Compare();
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(topology_override), 5, true);
    f.Compare();
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(global_base_instance_index), 9, true);
    f.Compare();
    REQUIRE_FALSE(f.fast.draw_manager.HasDeferredDraw());
}

TEST_CASE("Maxwell3D constant buffer word runs match single CallMethod writes",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    std::mt19937 rng{11};
    constexpr u32 subsequent = 1u << 26;
    for (u32 step = 0; step < 20000; ++step) {
        CAPTURE(step);
        switch (rng() % 8) {
        case 0:
            f.Call(ShadowControl, rng() % 4, true);
            break;
        case 1:
            // Near the end too, where words are dropped.
            f.Call(CbSize, rng() % 2 == 0 ? 0x10000 : 0x100 + rng() % 0x40, true);
            break;
        case 2:
            f.Call(CbOffset, rng() % 3 == 0 ? 0xF0 + (rng() % 8) * 4 : (rng() % 0x400) * 4, true);
            break;
        case 3:
            // An instanced loop the first word ends.
            f.Call(DrawBegin, rng() % 2 == 0 ? 0 : subsequent, true);
            f.Call(DrawEnd, 0, true);
            break;
        default: {
            const u32 count = 1 + rng() % 64;
            std::vector<u32> methods(count);
            std::vector<u32> arguments(count);
            const u32 first = CbData + rng() % 16;
            for (u32 i = 0; i < count; ++i) {
                methods[i] = rng() % 2 == 0 ? first : CbData + (first - CbData + i) % 16;
                arguments[i] = rng() % 2 == 0 ? rng() : rng() % 4;
            }
            for (u32 i = 0; i < count; ++i) {
                f.slow.CallMethod(f.system, methods[i], arguments[i], true);
            }
            f.fast.CallConstBufferWords(f.system, methods.data(), arguments.data(), count);
            break;
        }
        }
        f.Compare();
    }
}

TEST_CASE("GpuSplit CB batching respects device remaps within GPU big pages",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    auto* replacement = f.physical.GetPointerFromRaw<u8>(0x30000);
    for (u64 page = 0; page < 16; ++page) {
        table.entries.GetAndFault(4 + page)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing) - 0x4000);
    }
    table.entries.GetAndFault(0x20).Store(false, Common::PageType::Memory, 1,
                                        reinterpret_cast<uintptr_t>(replacement) - 0x20000);
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(0x10000, 0x10000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(0x10000, 0x4000, 0x10000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.device_memory.Unmap(0x11000, 0x1000);
    f.device_memory.Map(0x11000, 0x20000, 0x1000, asid);
    REQUIRE(f.memory.GetPointer(0x100ffc) == backing + 0xffc);
    REQUIRE(f.memory.GetPointer(0x101000) == replacement);

    for (auto* engine : {&f.slow, &f.fast}) {
        std::memset(backing, 0, 0x10000);
        std::memset(replacement, 0, 0x1000);
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.MacroWrite(engine, CbAddressHigh, 0);
        split.MacroWrite(engine, CbAddressLow, 0x100ffc);
        split.MacroWrite(engine, CbSize, 8);
        split.MacroWrite(engine, CbOffset, 0);
        split.MacroWrite(engine, CbData, 0x12345678);
        split.MacroWrite(engine, CbData + 1, 0x87654321);
        u64 position{};
        while (position < Access::End(split)) {
            position = Access::Execute(split, position, Access::End(split));
        }
        u32 first{}, second{}, stale{};
        std::memcpy(&first, backing + 0xffc, sizeof(first));
        std::memcpy(&second, replacement, sizeof(second));
        std::memcpy(&stale, backing + 0x1000, sizeof(stale));
        CAPTURE(engine == &f.fast);
        CAPTURE(stale);
        REQUIRE(first == 0x12345678);
        REQUIRE(second == 0x87654321);
        REQUIRE(stale == 0);
        REQUIRE(engine->regs.const_buffer.offset == 8);
    }
    f.Compare();
}

TEST_CASE("GpuSplit dispatch preserves mapped uploads and macro boundaries",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr DAddr device_address = 0x10000;
    for (u64 page = 0; page < 3; ++page) {
        const VAddr virtual_address = 0x4000 + page * 0x1000;
        auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000 + page * 0x8000);
        table.entries.GetAndFault(virtual_address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing) - virtual_address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x3000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(device_address, 0x4000, 0x3000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x3000, Tegra::PTEKind::PITCH, false);

    struct Snapshot {
        std::array<u8, 0x3000> bytes{};
        std::array<u32, Maxwell3D::Regs::NUM_REGS> regs{};
        std::array<u32, Maxwell3D::Regs::NUM_REGS> shadow{};
        Maxwell3D::DirtyState::Flags dirty{};
        GPUVAddr segment{};
    };
    const auto snapshot = [&](const Maxwell3D& engine) {
        Snapshot result;
        f.memory.ReadBlockUnsafe(gpu_address, result.bytes.data(), result.bytes.size());
        result.regs = engine.regs.reg_array;
        result.shadow = engine.shadow_state.reg_array;
        result.dirty = engine.dirty.flags;
        result.segment = engine.current_dma_segment;
        return result;
    };

    for (const bool wrap : {false, true}) {
        CAPTURE(wrap);
        std::vector<Snapshot> expected;
        for (auto* engine : {&f.slow, &f.fast}) {
            VideoCommon::GpuSplit split;
            struct ProbeEngine final : Tegra::Engines::EngineInterface {
                u32 writes{};
                void CallMethod(Core::System&, u32, u32, bool) override {
                    ++writes;
                }
                void CallMultiMethod(Core::System&, u32, const u32*, u32 amount, u32) override {
                    writes += amount;
                }
            } probe;
            const u64 start = wrap ? Access::NearWrap() : 0;
            Access::Initialize(split, f.system, start);
            std::vector<u64> boundaries;
            std::array<u8, 0x3000> cleared{};
            f.memory.WriteBlockUnsafe(gpu_address, cleared.data(), cleared.size());
            engine->current_dma_segment = 0;
            engine->dirty.flags.reset();
            const auto state = [&](u32 method, u32 value) {
                split.Method(engine, method, value, true, nullptr);
                boundaries.push_back(Access::End(split));
            };
            const auto direct = [&](u32 method, u32 value) {
                split.MacroWrite(engine, method, value);
                boundaries.push_back(Access::End(split));
            };
            size_t observed{};
            const auto inspect = [&] {
                split.Defer([&, engine] {
                    auto actual = snapshot(*engine);
                    if (engine == &f.slow) {
                        expected.push_back(std::move(actual));
                    } else {
                        REQUIRE(observed < expected.size());
                        const auto& reference = expected[observed];
                        REQUIRE(actual.bytes == reference.bytes);
                        REQUIRE(actual.regs == reference.regs);
                        REQUIRE(actual.shadow == reference.shadow);
                        REQUIRE(actual.dirty == reference.dirty);
                        REQUIRE(actual.segment == reference.segment);
                    }
                    ++observed;
                });
                boundaries.push_back(Access::End(split));
            };
            state(ShadowControl, 0);
            state(CbAddressHigh, 0);
            state(CbAddressLow, static_cast<u32>(gpu_address));
            state(CbSize, 0x3000);
            state(CbOffset, 0xffb);
            direct(MAXWELL3D_REG_INDEX(global_base_instance_index), 17);
            inspect();
            for (u32 i = 0; i < 129; ++i) {
                if (i % 2 == 0) {
                    direct(CbData + i % 16, 0x12340000 + i);
                } else {
                    state(CbData + i % 16, 0x56780000 + i);
                }
            }
            inspect();
            state(ShadowControl, 3);
            state(CbOffset, 0x1ffd);
            for (u32 i = 0; i < 33; ++i) {
                direct(CbData + i % 16, 0xdeadbeef);
            }
            inspect();
            state(ShadowControl, 2);
            state(CbOffset, 0x1ffd);
            direct(DrawBegin, 0);
            direct(DrawEnd, 0);
            direct(DrawBegin, 1u << 26);
            direct(DrawEnd, 0);
            direct(CbData, 0x11223344);
            direct(CbData + 1, 0xaabbccdd);
            const GPUVAddr segment = 0x12345678;
            split.Method(engine, CbData + 2, 0x87654321, false, &segment);
            boundaries.push_back(Access::End(split));
            direct(CbData + 3, 0x76543210);
            split.MacroWrite(&probe, 0x10, 42);
            boundaries.push_back(Access::End(split));
            direct(CbData + 4, 0x65432100);
            const std::array<u32, 3> multi{19, 20, 21};
            split.MultiMethod(engine, CbData, multi.data(), 3, 3, &segment);
            boundaries.push_back(Access::End(split));
            inspect();
            state(CbSize, 7);
            state(CbOffset, 0);
            direct(CbData, 0x99aabbcc);
            direct(CbData + 1, 0xddeeff00);
            inspect();

            u64 position = start;
            if (wrap) {
                for (const u64 available : boundaries) {
                    while (position < available) {
                        position = Access::Execute(split, position, available);
                        REQUIRE(position <= available);
                    }
                }
            } else {
                while (position < Access::End(split)) {
                    position = Access::Execute(split, position, Access::End(split));
                    REQUIRE(position <= Access::End(split));
                }
            }
            REQUIRE(observed == expected.size());
            REQUIRE(probe.writes == 1);
        }
        f.Compare();
    }
}

TEST_CASE("GpuSplit CB batching preserves words at the GPU address space end",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(backing) - 0x4000);
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(0x10000, 0x1000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(0x10000, 0x4000, 0x1000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(0xfffff000, 0x10000, 0x1000, Tegra::PTEKind::PITCH, false);
    for (auto* engine : {&f.slow, &f.fast}) {
        std::memset(backing, 0, 0x1000);
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.MacroWrite(engine, CbAddressHigh, 0);
        split.MacroWrite(engine, CbAddressLow, 0xfffffffcu);
        split.MacroWrite(engine, CbSize, 8);
        split.MacroWrite(engine, CbOffset, 0);
        split.MacroWrite(engine, CbData, 0x12345678);
        split.MacroWrite(engine, CbData + 1, 0x87654321);
        u64 position{};
        while (position < Access::End(split)) {
            position = Access::Execute(split, position, Access::End(split));
        }
        u32 word{};
        std::memcpy(&word, backing + 0xffc, sizeof(word));
        CAPTURE(engine == &f.fast);
        REQUIRE(word == 0x12345678);
        REQUIRE(engine->regs.const_buffer.offset == 8);
    }
    f.Compare();
}

TEST_CASE("Maxwell3D fast method dispatch cost", "[.][maxwell_fast_method_bench]") {
    Fixture f;
    // Per draw, as the back thread sees Ocarina's: state writes the front did not drop, a
    // constant buffer update, three binds and a draw.
    std::vector<std::pair<u32, u32>> stream;
    std::mt19937 rng{1};
    for (u32 draw = 0; draw < 4096; ++draw) {
        for (u32 i = 0; i < 12; ++i) {
            stream.emplace_back(f.state_methods[rng() % 512], rng() % 8);
        }
        stream.emplace_back(CbOffset, 0x640);
        stream.emplace_back(CbData, draw);
        stream.emplace_back(CbData + 1, 0);
        stream.emplace_back(CbData + 2, draw * 3);
        for (const u32 stage : {0u, 2u, 4u}) {
            stream.emplace_back(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config) + stage * 8,
                                1 | (rng() % 8) << 4);
        }
        stream.emplace_back(DrawBegin, 0);
        stream.emplace_back(MAXWELL3D_REG_INDEX(vertex_buffer.first), rng() % 4);
        stream.emplace_back(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3 + rng() % 4);
        stream.emplace_back(DrawEnd, 0);
    }
    const auto run = [&](Maxwell3D& engine) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 round = 0; round < 50; ++round) {
            for (const auto& [method, argument] : stream) {
                engine.CallMethod(f.system, method, argument, true);
            }
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               (50.0 * static_cast<double>(stream.size()));
    };
    run(f.slow);
    run(f.fast);
    const double slow_ns = run(f.slow);
    const double fast_ns = run(f.fast);
    WARN("CallMethod: " << slow_ns << " ns per method, fast path " << fast_ns << " ns");

    // Three constant buffer words (a draw macro's c[0][0x640..0x648]) one by one or as a run.
    constexpr std::array<u32, 3> methods{CbData, CbData + 1, CbData + 2};
    constexpr std::array<u32, 3> arguments{1, 2, 3};
    const auto words = [&](bool as_run) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 i = 0; i < 1'000'000; ++i) {
            f.fast.CallMethod(f.system, CbOffset, 0x640, true);
            if (as_run) {
                f.fast.CallConstBufferWords(f.system, methods.data(), arguments.data(), 3);
            } else {
                for (u32 w = 0; w < 3; ++w) {
                    f.fast.CallMethod(f.system, methods[w], arguments[w], true);
                }
            }
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               1'000'000.0;
    };
    words(false);
    words(true);
    const double single_ns = words(false);
    const double run_ns = words(true);
    WARN("Three constant buffer words: " << single_ns << " ns one by one, " << run_ns
                                         << " ns as a run (unmapped buffer)");
}

TEST_CASE("Full split drains preserve aliased uploads, partial coverage and queued remaps",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr VAddr cpu_address = 0x4000;
    constexpr DAddr device_address = 0x10000;
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr GPUVAddr alias = 0x200000;
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    for (u64 page = 0; page < 2; ++page) {
        const VAddr address = cpu_address + page * 0x1000;
        table.entries.GetAndFault(address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing + page * 0x1000) - address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    f.device_memory.Map(device_address, cpu_address, 0x2000, asid);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x2000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x2000, Tegra::PTEKind::PITCH, false);
    f.memory.Map(alias, device_address, 0x1000, Tegra::PTEKind::PITCH, false);
    std::memset(backing, 0, 0x2000);

    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    Access::EnableDmaSnapshots(split);
    SCOPE_EXIT { split.Stop(); };
    const auto state = [&](u32 method, u32 value) {
        split.Method(&f.fast, method, value, true, nullptr);
    };
    const auto upload = [&](GPUVAddr address, u32 size, std::vector<u32> words) {
        state(ShadowControl, 0);
        state(CbAddressHigh, static_cast<u32>(address >> 32));
        state(CbAddressLow, static_cast<u32>(address));
        state(CbSize, size);
        state(CbOffset, 0);
        split.MultiMethod(&f.fast, CbData, words.data(), static_cast<u32>(words.size()),
                          static_cast<u32>(words.size()), nullptr);
        std::ranges::fill(words, 0xdeadbeef);
    };

    upload(gpu_address, 16, {1, 2, 3, 4});
    upload(alias + 4, 4, {20, 30});
    upload(gpu_address + 0x1000, 8, {50, 60});
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    std::array<u32, 4> read{};
    f.memory.ReadBlockUnsafe(gpu_address, read.data(), sizeof(read));
    REQUIRE(read == std::array<u32, 4>{1, 20, 3, 4});
    REQUIRE(f.memory.Read<u32>(alias + 4) == 20);
    REQUIRE(f.memory.Read<u32>(gpu_address + 0x1000) == 50);

    // A snapshot certifies completed GPU work, not a copy of RAM: CPU updates remain visible
    // through both GPU aliases without another writer record.
    const u64 ready = split.DmaSnapshotWrites().Stamp();
    const u32 cpu_value = 123;
    std::memcpy(backing + 8, &cpu_value, sizeof(cpu_value));
    REQUIRE_FALSE(split.DmaSnapshotWrites().ChangedSince(ready));
    REQUIRE(f.memory.Read<u32>(gpu_address + 8) == cpu_value);
    REQUIRE(f.memory.Read<u32>(alias + 8) == cpu_value);
    const u64 mapping = f.memory.MappingGeneration();
    split.Defer([&] {
        f.memory.Map(alias, device_address + 0x1000, 0x1000, Tegra::PTEKind::PITCH, false);
    });
    REQUIRE(split.DmaSnapshotWrites().ChangedSince(ready));
    upload(alias, 4, {70});
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    REQUIRE(f.memory.Read<u32>(gpu_address) == 1);
    REQUIRE(f.memory.Read<u32>(gpu_address + 0x1000) == 70);
    REQUIRE(f.memory.Read<u32>(alias + 4) == 60);
    REQUIRE(f.memory.MappingGeneration() > mapping);
    const u64 before_stop = split.DmaSnapshotWrites().Stamp();
    split.Stop();
    REQUIRE(split.DmaSnapshotMode() == 0);
    REQUIRE(split.DmaSnapshotWrites().ChangedSince(before_stop));
}

TEST_CASE("Split consumption does not complete deferred downstream callbacks",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT { split.Stop(); };
    std::vector<std::function<void()>> deferred;
    std::array<u32, 3> output{};
    for (u32 i = 0; i < output.size(); ++i) {
        split.Defer([&, i] { deferred.emplace_back([&, i] { output[i] = i + 1; }); });
    }
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    REQUIRE(deferred.size() == 3);
    REQUIRE(output == std::array<u32, 3>{});
    split.Run([&] {
        for (auto& operation : deferred) {
            operation();
        }
        deferred.clear();
    });
    REQUIRE(output == std::array<u32, 3>{1, 2, 3});
}

TEST_CASE("Split restart resets writer positions without replaying old releases",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    u32 released{};
    u32 release_count{};
    for (u32 session = 1; session <= 3; ++session) {
        Access::Start(split, f.system);
        SCOPE_EXIT { split.Stop(); };
        REQUIRE(Access::End(split) == 0);
        REQUIRE_FALSE(split.WaitForWriters());
        REQUIRE_FALSE(split.WaitForSync());
        split.Defer([&, session] {
            released = session;
            ++release_count;
        });
        split.NoteWriter();
        split.NoteSync();
        split.Drain(VideoCommon::DrainSite::Acquire);
        REQUIRE(released == session);
        REQUIRE(release_count == session);
        split.Stop();
        REQUIRE_FALSE(split.Active());
        REQUIRE_FALSE(split.BackRunning());
        REQUIRE(Access::End(split) == 0);
    }
}

TEST_CASE("Maxwell3D takes its macros back from the split's front once the split stops",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    constexpr u32 StartAddressPointer = MAXWELL3D_REG_INDEX(load_mme.start_address_ptr);
    constexpr u32 StartAddress = MAXWELL3D_REG_INDEX(load_mme.start_address);
    const u32 state = f.state_methods.front();
    // The front takes the macros over and tracks a state write in its shadow RAM copy.
    engine.FrontCallMethod(f.system, split, state, 0x1234, true);
    REQUIRE(engine.shadow_state.reg_array[state] != 0x1234);
    // The split stopped: the next engine method runs on this thread, not the split's back.
    engine.CallMethod(f.system, StartAddressPointer, 0, true);
    engine.CallMethod(f.system, StartAddress, 0x40, true);
    CHECK(engine.shadow_state.reg_array[state] == 0x1234);
    CHECK(engine.GetNativeMacroPositions()[0] == 0x40);
}

TEST_CASE("Maxwell3D keeps state written after the split stops over the front's shadow RAM",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    const u32 state = f.state_methods.front();
    engine.FrontCallMethod(f.system, split, state, 0x1234, true);
    // After the stop, DmaPusher sinks a state write and consumes the sink before the next
    // executable method: the handback must come first so that write wins.
    engine.method_sink.emplace_back(state, 0x5678);
    engine.ConsumeSink(f.system);
    engine.CallMethod(f.system, MAXWELL3D_REG_INDEX(load_mme.start_address_ptr), 0, true);
    CHECK(engine.shadow_state.reg_array[state] == 0x5678);
    CHECK(engine.regs.reg_array[state] == 0x5678);
}

TEST_CASE("Every split entry is an unknown DMA snapshot writer unless explicitly classified",
          "[video_core][dma_snapshot][gpu_split_dispatch]") {
    Fixture f;
    VideoCommon::GpuSplit split;
    VideoCommon::GpuSplitTestAccess::Initialize(split, f.system);
    VideoCommon::GpuSplitTestAccess::EnableDmaSnapshots(split);
    constexpr u32 method = Maxwell3D::RootTables::SelectMethod;
    constexpr VideoCommon::DmaSnapshotWrite no_write{.no_write = true};
    split.Method(&f.fast, method, 0, true, nullptr, no_write); // engine selection invalidates
    const auto& writes = split.DmaSnapshotWrites();
    auto stamp = writes.Stamp();
    split.Method(&f.fast, method, 0, true, nullptr, no_write);
    split.MacroWrite(&f.fast, method, 0, no_write);
    const std::array<u32, 2> arguments{0, 0};
    split.MultiMethod(&f.fast, method, arguments.data(), 2, 2, nullptr, no_write);
    REQUIRE_FALSE(writes.ChangedSince(stamp));

    split.Method(&f.fast, method, 0, true, nullptr);
    REQUIRE(writes.ChangedSince(stamp));
    stamp = writes.Stamp();
    split.MacroWrite(&f.fast, method, 0);
    REQUIRE(writes.ChangedSince(stamp));
    stamp = writes.Stamp();
    split.MultiMethod(&f.fast, method, arguments.data(), 2, 2, nullptr);
    REQUIRE(writes.ChangedSince(stamp));
    stamp = writes.Stamp();
    split.Defer([] {});
    REQUIRE(writes.ChangedSince(stamp));
    stamp = writes.Stamp();
    bool called = false;
    split.DeferFunction([&called] { called = true; });
    REQUIRE(writes.ChangedSince(stamp));
    stamp = writes.Stamp();
    split.Method(&f.slow, method, 0, true, nullptr, no_write); // another engine/channel
    REQUIRE(writes.ChangedSince(stamp));

    // Execute real ring entries too, including freeing DeferFunction's owned callback.
    const auto end = VideoCommon::GpuSplitTestAccess::End(split);
    for (u64 pos = 0; pos < end;) {
        pos = VideoCommon::GpuSplitTestAccess::Execute(split, pos, end);
    }
    REQUIRE(called);
}

#ifdef ARCHITECTURE_x86_64
namespace {
class SnapshotRasterizer final : public VideoCore::RasterizerInterface {
public:
    std::atomic<u32> query_flushes{};
    std::atomic<bool> buffer_dirty{true};
    DAddr flushed_begin{};
    u64 flushed_bytes{};
    std::atomic<u32> query_checks{};
    std::atomic<bool> report_pending{};
    u8* change_on_flush{};
    u32 replacement{};
    std::function<void()> on_query_check;
    bool HasCachedReports(DAddr, u64) override {
        ++query_checks;
        if (on_query_check) {
            auto callback = std::move(on_query_check);
            callback();
        }
        return report_pending.load();
    }
    void FlushQueryResults(DAddr, u64) override {
        ++query_flushes;
        report_pending.store(false);
        if (change_on_flush) {
            std::memcpy(change_on_flush, &replacement, sizeof(replacement));
            change_on_flush = nullptr;
        }
    }
    void Draw(bool, u32) override {}
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&&) override {}
    void SyncOperation(std::function<void()>&&) override {}
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr begin, u64 bytes, VideoCommon::CacheType which) override {
        if (True(which & VideoCommon::CacheType::BufferCache)) {
            flushed_begin = begin;
            flushed_bytes = bytes;
            buffer_dirty.store(false);
        }
    }
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType which) override {
        return True(which & VideoCommon::CacheType::BufferCache) && buffer_dirty.load();
    }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(DAddr, u64) override {}
    bool OnCPUWrite(DAddr, u64) override {
        return false;
    }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override {
        std::terminate();
    }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}
};

std::array<u32, 3> SnapshotFifoInstruction(u32 method, bool end) {
    std::array<u32, 3> bits{};
    const auto field = [&](u32 offset, u32 count, u32 value) {
        for (u32 bit = 0; bit < count; ++bit) {
            bits[(offset + bit) / 32] |= ((value >> bit) & 1) << ((offset + bit) % 32);
        }
    };
    field(0, 1, end);
    for (u32 lane = 0; lane < 2; ++lane) {
        field(15 + lane * 36, 5, 24);
        field(20 + lane * 36, 5, 24);
        field(25 + lane * 36, 5, 24);
    }
    field(30, 16, method);
    field(82, 3, 5);
    field(85, 4, 3);
    return {bits[2], bits[1], bits[0]};
}
} // namespace

TEST_CASE("Real Maxwell front DMA snapshots hit, read live CPU RAM and reject invalidations",
          "[video_core][dma_snapshot][native_macro]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    for (const int mode : {9, 10}) {
        CAPTURE(mode);
        Fixture f;
        f.system.Initialize();
        SnapshotRasterizer rasterizer;
        f.fast.BindRasterizer(&rasterizer);
        f.memory.BindRasterizer(&rasterizer);
        f.device_memory.BindInterface(&rasterizer);
        Kernel::KProcess process{f.system.Kernel()};
        SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
        auto& table = process.GetPageTable().GetImpl();
        table.Resize(20, 12);
        constexpr VAddr cpu = 0x4000;
        constexpr DAddr device = 0x10000;
        constexpr GPUVAddr address = 0x100000;
        constexpr GPUVAddr alias = 0x200000;
        auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
        for (u64 page = 0; page < 2; ++page) {
            const VAddr vaddr = cpu + page * 4096;
            table.entries.GetAndFault(vaddr >> 12)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(backing + page * 4096) - vaddr);
        }
        process.GetMemory().SetCurrentPageTable(process);
        const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
        f.device_memory.Map(device, cpu, 8192, asid);
        SCOPE_EXIT {
            f.device_memory.Unmap(device, 8192);
            f.device_memory.UnregisterProcess(asid);
        };
        f.memory.Map(address, device, 8192, Tegra::PTEKind::PITCH, false);
        f.memory.Map(alias, device, 4096, Tegra::PTEKind::PITCH, false);
        const std::array<u32, 2> initial{11, 22};
        const std::array<u32, 2> second_page{55, 66};
        std::memcpy(backing, initial.data(), sizeof(initial));
        std::memcpy(backing + 4096, second_page.data(), sizeof(second_page));
        f.fast.SetNativeTuringMacroMode(true);
        auto& mme = f.fast.GetNativeMacroEngine();
        mme.ClearCode(0);
        // Same public synthetic instruction encoder as native_macro_fifo.cpp; no guest code.
        for (const auto instruction : {SnapshotFifoInstruction(0x158, false),
                                       SnapshotFifoInstruction(0x200, true),
                                       SnapshotFifoInstruction(0x201, false)}) {
            for (const u32 word : instruction) {
                mme.AddCode(0, word);
            }
        }
        f.fast.GetNativeMacroPositions()[0] = 0;
        REQUIRE_FALSE(f.fast.execution_mask[0x200]);
        REQUIRE_FALSE(f.fast.execution_mask[0x201]);
        VideoCommon::GpuSplit split;
        Access::Start(split, f.system);
        Access::EnableDmaSnapshots(split, mode);
        SCOPE_EXIT { split.Stop(); };
        const auto state = [&](u32 method, u32 value) {
            f.fast.FrontCallMethod(f.system, split, method, value, true);
        };
        state(0x154, 0);
        state(0x155, static_cast<u32>(address));
        const auto read = [&] {
            f.fast.FrontCallMethod(f.system, split, 0xe00, 2, true);
            split.Drain();
            return std::array<u32, 2>{f.fast.regs.reg_array[0x200], f.fast.regs.reg_array[0x201]};
        };
        REQUIRE(read() == initial);
        REQUIRE(rasterizer.flushed_begin == device);
        REQUIRE(rasterizer.flushed_bytes == 4096); // Whole page, not only the two words read.
        REQUIRE(read() == initial); // FinishFrontMacro from the first call was conservative.
        auto flushes = rasterizer.query_flushes.load();
        auto checks = rasterizer.query_checks.load();
        REQUIRE(read() == initial);
        REQUIRE(rasterizer.query_checks.load() > checks); // real eligible hit checked reports
        REQUIRE(rasterizer.query_flushes.load() == flushes + (mode == 10 ? 1 : 0));

        const u32 cpu_value = 77;
        std::memcpy(backing, &cpu_value, sizeof(cpu_value));
        flushes = rasterizer.query_flushes.load();
        REQUIRE(read() == std::array<u32, 2>{77, 22});
        REQUIRE(rasterizer.query_flushes.load() == flushes + (mode == 10 ? 1 : 0));

        split.Defer([] {}); // an unknown producer invalidates every snapshot
        flushes = rasterizer.query_flushes.load();
        REQUIRE(read() == std::array<u32, 2>{77, 22});
        REQUIRE(rasterizer.query_flushes.load() > flushes);

        // A real CB upload through another GPU alias must invalidate the device page.
        state(CbAddressHigh, 0);
        state(CbAddressLow, static_cast<u32>(alias));
        state(CbSize, 8);
        state(CbOffset, 0);
        const u32 upload = 88;
        f.fast.FrontCallMultiMethod(f.system, split, CbData, &upload, 1, 1);
        flushes = rasterizer.query_flushes.load();
        REQUIRE(read() == std::array<u32, 2>{88, 22});
        REQUIRE(rasterizer.query_flushes.load() > flushes);

        rasterizer.report_pending.store(true);
        flushes = rasterizer.query_flushes.load();
        REQUIRE(read() == std::array<u32, 2>{88, 22});
        REQUIRE(rasterizer.query_flushes.load() > flushes);
        REQUIRE_FALSE(rasterizer.report_pending.load());

        // Direct mapping-generation invalidation, without relying on the ring's unknown mark.
        f.memory.Map(address, device + 4096, 4096, Tegra::PTEKind::PITCH, false);
        flushes = rasterizer.query_flushes.load();
        REQUIRE(read() == second_page);
        REQUIRE(rasterizer.query_flushes.load() > flushes);
        if (mode == 10) {
            // Deliberately make the exact drained read differ: mode10 must consume that value.
            rasterizer.change_on_flush = backing + 4096;
            rasterizer.replacement = 99;
            checks = rasterizer.query_checks.load();
            REQUIRE(read() == std::array<u32, 2>{99, 66});
            REQUIRE(rasterizer.query_checks.load() > checks);
        }

        // Two distinct SMMU addresses share one physical page. A CB through that alias
        // invalidates globally, even though its DAddr differs from the snapshot's page.
        constexpr DAddr device_alias = 0x30000;
        f.device_memory.Map(device_alias, cpu, 4096, asid);
        SCOPE_EXIT { f.device_memory.Unmap(device_alias, 4096); };
        f.memory.Map(address, device, 4096, Tegra::PTEKind::PITCH, false);
        f.memory.Map(alias, device_alias, 4096, Tegra::PTEKind::PITCH, false);
        REQUIRE(read() == std::array<u32, 2>{88, 22});
        state(CbOffset, 0);
        const u32 smmu_upload = 123;
        f.fast.FrontCallMultiMethod(f.system, split, CbData, &smmu_upload, 1, 1);
        flushes = rasterizer.query_flushes.load();
        REQUIRE(read() == std::array<u32, 2>{123, 22});
        REQUIRE(rasterizer.query_flushes.load() > flushes);

        // Start a real mutation while FrontDmaRead is inside its hit path. Its query
        // callback runs after both mapping locks are held and before the RAM copy.
        // Completion must wait for that copy; checking epochs alone fails this test.
        for (const bool remap_backing : {false, true}) {
            CAPTURE(remap_backing);
            f.device_memory.Map(device, cpu, 4096, asid);
            f.memory.Map(address, device, 4096, Tegra::PTEKind::PITCH, false);
            std::memcpy(backing, initial.data(), sizeof(initial));
            std::memcpy(backing + 4096, second_page.data(), sizeof(second_page));
            REQUIRE(read() == initial);
            REQUIRE(read() == initial);
            std::promise<void> started;
            std::promise<void> completed;
            auto start = started.get_future();
            auto completion = completed.get_future();
            std::thread writer;
            SCOPE_EXIT { if (writer.joinable()) writer.join(); };
            bool mutation_blocked = false;
            rasterizer.on_query_check = [&] {
                writer = std::thread{[&] {
                    started.set_value();
                    if (remap_backing) {
                        f.device_memory.Map(device, cpu + 4096, 4096, asid);
                    } else {
                        f.memory.Map(address, device + 4096, 4096, Tegra::PTEKind::PITCH,
                                     false);
                    }
                    completed.set_value();
                }};
                start.wait();
                mutation_blocked = completion.wait_for(std::chrono::milliseconds{20}) ==
                                   std::future_status::timeout;
            };
            const auto during_remap = read();
            if (writer.joinable()) {
                writer.join();
            }
            REQUIRE(mutation_blocked);
            if (mode == 9) {
                REQUIRE(during_remap == initial);
            }
            // The next read must observe the new translation and re-certify: the mutation
            // completed after the previous read's initial mapping epochs were sampled.
            flushes = rasterizer.query_flushes.load();
            REQUIRE(read() == second_page);
            REQUIRE(rasterizer.query_flushes.load() > flushes);
            // The independent SMMU alias still names the original physical page.
            REQUIRE(f.memory.Read<u32>(alias) == initial[0]);
        }
    }
}
#endif
