
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <unordered_map>
#include <vector>
using u32 = std::uint32_t;
using u64 = std::uint64_t;
namespace Core { struct System {}; }
namespace VideoCommon { struct GpuSplit {
    unsigned batches = 0;
    template <typename... Args> void MultiMethod(Args...) { ++batches; }
}; }
struct MacroEngine {
    bool native_turing = true;
    u32 native_write_offset = 0;
    u64 native_code_generation = 0;
    std::array<u32, 0x8000> native_code{};
    std::unordered_map<u32, std::vector<u32>> uploaded_macro_code;
    std::unordered_map<u32, int> macro_cache;
inline void AddCode(u32 method, u32 data) noexcept {
        if (native_turing) {
            if (native_write_offset < native_code.size()) native_code[native_write_offset++] = data;
            ++native_code_generation;
            return;
        }
        uploaded_macro_code[method].push_back(data);
    }
inline void ClearCode(u32 method) noexcept {
        if (native_turing) { native_write_offset = u64{method} * 3 < native_code.size() ? method * 3 : static_cast<u32>(native_code.size()); return; }
        macro_cache.erase(method);
        uploaded_macro_code.erase(method);
    }
};
struct Maxwell3D {
    struct Regs {
        enum class ShadowRamControl { Track, TrackWithFilter, Passthrough, Replay };
    };
    struct ControlWrite { u32 argument, control, source; };
    struct FrontState {
        std::array<u32, 0x1000> regs{}, shadow{};
        Regs::ShadowRamControl control{};
        std::array<ControlWrite, 16> control_writes{};
        u32 control_count = 0, executing_macro = 0;
        bool instance_loop = false, draw_pending = false;
        std::vector<u32> parameters;
    } fs;
    static constexpr u32 MacroRegistersStart = 0xe00;
    std::array<bool, 0x1000> execution_mask{};
    MacroEngine engine;
    std::array<u32, 128> positions{};
    FrontState& Front(Core::System&, VideoCommon::GpuSplit&) { return fs; }
    MacroEngine& GetNativeMacroEngine() { return engine; }
    std::span<u32> GetNativeMacroPositions() { return positions; }
    void FrontRunMacro(Core::System&, VideoCommon::GpuSplit&, FrontState&) {}
    void FrontCallMultiMethod(Core::System&, VideoCommon::GpuSplit&, u32, const u32*, u32, u32);
    void FrontTrackInstanceLoop(FrontState&, u32, u32);
};
bool FrontVerifyEnabled() { return false; }
bool FrontHandledOutsideDrawManager(u32) { return true; }
void require(bool value) { if (!value) { std::fputs("FAIL\n", stderr); std::abort(); } }
[[gnu::noinline]] void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case 0x586:
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case 0x57a:
    case 0x57b:
    case 0x4c0:
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}
void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words.
    constexpr u32 cb_data = 0x8e4;
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    if (method >= cb_data && method < cb_data + 16) {
        fs.regs[0x8e3] += amount * sizeof(u32);
    } else if (method == 0x6d) {
        // Upload data only.
    } else if ((method == 0x57a ||
                method == 0x57b ||
                method == 0x4c0) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        FrontTrackInstanceLoop(fs, method, 0);
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == 0x49 && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            FrontTrackInstanceLoop(fs, method, value); // DrawBegin runs for every word
            if (!execution_mask[method]) {
                continue;
            }
            switch (method) {
            case 0x49:
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case 0x45:
                GetNativeMacroEngine().ClearCode(value);
                break;
            case 0x46:
                GetNativeMacroEngine().AddCode(fs.regs[0x45], value);
                break;
            case 0x48: {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[0x47];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case 0x8c4:
                fs.regs[0xd00] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
}
int main() {
    Core::System system;
    VideoCommon::GpuSplit split;
    Maxwell3D m;
    m.execution_mask.fill(true);
    auto send = [&](u32 method, const std::vector<u32>& words) {
        m.FrontCallMultiMethod(system, split, method, words.data(), words.size(), words.size());
    };
    auto set = [&](u32 method, u32 value) { send(method, {value}); };
    std::vector<u32> words(75);
    for (u32 i = 0; i < words.size(); ++i) words[i] = 0x12340000u + i;
    for (u32 start : {0u, 0x21u, 0x262u, 0x1000u}) {
        set(0x45, start);
        auto generation = m.engine.native_code_generation;
        send(0x46, words);
        require(m.engine.native_code_generation == generation + words.size());
        require(m.engine.native_write_offset == start * 3 + words.size());
        for (u32 i = 0; i < words.size(); ++i)
            require(m.engine.native_code[start * 3 + i] == words[i]);
    }
    set(0x47, 2);
    send(0x48, {0x21, 0x262, 0x372});
    require(m.positions[2] == 0x21 && m.positions[3] == 0x262 && m.positions[4] == 0x372);
    require(m.fs.regs[0x47] == 5);
    set(0x45, 0x100);
    m.fs.control = Maxwell3D::Regs::ShadowRamControl::Replay;
    m.fs.shadow[0x46] = 0xabcdef;
    send(0x46, {1, 2, 3});
    require(m.engine.native_code[0x300] == 0xabcdef && m.engine.native_code[0x302] == 0xabcdef);
    m.fs.control = Maxwell3D::Regs::ShadowRamControl::Track;
    set(0x586, 1u << 26);
    require(m.fs.instance_loop);
    set(0x586, 0);
    require(!m.fs.instance_loop);
    set(0x586, 1u << 26);
    send(0x57a, {1, 2, 3});
    require(!m.fs.instance_loop);
    require(split.batches == 16);
    std::puts("PASS uploads, nonzero offsets, positions, shadow replay, instance tracking, queue forwarding");
}
