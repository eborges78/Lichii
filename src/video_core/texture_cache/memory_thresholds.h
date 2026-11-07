// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"

namespace VideoCommon {

struct TextureMemoryThresholds {
    u64 expected;
    u64 critical;

    bool operator==(const TextureMemoryThresholds&) const = default;
};

// Both budgets must be known. Host memory is OS-visible RAM (after firmware UMA reservation);
// device_local_memory is the renderer's effective budget, not physical VRAM.
[[nodiscard]] constexpr bool UsesOotIntegratedMemoryBudget(
    u64 program_id, bool integrated, u64 device_local_memory, u64 host_memory) {
    constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
    constexpr u64 GiB = u64{1} << 30;
    return program_id == OcarinaOfTime && integrated && device_local_memory != 0 &&
           device_local_memory <= 4 * GiB && host_memory != 0 && host_memory <= 16 * GiB;
}

[[nodiscard]] constexpr TextureMemoryThresholds ClampOotIntegratedTextureThresholds(
    u64 program_id, bool integrated, u64 device_local_memory, u64 host_memory,
    TextureMemoryThresholds thresholds) {
    constexpr u64 GiB = u64{1} << 30;
    if (UsesOotIntegratedMemoryBudget(program_id, integrated, device_local_memory, host_memory)) {
        thresholds.expected = std::min(thresholds.expected, GiB);
        thresholds.critical = std::min(thresholds.critical, GiB + GiB / 2);
    }
    return thresholds;
}

} // namespace VideoCommon
