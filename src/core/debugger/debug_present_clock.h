// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace Core::Nx2Debug {

// Switch 1 debug input follows submitted, nonempty compositor stacks. This is not a
// completed GPU frame or a physical scanout. VI2 owns the Switch 2 counter separately.
inline void RecordLegacyDebugPresent(std::atomic<u64>& count, bool debug_requested,
                                     bool is_switch2, bool has_layers) {
    if (debug_requested && !is_switch2 && has_layers) {
        count.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace Core::Nx2Debug
