// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <limits>

#include "common/common_types.h"

namespace VideoCommon {

// Every executable method is a possible writer; only proven state writes are exempt.
struct DmaSnapshotWrite {
    bool no_write{};
};

// Shared by all channels of the split. No device-range classification: distinct SMMU
// addresses can alias the same RAM, and a CB target can change before queued execution.
class DmaSnapshotTracker {
public:
    void Record(DmaSnapshotWrite write = {}) {
        if (write.no_write) {
            return;
        }
        if (serial == std::numeric_limits<u64>::max()) {
            exhausted = true;
            return;
        }
        ++serial;
    }

    void Reset() {
        // Never reuse a stamp when the static split stops or a new game starts.
        Record();
    }

    [[nodiscard]] u64 Stamp() const {
        return serial;
    }

    [[nodiscard]] bool ChangedSince(u64 since) const {
        return exhausted || since != serial;
    }

private:
    friend struct DmaSnapshotTrackerTestAccess;
    u64 serial{};
    bool exhausted{};
};

} // namespace VideoCommon
