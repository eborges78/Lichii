// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "common/common_types.h"

namespace VideoCommon {

// Unknown by default: only the split front can certify a no-write or device-address range.
struct DmaSnapshotWrite {
    bool known{};
    u64 begin{};
    u64 size{};
};

// Conservative, bounded variant of Frizti 8ac3666f's MayBeWrittenSince. Device addresses
// join aliases and channels. Evicting a tracked page invalidates all older snapshots.
class DmaSnapshotTracker {
public:
    static constexpr size_t PageCount = 1024;
    static constexpr u64 PageSize = 4096;

    void Record(DmaSnapshotWrite write = {}) {
        if (write.known && write.size == 0) {
            return;
        }
        if (serial == std::numeric_limits<u64>::max()) {
            exhausted = true;
            return;
        }
        ++serial;
        if (!write.known || write.size > std::numeric_limits<u64>::max() - write.begin) {
            unknown = serial;
            return;
        }
        const u64 first = write.begin / PageSize;
        const u64 last = (write.begin + write.size - 1) / PageSize;
        if (last - first >= PageCount) {
            unknown = serial;
            return;
        }
        for (u64 page = first; page <= last; ++page) {
            auto& entry = pages[page % PageCount];
            if (entry.serial != 0 && entry.page != page) {
                unknown = serial;
            }
            entry = {page, serial};
        }
    }

    void Reset() {
        // Never reuse a stamp when the static split stops or a new game starts.
        Record();
        pages = {};
    }

    [[nodiscard]] u64 Stamp() const {
        return serial;
    }

    [[nodiscard]] bool ChangedSince(u64 since, u64 begin, u64 size) const {
        if (exhausted || since > serial || unknown > since || size == 0 ||
            size > std::numeric_limits<u64>::max() - begin) {
            return true;
        }
        const u64 first = begin / PageSize;
        const u64 last = (begin + size - 1) / PageSize;
        if (last - first >= PageCount) {
            return true;
        }
        for (u64 page = first; page <= last; ++page) {
            const auto& entry = pages[page % PageCount];
            if (entry.page == page && entry.serial > since) {
                return true;
            }
        }
        return false;
    }

private:
    struct Page {
        u64 page{};
        u64 serial{};
    };
    std::array<Page, PageCount> pages{};
    u64 serial{};
    u64 unknown{};
    bool exhausted{};
};

} // namespace VideoCommon
