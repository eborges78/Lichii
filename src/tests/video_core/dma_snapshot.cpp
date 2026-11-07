// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/dma_snapshot.h"

using VideoCommon::DmaSnapshotTracker;
using VideoCommon::DmaSnapshotWrite;

TEST_CASE("DMA snapshots track device aliases, page boundaries and unknown writers",
          "[video_core][dma_snapshot]") {
    DmaSnapshotTracker writes;
    const u64 ready = writes.Stamp();
    writes.Record({.known = true}); // register-only work
    REQUIRE_FALSE(writes.ChangedSince(ready, 0x4000, 4));
    writes.Record({.known = true, .begin = 0x8ffc, .size = 8});
    REQUIRE_FALSE(writes.ChangedSince(ready, 0x4000, 4));
    REQUIRE(writes.ChangedSince(ready, 0x8000, 4));
    REQUIRE(writes.ChangedSince(ready, 0x9000, 4));
    // Two GPU aliases / channels translate to this same device range.
    const u64 alias_a_device = 0x9004;
    const u64 alias_b_device = alias_a_device;
    const u64 alias_ready = writes.Stamp();
    writes.Record({.known = true, .begin = alias_a_device, .size = 4});
    REQUIRE(writes.ChangedSince(alias_ready, alias_b_device, 4));
    const u64 before_unknown = writes.Stamp();
    writes.Record(); // draw, compute, copy, remap, deferred function, opaque report
    REQUIRE(writes.ChangedSince(before_unknown, 0x4000, 4));
    const u64 before_reset = writes.Stamp();
    writes.Reset();
    REQUIRE(writes.ChangedSince(before_reset, 0x4000, 4));
    REQUIRE_FALSE(writes.ChangedSince(writes.Stamp(), 0x4000, 4));
}

TEST_CASE("DMA snapshot bounded history cannot forget an interfering write",
          "[video_core][dma_snapshot]") {
    DmaSnapshotTracker writes;
    const u64 old = writes.Stamp();
    writes.Record({.known = true, .begin = 0x1000, .size = 4});
    // Same table slot, distinct device page: evicting the first mark is a global barrier.
    writes.Record({.known = true,
                   .begin = 0x1000 + DmaSnapshotTracker::PageCount * DmaSnapshotTracker::PageSize,
                   .size = 4});
    REQUIRE(writes.ChangedSince(old, 0x1000, 4));
    REQUIRE(writes.ChangedSince(old, 0x4000, 4));
    auto stamp = writes.Stamp();
    writes.Record({.known = true, .begin = std::numeric_limits<u64>::max() - 3, .size = 8});
    REQUIRE(writes.ChangedSince(stamp, 0x4000, 4));
    stamp = writes.Stamp();
    writes.Record({.known = true, .begin = 0,
                   .size = (DmaSnapshotTracker::PageCount + 1) * DmaSnapshotTracker::PageSize});
    REQUIRE(writes.ChangedSince(stamp, 0x4000, 4));
    REQUIRE(writes.ChangedSince(writes.Stamp(), std::numeric_limits<u64>::max() - 3, 8));
    REQUIRE(writes.ChangedSince(writes.Stamp() + 1, 0x4000, 4));
}

TEST_CASE("DMA snapshot history is conservative against an independent interval oracle",
          "[video_core][dma_snapshot]") {
    struct Write {
        u64 stamp;
        u64 begin;
        u64 end;
        bool unknown;
    };
    DmaSnapshotTracker writes;
    std::vector<Write> history;
    std::mt19937 random{0x825491};
    for (size_t i = 0; i < 6000; ++i) {
        const u64 begin = u64{random() % 4096} * 4096 + random() % 4096;
        const u64 size = 1 + random() % 8192;
        const bool unknown = random() % 23 == 0;
        if (unknown) {
            writes.Reset();
        } else {
            writes.Record({.known = true, .begin = begin, .size = size});
        }
        history.push_back({writes.Stamp(), begin, begin + size, unknown});
        for (size_t j = 0; j < 4; ++j) {
            const u64 since = random() % (writes.Stamp() + 1);
            const u64 read = u64{random() % 4096} * 4096 + random() % 4096;
            const u64 bytes = 1 + random() % 16384;
            bool changed = false;
            for (const auto& write : history) {
                changed |= write.stamp > since &&
                           (write.unknown || (write.begin < read + bytes && read < write.end));
            }
            if (changed) {
                REQUIRE(writes.ChangedSince(since, read, bytes));
            }
        }
    }
}
