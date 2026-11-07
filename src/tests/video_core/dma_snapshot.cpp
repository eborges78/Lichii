// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/dma_snapshot.h"

namespace VideoCommon {
struct DmaSnapshotTrackerTestAccess {
    static void ExhaustNext(DmaSnapshotTracker& writes) {
        writes.serial = std::numeric_limits<u64>::max();
    }
};
} // namespace VideoCommon

using VideoCommon::DmaSnapshotTracker;

TEST_CASE("DMA snapshot writer epochs invalidate globally without reusing stamps",
          "[video_core][dma_snapshot]") {
    DmaSnapshotTracker writes;
    auto ready = writes.Stamp();
    writes.Record({.no_write = true});
    REQUIRE_FALSE(writes.ChangedSince(ready));
    writes.Record(); // CB, draw, compute, copy, query, another channel or callback
    REQUIRE(writes.ChangedSince(ready));
    ready = writes.Stamp();
    writes.Reset();
    REQUIRE(writes.ChangedSince(ready));
    REQUIRE_FALSE(writes.ChangedSince(writes.Stamp()));
    REQUIRE(writes.ChangedSince(writes.Stamp() + 1));
    VideoCommon::DmaSnapshotTrackerTestAccess::ExhaustNext(writes);
    ready = writes.Stamp();
    writes.Record();
    REQUIRE(writes.ChangedSince(ready));
    writes.Reset();
    writes.Record({.no_write = true});
    REQUIRE(writes.ChangedSince(writes.Stamp()));
}

TEST_CASE("DMA snapshot epochs preserve only certificates with no subsequent possible writer",
          "[video_core][dma_snapshot]") {
    struct Certificate { u64 stamp; size_t issued_at; };
    DmaSnapshotTracker writes;
    std::vector<Certificate> certificates;
    std::vector<bool> possible_writers;
    std::mt19937 random{0x825491};
    for (size_t i = 0; i < 6000; ++i) {
        certificates.push_back({writes.Stamp(), possible_writers.size()});
        const bool reset = random() % 23 == 0;
        const bool possible_writer = reset || random() % 7 == 0;
        possible_writers.push_back(possible_writer);
        if (reset) {
            writes.Reset();
        } else {
            writes.Record({.no_write = !possible_writer});
        }
        for (size_t j = 0; j < 4; ++j) {
            const auto& certificate = certificates[random() % certificates.size()];
            bool changed = false;
            for (size_t k = certificate.issued_at; k < possible_writers.size(); ++k) {
                changed |= possible_writers[k];
            }
            REQUIRE(writes.ChangedSince(certificate.stamp) == changed);
        }
    }
}
