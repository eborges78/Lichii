// SPDX-License-Identifier: GPL-3.0-or-later

#include <deque>
#include <limits>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/transcode_cache.h"

namespace {
constexpr u64 MiB = u64{1} << 20;
constexpr u64 CacheLimit = 256 * MiB;

struct Resource {
    explicit Resource(size_t& destroyed_) : destroyed{destroyed_} {}
    ~Resource() {
        ++destroyed;
    }
    size_t& destroyed;
};

struct Transcode {
    u64 tick;
    u32 width = 4096;
    u32 height = 4096;
    u32 levels = 1;
    u32 layers = 1;
    u32 unused_frames = 0;
    std::unique_ptr<Resource> resource;
};

Transcode MakeTranscode(u64 tick, size_t& destroyed, u32 layers = 1) {
    return {.tick = tick, .layers = layers, .resource = std::make_unique<Resource>(destroyed)};
}

using VideoCommon::CollectCompletedTranscodes;
using VideoCommon::DrainTranscodesToBudget;
using VideoCommon::TranscodeDecodedBytes;
using VideoCommon::TranscodesExceedBudget;
} // namespace

TEST_CASE("ASTC collection requires completed ticks, not recording or submission",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    pending.push_back(MakeTranscode(1, destroyed));
    u64 current_tick = 1;
    u64 completed_tick = 0;
    const auto collect = [&] {
        CollectCompletedTranscodes(pending, free,
                                   [&](u64 tick) { return completed_tick >= tick; }, CacheLimit);
    };
    collect();
    REQUIRE(pending.size() == 1);
    REQUIRE(free.empty());
    ++current_tick; // A flush advances the recording tick; completion is still unchanged.
    collect();
    REQUIRE(pending.size() == 1);
    REQUIRE(free.empty());
    REQUIRE(destroyed == 0);
    pending.push_back(MakeTranscode(current_tick, destroyed));
    completed_tick = 1;
    collect();
    REQUIRE(free.size() == 1);
    REQUIRE(free.front().tick == 1);
    REQUIRE(pending.size() == 1);
    REQUIRE(pending.front().tick == current_tick);
    REQUIRE(destroyed == 0);
}

TEST_CASE("ASTC collection returns completed storage for reuse without advancing a frame",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    auto transcode = MakeTranscode(1, destroyed);
    const auto* resource = transcode.resource.get();
    pending.push_back(std::move(transcode));
    for (u64 tick = 1; tick <= 128; ++tick) {
        CollectCompletedTranscodes(pending, free, [tick](u64 value) { return value <= tick; },
                                   CacheLimit);
        REQUIRE(pending.empty());
        REQUIRE(free.size() == 1);
        REQUIRE(free.front().resource.get() == resource);
        REQUIRE(free.front().unused_frames == 0);
        auto reused = std::move(free.front());
        free.clear();
        reused.tick = tick + 1;
        pending.push_back(std::move(reused));
    }
    REQUIRE(destroyed == 0);
}

TEST_CASE("ASTC free cache stays bounded and never destroys unfinished work",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    free.push_back(MakeTranscode(1, destroyed, 2)); // 128 MiB already reusable.
    free.front().unused_frames = 57;
    for (u64 tick = 2; tick <= 5; ++tick) {
        pending.push_back(MakeTranscode(tick, destroyed)); // 64 MiB each.
    }
    CollectCompletedTranscodes(pending, free, [](u64 tick) { return tick <= 4; }, CacheLimit);
    REQUIRE(free.size() == 3);
    REQUIRE(free.front().unused_frames == 57); // Collection is not a frame-age update.
    u64 free_bytes{};
    for (const auto& transcode : free) {
        free_bytes += TranscodeDecodedBytes(transcode);
    }
    REQUIRE(free_bytes == CacheLimit);
    REQUIRE(destroyed == 1); // Only the completed entry exceeding the cache budget.
    REQUIRE(pending.size() == 1);
    REQUIRE(pending.front().tick == 5);
    CollectCompletedTranscodes(pending, free, [](u64) { return false; }, CacheLimit);
    REQUIRE(free.front().unused_frames == 57);
    REQUIRE(destroyed == 1);
}

TEST_CASE("Oversized ASTC temporaries are destroyed only after completion",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    pending.push_back(MakeTranscode(1, destroyed, 8)); // 512 MiB, never cached.
    pending.push_back(MakeTranscode(2, destroyed, 8));
    CollectCompletedTranscodes(pending, free, [](u64 tick) { return tick <= 1; }, CacheLimit);
    REQUIRE(free.empty());
    REQUIRE(destroyed == 1);
    REQUIRE(pending.size() == 1);
    REQUIRE(pending.front().tick == 2);
}

TEST_CASE("ASTC accounting includes every mip and array layer",
          "[video_core][transcode_cache]") {
    Transcode transcode{.tick = 1, .width = 7, .height = 3, .levels = 3, .layers = 2};
    REQUIRE(TranscodeDecodedBytes(transcode) == 200); // (7*3 + 3*1 + 1*1) * RGBA8 * 2.
    transcode.width = 4096;
    transcode.height = 4096;
    transcode.levels = 1;
    transcode.layers = 4;
    REQUIRE(TranscodeDecodedBytes(transcode) == CacheLimit);
}

TEST_CASE("ASTC pending admission counts layers and accepts the exact budget without waiting",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    pending.push_back(MakeTranscode(8, destroyed, 4));
    REQUIRE_FALSE(TranscodesExceedBudget(pending, CacheLimit));
    size_t waits{}, collections{};
    REQUIRE(DrainTranscodesToBudget(pending, CacheLimit, [&](u64) { ++waits; },
                                    [&] { ++collections; }));
    REQUIRE(waits == 0);
    REQUIRE(collections == 0);
    pending.push_back(MakeTranscode(9, destroyed));
    REQUIRE(TranscodesExceedBudget(pending, CacheLimit));
    REQUIRE(destroyed == 0);
}

TEST_CASE("ASTC pending accounting cannot wrap below its budget",
          "[video_core][transcode_cache]") {
    constexpr auto Max = std::numeric_limits<u64>::max();
    constexpr auto Max32 = std::numeric_limits<u32>::max();
    std::deque<Transcode> pending;
    pending.push_back({.tick = 1, .width = Max32, .height = Max32});
    REQUIRE(TranscodeDecodedBytes(pending.back()) == Max);
    REQUIRE(TranscodesExceedBudget(pending, CacheLimit));
    pending.back().width = 1U << 30;
    pending.back().height = 1U << 30;
    pending.back().layers = 4;
    REQUIRE(TranscodeDecodedBytes(pending.back()) == Max); // Layer multiplication overflows.
    pending.back().layers = 1;
    pending.back().levels = 2;
    REQUIRE(TranscodeDecodedBytes(pending.back()) == (u64{5} << 60));
    for (size_t i = 0; i < 3; ++i) {
        pending.push_back({.tick = 2 + i, .width = 1U << 30, .height = 1U << 30,
                           .levels = 2});
    }
    REQUIRE(TranscodesExceedBudget(pending, Max)); // The sum, not an individual image, overflows.
    Transcode long_mips{.tick = 1, .width = 1, .height = 1, .levels = 33};
    REQUIRE(TranscodeDecodedBytes(long_mips) == 132); // No shift by 32.
}

TEST_CASE("ASTC backpressure waits oldest completion before retiring GPU resources",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    // A full free cache must never be selected as a wait target.
    free.push_back(MakeTranscode(1, destroyed, 4));
    for (u64 tick = 10; tick < 16; ++tick) {
        pending.push_back(MakeTranscode(tick, destroyed));
    }
    u64 completed_tick = 9;
    std::vector<u64> waits;
    REQUIRE(DrainTranscodesToBudget(
        pending, CacheLimit,
        [&](u64 tick) {
            REQUIRE(tick == pending.front().tick);
            REQUIRE(destroyed == waits.size());
            CollectCompletedTranscodes(pending, free,
                                       [&](u64 value) { return value <= completed_tick; }, CacheLimit);
            REQUIRE(pending.front().tick == tick); // Submission alone cannot retire it.
            waits.push_back(tick);
            completed_tick = tick;
        },
        [&] {
            CollectCompletedTranscodes(pending, free,
                                       [&](u64 tick) { return tick <= completed_tick; }, CacheLimit);
        }));
    REQUIRE(waits == std::vector<u64>{10, 11});
    REQUIRE(destroyed == 2);
    REQUIRE(pending.size() == 4);
    REQUIRE(pending.front().tick == 12);
    REQUIRE(free.size() == 1);
    REQUIRE_FALSE(TranscodesExceedBudget(pending, CacheLimit));
}

TEST_CASE("ASTC oversized transactions drain completely before another can accumulate",
          "[video_core][transcode_cache]") {
    size_t destroyed{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    u64 completed_tick{};
    std::vector<u64> waits;
    const auto drain = [&] {
        return DrainTranscodesToBudget(
            pending, CacheLimit,
            [&](u64 tick) {
                waits.push_back(tick);
                completed_tick = tick;
            },
            [&] {
                CollectCompletedTranscodes(pending, free,
                                           [&](u64 tick) { return tick <= completed_tick; },
                                           CacheLimit);
            });
    };
    pending.push_back(MakeTranscode(10, destroyed, 4)); // Previous budget, 256 MiB.
    pending.push_back(MakeTranscode(11, destroyed, 8)); // One oversized transaction, 512 MiB.
    REQUIRE(drain());
    REQUIRE(waits == std::vector<u64>{10, 11});
    REQUIRE(pending.empty());
    REQUIRE(free.size() == 1);
    REQUIRE(destroyed == 1);
    pending.push_back(MakeTranscode(12, destroyed, 8));
    REQUIRE(drain());
    REQUIRE(pending.empty());
    REQUIRE(waits.back() == 12);
    REQUIRE(destroyed == 2);
}

TEST_CASE("ASTC one completed submission can retire multiple transcodes",
          "[video_core][transcode_cache]") {
    size_t destroyed{}, waits{};
    std::deque<Transcode> pending;
    std::vector<Transcode> free;
    for (size_t i = 0; i < 5; ++i) {
        pending.push_back(MakeTranscode(2, destroyed));
    }
    u64 completed_tick{};
    REQUIRE(DrainTranscodesToBudget(
        pending, CacheLimit,
        [&](u64 tick) { ++waits; completed_tick = tick; },
        [&] {
            CollectCompletedTranscodes(pending, free,
                                       [&](u64 tick) { return tick <= completed_tick; }, CacheLimit);
        }));
    REQUIRE(waits == 1);
    REQUIRE(pending.empty());
    REQUIRE(free.size() == 4);
    REQUIRE(destroyed == 1);
}

TEST_CASE("ASTC backpressure refuses to spin if completion makes no progress",
          "[video_core][transcode_cache]") {
    size_t destroyed{}, waits{}, collections{};
    std::deque<Transcode> pending;
    pending.push_back(MakeTranscode(1, destroyed, 8));
    REQUIRE_FALSE(DrainTranscodesToBudget(pending, CacheLimit, [&](u64) { ++waits; },
                                          [&] { ++collections; }));
    REQUIRE(waits == 1);
    REQUIRE(collections == 1);
    REQUIRE(pending.size() == 1);
    REQUIRE(destroyed == 0);
}
