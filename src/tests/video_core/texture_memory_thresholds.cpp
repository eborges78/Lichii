// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/memory_thresholds.h"

namespace {
constexpr u64 Oot = 0x0400486024DEC000ULL;
constexpr u64 GiB = u64{1} << 30;
constexpr VideoCommon::TextureMemoryThresholds Original{2415ULL << 20, 3183ULL << 20};
constexpr VideoCommon::TextureMemoryThresholds OotBudget{GiB, GiB + GiB / 2};
using VideoCommon::ClampOotIntegratedTextureThresholds;
} // namespace

TEST_CASE("OOT texture collection applies only to an integrated GPU with a small effective budget",
          "[video_core][texture_memory_thresholds]") {
    for (const u64 memory : {u64{1}, GiB, 2 * GiB, 4 * GiB}) {
        REQUIRE(ClampOotIntegratedTextureThresholds(Oot, true, memory, 16 * GiB, Original) == OotBudget);
        REQUIRE(ClampOotIntegratedTextureThresholds(Oot, false, memory, 16 * GiB, Original) == Original);
    }
    for (const u64 memory : {u64{0}, 4 * GiB + 1, 6 * GiB, 16 * GiB,
                             std::numeric_limits<u64>::max()}) {
        REQUIRE(ClampOotIntegratedTextureThresholds(Oot, true, memory, 16 * GiB, Original) == Original);
    }
}

TEST_CASE("Other titles retain their texture thresholds on small integrated GPUs",
          "[video_core][texture_memory_thresholds]") {
    for (const u64 title : {u64{0}, u64{0x0100E95004038000ULL}, u64{0x0400C3F00006E000ULL},
                            Oot - 1, Oot + 1}) {
        for (const bool integrated : {false, true}) {
            REQUIRE(ClampOotIntegratedTextureThresholds(title, integrated, 4 * GiB, 16 * GiB, Original) ==
                    Original);
        }
    }
}

TEST_CASE("OOT texture collection requires a known small OS host memory budget",
          "[video_core][texture_memory_thresholds]") {
    for (const u64 host_memory : {u64{0}, 16 * GiB + 1, 27 * GiB, 32 * GiB,
                                  std::numeric_limits<u64>::max()}) {
        REQUIRE(ClampOotIntegratedTextureThresholds(Oot, true, 4 * GiB, host_memory, Original) ==
                Original);
    }
    for (const u64 host_memory : {11 * GiB, 15 * GiB, 16 * GiB}) {
        REQUIRE(ClampOotIntegratedTextureThresholds(Oot, true, 4 * GiB, host_memory, Original) ==
                OotBudget);
    }
}

TEST_CASE("OOT collection clamps preserve lower thresholds and repeated budget refreshes",
          "[video_core][texture_memory_thresholds]") {
    constexpr std::array thresholds{
        VideoCommon::TextureMemoryThresholds{0, 0},
        VideoCommon::TextureMemoryThresholds{GiB / 2, GiB}, OotBudget,
    };
    for (const auto values : thresholds) {
        REQUIRE(ClampOotIntegratedTextureThresholds(Oot, true, 4 * GiB, 16 * GiB, values) == values);
    }
    REQUIRE(ClampOotIntegratedTextureThresholds(
                Oot, true, 4 * GiB, 16 * GiB,
                ClampOotIntegratedTextureThresholds(Oot, true, 4 * GiB, 16 * GiB, Original)) == OotBudget);
    REQUIRE(ClampOotIntegratedTextureThresholds(Oot, true, 4 * GiB, 16 * GiB, {8 * GiB, 9 * GiB}) ==
            OotBudget);
}
