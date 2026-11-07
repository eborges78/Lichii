// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <utility>

#include "common/common_types.h"

namespace VideoCommon {

[[nodiscard]] constexpr u64 TranscodeByteSum(u64 left, u64 right) {
    constexpr u64 Max = std::numeric_limits<u64>::max();
    return right > Max - left ? Max : left + right;
}

[[nodiscard]] constexpr u64 TranscodeByteProduct(u64 left, u64 right) {
    constexpr u64 Max = std::numeric_limits<u64>::max();
    return right != 0 && left > Max / right ? Max : left * right;
}

template <class Transcode>
[[nodiscard]] u64 TranscodeDecodedBytes(const Transcode& transcode) {
    u64 bytes{};
    for (u32 level = 0; level < transcode.levels; ++level) {
        const u64 width = level < 32 ? (std::max)(transcode.width >> level, 1U) : 1;
        const u64 height = level < 32 ? (std::max)(transcode.height >> level, 1U) : 1;
        bytes = TranscodeByteSum(bytes, TranscodeByteProduct(width * height, 4));
    }
    return TranscodeByteProduct(bytes, transcode.layers);
}

template <class Pending>
[[nodiscard]] bool TranscodesExceedBudget(const Pending& pending, u64 budget) {
    for (const auto& transcode : pending) {
        const u64 bytes = TranscodeDecodedBytes(transcode);
        if (bytes > budget) {
            return true;
        }
        budget -= bytes;
    }
    return false;
}

// Called after recording the entire decode/encode/copy transaction. Wait must complete the
// oldest tick; collect must remove only completed entries. Refuse to spin if it makes no progress.
template <class Pending, class Wait, class Collect>
[[nodiscard]] bool DrainTranscodesToBudget(Pending& pending, u64 budget, Wait&& wait,
                                          Collect&& collect) {
    while (TranscodesExceedBudget(pending, budget)) {
        const auto count = pending.size();
        wait(pending.front().tick);
        collect();
        if (pending.size() >= count) {
            return false;
        }
    }
    return true;
}

// Completion is supplied by the GPU scheduler. Collection does not advance frame ages or wait.
template <class Pending, class Free, class IsFree>
void CollectCompletedTranscodes(Pending& pending, Free& free, IsFree&& is_free,
                                u64 max_free_bytes) {
    u64 free_bytes{};
    for (const auto& transcode : free) {
        free_bytes = TranscodeByteSum(free_bytes, TranscodeDecodedBytes(transcode));
    }
    while (!pending.empty() && is_free(pending.front().tick)) {
        auto& done = pending.front();
        const u64 bytes = TranscodeDecodedBytes(done);
        if (free_bytes <= max_free_bytes && bytes <= max_free_bytes - free_bytes) {
            done.unused_frames = 0;
            free_bytes += bytes;
            free.push_back(std::move(done));
        }
        pending.pop_front();
    }
}

} // namespace VideoCommon
