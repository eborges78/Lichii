// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCore {

[[nodiscard]] constexpr bool IsXenoblade2CloudCandidateTitle(u64 program_id) noexcept {
    // Other title IDs and regional variants have not been evaluated for this candidate.
    return program_id == 0x0100E95004038000ULL;
}

} // namespace VideoCore
