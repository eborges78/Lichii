// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_multi_range_buffer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

bool MemoryTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_MEMORY_TRACE");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

} // namespace

MultiRangeBufferCache::MultiRangeBufferCache(const Device& device) {
    sparse_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (device.IsBufferDeviceAddressSupported()) {
        sparse_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    if (!device.IsSparseBindingSupported()) {
        return;
    }
    u32 memory_type_bits = 0;
    const VkDeviceSize queried = QueryBlockSize(device, memory_type_bits);
    if (queried == 0 || memory_type_bits == 0) {
        return;
    }
    block_size = queried;
    sparse_memory_type_bits = memory_type_bits;
    use_sparse = true;
}

VkDeviceSize MultiRangeBufferCache::QueryBlockSize(const Device& device,
                                                   u32& memory_type_bits) const {
    const VkDevice logical = *device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    const VkBufferCreateInfo probe_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
        .size = DEFAULT_BLOCK_SIZE,
        .usage = sparse_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer probe{};
    if (dld.vkCreateBuffer(logical, &probe_ci, nullptr, &probe) != VK_SUCCESS) {
        return 0;
    }
    const SparseBuffer owned{probe, logical, dld};
    const VkBufferMemoryRequirementsInfo2 reqs_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = nullptr,
        .buffer = probe,
    };
    VkMemoryRequirements2 reqs2{
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = nullptr,
        .memoryRequirements = {},
    };
    dld.vkGetBufferMemoryRequirements2(logical, &reqs_info, &reqs2);
    memory_type_bits = reqs2.memoryRequirements.memoryTypeBits;
    return reqs2.memoryRequirements.alignment;
}

u64 MultiRangeBufferCache::HashSources(std::span<const MultiRangeSource> sources) const {
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&hash](u64 value) {
        hash ^= value;
        hash *= 0x100000001b3ULL;
    };
    for (const MultiRangeSource& source : sources) {
        mix(u64(source.handle));
        mix(u64(source.offset));
        mix(u64(source.size));
    }
    return hash;
}

u64 MultiRangeBufferCache::HashContent(std::span<const MultiRangeSource> sources) const {
    u64 hash = 0xcbf29ce484222325ULL;
    for (const MultiRangeSource& source : sources) {
        hash ^= source.write_tick;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

bool MultiRangeBufferCache::CanBindSparse(std::span<const MultiRangeSource> sources) const {
    return use_sparse &&
           std::none_of(sources.begin(), sources.end(),
                        [block = block_size, bits = sparse_memory_type_bits](auto const& e) {
                            const VkDeviceSize memory_offset = e.memory_offset + e.offset;
                            return e.memory == VK_NULL_HANDLE || e.memory_type >= 32 ||
                                   ((bits >> e.memory_type) & 1) == 0 ||
                                   (memory_offset % block) != 0 || (e.size % block) != 0;
                        });
}

SparseBuffer MultiRangeBufferCache::CreateSparse(const Device& device, Scheduler& scheduler,
                                                 std::span<const MultiRangeSource> sources,
                                                 VkDeviceSize total) {
    const VkDevice logical = *device.GetLogical();
    const auto& dld = device.GetDispatchLoader();
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
        .size = total,
        .usage = sparse_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    VkBuffer raw{};
    if (dld.vkCreateBuffer(logical, &buffer_ci, nullptr, &raw) != VK_SUCCESS) {
        return SparseBuffer{};
    }
    SparseBuffer handle{raw, logical, dld};
    std::vector<VkSparseMemoryBind> binds;
    binds.reserve(sources.size());
    VkDeviceSize resource_offset = 0;
    for (const MultiRangeSource& source : sources) {
        binds.push_back(VkSparseMemoryBind{
            .resourceOffset = resource_offset,
            .size = source.size,
            .memory = source.memory,
            .memoryOffset = source.memory_offset + source.offset,
            .flags = 0,
        });
        resource_offset += source.size;
    }
    const VkSparseBufferMemoryBindInfo buffer_bind{
        .buffer = raw,
        .bindCount = static_cast<u32>(binds.size()),
        .pBinds = binds.data(),
    };
    const VkBindSparseInfo bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .bufferBindCount = 1,
        .pBufferBinds = &buffer_bind,
        .imageOpaqueBindCount = 0,
        .pImageOpaqueBinds = nullptr,
        .imageBindCount = 0,
        .pImageBinds = nullptr,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    const VkFenceCreateInfo fence_ci{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    vk::Fence fence = device.GetLogical().CreateFence(fence_ci);
    VkResult bind_result = VK_ERROR_UNKNOWN;
    {
        std::scoped_lock lock{scheduler.submit_mutex};
        bind_result = device.GetGraphicsQueue().BindSparse(bind_info, *fence);
    }
    if (bind_result != VK_SUCCESS) {
        return SparseBuffer{};
    }
    fence.Wait();
    return handle;
}

void MultiRangeBufferCache::RetireEntry(Scheduler& scheduler, Entry& entry) {
    if (!entry.sparse_handle && !entry.gathered) {
        return;
    }
    if (retired.size() == retired.capacity()) {
        DrainRetired(scheduler);
    }
    if (retired.size() == retired.capacity()) {
        u64 oldest = retired.front().tick;
        for (const Retired& item : retired) {
            if (item.tick < oldest) {
                oldest = item.tick;
            }
        }
        scheduler.Wait(oldest);
        DrainRetired(scheduler);
    }
    retired.push_back(Retired{
        .handle = std::move(entry.sparse_handle),
        .gathered = std::move(entry.gathered),
        .tick = scheduler.CurrentTick(),
        .size = entry.size,
        .owner_count = entry.owners.size(),
    });
}

void MultiRangeBufferCache::DrainRetired(Scheduler& scheduler) {
    size_t index = 0;
    while (index < retired.size()) {
        if (scheduler.IsFree(retired[index].tick)) {
            if (index + 1 != retired.size()) {
                retired[index] = std::move(retired.back());
            }
            retired.pop_back();
            ++released_count;
        } else {
            ++index;
        }
    }
}

void MultiRangeBufferCache::TickFrame(Scheduler& scheduler) {
    DrainRetired(scheduler);
    if (!MemoryTraceEnabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_memory_trace != std::chrono::steady_clock::time_point{} &&
        now - last_memory_trace < std::chrono::seconds{1}) {
        return;
    }
    last_memory_trace = now;
    struct Totals {
        u64 gathered_count{};
        u64 gathered_requested_bytes{};
        u64 sparse_count{};
        u64 sparse_span_bytes{};
        u64 owner_links{};

        void Add(bool sparse, VkDeviceSize size, size_t owners) {
            if (sparse) {
                ++sparse_count;
                sparse_span_bytes += size;
            } else {
                ++gathered_count;
                gathered_requested_bytes += size;
            }
            owner_links += owners;
        }
    } active, pending;
    for (const auto& [key, entry] : entries) {
        active.Add(static_cast<bool>(entry.sparse_handle), entry.size, entry.owners.size());
    }
    for (const auto& entry : retired) {
        pending.Add(static_cast<bool>(entry.handle), entry.size, entry.owner_count);
    }
    // Sparse spans alias source allocations; gathered sizes exclude VMA alignment/block slack.
    LOG_INFO(Render_Vulkan,
             "Nx2 multi-range memory: active_gathered={} active_gathered_requested_bytes={} "
             "active_sparse={} active_sparse_span_bytes={} active_owner_links={} "
             "retired_gathered={} retired_gathered_requested_bytes={} retired_sparse={} "
             "retired_sparse_span_bytes={} retired_owner_links={} "
             "created_gathered={} created_gathered_requested_bytes={} created_sparse={} "
             "created_sparse_span_bytes={} released={}",
             active.gathered_count, active.gathered_requested_bytes, active.sparse_count,
             active.sparse_span_bytes, active.owner_links, pending.gathered_count,
             pending.gathered_requested_bytes, pending.sparse_count, pending.sparse_span_bytes,
             pending.owner_links, gathered_created_count, gathered_created_requested_bytes,
             sparse_created_count, sparse_created_span_bytes, released_count);
}

void MultiRangeBufferCache::TraceGatheredRequest(VkDeviceSize size, size_t owners,
                                               bool sparse_candidate) {
    if (!MemoryTraceEnabled()) {
        return;
    }
    constexpr u64 large_request = u64{64} << 20;
    constexpr u64 cumulative_step = u64{256} << 20;
    constexpr u32 max_events = 128;
    constexpr u32 max_small_events = 8;
    const u64 previous_bytes = gathered_requested_bytes;
    ++gathered_request_count;
    gathered_requested_bytes += size;
    const bool large = size >= large_request;
    const bool crossed_step = previous_bytes / cumulative_step !=
                              gathered_requested_bytes / cumulative_step;
    if (gathered_request_events >= max_events ||
        (!large && (!crossed_step || small_gathered_request_events >= max_small_events))) {
        return;
    }
    ++gathered_request_events;
    if (!large) {
        ++small_gathered_request_events;
    }
    // Before allocation: this reports a request, even if VMA fails or no frame completes.
    LOG_INFO(Render_Vulkan,
             "Nx2 gathered buffer request: requested_bytes={} owner_links={} requests={} "
             "cumulative_requested_bytes={} sparse_enabled={} sparse_candidate={} "
             "event={}/128 small_events={}/8",
             size, owners, gathered_request_count, gathered_requested_bytes, use_sparse,
             sparse_candidate, gathered_request_events, small_gathered_request_events);
}

MultiRangeRef MultiRangeBufferCache::Get(const Device& device, Scheduler& scheduler,
                                         MemoryAllocator& memory_allocator, u64 key,
                                         std::span<const MultiRangeSource> sources,
                                         VkDeviceSize total) {
    if (sources.empty() || total == 0) {
        return MultiRangeRef{};
    }
    if (!retired.empty()) {
        DrainRetired(scheduler);
    }
    const u64 geometry = HashSources(sources);
    const u64 content = HashContent(sources);
    const auto it = entries.find(key);
    if (it != entries.end() && it->second.geometry == geometry && it->second.size == total) {
        Entry& entry = it->second;
        if (entry.content != content) {
            entry.content = content;
            entry.dirty = true;
        }
        MultiRangeRef ref{
            .handle = *entry.sparse_handle,
            .address = entry.address,
            .size = entry.size,
            .sparse = true,
            .needs_gather = false,
        };
        if (!entry.sparse_handle) {
            ref.handle = *entry.gathered;
            ref.sparse = false;
            ref.needs_gather = entry.dirty;
        }
        return ref;
    }
    if (it != entries.end()) {
        RetireEntry(scheduler, it->second);
        entries.erase(it);
    }

    Entry entry{};
    entry.geometry = geometry;
    entry.content = content;
    entry.size = total;
    entry.owners.reserve(sources.size());
    for (const MultiRangeSource& source : sources) {
        entry.owners.push_back(source.handle);
    }
    const bool sparse_candidate = CanBindSparse(sources);
    if (sparse_candidate) {
        entry.sparse_handle = CreateSparse(device, scheduler, sources, total);
    }
    if (!entry.sparse_handle) {
        VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (device.IsBufferDeviceAddressSupported()) {
            flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
        const VkBufferCreateInfo gather_ci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = total,
            .usage = flags,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };
        TraceGatheredRequest(total, sources.size(), sparse_candidate);
        entry.gathered = memory_allocator.CreateBuffer(gather_ci, MemoryUsage::DeviceLocal);
        entry.dirty = true;
    }
    if (entry.sparse_handle) {
        ++sparse_created_count;
        sparse_created_span_bytes += total;
    } else {
        ++gathered_created_count;
        gathered_created_requested_bytes += total;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        VkBuffer address_handle = *entry.sparse_handle;
        if (!entry.sparse_handle) {
            address_handle = *entry.gathered;
        }
        entry.address = device.GetLogical().GetBufferDeviceAddress(address_handle);
    }

    MultiRangeRef ref{
        .handle = *entry.sparse_handle,
        .address = entry.address,
        .size = entry.size,
        .sparse = true,
        .needs_gather = false,
    };
    if (!entry.sparse_handle) {
        ref.handle = *entry.gathered;
        ref.sparse = false;
        ref.needs_gather = true;
    }
    entries.emplace(key, std::move(entry));
    return ref;
}

void MultiRangeBufferCache::MarkGathered(u64 key) {
    if (auto const it = entries.find(key); it != entries.end()) {
        it->second.dirty = false;
    }
}

void MultiRangeBufferCache::DropOwner(Scheduler& scheduler, VkBuffer owner) {
    if (owner == VK_NULL_HANDLE) {
        return;
    }
    for (auto it = entries.begin(); it != entries.end();) {
        Entry& entry = it->second;
        bool owned = false;
        for (const VkBuffer handle : entry.owners) {
            if (handle == owner) {
                owned = true;
                break;
            }
        }
        if (!owned) {
            ++it;
            continue;
        }
        RetireEntry(scheduler, entry);
        it = entries.erase(it);
    }
}

void MultiRangeBufferCache::Invalidate(u64 key) {
    if (auto const it = entries.find(key); it != entries.end()) {
        it->second.dirty = true;
    }
}

} // namespace Vulkan
