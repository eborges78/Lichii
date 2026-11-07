// SPDX-License-Identifier: GPL-3.0-or-later

// Standalone Vulkan runner for EDEN_TENSOR_TEST_OUTPUT compute fixtures.
// c++ -std=c++20 -O2 tensor_fixture_runner.cpp -lvulkan -o tensor_fixture_runner
// tensor_fixture_runner fixture.spv output.bin [32|64]

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string{operation} + ": VkResult " + std::to_string(result));
    }
}

struct Context {
    VkInstance instance{};
    VkDevice device{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    VkDescriptorSetLayout set_layout{};
    VkDescriptorPool descriptor_pool{};
    VkPipelineLayout pipeline_layout{};
    VkShaderModule shader{};
    VkPipeline pipeline{};
    VkCommandPool command_pool{};
    VkFence fence{};
    void* mapped{};

    ~Context() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (mapped) vkUnmapMemory(device, memory);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, shader, nullptr);
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            vkDestroyBuffer(device, buffer, nullptr);
            vkFreeMemory(device, memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};
} // namespace

int main(int argc, char** argv) try {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: tensor_fixture_runner fixture.spv output.bin [32|64]\n";
        return 2;
    }
    const uint32_t subgroup = argc == 4 ? static_cast<uint32_t>(std::stoul(argv[3])) : 32;
    if (subgroup != 32 && subgroup != 64) {
        throw std::runtime_error("Subgroup size must be 32 or 64");
    }
    std::ifstream input{argv[1], std::ios::binary | std::ios::ate};
    const auto bytes = input.tellg();
    if (!input || bytes < 20 || bytes % 4 != 0) {
        throw std::runtime_error("Invalid SPIR-V file");
    }
    std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(code.data()), bytes);
    if (!input || code.front() != 0x07230203) {
        throw std::runtime_error("Could not read SPIR-V");
    }
    bool uses_float16 = false;
    for (size_t offset = 5; offset < code.size();) {
        const uint32_t length = code[offset] >> 16;
        if (length == 0 || length > code.size() - offset) {
            throw std::runtime_error("Malformed SPIR-V instruction");
        }
        if ((code[offset] & 0xffff) == 17 && length == 2 && code[offset + 1] == 9) {
            uses_float16 = true;
        }
        offset += length;
    }

    Context ctx;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Lichii tensor fixture validation";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    Check(vkCreateInstance(&instance_info, nullptr, &ctx.instance), "vkCreateInstance");
    uint32_t count = 0;
    Check(vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr), "Enumerate device count");
    std::vector<VkPhysicalDevice> devices(count);
    Check(vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data()), "Enumerate devices");
    VkPhysicalDevice physical{};
    uint32_t queue_family = 0;
    VkPhysicalDeviceProperties properties{};
    for (const auto candidate : devices) {
        VkPhysicalDeviceSubgroupSizeControlProperties sizes{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroups{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        subgroups.pNext = &sizes;
        VkPhysicalDeviceFloatControlsProperties float_controls{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        sizes.pNext = &float_controls;
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &subgroups;
        vkGetPhysicalDeviceProperties2(candidate, &props);
        VkPhysicalDeviceVulkan13Features features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features.pNext = &features12;
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &features;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);
        constexpr auto required_ops = VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT;
        if (props.properties.apiVersion < VK_API_VERSION_1_3 ||
            !features.subgroupSizeControl || !features.computeFullSubgroups ||
            (uses_float16 && (!features12.shaderFloat16 ||
                             !float_controls.shaderDenormPreserveFloat16 ||
                             !float_controls.shaderSignedZeroInfNanPreserveFloat16 ||
                             !float_controls.shaderRoundingModeRTEFloat16)) ||
            subgroup < sizes.minSubgroupSize || subgroup > sizes.maxSubgroupSize ||
            !(sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            !(subgroups.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            (subgroups.supportedOperations & required_ops) != required_ops) {
            continue;
        }
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queue_count, queues.data());
        for (uint32_t i = 0; i < queue_count; ++i) {
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical = candidate;
                properties = props.properties;
                queue_family = i;
                break;
            }
        }
        if (physical) break;
    }
    if (!physical) {
        throw std::runtime_error("No Vulkan 1.3 compute device supports the requested full subgroup and quad/shuffle operations");
    }
    std::cout << "Device: " << properties.deviceName << "; driver=" << properties.driverVersion
              << "; subgroup=" << subgroup << "; float16=" << uses_float16
              << "; workgroup=64; dispatch=1x1x1\n";
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan13Features enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enabled.subgroupSizeControl = VK_TRUE;
    enabled.computeFullSubgroups = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enabled12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enabled12.shaderFloat16 = uses_float16;
    enabled.pNext = &enabled12;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &enabled;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    Check(vkCreateDevice(physical, &device_info, nullptr, &ctx.device), "vkCreateDevice");
    VkQueue queue{};
    vkGetDeviceQueue(ctx.device, queue_family, 0, &queue);

    constexpr VkDeviceSize output_bytes = 64 * 2 * sizeof(uint32_t);
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = output_bytes;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateBuffer(ctx.device, &buffer_info, nullptr, &ctx.buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.device, ctx.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    uint32_t memory_type = memory_properties.memoryTypeCount;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1U << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memory_type = i;
            break;
        }
    }
    if (memory_type == memory_properties.memoryTypeCount) {
        throw std::runtime_error("No host-visible memory type");
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type;
    Check(vkAllocateMemory(ctx.device, &allocation, nullptr, &ctx.memory), "vkAllocateMemory");
    Check(vkBindBufferMemory(ctx.device, ctx.buffer, ctx.memory, 0), "vkBindBufferMemory");
    Check(vkMapMemory(ctx.device, ctx.memory, 0, VK_WHOLE_SIZE, 0, &ctx.mapped), "vkMapMemory");
    std::memset(ctx.mapped, 0xcd, output_bytes);
    VkMappedMemoryRange mapped_range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    mapped_range.memory = ctx.memory;
    mapped_range.size = VK_WHOLE_SIZE;
    Check(vkFlushMappedMemoryRanges(ctx.device, 1, &mapped_range), "vkFlushMappedMemoryRanges");

    VkDescriptorSetLayoutBinding binding{};
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 1;
    set_info.pBindings = &binding;
    Check(vkCreateDescriptorSetLayout(ctx.device, &set_info, nullptr, &ctx.set_layout), "vkCreateDescriptorSetLayout");
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    Check(vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &ctx.descriptor_pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo set_allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_allocation.descriptorPool = ctx.descriptor_pool;
    set_allocation.descriptorSetCount = 1;
    set_allocation.pSetLayouts = &ctx.set_layout;
    VkDescriptorSet set{};
    Check(vkAllocateDescriptorSets(ctx.device, &set_allocation, &set), "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo descriptor{ctx.buffer, 0, output_bytes};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptor;
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &ctx.set_layout;
    Check(vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &ctx.pipeline_layout), "vkCreatePipelineLayout");
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = code.size() * sizeof(uint32_t);
    shader_info.pCode = code.data();
    Check(vkCreateShaderModule(ctx.device, &shader_info, nullptr, &ctx.shader), "vkCreateShaderModule");
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    required.requiredSubgroupSize = subgroup;
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.pNext = &required;
    pipeline_info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = ctx.shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = ctx.pipeline_layout;
    Check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &ctx.pipeline), "vkCreateComputePipelines");

    VkCommandPoolCreateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_info.queueFamilyIndex = queue_family;
    Check(vkCreateCommandPool(ctx.device, &command_info, nullptr, &ctx.command_pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo command_allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocation.commandPool = ctx.command_pool;
    command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocation.commandBufferCount = 1;
    VkCommandBuffer command{};
    Check(vkAllocateCommandBuffers(ctx.device, &command_allocation, &command), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, 1, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    Check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    Check(vkCreateFence(ctx.device, &fence_info, nullptr, &ctx.fence), "vkCreateFence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    Check(vkQueueSubmit(queue, 1, &submit, ctx.fence), "vkQueueSubmit");
    Check(vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, 30'000'000'000ULL), "vkWaitForFences");
    Check(vkInvalidateMappedMemoryRanges(ctx.device, 1, &mapped_range), "vkInvalidateMappedMemoryRanges");
    std::ofstream output{argv[2], std::ios::binary};
    output.write(static_cast<const char*>(ctx.mapped), output_bytes);
    if (!output) throw std::runtime_error("Could not write output file");
    std::cout << "Wrote " << output_bytes << " bytes to " << argv[2] << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
