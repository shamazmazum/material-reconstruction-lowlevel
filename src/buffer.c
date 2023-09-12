#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vulkan/vulkan.h>

#include "annealing-lowlevel.h"
#include "internal.h"

void
an_destroy_buffer (struct an_gpu_context *ctx, struct an_image_memory *imemory) {
    assert (ctx->device != VK_NULL_HANDLE);

    if (imemory->memory != VK_NULL_HANDLE) {
        vkFreeMemory (ctx->device, imemory->memory, NULL);
    }

    if (imemory->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer (ctx->device, imemory->buffer, NULL);
    }
}

struct an_image_memory*
an_create_buffer (struct an_gpu_context *ctx, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties, size_t size) {
    assert (ctx->physDev != VK_NULL_HANDLE && ctx->device != VK_NULL_HANDLE);

    VkResult result;
    struct an_image_memory *imemory = malloc (sizeof (struct an_image_memory));
    memset (imemory, 0, sizeof (struct an_image_memory));
    
    VkBufferCreateInfo bufferInfo;
    ZERO(bufferInfo);
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(ctx->device, &bufferInfo, NULL, &imemory->buffer);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create buffer, code = %i\n", result);
        goto cleanup;
    }

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(ctx->physDev, &memProperties);
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(ctx->device, imemory->buffer, &memRequirements);

    int i;
    for (i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            break;
        }
    }

    if (i == memProperties.memoryTypeCount) {
        fprintf (stderr, "Cannot find memory with needed requirements\n");
        goto cleanup;
    }

    VkMemoryAllocateInfo allocInfo;
    ZERO(allocInfo);
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = i;

    result = vkAllocateMemory(ctx->device, &allocInfo, NULL, &imemory->memory);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate buffer memory, code = %i\n", result);
        goto cleanup;
    }

    vkBindBufferMemory (ctx->device, imemory->buffer, imemory->memory, 0);
    return imemory;

cleanup:
    an_destroy_buffer (ctx, imemory);
    return NULL;
}

static int
copy_buffer (struct an_gpu_context *ctx, VkBuffer source, VkBuffer destination,
             VkDeviceSize size) {
    VkResult result;
    VkCommandBuffer commandBuffer;
    result = an_create_command_buffer (ctx, &commandBuffer);
    if (result != VK_SUCCESS) {
        return 0;
    }

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion;
    ZERO(copyRegion);
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, source, destination, 1, &copyRegion);
    vkEndCommandBuffer (commandBuffer);

    VkSubmitInfo submitInfo;
    ZERO(submitInfo);
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
    vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &commandBuffer);

    return 1;
}

int
an_write_data (struct an_gpu_context *ctx, struct an_image_memory *imageMemory,
               const void *data, size_t size) {
    int code = 1;
    void *ptr;
    VkResult result;

    struct an_image_memory *tmp =
        an_create_buffer (ctx, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          size);
    if (tmp == NULL) {
        fprintf (stderr, "Cannot create staging buffer\n");
        code = 0;
        goto cleanup;
    }

    result = vkMapMemory (ctx->device, tmp->memory, 0, size, 0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map memory\n");
        code = 0;
        goto cleanup;
    }

    memcpy(ptr, data, size);
    vkUnmapMemory (ctx->device, tmp->memory);

    if (!copy_buffer (ctx, tmp->buffer, imageMemory->buffer, size)) {
        fprintf (stderr, "Cannot copy data from staging buffer\n");
        code = 0;
        goto cleanup;
    }

cleanup:
    if (tmp != NULL) {
        an_destroy_buffer (ctx, tmp);
    }

    return code;
}

int
an_read_data (struct an_gpu_context *ctx, struct an_image_memory *imageMemory,
              void *data, size_t size) {
    int code = 1;
    void *ptr;
    VkResult result;

    struct an_image_memory *tmp =
        an_create_buffer (ctx, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          size);
    if (tmp == NULL) {
        fprintf (stderr, "Cannot create staging buffer\n");
        code = 0;
        goto cleanup;
    }

    if (!copy_buffer (ctx, imageMemory->buffer, tmp->buffer, size)) {
        fprintf (stderr, "Cannot copy data to staging buffer\n");
        code = 0;
        goto cleanup;
    }

    result = vkMapMemory (ctx->device, tmp->memory, 0, size, 0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map memory\n");
        code = 0;
        goto cleanup;
    }

    memcpy(data, ptr, size);
    vkUnmapMemory (ctx->device, tmp->memory);

cleanup:
    if (tmp != NULL) {
        an_destroy_buffer (ctx, tmp);
    }

    return code;
}
