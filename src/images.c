#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <vulkan/vulkan.h>

#include "annealing-lowlevel.h"
#include "internal.h"

static void
update_descriptors (struct an_image *image) {
    struct an_gpu_context *ctx = image->ctx;

    VkDescriptorBufferInfo memoryInfo;
    ZERO(memoryInfo);
    memoryInfo.buffer = image->imageMemory->buffer;
    memoryInfo.offset = 0;
    memoryInfo.range = sizeof (mycomplex) * image->actual_size;

    VkWriteDescriptorSet dsSet;
    ZERO (dsSet);
    dsSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSet.dstSet = image->descriptorSet;
    dsSet.dstBinding = 0; // binding #
    dsSet.dstArrayElement = 0;
    dsSet.descriptorCount = 1;
    dsSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSet.pBufferInfo = &memoryInfo;

    vkUpdateDescriptorSets (ctx->device, 1, &dsSet, 0, NULL);
}

void
an_image_synchronize (struct an_image *image) {
    struct an_gpu_context *ctx = image->ctx;

    if (image->computationLaunched) {
        image->computationLaunched = 0;

        vkWaitForFences (ctx->device, 1, &image->fence, VK_TRUE, -1);
        vkResetFences (ctx->device, 1, &image->fence);
    }
}

void
an_destroy_image (struct an_image *image) {
    struct an_gpu_context *ctx = image->ctx;

    if (image->fence != VK_NULL_HANDLE) {
        vkDestroyFence (ctx->device, image->fence, NULL);
    }

    if (image->descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets (ctx->device, ctx->descPool, 1, &image->descriptorSet);
    }

    if (image->commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers (ctx->device, ctx->cmdPool, 1, &image->commandBuffer);
    }

    if (image->imageMemory != NULL) {
        an_destroy_buffer (ctx, image->imageMemory);
    }
        
    free (image);
}

struct an_image*
an_create_image (struct an_gpu_context *ctx,
                 const float           *real,
                 const float           *imag,
                 const unsigned int    *dimensions,
                 unsigned int           ndim) {
    if (ndim != ctx->ndim) {
        fprintf (stderr, "Context dimensions must match image dimensions\n");
        return NULL;
    }

    mycomplex *data = NULL;
    VkResult result;
    struct an_image *image = malloc (sizeof (struct an_image));
    memset (image, 0, sizeof (struct an_image));

    image->ctx = ctx;
    image->updateData.ndim = ndim;
    memcpy(image->updateData.logical_dimensions, dimensions,
           sizeof (unsigned int) * MAX_DIMENSIONS);
    memcpy(image->updateData.actual_dimensions, dimensions,
           sizeof (unsigned int) * MAX_DIMENSIONS);
    image->updateData.actual_dimensions[image->updateData.ndim-1] =
        image->updateData.actual_dimensions[image->updateData.ndim-1] / 2 + 1;
    image->actual_size = 1;

    /* FIXME: Isn't it all the way around? */
    for (int i = 0; i < image->updateData.ndim; i++) {
        image->updateData.stride[i] = 1;
        for (int j = i + 1; j < image->updateData.ndim; j++) {
            image->updateData.stride[i] *= image->updateData.actual_dimensions[j];
        }
    }

    for (int i = 0; i < image->updateData.ndim; i++) {
        image->actual_size *= image->updateData.actual_dimensions[i];
    }

    image->imageMemory =
        an_create_buffer (ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          image->actual_size * sizeof(mycomplex));
    if (image->imageMemory == NULL) {
        fprintf (stderr, "Cannot create input buffer\n");
        goto cleanup;
    }

    data = malloc (image->actual_size * sizeof (mycomplex));
    for (int i = 0; i < image->actual_size; i++) {
        data[i].re = real[i];
        data[i].im = imag[i];
    }

    if (!an_write_data (ctx, image->imageMemory, data,
                        sizeof (mycomplex) * image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        free (data);
        goto cleanup;
    }
    free (data);

    /* Number of work groups */
    const unsigned int *grpSize = &update_group_sizes[MAX_DIMENSIONS * (ndim - 1)];
    for (uint32_t i = 0; i < MAX_DIMENSIONS; i++) {
        image->ngroups[i] = (i < ndim) ?
            ceil((double)image->updateData.actual_dimensions[i] / (double) grpSize[i]) : 1;
    }

    result = an_create_command_buffer (ctx, &image->commandBuffer);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate command buffer, code = %i\n", result);
        goto cleanup;
    }

    result = an_allocate_descriptor_set (ctx, ctx->pipelines[PIPELINE_CFUPDATE],
                                         &image->descriptorSet);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate descriptor set, code = %i\n", result);
        goto cleanup;
    }

    result = an_create_fence (ctx, &image->fence);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create a fence, code = %i\n", result);
        goto cleanup;
    }

    update_descriptors (image);
    return image;

cleanup:
    an_destroy_image (image);
    return NULL;
}

int
an_image_get (struct an_image *image,
              float           *real,
              float           *imag) {
    struct an_gpu_context *ctx = image->ctx;

    an_image_synchronize (image);
    mycomplex *data = malloc (sizeof (mycomplex) * image->actual_size);
    int res = an_read_data (ctx, image->imageMemory, data,
                            sizeof (mycomplex) * image->actual_size);
    for (int i = 0; i < image->actual_size; i++) {
        real[i] = data[i].re;
        imag[i] = data[i].im;
    }
    free (data);
    return res;
}

int
an_image_update_fft (struct an_image    *image,
                     const unsigned int *coord,
                     unsigned int        ndim,
                     float               delta) {
    if (ndim != image->updateData.ndim) {
        fprintf (stderr, "Wrong dimensions\n");
        return 0;
    }

    memcpy(image->updateData.point, coord, sizeof (unsigned int) * ndim);
    image->updateData.c = delta;

    struct an_gpu_context *ctx = image->ctx;
    struct pipeline *updPipeline = ctx->pipelines[PIPELINE_CFUPDATE];

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    an_image_synchronize (image);
    vkBeginCommandBuffer (image->commandBuffer, &beginInfo);
    vkCmdBindPipeline (image->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       updPipeline->pipeline);
    vkCmdBindDescriptorSets (image->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             updPipeline->pipelineLayout,
                             0, 1, &image->descriptorSet, 0, NULL);
    vkCmdPushConstants (image->commandBuffer, updPipeline->pipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof (struct CFUpdateData), &image->updateData);
    vkCmdDispatch (image->commandBuffer,
                   image->ngroups[0], image->ngroups[1], image->ngroups[2]);
    vkEndCommandBuffer (image->commandBuffer);

    VkSubmitInfo submitInfo;
    ZERO(submitInfo);
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &image->commandBuffer;
    vkQueueSubmit (ctx->queue, 1, &submitInfo, image->fence);
    image->computationLaunched = 1;

    return 1;
}
