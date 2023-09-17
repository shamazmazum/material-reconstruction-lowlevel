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

    VkDescriptorBufferInfo uniInfo;
    ZERO(uniInfo);
    uniInfo.buffer = image->uniformMemory->buffer;
    uniInfo.offset = 0;
    uniInfo.range = sizeof (struct CFUpdateDataUni);

    VkWriteDescriptorSet dsSets[2];
    memset (dsSets, 0, sizeof (dsSets));
    dsSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[0].dstSet = image->descriptorSet;
    dsSets[0].dstBinding = 0; // binding #
    dsSets[0].dstArrayElement = 0;
    dsSets[0].descriptorCount = 1;
    dsSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[0].pBufferInfo = &memoryInfo;
    dsSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[1].dstSet = image->descriptorSet;
    dsSets[1].dstBinding = 1; // binding #
    dsSets[1].dstArrayElement = 0;
    dsSets[1].descriptorCount = 1;
    dsSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    dsSets[1].pBufferInfo = &uniInfo;

    vkUpdateDescriptorSets (ctx->device, 2, dsSets, 0, NULL);
}

static void
record_command_buffer (struct an_image *image) {
    struct an_gpu_context *ctx = image->ctx;
    struct pipeline *updPipeline = ctx->pipelines[PIPELINE_CFUPDATE];

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer (image->commandBuffer, &beginInfo);
    vkCmdBindPipeline (image->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       updPipeline->pipeline);
    vkCmdBindDescriptorSets (image->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             updPipeline->pipelineLayout,
                             0, 1, &image->descriptorSet, 0, NULL);
    vkCmdPushConstants (image->commandBuffer, updPipeline->pipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof (struct CFUpdateDataConst), &image->updateData);
    vkCmdDispatch (image->commandBuffer,
                   image->ngroups[0], image->ngroups[1], image->ngroups[2]);
    vkEndCommandBuffer (image->commandBuffer);
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

    if (image->uniformPtr != NULL) {
        vkUnmapMemory (ctx->device, image->uniformMemory->memory);
    }

    if (image->uniformMemory != NULL) {
        an_destroy_buffer (ctx, image->uniformMemory);
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
    image->actual_size = 1;

    for (int i = 0; i < image->updateData.ndim; i++) {
        image->updateData.logical_dimensions[i] = dimensions[image->updateData.ndim - i - 1];
        image->updateData.actual_dimensions[i] = dimensions[image->updateData.ndim - i - 1];
    }

    image->updateData.actual_dimensions[0] =
        image->updateData.actual_dimensions[0] / 2 + 1;

    for (int i = 0; i < image->updateData.ndim; i++) {
        image->updateData.stride[i] = 1;
    }

    for (int i = 1; i < image->updateData.ndim; i++) {
        image->updateData.stride[i] = image->updateData.stride[i-1] *
            image->updateData.actual_dimensions[i-1];
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

    image->uniformMemory =
        an_create_buffer (ctx, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          sizeof (struct CFUpdateDataUni));
    if (image->uniformMemory == NULL) {
        fprintf (stderr, "Cannot create uniform buffer\n");
        goto cleanup;
    }

    void *ptr;
    result = vkMapMemory (ctx->device, image->uniformMemory->memory, 0,
                          sizeof (struct CFUpdateDataUni), 0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map uniform buffer, code = %i\n", result);
        goto cleanup;
    }
    image->uniformPtr = ptr;

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
    record_command_buffer (image);
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

    struct an_gpu_context *ctx = image->ctx;
    an_image_synchronize (image);
    image->uniformPtr->c = delta;
    for (int i = 0; i < image->updateData.ndim; i++) {
        image->uniformPtr->point[i] = coord[image->updateData.ndim - i - 1];
    }

    VkSubmitInfo submitInfo;
    ZERO(submitInfo);
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &image->commandBuffer;
    vkQueueSubmit (ctx->queue, 1, &submitInfo, image->fence);
    image->computationLaunched = 1;

    return 1;
}
