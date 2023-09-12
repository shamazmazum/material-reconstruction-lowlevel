#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <vulkan/vulkan.h>

#include "annealing-lowlevel.h"
#include "internal.h"

#define METRIC_GRP_SIZE 64

/* =================
 * =================
 * = Image Buffers =
 * =================
 * =================
 */

typedef union
{
    float __attribute__((aligned (8))) s[2];
    __extension__ struct{ float  re, im; };
} mycomplex;

enum image_type {
    IMAGE_ABS,
    IMAGE_COMPLEX
};

struct an_image {
    enum image_type type;
    struct an_gpu_context *ctx;
    VkCommandBuffer commandBuffer;

    struct CFUpdateData updateData;
    size_t actual_size;
    uint32_t ngroups[MAX_DIMENSIONS];

    struct an_image_memory *metricMemory;
    struct an_image_memory *inputMemory;
    struct an_image_memory *outputMemory;
    /* Not a separate buffer, just switches between other two */
    struct an_image_memory *savedMemory;
};

static int
create_command_buffer (VkDevice device, VkCommandPool pool, VkCommandBuffer *buffer) {
    VkResult result;

    VkCommandBufferAllocateInfo cbAllocInfo;
    ZERO(cbAllocInfo);
    cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAllocInfo.commandPool = pool;
    cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAllocInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers (device, &cbAllocInfo, buffer);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate command buffer, code = %i\n", result);
    }

    return result == VK_SUCCESS;
}

void
an_destroy_image (struct an_image *image) {
    assert (image->savedMemory != image->outputMemory);
    struct an_gpu_context *ctx = image->ctx;

    if (image->commandBuffer != NULL) {
        vkFreeCommandBuffers (ctx->device, ctx->cmdPool, 1, &image->commandBuffer);
    }

    if (image->metricMemory != NULL) {
        an_destroy_buffer (ctx, image->metricMemory);
    }

    if (image->savedMemory != NULL) {
        an_destroy_buffer (ctx, image->savedMemory);
    }

    if (image->outputMemory != NULL) {
        an_destroy_buffer (ctx, image->outputMemory);
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

    struct an_image *image = malloc (sizeof (struct an_image));
    memset (image, 0, sizeof (struct an_image));

    image->type = IMAGE_COMPLEX;
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

    printf ("Actual buffer size: %lu\n", image->actual_size * sizeof (mycomplex));
    image->inputMemory =
        an_create_buffer (ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          image->actual_size * sizeof(mycomplex));
    if (image->inputMemory == NULL) {
        fprintf (stderr, "Cannot create input buffer\n");
        goto cleanup;
    }

    image->outputMemory =
        an_create_buffer (ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          image->actual_size * sizeof(mycomplex));
    if (image->outputMemory == NULL) {
        fprintf (stderr, "Cannot create output buffer\n");
        goto cleanup;
    }

    data = malloc (image->actual_size * sizeof (mycomplex));
    for (int i = 0; i < image->actual_size; i++) {
        data[i].re = real[i];
        data[i].im = imag[i];
    }

    if (!an_write_data (ctx, image->inputMemory, data,
                        sizeof (mycomplex) * image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    if (!an_write_data (ctx, image->outputMemory, data,
                        sizeof (mycomplex) * image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    /* Number of work groups */
    const unsigned int *grpSize = &update_group_sizes[MAX_DIMENSIONS * (ndim - 1)];
    for (uint32_t i = 0; i < MAX_DIMENSIONS; i++) {
        image->ngroups[i] = (i < ndim) ?
            ceil((double)image->updateData.actual_dimensions[i] / (double) grpSize[i]) : 1;
    }

    if (!create_command_buffer (ctx->device, ctx->cmdPool, &image->commandBuffer)) {
        goto cleanup;
    }

    image->savedMemory = image->inputMemory;
    free (data);
    return image;

cleanup:
    free (data);
    an_destroy_image (image);
    return NULL;
}

int
an_image_get (struct an_image *image,
              float           *real,
              float           *imag) {
    mycomplex *data = malloc (sizeof (mycomplex) * image->actual_size);
    int res = an_read_data (image->ctx, image->outputMemory, data,
                            sizeof (mycomplex) * image->actual_size);
    for (int i = 0; i < image->actual_size; i++) {
        real[i] = data[i].re;
        imag[i] = data[i].im;
    }
    free (data);
    return res;
}

void
an_image_store_state (struct an_image *image) {
    assert (image->savedMemory != image->outputMemory);

    struct an_image_memory *tmp = image->savedMemory;
    image->savedMemory  = image->outputMemory;
    image->outputMemory = tmp;
    image->inputMemory  = image->savedMemory;
}

void
an_image_rollback (struct an_image *image) {
    assert (image->savedMemory != image->outputMemory);

    struct an_image_memory *tmp = image->outputMemory;
    image->inputMemory  = image->savedMemory;
    image->outputMemory = image->savedMemory;
    image->savedMemory  = tmp;
}

int
an_image_update_fft (struct an_image    *image,
                     const unsigned int *coord,
                     unsigned int        ndim,
                     float               delta) {
    VkResult result;

    if (ndim != image->updateData.ndim) {
        fprintf (stderr, "Wrong dimensions\n");
        return 0;
    }

    if (image->type != IMAGE_COMPLEX) {
        fprintf (stderr, "Wrong image type\n");
        return 0;
    }

    assert (image->savedMemory != image->outputMemory);
    memcpy(image->updateData.point, coord, sizeof (unsigned int) * ndim);
    image->updateData.c = delta;

    struct an_gpu_context *ctx = image->ctx;
    struct pipeline *updPipeline = ctx->pipelines[PIPELINE_CFUPDATE];

    VkDescriptorBufferInfo inputInfo;
    ZERO(inputInfo);
    inputInfo.buffer = image->inputMemory->buffer;
    inputInfo.offset = 0;
    inputInfo.range = sizeof (mycomplex) * image->actual_size;

    VkDescriptorBufferInfo outputInfo;
    ZERO(outputInfo);
    outputInfo.buffer = image->outputMemory->buffer;
    outputInfo.offset = 0;
    outputInfo.range = sizeof (mycomplex) * image->actual_size;

    VkWriteDescriptorSet dsSets[2];
    memset (dsSets, 0, sizeof (dsSets));
    dsSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[0].dstSet = updPipeline->descriptorSet;
    dsSets[0].dstBinding = 0; // binding #
    dsSets[0].dstArrayElement = 0;
    dsSets[0].descriptorCount = 1;
    dsSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[0].pBufferInfo = &inputInfo;

    dsSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[1].dstSet = updPipeline->descriptorSet;
    dsSets[1].dstBinding = 1; // binding #
    dsSets[1].dstArrayElement = 0;
    dsSets[1].descriptorCount = 1;
    dsSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[1].pBufferInfo = &outputInfo;

    vkUpdateDescriptorSets (ctx->device, 2, dsSets, 0, NULL);

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer (image->commandBuffer, &beginInfo);
    vkCmdBindPipeline (image->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       updPipeline->pipeline);
    vkCmdBindDescriptorSets (image->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             updPipeline->pipelineLayout,
                             0, 1, &updPipeline->descriptorSet, 0, NULL);
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
    vkQueueSubmit (ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);

    /* End */
    image->inputMemory = image->outputMemory;

    return 1;
}

struct an_image*
an_create_corrfn (struct an_gpu_context *ctx,
                  const float           *corrfn,
                  const unsigned int    *dimensions,
                  unsigned int           ndim) {
    if (ndim != ctx->ndim) {
        fprintf (stderr, "Context dimensions must match image dimensions\n");
        return NULL;
    }

    struct an_image *image = malloc (sizeof (struct an_image));
    memset (image, 0, sizeof (struct an_image));

    image->type = IMAGE_ABS;
    image->ctx = ctx;
    image->actual_size = 1;

    for (int i = 0; i < ndim-1; i++) {
        image->actual_size *= dimensions[i];
    }
    image->actual_size *= dimensions[ndim-1] / 2 + 1;

    printf ("Actual buffer size: %lu\n", image->actual_size * sizeof (mycomplex));

    /* Store correlation data here */
    image->outputMemory =
        an_create_buffer (ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          image->actual_size * sizeof(float));
    if (image->outputMemory == NULL) {
        fprintf (stderr, "Cannot create output buffer\n");
        goto cleanup;
    }

    /* Temporary buffer for metric */
    image->metricMemory =
        an_create_buffer (ctx,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          image->actual_size * sizeof(float));
    if (image->metricMemory == NULL) {
        fprintf (stderr, "Cannot create metric buffer\n");
        goto cleanup;
    }

    if (!an_write_data (ctx, image->outputMemory, corrfn,
                        sizeof (float) * image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    if (!create_command_buffer (ctx->device, ctx->cmdPool, &image->commandBuffer)) {
        goto cleanup;
    }

    return image;

cleanup:
    an_destroy_image (image);
    return NULL;
}

static void
invoke_metric_kernel (struct an_image *target,
                      struct an_image *recon) {
    struct MetricUpdateData params;
    params.length = recon->actual_size;

    struct an_gpu_context *ctx = recon->ctx;
    struct pipeline *metricPipeline = ctx->pipelines[PIPELINE_METRIC];

    VkDescriptorBufferInfo cfInfo;
    ZERO(cfInfo);
    cfInfo.buffer = target->outputMemory->buffer;
    cfInfo.offset = 0;
    cfInfo.range = sizeof (float) * target->actual_size;

    VkDescriptorBufferInfo reconInfo;
    ZERO(reconInfo);
    reconInfo.buffer = recon->outputMemory->buffer;
    reconInfo.offset = 0;
    reconInfo.range = sizeof (mycomplex) * recon->actual_size;

    VkDescriptorBufferInfo outputInfo;
    ZERO(outputInfo);
    outputInfo.buffer = target->metricMemory->buffer;
    outputInfo.offset = 0;
    outputInfo.range = sizeof (float) * recon->actual_size;

    VkWriteDescriptorSet dsSets[3];
    memset (dsSets, 0, sizeof (dsSets));
    dsSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[0].dstSet = metricPipeline->descriptorSet;
    dsSets[0].dstBinding = 0; // binding #
    dsSets[0].dstArrayElement = 0;
    dsSets[0].descriptorCount = 1;
    dsSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[0].pBufferInfo = &cfInfo;

    dsSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[1].dstSet = metricPipeline->descriptorSet;
    dsSets[1].dstBinding = 1; // binding #
    dsSets[1].dstArrayElement = 0;
    dsSets[1].descriptorCount = 1;
    dsSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[1].pBufferInfo = &reconInfo;

    dsSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[2].dstSet = metricPipeline->descriptorSet;
    dsSets[2].dstBinding = 2; // binding #
    dsSets[2].dstArrayElement = 0;
    dsSets[2].descriptorCount = 1;
    dsSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[2].pBufferInfo = &outputInfo;

    vkUpdateDescriptorSets (ctx->device, 3, dsSets, 0, NULL);

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer (target->commandBuffer, &beginInfo);
    vkCmdBindPipeline (target->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       metricPipeline->pipeline);
    vkCmdBindDescriptorSets (target->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             metricPipeline->pipelineLayout,
                             0, 1, &metricPipeline->descriptorSet, 0, NULL);
    vkCmdPushConstants (target->commandBuffer, metricPipeline->pipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof (struct MetricUpdateData), &params);
    vkCmdDispatch (target->commandBuffer,
                   ceil((double) target->actual_size / (double)METRIC_GRP_SIZE), 1, 1);
    vkEndCommandBuffer (target->commandBuffer);

    VkSubmitInfo submitInfo;
    ZERO(submitInfo);
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &target->commandBuffer;
    vkQueueSubmit (ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
}

static void
invoke_reduce_kernel (struct an_image *target) {
    struct an_gpu_context *ctx = target->ctx;
    struct pipeline *reducePipeline = ctx->pipelines[PIPELINE_REDUCE];

    VkDescriptorBufferInfo mInfo;
    ZERO(mInfo);
    mInfo.buffer = target->metricMemory->buffer;
    mInfo.offset = 0;
    mInfo.range = sizeof (float) * target->actual_size;

    VkWriteDescriptorSet dsSets;
    ZERO (dsSets);
    dsSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets.dstSet = reducePipeline->descriptorSet;
    dsSets.dstBinding = 0; // binding #
    dsSets.dstArrayElement = 0;
    dsSets.descriptorCount = 1;
    dsSets.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets.pBufferInfo = &mInfo;

    vkUpdateDescriptorSets (ctx->device, 1, &dsSets, 0, NULL);

    VkMemoryBarrier memoryBarrier;
    ZERO (memoryBarrier);
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    struct MetricUpdateData params;
    params.length = target->actual_size;

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer (target->commandBuffer, &beginInfo);
    vkCmdBindPipeline (target->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       reducePipeline->pipeline);
    vkCmdBindDescriptorSets (target->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             reducePipeline->pipelineLayout,
                             0, 1, &reducePipeline->descriptorSet, 0, NULL);
    while (params.length > 0) {
        int groups = ceil((double)params.length / (double)METRIC_GRP_SIZE);
        vkCmdPushConstants (target->commandBuffer, reducePipeline->pipelineLayout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof (struct MetricUpdateData), &params);
        vkCmdDispatch (target->commandBuffer, groups, 1, 1);
        vkCmdPipelineBarrier(target->commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &memoryBarrier, 0, NULL, 0, NULL);
        params.length = (groups == 1) ? 0 : groups;
    }
    vkEndCommandBuffer (target->commandBuffer);

    VkSubmitInfo submitInfo;
    ZERO(submitInfo);
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &target->commandBuffer;
    vkQueueSubmit (ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
}

int
an_distance (struct an_image *target,
             struct an_image *recon,
             float           *distance) {
    if (target->type != IMAGE_ABS || recon->type != IMAGE_COMPLEX ||
        target->ctx  != recon->ctx ||
        target->actual_size != recon->actual_size) {
        fprintf (stderr, "Incompatible images\n");
        return 0;
    }

    invoke_metric_kernel (target, recon);
    invoke_reduce_kernel (target);

    struct an_gpu_context *ctx = recon->ctx;
    if (!an_read_data (ctx, target->metricMemory, distance, sizeof (float))) {
        fprintf (stderr, "Cannot read metric\n");
        return 0;
    }
    return 1;
}
