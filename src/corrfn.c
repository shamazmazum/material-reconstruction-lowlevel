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

void
an_destroy_corrfn (struct an_corrfn *image) {
    struct an_gpu_context *ctx = image->ctx;

    if (image->commandBuffer != NULL) {
        vkFreeCommandBuffers (ctx->device, ctx->cmdPool, 1, &image->commandBuffer);
    }

    if (image->corrfnMemory != NULL) {
        an_destroy_buffer (ctx, image->corrfnMemory);
    }

    if (image->metricMemory != NULL) {
        an_destroy_buffer (ctx, image->metricMemory);
    }
        
    free (image);
}

struct an_corrfn*
an_create_corrfn (struct an_gpu_context *ctx,
                  const float           *corrfn,
                  const unsigned int    *dimensions,
                  unsigned int           ndim) {
    if (ndim != ctx->ndim) {
        fprintf (stderr, "Context dimensions must match image dimensions\n");
        return NULL;
    }

    struct an_corrfn *image = malloc (sizeof (struct an_corrfn));
    memset (image, 0, sizeof (struct an_corrfn));

    image->ctx = ctx;
    image->actual_size = 1;

    for (int i = 0; i < ndim-1; i++) {
        image->actual_size *= dimensions[i];
    }
    image->actual_size *= dimensions[ndim-1] / 2 + 1;

    image->corrfnMemory =
        an_create_buffer (ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          image->actual_size * sizeof(float));
    if (image->corrfnMemory == NULL) {
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

    if (!an_write_data (ctx, image->corrfnMemory, corrfn,
                        sizeof (float) * image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    VkResult result = an_create_command_buffer (ctx, &image->commandBuffer);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate command buffer, code = %i\n", result);
        goto cleanup;
    }

    return image;

cleanup:
    an_destroy_corrfn (image);
    return NULL;
}

static void
invoke_metric_kernel (struct an_corrfn *target,
                      struct an_image  *recon) {
    struct MetricUpdateData params;
    params.length = recon->actual_size;

    struct an_gpu_context *ctx = recon->ctx;
    struct pipeline *metricPipeline = ctx->pipelines[PIPELINE_METRIC];

    VkDescriptorBufferInfo cfInfo;
    ZERO(cfInfo);
    cfInfo.buffer = target->corrfnMemory->buffer;
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
invoke_reduce_kernel (struct an_corrfn *target) {
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
an_distance (struct an_corrfn *target,
             struct an_image  *recon,
             float            *distance) {
    if (target->ctx  != recon->ctx ||
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
