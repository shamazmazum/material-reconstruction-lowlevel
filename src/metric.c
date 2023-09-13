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

static void
update_metric_descriptors (struct an_metric *metric) {
    struct an_gpu_context *ctx = metric->ctx;
    struct an_corrfn *target = metric->target;
    struct an_image *recon = metric->recon;

    VkDescriptorBufferInfo cfInfo;
    ZERO(cfInfo);
    cfInfo.buffer = target->corrfnMemory->buffer;
    cfInfo.offset = 0;
    cfInfo.range = sizeof (float) * target->actual_size;

    VkDescriptorBufferInfo reconInfo;
    ZERO(reconInfo);
    reconInfo.buffer = recon->imageMemory->buffer;
    reconInfo.offset = 0;
    reconInfo.range = sizeof (mycomplex) * recon->actual_size;

    VkDescriptorBufferInfo outputInfo;
    ZERO(outputInfo);
    outputInfo.buffer = metric->metricMemory->buffer;
    outputInfo.offset = 0;
    outputInfo.range = sizeof (float) * target->actual_size;

    VkWriteDescriptorSet dsSets[3];
    memset (dsSets, 0, sizeof (dsSets));
    dsSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[0].dstSet = metric->metricSet;
    dsSets[0].dstBinding = 0; // binding #
    dsSets[0].dstArrayElement = 0;
    dsSets[0].descriptorCount = 1;
    dsSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[0].pBufferInfo = &cfInfo;

    dsSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[1].dstSet = metric->metricSet;
    dsSets[1].dstBinding = 1; // binding #
    dsSets[1].dstArrayElement = 0;
    dsSets[1].descriptorCount = 1;
    dsSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[1].pBufferInfo = &reconInfo;

    dsSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[2].dstSet = metric->metricSet;
    dsSets[2].dstBinding = 2; // binding #
    dsSets[2].dstArrayElement = 0;
    dsSets[2].descriptorCount = 1;
    dsSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[2].pBufferInfo = &outputInfo;

    vkUpdateDescriptorSets (ctx->device, 3, dsSets, 0, NULL);
}

static void
update_reduce_descriptors (struct an_metric *metric) {
    struct an_gpu_context *ctx = metric->ctx;

    VkDescriptorBufferInfo mInfo;
    ZERO(mInfo);
    mInfo.buffer = metric->metricMemory->buffer;
    mInfo.offset = 0;
    mInfo.range = sizeof (float) * metric->target->actual_size;

    VkWriteDescriptorSet dsSets;
    ZERO (dsSets);
    dsSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets.dstSet = metric->reduceSet;
    dsSets.dstBinding = 0; // binding #
    dsSets.dstArrayElement = 0;
    dsSets.descriptorCount = 1;
    dsSets.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets.pBufferInfo = &mInfo;

    vkUpdateDescriptorSets (ctx->device, 1, &dsSets, 0, NULL);
}

static void
record_command_buffer (struct an_metric *metric) {
    struct an_gpu_context *ctx = metric->ctx;
    struct pipeline *metricPipeline = ctx->pipelines[PIPELINE_METRIC];
    struct pipeline *reducePipeline = ctx->pipelines[PIPELINE_REDUCE];
    unsigned int actual_size = metric->recon->actual_size;

    struct MetricUpdateData params;
    params.length = actual_size;

    VkMemoryBarrier memoryBarrier;
    ZERO (memoryBarrier);
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer (metric->commandBuffer, &beginInfo);
    /* Calculate squared difference */
    vkCmdBindPipeline (metric->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       metricPipeline->pipeline);
    vkCmdBindDescriptorSets (metric->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             metricPipeline->pipelineLayout,
                             0, 1, &metric->metricSet, 0, NULL);
    vkCmdPushConstants (metric->commandBuffer, metricPipeline->pipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof (struct MetricUpdateData), &params);
    vkCmdDispatch (metric->commandBuffer,
                   ceil((double) actual_size / (double)METRIC_GRP_SIZE), 1, 1);
    vkCmdPipelineBarrier(metric->commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &memoryBarrier, 0, NULL, 0, NULL);

    /* Reduce */
    vkCmdBindPipeline (metric->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       reducePipeline->pipeline);
    vkCmdBindDescriptorSets (metric->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             reducePipeline->pipelineLayout,
                             0, 1, &metric->reduceSet, 0, NULL);
    while (params.length > 0) {
        int groups = ceil((double)params.length / (double)METRIC_GRP_SIZE);
        vkCmdPushConstants (metric->commandBuffer, reducePipeline->pipelineLayout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof (struct MetricUpdateData), &params);
        vkCmdDispatch (metric->commandBuffer, groups, 1, 1);
        params.length = (groups == 1) ? 0 : groups;
        if (params.length > 0) {
            vkCmdPipelineBarrier(metric->commandBuffer,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &memoryBarrier, 0, NULL, 0, NULL);
        }
    }
    vkEndCommandBuffer (metric->commandBuffer);
}

void
an_destroy_metric (struct an_metric *metric) {
    struct an_gpu_context *ctx = metric->ctx;

    if (metric->metricSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets (ctx->device, ctx->descPool, 1, &metric->metricSet);
    }

    if (metric->reduceSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets (ctx->device, ctx->descPool, 1, &metric->reduceSet);
    }

    if (metric->commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers (ctx->device, ctx->cmdPool, 1, &metric->commandBuffer);
    }

    if (metric->metricMemory != NULL) {
        an_destroy_buffer (ctx, metric->metricMemory);
    }
        
    free (metric);
}

struct an_metric*
an_create_metric (struct an_gpu_context *ctx,
                  struct an_corrfn      *target,
                  struct an_image       *recon) {
    if (target->ctx != recon->ctx ||
        target->ctx != ctx ||
        target->actual_size != recon->actual_size) {
        fprintf (stderr, "Incompatible images\n");
        return 0;
    }

    VkResult result;
    struct an_metric *metric = malloc (sizeof (struct an_metric));
    memset (metric, 0, sizeof (struct an_metric));
    metric->ctx = ctx;
    metric->recon  = recon;
    metric->target = target;

    metric->metricMemory =
        an_create_buffer (ctx,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          recon->actual_size * sizeof(float));
    if (metric->metricMemory == NULL) {
        fprintf (stderr, "Cannot create metric buffer\n");
        goto cleanup;
    }

    result = an_create_command_buffer (ctx, &metric->commandBuffer);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate command buffer, code = %i\n", result);
        goto cleanup;
    }

    result = an_allocate_descriptor_set (ctx, ctx->pipelines[PIPELINE_METRIC],
                                         &metric->metricSet);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate descriptor set, code = %i\n", result);
        goto cleanup;
    }

    result = an_allocate_descriptor_set (ctx, ctx->pipelines[PIPELINE_REDUCE],
                                         &metric->reduceSet);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate descriptor set, code = %i\n", result);
        goto cleanup;
    }

    update_metric_descriptors (metric);
    update_reduce_descriptors (metric);
    record_command_buffer (metric);

    return metric;

cleanup:
    an_destroy_metric (metric);
    return NULL;
}

static void
invoke_kernels (struct an_metric *metric) {
    struct an_gpu_context *ctx = metric->ctx;
    an_image_synchronize (metric->recon);

    VkSubmitInfo submitInfo;
    ZERO(submitInfo);
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &metric->commandBuffer;
    vkQueueSubmit (ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);
}

int
an_distance (struct an_metric *metric,
             float            *distance) {
    invoke_kernels (metric);

    struct an_gpu_context *ctx = metric->ctx;
    if (!an_read_data (ctx, metric->metricMemory, distance, sizeof (float))) {
        fprintf (stderr, "Cannot read metric\n");
        return 0;
    }
    return 1;
}
