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
    dsSets[0].dstSet = image->descriptorSet;
    dsSets[0].dstBinding = 0; // binding #
    dsSets[0].dstArrayElement = 0;
    dsSets[0].descriptorCount = 1;
    dsSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[0].pBufferInfo = &inputInfo;

    dsSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    dsSets[1].dstSet = image->descriptorSet;
    dsSets[1].dstBinding = 1; // binding #
    dsSets[1].dstArrayElement = 0;
    dsSets[1].descriptorCount = 1;
    dsSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dsSets[1].pBufferInfo = &outputInfo;

    vkUpdateDescriptorSets (ctx->device, 2, dsSets, 0, NULL);
}

void
an_destroy_image (struct an_image *image) {
    assert (image->savedMemory != image->outputMemory);
    struct an_gpu_context *ctx = image->ctx;

    if (image->descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets (ctx->device, ctx->descPool, 1, &image->descriptorSet);
    }

    if (image->commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers (ctx->device, ctx->cmdPool, 1, &image->commandBuffer);
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
        free (data);
        goto cleanup;
    }

    if (!an_write_data (ctx, image->outputMemory, data,
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

    update_descriptors (image);

    image->savedMemory = image->inputMemory;
    return image;

cleanup:
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
    if (ndim != image->updateData.ndim) {
        fprintf (stderr, "Wrong dimensions\n");
        return 0;
    }

    assert (image->savedMemory != image->outputMemory);
    memcpy(image->updateData.point, coord, sizeof (unsigned int) * ndim);
    image->updateData.c = delta;

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
