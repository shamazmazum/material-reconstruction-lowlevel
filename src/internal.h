#pragma once

#define ZERO(x) memset(&(x), 0, sizeof(x))
#define DESCRIPTORS_IN_POOL 10

extern const unsigned int update_group_sizes[];

/* Pipelines */

enum pipeline_type {
    PIPELINE_CFUPDATE = 0,
    PIPELINE_METRIC,
    PIPELINE_REDUCE,
    PIPELINE_COUNT
};

struct pipeline {
    VkShaderModule shader;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    /* TODO: Move this to image objects */
    VkDescriptorSet descriptorSet;
};

struct an_gpu_context {
    uint32_t ndim;
    VkInstance instance;
    uint32_t queueFamilyID;
    VkPhysicalDevice physDev;
    VkDevice device;
    VkPipelineCache cache;
    VkDescriptorPool descPool;
    VkCommandPool cmdPool;
    VkQueue queue;

    struct pipeline *pipelines[PIPELINE_COUNT];
};

/* Command buffers */
int
an_create_command_buffer (struct an_gpu_context *ctx, VkCommandBuffer *buffer);

/* Memory buffers */

struct an_image_memory {
    VkBuffer buffer;
    VkDeviceMemory memory;
};

typedef union
{
    float __attribute__((aligned (8))) s[2];
    __extension__ struct{ float  re, im; };
} mycomplex;

struct an_image {
    struct an_gpu_context *ctx;
    VkCommandBuffer commandBuffer;

    struct CFUpdateData updateData;
    size_t actual_size;
    uint32_t ngroups[MAX_DIMENSIONS];

    struct an_image_memory *inputMemory;
    struct an_image_memory *outputMemory;
    /* Not a separate buffer, just switches between other two */
    struct an_image_memory *savedMemory;
};

struct an_corrfn {
    struct an_gpu_context *ctx;
    VkCommandBuffer commandBuffer;
    size_t actual_size;

    struct an_image_memory *corrfnMemory;
    struct an_image_memory *metricMemory;
};

struct an_image_memory*
an_create_buffer (struct an_gpu_context *ctx, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties, size_t size);

void
an_destroy_buffer (struct an_gpu_context *ctx, struct an_image_memory *imemory);

int
an_write_data (struct an_gpu_context *ctx, struct an_image_memory *imageMemory,
               const void *data, size_t size);

int
an_read_data (struct an_gpu_context *ctx, struct an_image_memory *imageMemory,
              void *data, size_t size);
