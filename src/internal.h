#pragma once

#define ZERO(x) memset(&(x), 0, sizeof(x))
#define DESCRIPTORS_IN_POOL 10

extern const unsigned int update_group_sizes[];

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
