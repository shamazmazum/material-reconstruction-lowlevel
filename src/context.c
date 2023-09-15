/* Context (pipelines, descriptor layouts etc.) creation and destruction */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vulkan/vulkan.h>

#include <shader-source.h>
#include "annealing-lowlevel.h"
#include "internal.h"

const unsigned int update_group_sizes[] = {
    64,  1, 1,
    16, 16, 1,
    8,   8, 8
};

const char *validationLayer = "VK_LAYER_KHRONOS_validation";

static int
hasValidationLayer () {
    int result = 0;
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, NULL);

    VkLayerProperties *layers = malloc (sizeof (VkLayerProperties) * layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers);

    for (int i=0; i<layerCount; i++) {
        if (strcmp (validationLayer, layers[i].layerName) == 0) {
            result = 1;
            break;
        }
    }

    free (layers);
    return result;
}

static VkResult
create_command_pool (struct an_gpu_context *ctx) {
    assert (ctx->device != VK_NULL_HANDLE);

    VkCommandPoolCreateInfo poolCmdInfo;
    ZERO(poolCmdInfo);
    poolCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCmdInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCmdInfo.queueFamilyIndex = ctx->queueFamilyID;

    return vkCreateCommandPool(ctx->device, &poolCmdInfo, NULL, &ctx->cmdPool);
}

static VkResult
create_descriptor_pool (struct an_gpu_context *ctx) {
    assert (ctx->device != VK_NULL_HANDLE);

    VkDescriptorPoolSize poolSizes[2];
    memset (poolSizes, 0, sizeof (poolSizes));
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = DESCRIPTORS_IN_POOL;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = DESCRIPTORS_IN_POOL;

    VkDescriptorPoolCreateInfo poolInfo;
    ZERO(poolInfo);
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = DESCRIPTORS_IN_POOL;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    return vkCreateDescriptorPool(ctx->device, &poolInfo, NULL, &ctx->descPool);
}

static VkResult
create_pipelines (struct an_gpu_context *ctx) {
    /* TODO: Good idea to check pipeline structures */
    assert (ctx->ndim > 0 && ctx->ndim <= MAX_DIMENSIONS &&
            ctx->device != VK_NULL_HANDLE &&
            ctx->cache != VK_NULL_HANDLE);

    VkSpecializationInfo specInfos[PIPELINE_COUNT];
    memset (specInfos, 0, sizeof (specInfos));

    VkSpecializationMapEntry specEntry[MAX_DIMENSIONS];
    for (int i = 0; i < MAX_DIMENSIONS; i++) {
        specEntry[i].constantID = i;
        specEntry[i].offset = sizeof(int) * i;
        specEntry[i].size = sizeof(int);
    }

    const unsigned int *groupSize = &update_group_sizes[MAX_DIMENSIONS * (ctx->ndim - 1)];
    specInfos[PIPELINE_CFUPDATE].mapEntryCount = 3;
    specInfos[PIPELINE_CFUPDATE].pMapEntries = specEntry;
    specInfos[PIPELINE_CFUPDATE].dataSize = 3 * sizeof(int);
    specInfos[PIPELINE_CFUPDATE].pData = groupSize;

    VkPipelineShaderStageCreateInfo stageInfos[PIPELINE_COUNT];
    memset (stageInfos, 0, sizeof (stageInfos));
    for (int i = 0; i < PIPELINE_COUNT; i++) {
        stageInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfos[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfos[i].module = ctx->pipelines[i]->shader;
        stageInfos[i].pName = "main";
        stageInfos[i].pSpecializationInfo = &specInfos[i];
    }

    VkComputePipelineCreateInfo pipelineInfos[PIPELINE_COUNT];
    memset (pipelineInfos, 0, sizeof (pipelineInfos));
    for (int i = 0; i < PIPELINE_COUNT; i++) {
        pipelineInfos[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfos[i].stage = stageInfos[i];
        pipelineInfos[i].layout = ctx->pipelines[i]->pipelineLayout;
    }

    VkPipeline ps[PIPELINE_COUNT];
    VkResult result;
    result = vkCreateComputePipelines (ctx->device, ctx->cache, 3, pipelineInfos, NULL, ps);

    for (int i = 0; i < PIPELINE_COUNT; i++) {
        ctx->pipelines[i]->pipeline = ps[i];
    }

    return result;
}

static VkResult
load_shader (VkDevice device, const char *source, VkShaderModule *shaderModule) {
    FILE *stream = fopen (source, "rb");
    if (stream == NULL) {
        fprintf (stderr, "Cannot open a shader\n");
        return 0;
    }

    long size;
    fseek (stream, 0, SEEK_END);
    size = ftell (stream);
    fseek (stream, 0, SEEK_SET);

    void *code = malloc (size);
    fread (code, size, 1, stream);
    fclose (stream);

    VkShaderModuleCreateInfo createInfo;
    ZERO (createInfo);
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = code;

    VkResult result = vkCreateShaderModule(device, &createInfo, NULL, shaderModule);
    free (code);
    return result;
}

static void
pipeline_cleanup (struct an_gpu_context *ctx, struct pipeline *pipeline) {
    assert (ctx->device != VK_NULL_HANDLE && ctx->descPool != VK_NULL_HANDLE);

    if (pipeline->pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline (ctx->device, pipeline->pipeline, NULL);
    }

    if (pipeline->pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout (ctx->device, pipeline->pipelineLayout, NULL);
    }

    if (pipeline->descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout (ctx->device, pipeline->descriptorSetLayout, NULL);
    }

    if (pipeline->shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule (ctx->device, pipeline->shader, NULL);
    }

    free (pipeline);
}

VkResult
an_allocate_descriptor_set (struct an_gpu_context *ctx,
                            struct pipeline *pipeline,
                            VkDescriptorSet *descriptorSet) {
    assert (ctx->descPool != VK_NULL_HANDLE &&
            pipeline->descriptorSetLayout != VK_NULL_HANDLE);

    VkDescriptorSetAllocateInfo allocateInfo;
    ZERO(allocateInfo);
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = ctx->descPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &pipeline->descriptorSetLayout;

    return vkAllocateDescriptorSets (ctx->device, &allocateInfo, descriptorSet);
}

VkResult
an_create_fence (struct an_gpu_context *ctx, VkFence *fence) {
    VkFenceCreateInfo info;
    ZERO (info);
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    return vkCreateFence (ctx->device, &info, NULL, fence);
}

/* Create VkPipelineLayout */
static struct pipeline*
create_pipeline_layout (struct an_gpu_context *ctx, const char *shaderPath,
                        unsigned int storageBuffers, unsigned int uniformBuffers,
                        size_t pushConstantsSize) {
    assert (ctx->device != VK_NULL_HANDLE);

    VkResult result;
    struct pipeline *pipeline = malloc (sizeof (struct pipeline));
    memset (pipeline, 0, sizeof (struct pipeline));

    result = load_shader (ctx->device, shaderPath, &pipeline->shader);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create shader module, code = %i\n", result);
        goto cleanup;
    }

    unsigned int nbuffers = storageBuffers + uniformBuffers;
    VkDescriptorSetLayoutBinding *bindings;
    bindings = malloc (nbuffers * sizeof (VkDescriptorSetLayoutBinding));
    memset (bindings, 0, nbuffers * sizeof (VkDescriptorSetLayoutBinding));

    for (int i = 0; i < storageBuffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    for (int i = storageBuffers; i < nbuffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo;
    ZERO(layoutInfo);
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = nbuffers;
    layoutInfo.pBindings = bindings;

    result = vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, NULL,
                                         &pipeline->descriptorSetLayout);
    free (bindings);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create descriptor set layout, code = %i\n", result);
        goto cleanup;
    }

    VkPushConstantRange pushConstantRange;
    ZERO(pushConstantRange);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantsSize;

    VkPipelineLayoutCreateInfo plInfo;
    ZERO(plInfo);
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &pipeline->descriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushConstantRange;

    result = vkCreatePipelineLayout (ctx->device, &plInfo, NULL, &pipeline->pipelineLayout);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create pipeline layout, code = %i\n", result);
        goto cleanup;
    }

    return pipeline;

cleanup:
    pipeline_cleanup (ctx, pipeline);
    return NULL;
}

static int
create_cfupdate_pipeline (struct an_gpu_context *ctx) {
    ctx->pipelines[PIPELINE_CFUPDATE] =
        create_pipeline_layout (ctx, SHADER_SOURCE "update-s2.spv",
                                1, 1, sizeof (struct CFUpdateDataConst));
    return ctx->pipelines[PIPELINE_CFUPDATE] != NULL;
}

static int
create_metric_pipeline (struct an_gpu_context *ctx) {
    ctx->pipelines[PIPELINE_METRIC] =
        create_pipeline_layout (ctx, SHADER_SOURCE "metric.spv",
                                3, 0, sizeof (struct MetricUpdateData));
    return ctx->pipelines[PIPELINE_METRIC] != NULL;
}

static int
create_reduce_pipeline (struct an_gpu_context *ctx) {
    ctx->pipelines[PIPELINE_REDUCE] =
        create_pipeline_layout (ctx, SHADER_SOURCE "reduce.spv",
                                2, 0, sizeof (struct MetricUpdateData));
    return ctx->pipelines[PIPELINE_REDUCE] != NULL;
}

static int
find_queue_family_id (struct an_gpu_context *ctx) {
    assert (ctx->physDev != VK_NULL_HANDLE);

    int found = 0;
    uint32_t prop_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties (ctx->physDev, &prop_count, NULL);
    VkQueueFamilyProperties *properties =
        malloc (sizeof (VkQueueFamilyProperties) * prop_count);
    vkGetPhysicalDeviceQueueFamilyProperties (ctx->physDev, &prop_count, properties);

    for (uint32_t i=0; i<prop_count; i++) {
        /* Do we need the first conditional? */
        if (properties[i].queueCount > 0 &&
            (properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            found = 1;
            ctx->queueFamilyID = i;
            break;
        }
    }

    free (properties);
    return found;
}

static int
find_device (struct an_gpu_context *ctx) {
    assert (ctx->instance != VK_NULL_HANDLE &&
            ctx->physDev == VK_NULL_HANDLE);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, NULL);

    VkPhysicalDevice *devices = malloc (sizeof (VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, devices);

    for (int i=0; i<deviceCount; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties (devices[i], &properties);
        printf ("device %i: %s\n", i, properties.deviceName);

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx->physDev = devices[i];
        }
    }
    
    // Use the first device if non is selected yet
    if (deviceCount > 0 && ctx->physDev == VK_NULL_HANDLE) {
        ctx->physDev = devices[0];
    }

    free (devices);
    return ctx->physDev != VK_NULL_HANDLE;
}

static VkResult
create_instance (struct an_gpu_context *ctx, int enableValidation) {
    int validation = hasValidationLayer () && enableValidation;
    if (validation) {
        printf ("Enabling validation layer\n");
    }

    VkApplicationInfo appInfo;
    ZERO(appInfo);
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Material reconstruction library";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo;
    ZERO(createInfo);
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = validation ? 1 : 0;
    createInfo.ppEnabledLayerNames = validation ? &validationLayer : NULL;

    return vkCreateInstance(&createInfo, NULL, &ctx->instance);
}

static VkResult
create_device (struct an_gpu_context *ctx) {
    assert (ctx->physDev != VK_NULL_HANDLE);

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo;
    ZERO(queueCreateInfo);
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = ctx->queueFamilyID;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures;
    ZERO(deviceFeatures);

    VkDeviceCreateInfo createDevInfo;
    ZERO(createDevInfo);
    createDevInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createDevInfo.pQueueCreateInfos = &queueCreateInfo;
    createDevInfo.queueCreateInfoCount = 1;
    createDevInfo.pEnabledFeatures = &deviceFeatures;

    return vkCreateDevice(ctx->physDev, &createDevInfo, NULL, &ctx->device);
}

static VkResult
create_pipeline_cache (struct an_gpu_context *ctx) {
    assert (ctx->device != VK_NULL_HANDLE);

    VkPipelineCacheCreateInfo cacheInfo;
    ZERO(cacheInfo);
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    return vkCreatePipelineCache (ctx->device, &cacheInfo, NULL, &ctx->cache);
}

struct an_gpu_context* an_create_context(unsigned int ndim, int validation) {
    VkResult result;
    struct an_gpu_context *ctx = malloc (sizeof (struct an_gpu_context));
    memset (ctx, 0, sizeof (struct an_gpu_context));

    ctx->ndim = ndim;

    /* Create an instance */
    result = create_instance (ctx, validation);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create vulkan instance, code = %i\n", result);
        goto cleanup;
    }

    /* Find a physical device */
    if (!find_device (ctx)) {
        fprintf (stderr, "Cannot find an appropriate device\n");
        goto cleanup;
    }

    /* Find appropriate queue family */
    if (!find_queue_family_id (ctx)) {
        fprintf (stderr, "Cannot find an appropriate queue family\n");
        goto cleanup;
    }

    /* Create device */
    result = create_device (ctx);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create a logical device, code = %i\n", result);
        goto cleanup;
    }

    /* Create pipeline cache */
    result = create_pipeline_cache (ctx);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create pipeline cache, code = %i\n", result);
    }

    /* Create descriptor pool */
    result = create_descriptor_pool (ctx);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot descriptor pool, code = %i\n", result);
        goto cleanup;
    }

    /* Create pipeline layout for updation of correlation function */
    if (!create_cfupdate_pipeline (ctx)) {
        fprintf (stderr, "Cannot create cf updating pipeline layout\n");
        goto cleanup;
    }

    /* Create pipeline layout for calculation difference with the original */
    if (!create_metric_pipeline (ctx)) {
        fprintf (stderr, "Cannot create metric pipeline layout\n");
        goto cleanup;
    }

    /* Create pipeline layout for reduction */
    if (!create_reduce_pipeline (ctx)) {
        fprintf (stderr, "Cannot create reduction pipeline layout\n");
        goto cleanup;
    }

    /* Create pipelines */
    result = create_pipelines (ctx);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create pipelines, code = %i\n", result);
        goto cleanup;
    }

    /* Get queue */
    vkGetDeviceQueue (ctx->device, ctx->queueFamilyID, 0, &ctx->queue);

    /* Create command pool */
    result = create_command_pool (ctx);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create command pool, code =%i\n", result);
        goto cleanup;
    }

    return ctx;

cleanup:
    an_destroy_context (ctx);
    return NULL;
}

void an_destroy_context (struct an_gpu_context *ctx) {
    if (ctx->cmdPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool (ctx->device, ctx->cmdPool, NULL);
    }

    for (int i = 0; i < PIPELINE_COUNT; i++) {
        if (ctx->pipelines[i] != NULL) {
            pipeline_cleanup (ctx, ctx->pipelines[i]);
        }
    }

    if (ctx->descPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool (ctx->device, ctx->descPool, NULL);
    }

    if (ctx->cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache (ctx->device, ctx->cache, NULL);
    }

    if (ctx->device != VK_NULL_HANDLE) {
        vkDestroyDevice (ctx->device, NULL);
    }

    if (ctx->instance != VK_NULL_HANDLE) {
        vkDestroyInstance (ctx->instance, NULL);
    }

    free (ctx);
}

VkResult
an_create_command_buffer (struct an_gpu_context *ctx, VkCommandBuffer *buffer) {
    assert (ctx->device != VK_NULL_HANDLE && ctx->cmdPool != VK_NULL_HANDLE);

    VkResult result;
    VkCommandBufferAllocateInfo cbAllocInfo;
    ZERO(cbAllocInfo);
    cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAllocInfo.commandPool = ctx->cmdPool;
    cbAllocInfo.commandBufferCount = 1;

    return vkAllocateCommandBuffers (ctx->device, &cbAllocInfo, buffer);
}
