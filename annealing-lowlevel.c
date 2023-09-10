#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <vulkan/vulkan.h>
#include "annealing-lowlevel.h"

#include <shader-source.h>

#define ZERO(x) memset(&(x), 0, sizeof(x))

#define DESCRIPTORS_IN_POOL 10
#define UPDATE_GRP_SIZE 16
#define METRIC_GRP_SIZE 64

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

const char *validationLayer = "VK_LAYER_KHRONOS_validation";
const int enableValidation = 0;

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

/* =================
 * =================
 * === Pipelines ===
 * =================
 * =================
 */

static VkResult
create_command_pool (VkDevice device, VkCommandPool *pool, uint32_t queueFamilyID) {
    VkCommandPoolCreateInfo poolCmdInfo;
    ZERO(poolCmdInfo);
    poolCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCmdInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCmdInfo.queueFamilyIndex = queueFamilyID;

    return vkCreateCommandPool(device, &poolCmdInfo, NULL, pool);
}

static VkResult
create_descriptor_pool (VkDevice device, VkDescriptorPool *pool) {
    VkDescriptorPoolSize poolSize;
    ZERO(poolSize);
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = DESCRIPTORS_IN_POOL;

    VkDescriptorPoolCreateInfo poolInfo;
    ZERO(poolInfo);
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = DESCRIPTORS_IN_POOL;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    return vkCreateDescriptorPool(device, &poolInfo, NULL, pool);
}

static VkResult
create_pipelines (VkDevice device, VkPipelineCache cache, struct pipeline *pipelines[],
                                  unsigned int ndim) {
    assert (ndim > 0 && ndim <= MAX_DIMENSIONS);

    VkSpecializationInfo specInfos[PIPELINE_COUNT];
    memset (specInfos, 0, sizeof (specInfos));

    VkSpecializationMapEntry specEntry[MAX_DIMENSIONS];
    for (int i = 0; i < MAX_DIMENSIONS; i++) {
        specEntry[i].constantID = i;
        specEntry[i].offset = sizeof(int) * i;
        specEntry[i].size = sizeof(int);
    }

    int groupSize[MAX_DIMENSIONS];
    for (int i = 0; i < MAX_DIMENSIONS; i++) {
        groupSize[i] = (i < ndim) ? UPDATE_GRP_SIZE : 1;
    }

    specInfos[PIPELINE_CFUPDATE].mapEntryCount = 3;
    specInfos[PIPELINE_CFUPDATE].pMapEntries = specEntry;
    specInfos[PIPELINE_CFUPDATE].dataSize = 3 * sizeof(int);
    specInfos[PIPELINE_CFUPDATE].pData = groupSize;

    VkPipelineShaderStageCreateInfo stageInfos[PIPELINE_COUNT];
    memset (stageInfos, 0, sizeof (stageInfos));
    for (int i = 0; i < PIPELINE_COUNT; i++) {
        stageInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfos[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfos[i].module = pipelines[i]->shader;
        stageInfos[i].pName = "main";
        stageInfos[i].pSpecializationInfo = &specInfos[i];
    }

    VkComputePipelineCreateInfo pipelineInfos[PIPELINE_COUNT];
    memset (pipelineInfos, 0, sizeof (pipelineInfos));
    for (int i = 0; i < PIPELINE_COUNT; i++) {
        pipelineInfos[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfos[i].stage = stageInfos[i];
        pipelineInfos[i].layout = pipelines[i]->pipelineLayout;
    }

    VkPipeline ps[PIPELINE_COUNT];
    VkResult result;
    result = vkCreateComputePipelines (device, cache, 3, pipelineInfos, NULL, ps);

    for (int i = 0; i < PIPELINE_COUNT; i++) {
        pipelines[i]->pipeline = ps[i];
    }

    return result;
}

static int
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

    VkResult result;
    result = vkCreateShaderModule(device, &createInfo, NULL, shaderModule);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create shader module, code = %i\n", result);
    }

    free (code);
    return result == VK_SUCCESS;
}

static void
pipeline_cleanup (VkDevice device, VkDescriptorPool pool, struct pipeline *pipeline) {
    if (pipeline->pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline (device, pipeline->pipeline, NULL);
    }

    if (pipeline->descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets (device, pool, 1, &pipeline->descriptorSet);
    }

    if (pipeline->pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout (device, pipeline->pipelineLayout, NULL);
    }

    if (pipeline->descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout (device, pipeline->descriptorSetLayout, NULL);
    }

    if (pipeline->shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule (device, pipeline->shader, NULL);
    }

    free (pipeline);
}

/* Create VkPipelineLayout AND VkDescriptorSet */
static struct pipeline*
create_pipeline_layout (VkDevice device, VkDescriptorPool descPool,
                        const char *shaderPath,
                        unsigned int nbuffers, size_t pushConstantsSize) {
    VkResult result;
    struct pipeline *pipeline = malloc (sizeof (struct pipeline));
    memset (pipeline, 0, sizeof (struct pipeline));
    VkDescriptorSetLayoutBinding *bindings = NULL;

    if (!load_shader (device, shaderPath, &pipeline->shader)) {
        goto cleanup;
    }

    bindings = malloc (nbuffers * sizeof (VkDescriptorSetLayoutBinding));
    memset (bindings, 0, nbuffers * sizeof (VkDescriptorSetLayoutBinding));

    for (int i = 0; i < nbuffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo;
    ZERO(layoutInfo);
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = nbuffers;
    layoutInfo.pBindings = bindings;

    result = vkCreateDescriptorSetLayout(device, &layoutInfo, NULL,
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

    result = vkCreatePipelineLayout (device, &plInfo, NULL, &pipeline->pipelineLayout);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create pipeline layout, code = %i\n", result);
        goto cleanup;
    }

    VkDescriptorSetAllocateInfo allocateInfo;
    ZERO(allocateInfo);
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &pipeline->descriptorSetLayout;

    result = vkAllocateDescriptorSets (device, &allocateInfo, &pipeline->descriptorSet);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create descriptor set, code = %i\n", result);
    }

    return pipeline;

cleanup:
    pipeline_cleanup (device, descPool, pipeline);
    return NULL;
}

static struct pipeline*
create_cfupdate_pipeline (VkDevice device, VkDescriptorPool descPool) {
    return create_pipeline_layout (device, descPool,
                                   SHADER_SOURCE "update-s2.spv",
                                   2, sizeof (struct CFUpdateData));
}

static struct pipeline*
create_metric_pipeline (VkDevice device, VkDescriptorPool descPool) {
    return create_pipeline_layout (device, descPool,
                                   SHADER_SOURCE "metric.spv",
                                   3, sizeof (struct MetricUpdateData));
}

static struct pipeline*
create_reduce_pipeline (VkDevice device, VkDescriptorPool descPool) {
    return create_pipeline_layout (device, descPool,
                                   SHADER_SOURCE "reduce.spv",
                                   1, sizeof (struct MetricUpdateData));
}

static int
find_queue_family_id (VkPhysicalDevice device, uint32_t *family) {
    int found = 0;
    uint32_t prop_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties (device, &prop_count, NULL);
    VkQueueFamilyProperties *properties =
        malloc (sizeof (VkQueueFamilyProperties) * prop_count);
    vkGetPhysicalDeviceQueueFamilyProperties (device, &prop_count, properties);

    for (uint32_t i=0; i<prop_count; i++) {
        /* Do we need the first conditional? */
        if (properties[i].queueCount > 0 &&
            (properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            found = 1;
            *family = i;
            break;
        }
    }

    free (properties);
    return found;
}

static VkPhysicalDevice
find_device (VkInstance instance) {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

    VkPhysicalDevice *devices = malloc (sizeof (VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    for (int i=0; i<deviceCount; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties (devices[i], &properties);
        printf ("device %i: %s\n", i, properties.deviceName);

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            device = devices[i];
        }
    }
    
    // Use the first device if non is selected yet
    if (deviceCount > 0 && device == VK_NULL_HANDLE) {
        device = devices[0];
    }

    free (devices);
    return device;
}

static VkResult
create_instance (VkInstance *instance) {
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

    return vkCreateInstance(&createInfo, NULL, instance);
}

static VkResult
create_device (VkPhysicalDevice physDev, uint32_t familyID, VkDevice *device) {
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo;
    ZERO(queueCreateInfo);
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = familyID;
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

    return vkCreateDevice(physDev, &createDevInfo, NULL, device);
}

static VkResult
create_pipeline_cache (VkDevice device, VkPipelineCache *cache) {
    VkPipelineCacheCreateInfo cacheInfo;
    ZERO(cacheInfo);
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    return vkCreatePipelineCache (device, &cacheInfo, NULL, cache);
}

struct an_gpu_context* an_create_context(unsigned int ndim) {
    VkResult result;
    struct an_gpu_context *ctx = malloc (sizeof (struct an_gpu_context));
    memset (ctx, 0, sizeof (struct an_gpu_context));

    ctx->ndim = ndim;

    /* Create an instance */
    result = create_instance (&ctx->instance);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create vulkan instance, code = %i\n", result);
        goto cleanup;
    }

    /* Find a physical device */
    ctx->physDev = find_device (ctx->instance);
    if (ctx->physDev == VK_NULL_HANDLE) {
        fprintf (stderr, "Cannot find an appropriate device\n");
        goto cleanup;
    }

    /* Find appropriate queue family */
    if (!find_queue_family_id (ctx->physDev, &ctx->queueFamilyID)) {
        fprintf (stderr, "Cannot find an appropriate queue family\n");
        goto cleanup;
    }

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties (ctx->physDev, &properties);
    printf ("Using %s\n", properties.deviceName);
    printf ("Using queue family %i\n", ctx->queueFamilyID);

    /* Create device */
    result = create_device (ctx->physDev, ctx->queueFamilyID, &ctx->device);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create a logical device, code = %i\n", result);
        goto cleanup;
    }

    /* Create pipeline cache */
    result = create_pipeline_cache (ctx->device, &ctx->cache);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create pipeline cache, code = %i\n", result);
    }

    /* Create descriptor pool */
    result = create_descriptor_pool (ctx->device, &ctx->descPool);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot descriptor pool, code = %i\n", result);
        goto cleanup;
    }

    /* Create pipeline layout for updation of correlation function */
    ctx->pipelines[PIPELINE_CFUPDATE] =
        create_cfupdate_pipeline (ctx->device, ctx->descPool);
    if (ctx->pipelines[PIPELINE_CFUPDATE] == NULL) {
        fprintf (stderr, "Cannot create cf updating pipeline layout\n");
        goto cleanup;
    }

    /* Create pipeline layout for calculation difference with the original */
    ctx->pipelines[PIPELINE_METRIC] =
        create_metric_pipeline (ctx->device, ctx->descPool);
    if (ctx->pipelines[PIPELINE_METRIC] == NULL) {
        fprintf (stderr, "Cannot create metric pipeline layout\n");
        goto cleanup;
    }

    /* Create pipeline layout for reduction */
    ctx->pipelines[PIPELINE_REDUCE] =
        create_reduce_pipeline (ctx->device, ctx->descPool);
    if (ctx->pipelines[PIPELINE_REDUCE] == NULL) {
        fprintf (stderr, "Cannot create reduction pipeline layout\n");
        goto cleanup;
    }

    /* Create pipelines */
    result = create_pipelines (ctx->device, ctx->cache, ctx->pipelines, ctx->ndim);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create pipelines, code = %i\n", result);
        goto cleanup;
    }

    /* Get queue */
    vkGetDeviceQueue (ctx->device, ctx->queueFamilyID, 0, &ctx->queue);

    /* Create command pool */
    result = create_command_pool (ctx->device, &ctx->cmdPool, ctx->queueFamilyID);
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
            pipeline_cleanup (ctx->device, ctx->descPool, ctx->pipelines[i]);
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

struct image_memory {
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct an_image {
    enum image_type type;
    struct an_gpu_context *ctx;
    VkCommandBuffer commandBuffer;

    struct CFUpdateData updateData;
    size_t actual_size;
    uint32_t ngroups[MAX_DIMENSIONS];

    struct image_memory *metricMemory;
    struct image_memory *inputMemory;
    struct image_memory *outputMemory;
    /* Not a separate buffer, just switches between other two */
    struct image_memory *savedMemory;
};

static void
free_image_memory (VkDevice device, struct image_memory *imemory) {
    if (imemory->memory != VK_NULL_HANDLE) {
        vkFreeMemory (device, imemory->memory, NULL);
    }

    if (imemory->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer (device, imemory->buffer, NULL);
    }
}

static int
write_data (VkDevice device, struct image_memory *imageMemory,
            const float *real, const float *imag,
            size_t actual_size) {
    void *ptr;
    mycomplex *data;
    VkResult result;

    result = vkMapMemory (device, imageMemory->memory,
                          0, actual_size * sizeof (mycomplex),
                          0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map memory\n");
        return 0;
    }

    data = ptr;
    for (size_t i = 0; i < actual_size; i++) {
        data[i].re = real[i];
        data[i].im = imag[i];
    }

    vkUnmapMemory (device, imageMemory->memory);
    return 1;
}

static int
write_cf_data (VkDevice device, struct image_memory *imageMemory,
               const float *cf, size_t actual_size) {
    void *ptr;
    VkResult result;

    result = vkMapMemory (device, imageMemory->memory,
                          0, actual_size * sizeof (float),
                          0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map memory\n");
        return 0;
    }

    memcpy(ptr, cf, sizeof(float) * actual_size);

    vkUnmapMemory (device, imageMemory->memory);
    return 1;
}

static int
read_data (VkDevice device, struct image_memory *imageMemory,
           float *real, float *imag,
           size_t actual_size) {
    void *ptr;
    mycomplex *data;
    VkResult result;

    result = vkMapMemory (device, imageMemory->memory,
                          0, actual_size * sizeof (mycomplex),
                          0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map memory\n");
        return 0;
    }

    data = ptr;
    for (size_t i = 0; i < actual_size; i++) {
        real[i] = data[i].re;
        imag[i] = data[i].im;
    }

    vkUnmapMemory (device, imageMemory->memory);
    return 1;
}

static struct image_memory*
create_buffer (VkPhysicalDevice physicalDevice, VkDevice device,
               VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
               size_t size) {
    VkResult result;
    int i;

    struct image_memory *imemory = malloc (sizeof (struct image_memory));
    memset (imemory, 0, sizeof (struct image_memory));
    
    VkBufferCreateInfo bufferInfo;
    ZERO(bufferInfo);
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    result = vkCreateBuffer(device, &bufferInfo, NULL, &imemory->buffer);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot create buffer, code = %i\n", result);
        goto cleanup;
    }

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, imemory->buffer, &memRequirements);

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

    result = vkAllocateMemory(device, &allocInfo, NULL, &imemory->memory);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot allocate buffer memory, code = %i\n", result);
        goto cleanup;
    }

    vkBindBufferMemory (device, imemory->buffer, imemory->memory, 0);
    return imemory;

cleanup:
    free_image_memory (device, imemory);
    return NULL;
}

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
        free_image_memory (ctx->device, image->metricMemory);
    }

    if (image->savedMemory != NULL) {
        free_image_memory (ctx->device, image->savedMemory);
    }

    if (image->outputMemory != NULL) {
        free_image_memory (ctx->device, image->outputMemory);
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
    image->inputMemory = create_buffer (ctx->physDev, ctx->device,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                        image->actual_size * sizeof(mycomplex));
    if (image->inputMemory == NULL) {
        fprintf (stderr, "Cannot create input buffer\n");
        goto cleanup;
    }

    image->outputMemory = create_buffer (ctx->physDev, ctx->device,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         image->actual_size * sizeof(mycomplex));
    if (image->outputMemory == NULL) {
        fprintf (stderr, "Cannot create output buffer\n");
        goto cleanup;
    }

    if (!write_data (ctx->device, image->inputMemory, real, imag, image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    if (!write_data (ctx->device, image->outputMemory, real, imag, image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    /* Number of work groups */
    for (uint32_t i = 0; i < MAX_DIMENSIONS; i++) {
        image->ngroups[i] = (i < ndim) ?
            ceil((double)image->updateData.actual_dimensions[i] / (double) UPDATE_GRP_SIZE) :
            1;
    }

    if (!create_command_buffer (ctx->device, ctx->cmdPool, &image->commandBuffer)) {
        goto cleanup;
    }

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
    return read_data (image->ctx->device, image->outputMemory, real, imag,
                      image->actual_size);
}

void
an_image_store_state (struct an_image *image) {
    assert (image->savedMemory != image->outputMemory);

    struct image_memory *tmp = image->savedMemory;
    image->savedMemory  = image->outputMemory;
    image->outputMemory = tmp;
    image->inputMemory  = image->savedMemory;
}

void
an_image_rollback (struct an_image *image) {
    assert (image->savedMemory != image->outputMemory);

    struct image_memory *tmp = image->outputMemory;
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
    image->outputMemory = create_buffer (ctx->physDev, ctx->device,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         image->actual_size * sizeof(float));
    if (image->outputMemory == NULL) {
        fprintf (stderr, "Cannot create output buffer\n");
        goto cleanup;
    }

    /* Temporary buffer for metric */
    image->metricMemory = create_buffer (ctx->physDev, ctx->device,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         image->actual_size * sizeof(float));
    if (image->metricMemory == NULL) {
        fprintf (stderr, "Cannot create metric buffer\n");
        goto cleanup;
    }

    if (!write_cf_data (ctx->device, image->outputMemory, corrfn, image->actual_size)) {
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
    struct MetricUpdateData params;
    params.length = target->actual_size;

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

    VkCommandBufferBeginInfo beginInfo;
    ZERO(beginInfo);
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer (target->commandBuffer, &beginInfo);
    vkCmdBindPipeline (target->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                       reducePipeline->pipeline);
    vkCmdBindDescriptorSets (target->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             reducePipeline->pipelineLayout,
                             0, 1, &reducePipeline->descriptorSet, 0, NULL);
    vkCmdPushConstants (target->commandBuffer, reducePipeline->pipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof (struct MetricUpdateData), &params);
    vkCmdDispatch (target->commandBuffer, METRIC_GRP_SIZE, 1, 1);
    vkCmdDispatch (target->commandBuffer, 1, 1, 1);
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
    void *ptr;
    VkResult result;
    result = vkMapMemory (ctx->device, target->metricMemory->memory,
                          0, sizeof (float), 0, &ptr);
    if (result != VK_SUCCESS) {
        fprintf (stderr, "Cannot map memory\n");
        return 0;
    }

    *distance = *((float*)ptr);
    vkUnmapMemory (ctx->device, target->metricMemory->memory);

    return 1;
}
