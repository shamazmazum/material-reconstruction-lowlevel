#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "annealing-lowlevel.h"
#include "internal.h"

void
an_destroy_corrfn (struct an_corrfn *image) {
    struct an_gpu_context *ctx = image->ctx;

    if (image->corrfnMemory != NULL) {
        an_destroy_buffer (ctx, image->corrfnMemory);
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

    if (!an_write_data (ctx, image->corrfnMemory, corrfn,
                        sizeof (float) * image->actual_size)) {
        fprintf (stderr, "Cannot write data to image memory\n");
        goto cleanup;
    }

    return image;

cleanup:
    an_destroy_corrfn (image);
    return NULL;
}
