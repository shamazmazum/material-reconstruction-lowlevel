#ifndef __ANNEALING_LL__
#define __ANNEALING_LL__

/* Fourier transform */
int an_rfft (const float         *array,
             float               *real,
             float               *imag,
             const unsigned int  *dimensions,
             unsigned int         ndims);

int an_irfft (float              *array,
              const float        *real,
              const float        *imag,
              const unsigned int *dimensions,
              unsigned int        ndims);

#define MAX_DIMENSIONS 3

struct an_gpu_context;

struct an_image;

struct CFUpdateData {
    unsigned int actual_dimensions[MAX_DIMENSIONS + 1];
    unsigned int logical_dimensions[MAX_DIMENSIONS + 1];
    unsigned int stride[MAX_DIMENSIONS + 1];
    unsigned int point[MAX_DIMENSIONS + 1];

    float c;
    unsigned int ndim;
};

struct MetricUpdateData {
    unsigned int length;
};

struct an_gpu_context*
an_create_context(unsigned int ndim);

void
an_destroy_context (struct an_gpu_context *ctx);

struct an_image*
an_create_image (struct an_gpu_context *ctx,
                 const float           *real,
                 const float           *imag,
                 const unsigned int    *dimensions,
                 unsigned int           ndim);

struct an_image*
an_create_corrfn (struct an_gpu_context *ctx,
                  const float           *corrfn,
                  const unsigned int    *dimensions,
                  unsigned int           ndim);

void
an_destroy_image (struct an_image *image);

void
an_image_store_state (struct an_image *image);

void
an_image_rollback (struct an_image *image);

int
an_image_update_fft (struct an_image    *image,
                     const unsigned int *coord,
                     unsigned int        ndim,
                     float               delta);

/* Testing */
int
an_image_get (struct an_image *image,
              float           *real,
              float           *imag);

/* Distance measurement */
int
an_distance (struct an_image *target,
             struct an_image *recon,
             float           *distance);

#endif
