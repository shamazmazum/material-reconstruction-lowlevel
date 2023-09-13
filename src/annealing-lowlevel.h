#pragma once

#define AN_EXPORT __attribute__((visibility ("default")))

/* Fourier transform */
AN_EXPORT int
an_rfft (const float         *array,
         float               *real,
         float               *imag,
         const unsigned int  *dimensions,
         unsigned int         ndims);

AN_EXPORT int
an_irfft (float              *array,
          const float        *real,
          const float        *imag,
          const unsigned int *dimensions,
          unsigned int        ndims);

#define MAX_DIMENSIONS 3

struct an_gpu_context;
struct an_image;
struct an_corrfn;
struct an_metric;

AN_EXPORT struct an_gpu_context*
an_create_context(unsigned int ndim, int validation);

AN_EXPORT void
an_destroy_context (struct an_gpu_context *ctx);

AN_EXPORT struct an_image*
an_create_image (struct an_gpu_context *ctx,
                 const float           *real,
                 const float           *imag,
                 const unsigned int    *dimensions,
                 unsigned int           ndim);

AN_EXPORT void
an_destroy_image (struct an_image *image);

AN_EXPORT struct an_corrfn*
an_create_corrfn (struct an_gpu_context *ctx,
                  const float           *corrfn,
                  const unsigned int    *dimensions,
                  unsigned int           ndim);

AN_EXPORT void
an_destroy_corrfn (struct an_corrfn *corrfn);

AN_EXPORT int
an_image_update_fft (struct an_image    *image,
                     const unsigned int *coord,
                     unsigned int        ndim,
                     float               delta);

/* Testing */
AN_EXPORT int
an_image_get (struct an_image *image,
              float           *real,
              float           *imag);

/* Distance measurement */
AN_EXPORT struct an_metric*
an_create_metric (struct an_gpu_context *ctx,
                  struct an_corrfn      *target,
                  struct an_image       *recon);

AN_EXPORT void
an_destroy_metric (struct an_metric *metric);

AN_EXPORT int
an_distance (struct an_metric *metric,
             float            *distance);
