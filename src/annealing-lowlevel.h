#pragma once

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
struct an_corrfn;
struct an_metric;

struct an_gpu_context*
an_create_context(unsigned int ndim, int validation);

void
an_destroy_context (struct an_gpu_context *ctx);

struct an_image*
an_create_image (struct an_gpu_context *ctx,
                 const float           *real,
                 const float           *imag,
                 const unsigned int    *dimensions,
                 unsigned int           ndim);

void
an_destroy_image (struct an_image *image);

struct an_corrfn*
an_create_corrfn (struct an_gpu_context *ctx,
                  const float           *corrfn,
                  const unsigned int    *dimensions,
                  unsigned int           ndim);

void
an_destroy_corrfn (struct an_corrfn *corrfn);

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
struct an_metric*
an_create_metric (struct an_gpu_context *ctx,
                  struct an_corrfn      *target,
                  struct an_image       *recon);

void
an_destroy_metric (struct an_metric *metric);

int
an_distance (struct an_metric *metric,
             float            *distance);
