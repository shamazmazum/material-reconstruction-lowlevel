#include <fftw3.h>
#include <string.h>
#include "annealing-lowlevel.h"

struct an_array_sizes {
    size_t real;
    size_t complex;
};

static struct an_array_sizes
an_get_array_sizes (const unsigned int *dimensions,
                    unsigned int        ndims) {
    struct an_array_sizes result;
    size_t size = 1;
    unsigned int i;

    for (i=0; i<ndims; i++) {
        size *= dimensions[i];
    }

    result.real = size;
    result.complex = result.real / dimensions[i-1] * (dimensions[i-1]/2 + 1);

    return result;
}

int an_rfft (const float *array,
             float       *real,
             float       *imag,
             const unsigned int  *dimensions,
             unsigned int    ndims) {
    // Input and output arrays
    float      *in;
    fftwf_complex *out;

    // FFT plan
    fftwf_plan p;

    // Dimensions
    size_t i;
    struct an_array_sizes asizes = an_get_array_sizes (dimensions, ndims);

    // Success
    int ok = 1;

    in  = fftwf_malloc(sizeof(float)         * asizes.real);
    out = fftwf_malloc(sizeof(fftwf_complex) * asizes.complex);
    p   = fftwf_plan_dft_r2c (ndims, (const int*)dimensions, in, out, FFTW_ESTIMATE);

    if (p == NULL) {
        ok = 0;
        goto cleanup;
    }

    // Copy data to the input array and calculate FFT
    memcpy (in, array, sizeof(float)*asizes.real);
    fftwf_execute(p);

    for (i=0; i < asizes.complex; i++) {
        real[i] = out[i][0];
        imag[i] = out[i][1];
    }

cleanup:
    if (p != NULL) {
        fftwf_destroy_plan (p);
    }

    fftwf_free (in);
    fftwf_free (out);

    return ok;
}

int an_irfft (float       *array,
              const float *real,
              const float *imag,
              const unsigned int  *dimensions,
              unsigned int    ndims) {
    // Input and output arrays
    fftwf_complex *in;
    float      *out;

    // FFT plan
    fftwf_plan p;

    // Dimensions
    size_t i;
    struct an_array_sizes asizes = an_get_array_sizes (dimensions, ndims);

    // Success
    int ok = 1;

    in  = fftwf_malloc(sizeof(fftwf_complex) * asizes.complex);
    out = fftwf_malloc(sizeof(float)      * asizes.real);
    p   = fftwf_plan_dft_c2r (ndims, (const int*)dimensions, in, out, FFTW_ESTIMATE);

    if (p == NULL) {
        ok = 0;
        goto cleanup;
    }

    for (i=0; i < asizes.complex; i++) {
        in[i][0] = real[i];
        in[i][1] = imag[i];
    }

    fftwf_execute(p);
    memcpy (array, out, sizeof(float)*asizes.real);

cleanup:
    if (p != NULL) {
        fftwf_destroy_plan (p);
    }

    fftwf_free (in);
    fftwf_free (out);

    return ok;
}
