#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float **gen_random_float_arrays(unsigned long long count,
                                unsigned long long N,
                                float minv,
                                float maxv,
                                unsigned int seed) {
    if (!count || !N || !isfinite(minv) || !isfinite(maxv) || maxv < minv)
        return NULL;

    if (!seed) seed = (unsigned int)time(NULL);
    srand(seed);

    float **arrs = (float**)calloc(count, sizeof(*arrs));
    if (!arrs) return NULL;

    for (unsigned long long i = 0; i < count; ++i) {
        arrs[i] = (float*)malloc(N * sizeof(float));
        if (!arrs[i]) {
            for (unsigned long long j = 0; j < i; ++j) free(arrs[j]);
            free(arrs);
            return NULL;
        }
        for (unsigned long long j = 0; j < N; ++j) {
            float u = (float)rand() / (float)RAND_MAX;
            arrs[i][j] = minv + u * (maxv - minv);
        }
    }
    return arrs;
}

void free_random_float_arrays(float **arrs, unsigned long long count) {
    if (!arrs) return;
    for (unsigned long long i = 0; i < count; ++i) free(arrs[i]);
    free(arrs);
}