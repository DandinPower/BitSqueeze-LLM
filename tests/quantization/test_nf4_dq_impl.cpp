#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <quantization/quantization.hpp>
#include <quantization/utils/random.hpp>
#include <quantization/utils/evaluation.hpp>
#include <inttypes.h>

int main(void) {
    const unsigned long long X   = 5;            /* number of random arrays            */
    const unsigned long long N   = 4194304;      /* length of each array               */
    const float  MINV  = -10.0f;
    const float  MAXV  =  10.0f;
    const unsigned int SEED = 12345;

    float **inputs = gen_random_float_arrays(X, N, MINV, MAXV, SEED);
    if (!inputs) {
        fprintf(stderr, "failed to allocate random inputs\n");
        return EXIT_FAILURE;
    }

    for (unsigned long long k = 0; k < X; ++k) {
        quantization_buffer_t *buf = NULL;
        double t0 = get_time_ms();
        int c_res = quantization_compress(inputs[k], N, NF4_DQ, &buf);
        double t1 = get_time_ms();
        double comp_time = t1 - t0;
        if (c_res || !buf) {
            fprintf(stderr, "nf4_dq compress failed on array %llu \n", (unsigned long long)k);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        float *deq = (float *)malloc(N * sizeof(float));
        if (!deq) {
            fprintf(stderr, "malloc failed for nf4_dq dequant buffer\n");
            quantization_free(buf);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double t2 = get_time_ms();
        int d_res = quantization_decompress(buf, deq, N);
        double t3 = get_time_ms();
        double decomp_time = t3 - t2;
        if (d_res) {
            fprintf(stderr, "nf4_dq decompress failed on array %llu \n", (unsigned long long)k);
            free(deq);
            quantization_free(buf);
            free_random_float_arrays(inputs, X);
            return EXIT_FAILURE;
        }

        double mae, mse, maxabs;
        measure_metrics(inputs[k], deq, N, &mae, &mse, &maxabs);

        double size_kb = quantization_get_packed_size(buf) / 1024.0;
        double bw = 8.0 * size_kb * 1024.0 / (double)N;

        printf("[array %llu] N=%llu, original_size=%.3f KB\n",
               (unsigned long long)k, (unsigned long long)N, N * sizeof(float) / 1024.0);
        printf("   NF4_DQ: size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               size_kb, bw, mae, mse, maxabs);
        printf("          CompTime=%.3f ms, DecompTime=%.3f ms\n", comp_time, decomp_time);

        free(deq);
        quantization_free(buf);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
