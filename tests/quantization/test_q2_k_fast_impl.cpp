#include <stdio.h>
#include <stdlib.h>

#include <quantization/quantization.hpp>
#include <quantization/utils/random.hpp>
#include <quantization/utils/evaluation.hpp>

int main(void) {
    const unsigned long long X = 5;
    const unsigned long long N = 4194304;
    float **inputs = gen_random_float_arrays(X, N, -10.0f, 10.0f, 12345);
    if (!inputs) return EXIT_FAILURE;

    for (unsigned long long k = 0; k < X; ++k) {
        quantization_buffer_t *buf = NULL;
        double t0 = get_time_ms();
        if (quantization_compress(inputs[k], N, Q2_K_FAST, &buf) || !buf) return EXIT_FAILURE;
        double t1 = get_time_ms();
        double comp_time = t1 - t0;

        float *deq = (float *)malloc(N * sizeof(float));
        if (!deq) return EXIT_FAILURE;

        double t2 = get_time_ms();
        if (quantization_decompress(buf, deq, N)) return EXIT_FAILURE;
        double t3 = get_time_ms();
        double decomp_time = t3 - t2;

        double mae, mse, maxabs;
        measure_metrics(inputs[k], deq, N, &mae, &mse, &maxabs);
        double size_kb = quantization_get_packed_size(buf) / 1024.0;
        double bw = 8.0 * size_kb * 1024.0 / (double)N;
        printf("[array %llu] Q2_K_FAST: size=%.3f KB, B/W=%.5f, MAE=%.6f, MSE=%.6f, MaxAbs=%.6f\n",
               k, size_kb, bw, mae, mse, maxabs);
        printf("           CompTime=%.3f ms, DecompTime=%.3f ms\n", comp_time, decomp_time);

        free(deq);
        quantization_free(buf);
    }

    free_random_float_arrays(inputs, X);
    return EXIT_SUCCESS;
}
