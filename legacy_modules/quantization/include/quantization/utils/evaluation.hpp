#include <math.h>
#include <stdlib.h>
#include <time.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void measure_metrics(const float *orig, const float *deq, unsigned long long N,
                            double *mae, double *mse, double *max_abs) {
    double m = 0.0, s = 0.0, mx = 0.0;
    for (unsigned long long i = 0; i < N; ++i) {
        double e = (double)deq[i] - (double)orig[i];
        double ae = fabs(e);
        m    += ae;
        s    += e * e;
        if (ae > mx) mx = ae;
    }
    *mae     = m / (double)N;
    *mse     = s / (double)N;
    *max_abs = mx;
}
