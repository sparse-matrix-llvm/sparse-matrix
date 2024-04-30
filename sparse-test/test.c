#include <stdio.h>


typedef float m2x2_t __attribute__((matrix_type(2, 2)));

void f(float *pa, float *pb) {
        m2x2_t a = __builtin_sparse_matrix_load(pa, 2, 2, 4);
        m2x2_t b = __builtin_sparse_matrix_load(pb, 2, 2, 4);
        m2x2_t r = a * b + 10.0;
        __builtin_sparse_matrix_store(r, pa, 4);
}

int main() {
        float pa[4] = {1.0, 2.0, 3.0, 4.0};
        float pb[4] = {1.0, 0.0, 0.0, 1.0};
        printf("%.2f %.2f %.2f %.2f\n", pa[0], pa[1], pa[2], pa[3]);
        return 0;
}
