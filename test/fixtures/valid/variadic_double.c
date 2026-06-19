/* Variadic function summing doubles, tests XMM register path in va_arg. */
#include <stdarg.h>

double dsum(int count, ...) {
    va_list ap;
    va_start(ap, count);
    double total = 0.0;
    for (int i = 0; i < count; i++) {
        total += va_arg(ap, double);
    }
    va_end(ap);
    return total;
}

int main(void) {
    double result = dsum(3, 1.5, 2.5, 6.0);
    /* result == 10.0; return 0 if correct */
    return (int)(result - 10.0);
}
