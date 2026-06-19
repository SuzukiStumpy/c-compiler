/* Variadic function with args that spill to overflow area (> 6 GP regs). */
#include <stdarg.h>

int sum7(int a, int b, int c, int d, int e, int f, ...) {
    va_list ap;
    va_start(ap, f);
    int g = va_arg(ap, int);  /* 7th arg -- must come from overflow area */
    va_end(ap);
    return a + b + c + d + e + f + g;
}

int main(void) {
    return sum7(1, 2, 3, 4, 5, 6, 7) - 28;
}
