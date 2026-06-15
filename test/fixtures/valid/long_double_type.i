# 0 "test/fixtures/valid/long_double_type.c"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3
# 0 "<command-line>" 2
# 1 "test/fixtures/valid/long_double_type.c"
long double x;
int main(void) {
    long double a = 1.5L;
    long double b = 1.5e10L;
    long double c = 1e5l;
    long double d = 1.L;
    return 0;
}
