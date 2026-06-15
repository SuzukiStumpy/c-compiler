# 0 "test/fixtures/valid/float_type.c"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3
# 0 "<command-line>" 2
# 1 "test/fixtures/valid/float_type.c"
float x;
int main(void) {
    float a = 1.5f;
    float b = 1.5e10f;
    float c = 1e5F;
    float d = 1.F;
    return 0;
}
