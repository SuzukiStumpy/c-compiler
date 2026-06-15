# 0 "test/fixtures/valid/variadic_decl.c"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3
# 0 "<command-line>" 2
# 1 "test/fixtures/valid/variadic_decl.c"
int printf(char *fmt, ...);

int main(void) {
    printf("hello %d\n", 42);
    return 0;
}
