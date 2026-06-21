/* Fixture for __asm__ symbol-renaming lexer tokenisation tests. */
extern int my_scanf(const char *fmt, ...) __asm__("scanf");
int main(void) { return 0; }
