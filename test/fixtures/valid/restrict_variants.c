/* Fixture for __restrict and __restrict__ lexer tokenisation tests.
   Both GCC extension spellings should produce a RESTRICT token. */
int copy(char * __restrict dst, const char * __restrict__ src);
int main(void) { return 0; }
