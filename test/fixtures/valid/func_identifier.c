/* Integration test for the __func__ predefined identifier (C99 §6.4.2.2).
   Uses an inline strcmp to avoid pulling in string.h (which uses __restrict).
   Returns 0 if __func__ equals the enclosing function name, non-zero otherwise. */

static int str_eq(const char *a, const char *b) {
    while (*a && *a == *b) {
        a++;
        b++;
    }
    return *a - *b;
}

static int named_function(void) {
    return str_eq(__func__, "named_function");
}

int main(void) {
    return named_function();
}
