// Naughty, but we don't currently support const char * declarations so
// have to redeclare here

int puts(char *c);

int main(void) {
    puts("Hello World!");
    return 0;
}