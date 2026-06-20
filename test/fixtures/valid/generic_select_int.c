int main(void) {
    int x = 42;
    int r = _Generic(x, int: 0, double: 1, default: 2);
    return r;
}
