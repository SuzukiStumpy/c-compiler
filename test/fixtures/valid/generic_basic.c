int main(void) {
    int x = 0;
    return _Generic(x, int: 0, double: 1, default: 2);
}
