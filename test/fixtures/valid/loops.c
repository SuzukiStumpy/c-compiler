int main(void) {
    int i = 0;
    while (i < 3) {
        if (i == 1) continue;
        if (i == 2) break;
        i++;
    }
    for (int j = 0; j < 1; j++) {
        break;
    }
    do {
        i++;
    } while (i < 5);
    return 0;
}
