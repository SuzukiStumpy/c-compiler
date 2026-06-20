/* Lexer/parser test fixture for wide and Unicode string/char literals. */
int main(void) {
    wchar_t *ws = L"hello";
    wchar_t wc = L'A';
    unsigned short *u16s = u"hi";
    unsigned short u16c = u'B';
    unsigned int *u32s = U"bye";
    unsigned int u32c = U'C';
    char *u8s = u8"utf8";
    return 0;
}
