// Demonstration of declaring a pointer to a function which adds two numbers
int add(int a, int b) {
  return a + b;
}

int main() {
  int (*ptr)(int, int) = add;
  int result = ptr(5, 3);
  return result;
}