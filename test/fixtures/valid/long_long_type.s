      .globl a
      .bss
      .balign 8
a:
      .zero 8
      .globl b
      .bss
      .balign 8
b:
      .zero 8

###
# Function: main
# Source: test/fixtures/valid/long_long_type.c:3
###
      .globl main
      .text
main:
      pushq %rbp
      movq  %rsp, %rbp
      subq  $16, %rsp
      movq  $42, -8(%rbp)
      movq  $100, -16(%rbp)
      movl  $0, %eax
      movq  %rbp, %rsp
      popq  %rbp
      ret
      .section .note.GNU-stack,"",@progbits
