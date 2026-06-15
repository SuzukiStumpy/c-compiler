      .section .rodata
      .balign 4
float_const.1069547520:
      .long 1069547520
      .section .rodata
      .balign 4
float_const.1348437110:
      .long 1348437110
      .section .rodata
      .balign 4
float_const.1203982336:
      .long 1203982336
      .section .rodata
      .balign 4
float_const.1065353216:
      .long 1065353216
      .globl x
      .bss
      .balign 4
x:
      .zero 4

###
# Function: main
# Source: test/fixtures/valid/float_type.c:2
###
      .globl main
      .text
main:
      pushq %rbp
      movq  %rsp, %rbp
      movq  float_const.1069547520(%rip), %xmm0
      movq  float_const.1348437110(%rip), %xmm0
      movq  float_const.1203982336(%rip), %xmm0
      movq  float_const.1065353216(%rip), %xmm0
      movl  $0, %eax
      movq  %rbp, %rsp
      popq  %rbp
      ret
      .section .note.GNU-stack,"",@progbits
