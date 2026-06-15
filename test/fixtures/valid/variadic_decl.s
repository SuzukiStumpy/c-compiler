      .section .rodata
      .balign 4
.Lstr.0:
      # String: hello %d\n\0
      .byte 104, 101, 108, 108, 111, 32, 37, 100, 10, 0

###
# Function: main
# Source: test/fixtures/valid/variadic_decl.c:3
###
      .globl main
      .text
main:
      pushq %rbp
      movq  %rsp, %rbp
      leaq  .Lstr.0(%rip), %rdi
      movl  $42, %esi
      movb  $0, %al
      call  printf@PLT
      movl  $0, %eax
      movq  %rbp, %rsp
      popq  %rbp
      ret
      .section .note.GNU-stack,"",@progbits
