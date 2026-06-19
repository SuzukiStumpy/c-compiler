# C11 Compatibility Roadmap

Features remaining to implement. Effort: L/M/H/XH. Impact on C11 compatibility: H/M/L.

## Unimplemented features

| Feature | Effort | Impact | Notes |
|---|---|---|---|
| `__func__` predefined identifier | L | M | C99/C11 §6.4.2.2 — magic `const char[]` variable in every function body containing the function name; commonly used in assert macros and debug logging |
| `register` / `auto` storage-class specifiers | L | L | C89/C11; both are effectively no-ops (register is deprecated in C11); parse and ignore to avoid errors on real C code that uses them |
| `_Pragma` operator | L | L | C99/C11 §6.10.9 — turns a string into a `#pragma`; mainly used in macros that conditionally emit pragmas; can be lowered to a no-op or forwarded to the preprocessor |
| Static array parameters `int foo(int a[static 10])` | L | L | C99/C11 §6.7.6.3 — `static` inside array parameter declarator indicates minimum array size; parse + semantic annotation only; used as an optimisation hint by compilers |
| `__int128` full arithmetic | H | L | Multi-word instruction sequences (`adcq`/`sbbq`), two-register SysV calling convention; niche use outside cryptography/numerics |
| Hex floating-point literals (`0x1.8p10`) | M | L | C99/C11 §6.4.4.2 — exact FP representation via hex mantissa and binary exponent; used in some numerical libraries and standard headers |
| Wide / Unicode string and char literals | M | L | C11 §6.4.4.4, §6.4.5 — `L'x'`, `L"..."`, `u'x'`, `u"..."`, `U'x'`, `U"..."`, `u8"..."` prefixes; needed to parse i18n and platform code without errors |
| `char16_t` / `char32_t` types | M | L | C11 §6.7.2 via `<uchar.h>`; natural companion to Unicode string literals; effectively `uint16_t`/`uint32_t` aliases |
| GCC builtins (`__builtin_expect`, `__builtin_unreachable`, `__builtin_offsetof`) | M | M | Widely used in Linux/glibc headers; `__builtin_expect` is a branch-hint no-op, `__builtin_unreachable` marks dead paths, `__builtin_offsetof` computes member offsets; required to parse real-world system headers without errors |
| Multi-file compilation (multiple `.c` per invocation) | M | M | Tool feature — `compiler a.c b.c` in a single call; separate compilation of individual files and linking already works |
| Inline function external linkage rules | M | L | C99/C11 §6.7.4 p6–7 — `inline` without `static` or `extern` has complex one-definition-rule implications; currently `inline` is stored but no linkage enforcement is performed |
| `_Generic` selection | H | L | C11 §6.5.1.1 — type-generic macro dispatch used by `<tgmath.h>`; requires type-matching at parse/SA time; complex to implement correctly |
| VLAs | H | L | C99 mandatory, optional in C11 (Annex J.2), removed in C17; requires runtime stack allocation; low priority given C17 trajectory |
| `_Thread_local` storage-class specifier | H | L | C11 §6.7.1 — per-thread static storage via the TLS segment; requires linker support and thread-local relocation model |
| `_Atomic` type qualifier / specifier | XH | L | C11 §6.7.2.4 — lock-free atomic reads, writes, and read-modify-write operations; requires instruction-level memory ordering and a full concurrency model |
| `_Complex` / `_Imaginary` | XH | L | C11 §6.2.5 p11 — complex and imaginary arithmetic; full ABI support; near-zero real-world demand outside scientific computing |

## Priority grouping

**Quick wins** (L effort, low risk): `__func__`, `register`/`auto`, `_Pragma`, static array params.  
Implement these to silence parse errors on common real-world code.

**Medium payoff**: GCC builtins and multi-file compilation have the highest practical impact — they unblock parsing real system headers and building multi-file projects.

**Deferred**: `__int128`, `_Generic`, VLAs are each self-contained but high effort for niche benefit.

**Skip for now**: `_Thread_local`, `_Atomic`, `_Complex`/`_Imaginary` — effort-to-benefit ratio is unfavourable for a C11-subset compiler.
