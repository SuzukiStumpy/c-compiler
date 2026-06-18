# C11 Compatibility Roadmap

Effort: L/M/H/XH. Effectiveness for C11 compat: H/M/L.

| Feature | Effort | Effect | Notes |
|---|---|---|---|
| `-I`/`-D`/`-U` passthrough | L | H | ✓ Done — forwarded to GCC preprocessor |
| `inline` specifier | L | M | Parse + mostly ignore; affects external linkage rules only |
| `_Noreturn` | L | L | Parse + ignore; hint only |
| `_Static_assert` | L | M | Constant expr eval → error; already have const eval |
| Inline enum `enum E { A } x;` | L | M | Parser-only change |
| Anonymous enum `enum { A, B }` standalone | L | L | Small semantic tweak |
| Full constant-expr enum values | L | M | Extend existing const evaluator |
| `_Alignof` | M | M | New operator; type-size lookup |
| `_Alignas` | M | M | Parse + emit `.align`; affects struct layout |
| Designated initialisers `{.x=1}` | M | H | Common in real code; need field-order tracking |
| Compound literals `(T){...}` | M | M | Parse as temp + existing init machinery |
| Flexible array members | M | M | Incomplete-type size calc |
| Anonymous struct/union members | M | H | Field promotion; common in Linux-style code |
| `__int128` full arithmetic | H | L | Multi-word insn sequences; niche use |
| Variadic fn definition (`va_list` etc.) | H | H | `__builtin_va_*` intrinsics + ABI stack layout |
| `_Generic` selection | H | L | Mainly `<tgmath.h>`; complex type dispatch |
| VLAs | H | L | Optional in C11; dropped in C17 |
| `_Thread_local` | H | L | TLS segment; niche |
| `_Atomic` | XH | L | Lock-free sequences + memory ordering; concurrency model |
| `_Complex`/`_Imaginary` | XH | L | Full complex ABI; near-zero real-world demand |

## Recommended order

1. `-I/-D/-U` passthrough — trivial, unlocks real headers
2. `inline` + `_Noreturn` + `_Static_assert` — parse-level, low risk
3. Designated initialisers + anonymous struct/union — high payoff for real code
4. `_Alignas`/`_Alignof` — needed for `<stddef.h>` patterns
5. Enum gaps — small cleanup
6. Compound literals + flexible arrays — fills out struct/init coverage
7. Variadic fn definition — high effort but unlocks writing `printf`-style fns

Stop before `_Atomic`, `_Thread_local`, `_Complex`, VLAs — effort exceeds benefit for a C11-subset compiler.
