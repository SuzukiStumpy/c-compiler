# C11 Compatibility Roadmap

Features remaining to implement. Effort: L/M/H/XH. Impact on C11 compatibility: H/M/L.

## Unimplemented / missing features

| Feature | Effort | Impact | Notes |
|---|---|---|---|
| `_Alignas` on struct/union members | M | L | C11 §6.7.5 p3 — alignment specifiers are supported on variable declarations but `_parse_member_declaration` calls `_parse_type()` rather than `_parse_specifiers()`, so `_Alignas` on a struct field is silently ignored |
| `inline` body inlining | H | M | C11 §6.7.4 — inline linkage rules are fully enforced; actual substitution of the callee body at call sites is not performed; cross-TU (LTO) inlining is out of scope |
| `_Noreturn` contract enforcement | L | L | C11 §6.7.4 — `_Noreturn` is lexed, parsed, and stored on `Function.is_noreturn`; no diagnostic is emitted when a `_Noreturn` function has a reachable `return` path |
| VLAs | H | L | C99 mandatory, optional in C11 (Annex J.2), removed in C17; requires runtime stack allocation; low priority given C17 trajectory |
| `_Thread_local` storage-class specifier | H | L | C11 §6.7.1 — per-thread static storage via the TLS segment; requires linker support and thread-local relocation model |
| `_Atomic` type qualifier / specifier | XH | L | C11 §6.7.2.4 — lock-free atomic reads, writes, and read-modify-write operations; requires instruction-level memory ordering and a full concurrency model |
| `_Complex` / `_Imaginary` | XH | L | C11 §6.2.5 p11 — complex and imaginary arithmetic; full ABI support; near-zero real-world demand outside scientific computing |

## Priority grouping

**Deferred**: VLAs, `_Alignas` on struct/union members, `inline` body inlining — self-contained but high effort or niche benefit.

**Low priority**: `_Noreturn` enforcement — trivial to add but rarely matters in practice (UB-on-return is already undefined, and most callers rely on attributes for optimisation hints only).

**Skip for now**: `_Thread_local`, `_Atomic`, `_Complex`/`_Imaginary` — effort-to-benefit ratio is unfavourable for a C11-subset compiler.
