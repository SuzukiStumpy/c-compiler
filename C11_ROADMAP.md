# C11 Compatibility Roadmap

Features remaining to implement. Effort: L/M/H/XH. Impact on C11 compatibility: H/M/L.

## Unimplemented features

| Feature | Effort | Impact | Notes |
|---|---|---|---|
| Hex floating-point literals (`0x1.8p10`) | M | L | C99/C11 §6.4.4.2 — exact FP representation via hex mantissa and binary exponent; used in some numerical libraries and standard headers |
| Inline function external linkage rules | M | L | C99/C11 §6.7.4 p6–7 — `inline` without `static` or `extern` has complex one-definition-rule implications; currently `inline` is stored but no linkage enforcement is performed |
| VLAs | H | L | C99 mandatory, optional in C11 (Annex J.2), removed in C17; requires runtime stack allocation; low priority given C17 trajectory |
| `_Thread_local` storage-class specifier | H | L | C11 §6.7.1 — per-thread static storage via the TLS segment; requires linker support and thread-local relocation model |
| `_Atomic` type qualifier / specifier | XH | L | C11 §6.7.2.4 — lock-free atomic reads, writes, and read-modify-write operations; requires instruction-level memory ordering and a full concurrency model |
| `_Complex` / `_Imaginary` | XH | L | C11 §6.2.5 p11 — complex and imaginary arithmetic; full ABI support; near-zero real-world demand outside scientific computing |

## Priority grouping

**Medium payoff**: Remaining medium items: `inline` external linkage rules, hex floating-point literals.

**Deferred**: VLAs — self-contained but high effort for niche benefit.

**Skip for now**: `_Thread_local`, `_Atomic`, `_Complex`/`_Imaginary` — effort-to-benefit ratio is unfavourable for a C11-subset compiler.
