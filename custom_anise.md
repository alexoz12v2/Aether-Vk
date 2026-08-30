# Custom Anise Fork — AetherVk

This is a custom fork of [nyx-space/anise](https://github.com/nyx-space/anise) (based on **0.9.6**) adapted
for use in AetherVk. The main divergence from upstream is `no_std` compatibility so that the library can
be used inside the Rust `rlib` crate that compiles to a C-ABI shared library without a standard-library
allocator assumption.

> [!NOTE]
> Upstream 0.10.6 was evaluated for rebasing but does **not** add SPK Type 21 support either, so rebasing
> was deferred. New types are being back-ported from CSPICE directly into this fork instead.

---

## Changes vs. Upstream 0.9.6

### Added in this fork

| File | What was added |
|------|----------------|
| `anise/src/naif/daf/datatypes/extended_modified_diff.rs` | **SPK Type 21** (Extended Modified Difference Array) — full evaluator ported from [`spke21.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke21.c) and [`spkr21.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spkr21.c). Handles runtime-variable `MAXDIM` (up to 25) unlike the hardcoded-15 Type 1. |
| `anise/src/naif/daf/datatypes/mod.rs` | Exports `extended_modified_diff` module. |
| `anise/src/ephemerides/translate_to_parent.rs` | `DafDataType::Type21ExtendedModifiedDifferenceArray` match arm wired to `ExtendedModifiedDiffType21`. |

### Why Type 21 matters

JPL Horizons generates SPK Type 21 files for **all comet and asteroid custom ephemerides**. Without this,
`spk_ezr` on any Horizons-downloaded `.bsp` file would fail with:

```
UnsupportedDatatype { dtype: Type21ExtendedModifiedDifferenceArray, kind: "SPK computations" }
```

67P/Churyumov-Gerasimenko (NAIF ID `1000012`) is the primary use case.

---

## SPK Type Support Status

Types already supported by anise 0.9.6 upstream are marked ✅. Types added in this fork are marked 🆕.
Types not yet implemented are ranked by implementation effort.

| Type | Name | Status | Source file | Effort to implement |
|------|------|--------|-------------|---------------------|
| 1  | Modified Difference Array | ✅ upstream | `spke01.c` | — |
| 2  | Chebyshev (position only) | ✅ upstream | — | — |
| 3  | Chebyshev (pos + vel)     | ✅ upstream | — | — |
| 5  | Two-body propagation (discrete states) | ❌ missing | [`spke05.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke05.c) | 🔴 **Hard** — requires full Kepler universal-variable solver (`prop2b`) |
| 8  | Lagrange (equal step)      | ✅ upstream | — | — |
| 9  | Lagrange (unequal step)    | ✅ upstream | — | — |
| 10 | NORAD TLE (SGP4)           | ❌ missing | [`spke10.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke10.c) | 🔴 **Very hard** — full SGP4 propagator + TEME frame transform; practically a separate crate |
| 12 | Hermite (equal step)       | ✅ upstream | — | — |
| 13 | Hermite (unequal step)     | ✅ upstream | — | — |
| 14 | Chebyshev (pos + vel, unequal step) | ❌ missing | [`spke14.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke14.c) | 🟢 **Easy** — identical to Type 2/3, just different record layout (`ncoeff, midpt, halfwidth, coeffs×6`). Uses `chbval`. |
| 15 | Precessing conic propagation | ❌ missing | [`spke15.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke15.c) | 🔴 **Hard** — precessing ellipse with J2 node/peri regression; needs `prop2b`, `vrotv`, anomaly solver |
| 17 | Equinoctial elements       | ❌ missing | [`spke17.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke17.c) | 🟡 **Medium** — validates then delegates to `eqncpv` (equinoctial → Cartesian, pure trig, ~100 lines) |
| 18 | ESOC Hermite/Lagrange      | ❌ missing | [`spke18.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke18.c) | 🟡 **Medium** — subtype dispatch (0 = Hermite, 1 = Lagrange); can reuse existing anise Hermite/Lagrange kernels |
| 19 | ESOC piecewise (generalised 18) | ❌ missing | [`spke19.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke19.c) | 🟡 **Medium** — nearly identical structure to Type 18 with piecewise boundary logic added |
| 20 | Chebyshev (velocity only)  | ❌ missing | [`spke20.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke20.c) | 🟡 **Medium** — integrates velocity Chebyshev polynomials for position; needs `chbigr` (Chebyshev indefinite integral, ~50 lines) |
| 21 | Extended Modified Difference Array | 🆕 **this fork** | [`spke21.c`](https://github.com/NablaZeroLabs/cspice/blob/master/src/spke21.c) | ✅ done |

### Recommended implementation order

1. **Type 14** — trivial Chebyshev adaptation, no new math
2. **Type 20** — add `chbigr` (Chebyshev integral), then straightforward
3. **Type 17** — implement `eqncpv` (pure orbital mechanics, self-contained)
4. **Type 18 / 19** — reuse existing anise Hermite/Lagrange with subtype dispatch
5. **Type 15** — after `prop2b` is available
6. **Type 5** — depends on `prop2b`
7. **Type 10** — consider pulling in an existing SGP4 crate instead

---

## Key Architecture Notes

### MAXDIM (Type 21 vs. Type 1)

| Aspect | Type 1 | Type 21 |
|--------|--------|---------|
| Max difference table dim | Hardcoded `15` | Runtime `MAXDIM` (1–25), stored at `slice[len-2]` |
| Record size | Fixed `71` doubles | `4 * MAXDIM + 11` doubles |
| `DT` array layout | `[15][3]` row-major | `[MAXDIM][3]` **col-major** |
| W-loop (position) | `for jx in 1..=mq2` | `while ks >= 2` (saves `jx` for velocity) |

### Segment tail layout (both Type 1 and Type 21)

```
[... records × dflsiz doubles ...]  ← record_data
[... num_records epoch f64s ...]    ← epoch_data  
[... num_records/100 dir epochs ...]← epoch_registry
[maxdim as f64]                     ← only Type 21; absent in Type 1
[num_records as f64]
```

### CSPICE reference repository

All missing evaluators can be ported from:
<https://github.com/NablaZeroLabs/cspice/tree/master/src>

Pattern: `spke{NN}.c` = evaluator, `spkr{NN}.c` = record reader (tells you the segment layout).
