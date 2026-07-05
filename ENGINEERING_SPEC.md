# Janus — Cross-Scale Fluid Dynamics Simulator

**Engineering Design Specification (v1)**

Janus is a pure-Rust, cross-scale (continuum → transition → free-molecular) fluid
dynamics simulator built around the **Unified Gas-Kinetic Wave-Particle (UGKWP)**
method. It targets the Knudsen range **Kn ≈ 0.001 – 100** in a single solver, with a
first-class modern desktop UI and visualization. It exists because existing tools
(Elmer, etc.) do not support seamless cross-scale simulation and have poor UX.

This document is the authoritative build blueprint. Subagents implementing Janus
must follow it. Where it is silent, prefer the simplest correct thing and leave a
`// DESIGN:` note rather than inventing scope.

---

## 0. Non-negotiable constraints

- **Language:** Rust (stable, latest). Edition 2021+.
- **No C bindings / no FFI to C libraries.** No BLAS/LAPACK, no VTK C libs, no
  ffmpeg linking. Pure-Rust crates only. (Calling out to an *external* ffmpeg
  *process* is also disallowed for core features; PNG-sequence export is the
  baseline video path.)
- **Linear algebra:** `faer` only. Do not pull in `nalgebra`/`ndarray` for the
  solver hot path (a thin local vector type is fine; see §5).
- **Serialization:** `bincode` (bincode-next), using its zerocopy path for bulk
  field blocks.
- **Scheduling / parallelism:** the in-house `dtact` crate (M:N topology-aware
  fiber runtime). Do **not** add `rayon`/`tokio` to the solver. `dtact` is at
  `D:\dev\janus-dev\dtact` — see §7 for its API.
- **Approved dependency whitelist:** `iced`, `faer`, `dtact`, `bincode`,
  `memmap2`, `bytemuck`, `wgpu`, `glam`, plus widely-used well-maintained pure-Rust
  crates as needed (e.g. `serde` for the JSON header only, a colormap helper, a
  logging facade). **Before adding anything outside this list, stop and ask the
  user.** Prefer hand-rolling small utilities over new deps.
- **Rendering:** `wgpu` GPU pipeline (Iced already depends on wgpu; reuse its device).
- **unsafe is allowed** where it buys performance or correctness, but must be
  wrapped in a sound, documented abstraction with a `// SAFETY:` comment. Building
  bespoke abstractions instead of leaning on `std` is acceptable when justified.
- **mmap first:** large field/particle data should be memory-mapped (`memmap2`)
  and cast zero-copy via `bytemuck`, not read into `Vec` then parsed.

---

## 1. Discretization decision (settled)

**Finite Volume Method (FVM), cell-centered, structured Cartesian grid (2D first,
3D later).** Rationale: UGKWP/DUGKS interface fluxes are constructed directly from
the integral solution of the kinetic equation across cell faces — this *is* a
finite-volume construction. FDM is non-conservative (fails at shocks); FEM is
unnatural and expensive for hyperbolic kinetic transport. Do not use FDM or FEM.

Grid: uniform Cartesian to start. Structure the mesh API so non-uniform / AMR /
unstructured can be added later behind the same `Mesh` trait, but **do not build
AMR now**.

---

## 2. Physics model (what to implement, in order)

The solver spans scales via UGKWP; there is **no domain-decomposition interface**
between models — the wave/particle split is local per cell and driven by local Kn.
That deliberately avoids the boundary-coupling / discontinuity problem.

**Full UGKWP is mandatory (not optional).** The final solver MUST implement the
complete Unified Gas-Kinetic Wave-Particle scheme: the *full* multiscale flux with a
genuine wave (deterministic) + particle (stochastic) decomposition whose split is
governed by the local accumulated collision probability `e^{-dt/tau}`, valid and
accurate across the entire Kn = 0.001–100 range. A partial UGKWP (e.g. UGKWP only in
some regime with a plain DUGKS simplification elsewhere) is NOT acceptable as the
final state. **DUGKS may remain only as an *optional* kernel** — selectable, and used
only where it is genuinely appropriate (e.g. a cheap near-continuum fast path) — never
as a silent substitute for the full UGKWP flux. Any place currently using a DUGKS
simplification in lieu of full UGKWP must carry a `// PHYSICS-DEBT:` marker until the
full UGKWP path replaces it.

1. **Collision operator: Shakhov (S-model) BGK.** Correct Prandtl number, correct
   viscosity + heat flux for monatomic gas, cost of a local relaxation. Do **not**
   implement the full Boltzmann collision integral now. Keep the collision term
   behind a `Collision` trait so a fast-spectral full operator can be slotted in later.
2. **Wave part = Discrete Velocity Method (DVM) + DUGKS flux.** Deterministic,
   finite-volume. This alone must correctly recover Navier–Stokes in the continuum
   limit (validate on Sod & Couette).
3. **Particle part.** Stochastic particles sampled from the residual distribution
   in rarefied cells; wave↔particle mass is exchanged each step. Local Kn (via a
   gradient-length ratio, `Kn_loc = λ·|∇ρ|/ρ` or equivalent) governs the split so
   transition is smooth and continuous — no fault line.
4. Gas: monatomic, single species, ideal, VHS/hard-sphere viscosity law. Multi-
   species / polyatomic / reactions are explicitly out of scope for v1.

**Physics-completeness mandate (hard requirement for the final model).** Early
milestones may simplify (M1 used a strictly-2D velocity space, γ=2). But the
*final* physics model MUST be complete and correct:
- Use the **reduced (g, h) two-distribution formulation** so a 2D-space simulation
  keeps the correct monatomic 3 translational DOF and **γ = 5/3** (not the γ=2 of a
  strictly-2D velocity space). This is required no later than M2.
- The full Boltzmann collision operator (fast-spectral) must be available behind the
  `Collision` trait by the end (M4), not only the Shakhov model.
- No permanent shortcuts: every simplification carries a `// PHYSICS-DEBT:` marker
  and must be retired before v1 is called complete.

Boundary conditions to support: diffuse wall (Maxwell accommodation coeff), specular
wall, mixed Maxwell, velocity/pressure inlet, outlet, symmetry, periodic.

**Key correctness hazards (write tests around these):** wave↔particle sampling must
conserve mass/momentum/energy exactly; the Shakhov relaxation time and the DUGKS
half-step reconstruction are the two most error-prone pieces. Validate in 1D/2D
where a bug is found in a day, not a week.

---

## 3. Workspace layout

```
janus/                     (cargo workspace)
├─ Cargo.toml              # [workspace] members + shared deps table
├─ ENGINEERING_SPEC.md     # this file
├─ crates/
│  ├─ janus-core/          # Mesh, fields (SoA), units, Kn field, geometry, config
│  ├─ janus-kinetic/       # UGKWP: DVM wave + particles + Shakhov; time stepping
│  ├─ janus-io/            # .jvtk format read/write (mmap + bytemuck + bincode)
│  ├─ janus-sched/         # dtact adapter: mesh blocks -> fibers, load rebalancing
│  ├─ janus-viz/           # wgpu render backend (fields, particles, Kn overlay)
│  ├─ janus-ui/            # Iced desktop app; embeds janus-viz via wgpu primitive
│  └─ janus-cli/           # headless batch runner (HPC, no UI) -> writes .jvtk
```

**Hard rule: the solver (`janus-kinetic`) must not depend on `janus-ui`/`janus-viz`.**
The UI consumes solver snapshots over a channel; the CLI runs the same solver
headless. This decoupling is a primary design goal (Elmer's coupling of the two is
a known failure).

Snapshot flow: solver produces immutable `FrameSnapshot` (Arc'd field slices +
particle point cloud + metadata) every N steps → bounded channel → UI/viz. Solver
never blocks on the UI; drop frames if the consumer is behind.

---

## 4. `.jvtk` file format (define and own it)

A minimal binary VTK-subset for structured grids + particle clouds. Goals: mmap-
friendly, zero-copy, ParaView escape hatch.

Layout of a `.jvtk` file:

```
[8 bytes]  magic  = b"JVTK\x01\x00\x00\x00"   (format version 1)
[u64 LE]   header_len
[header_len bytes] JSON header (UTF-8), serde:
    {
      "dims": [nx, ny, nz],          // nz=1 for 2D
      "spacing": [dx, dy, dz],
      "origin": [x0, y0, z0],
      "time": 1.234,  "step": 42,
      "kn_range": [min, max],
      "cell_fields": [ {"name":"rho","comps":1,"dtype":"f64","offset":B,"len":L}, ... ],
      "point_fields": [ ... ],
      "particles": {"count":N,"offset":B,"stride":..., "layout":["pos3","vel3","weight"]} | null
    }
[padding to 64-byte alignment]
[raw binary blocks]  // each field a contiguous f64/f32 array at its offset; mmap+bytemuck cast
```

Rules: all binary blocks 64-byte aligned (cache line + SIMD). Field arrays are plain
`[f64]`/`[f32]` in C order, cast with `bytemuck::cast_slice` from an mmap. The JSON
header is small and human-inspectable. Provide:

- `JvtkWriter` (streaming: write header, then append blocks; supports time series as
  separate numbered files `case.0000.jvtk`, `case.0001.jvtk`, …).
- `JvtkReader` (mmap the file, expose `&[f64]` views without copying).
- `export_legacy_vtk()` — optional ASCII legacy-VTK writer so users can open output
  in ParaView as a fallback. Keep it out of the hot path.

Use `bincode` zerocopy only for auxiliary structured metadata if needed; the bulk
field arrays are raw aligned blocks, not bincode-encoded, so mmap casting is trivial.

---

## 5. Data structures (performance-critical — read §8 first)

**Structure-of-Arrays everywhere in the hot path.** No `Vec<Cell>` of fat structs.

```rust
// janus-core
pub struct Grid2D { pub nx: usize, pub ny: usize, pub dx: f64, pub dy: f64, pub origin: [f64;2] }

pub struct MacroFields {           // one entry per cell, SoA
    pub rho:    Vec<f64>,
    pub mom:    [Vec<f64>; 2],      // or 3
    pub energy: Vec<f64>,
    pub stress: Vec<[f64; 3]>,      // optional higher moments for transition regime
    pub heat:   [Vec<f64>; 2],
    pub kn_loc: Vec<f64>,           // local Knudsen, drives wave/particle split
}

pub struct Distribution {          // DVM: f[cell][velocity_node], SoA by velocity node
    pub nv: usize,                  // number of discrete velocities
    pub vgrid: Vec<[f64;2]>,        // velocity ordinates + weights
    pub vw: Vec<f64>,
    pub f: Vec<f64>,                // len = ncell * nv, cell-major; index(cell,k)=cell*nv+k
}

pub struct Particles {             // SoA; only populated in rarefied cells
    pub pos: Vec<[f64;2]>, pub vel: Vec<[f64;2]>, pub weight: Vec<f64>, pub cell: Vec<u32>,
}
```

Local math: a tiny `Vec2`/`Vec3` POD type with `#[repr(C)]` + `bytemuck::Pod`;
do not pull nalgebra. Use `faer` only for the (small, dense) implicit/boundary
solves and any future implicit time integration.

---

## 6. Solver architecture (`janus-kinetic`)

Traits for extensibility:

```rust
pub trait Collision { fn relax(&self, /* cell macro state, dt, tau */) -> /* post-collision */; }
pub struct Shakhov { pub pr: f64, /* ... */ }   // implement this now

pub trait BoundaryCondition { fn apply(&self, /* face, ghost state */); }

pub trait TimeStepper { fn step(&mut self, dt: f64); }
```

DUGKS wave step (per cell, finite volume):
1. Reconstruct distribution at cell faces at the half time step using the kinetic
   characteristic solution (this is the crux — document the derivation in code).
2. Compute interface fluxes by integrating face distribution over velocity space.
3. Update cell-averaged distribution; take moments → updated macro fields.
4. Apply Shakhov relaxation toward local equilibrium.

UGKWP wave/particle split (per cell): from `kn_loc`, decide the fraction carried by
deterministic wave vs stochastic particles; sample particles from the free-transport
residual; transport + collide particles; re-aggregate to moments; **assert conservation**.

Time step chosen by CFL over max discrete velocity. Explicit first; keep a hook for
implicit later (faer solve).

---

## 7. Scheduling (`janus-sched`) — dtact integration

`dtact` is an M:N topology-aware fiber runtime. Confirmed public API (from
`dtact/src/api.rs`):

- `dtact::spawn(fut) -> handle` and `dtact::spawn_with()` builder:
  `.kind(WorkloadKind::{Compute,IO,Memory,System})`, `.priority(Priority::{Low,Normal,High,Critical})`,
  `.affinity(Affinity::{SameCore,SameCCX,SameNUMA,Any})`, `.name(..)`, `.switcher::<S>()`,
  `.spawn(fut) -> dtact_handle_t`.
- `dtact::yield_now().await`, `dtact::fiber::*`, `config::set_deflection_threshold(core, thr)`.
- Runtime must be initialized before spawning (`GLOBAL_RUNTIME`). Check the crate's
  README/lib.rs for the init entry point and wire it in `janus-sched` setup.

Design: partition the mesh into **blocks**; each block advanced by one fiber per step.
- **Wave-dominated blocks** (continuum): cheap, regular →
  `.kind(Compute).affinity(SameCCX)` `.priority(Normal)`, deflectable so idle cores steal.
- **Particle-dominated blocks** (rarefied): expensive, irregular →
  `.kind(Compute).affinity(SameNUMA).priority(High)` to keep data NUMA-local.
- After each step, reweight blocks by measured cost (particle count is a good proxy)
  and let dtact's work-deflection rebalance. This dynamic, spatially-drifting load is
  exactly what static MPI partitioning fails at and where dtact should shine.

Deliverable microbench: a case that is half continuum / half rarefied; compare wall
time of static block assignment vs dtact deflection. This is a useful applied data
point.

Beware **false sharing** at block boundaries: pad per-block accumulators to 64 bytes
(`#[repr(align(64))]`), never let two fibers write adjacent cache lines of the same
array. Halo/ghost exchange between blocks must be double-buffered.

---

## 8. Performance rules (mandatory)

- SoA layouts; cell-major distribution indexing `cell*nv+k` for velocity-space locality.
- 64-byte alignment on shared arrays and per-fiber accumulators; pad to avoid false
  sharing (`#[repr(align(64))]` wrapper for hot atomics/counters).
- mmap (`memmap2`) + `bytemuck::cast_slice` for all bulk field/particle I/O; no
  parse-into-Vec.
- Prefer flat `Vec<f64>` + index math over nested `Vec<Vec<..>>`.
- Keep allocations out of the time-step loop; reuse scratch buffers (double buffer
  the distribution / halos).
- `unsafe` allowed for bounds-check elision in the innermost kernels *after* a safe
  version works and is benchmarked — always behind a documented `// SAFETY:` wrapper.
- Add `criterion`-style benches for the flux kernel and the particle push.

---

## 9. UI / UX (`janus-ui` + `janus-viz`) — the product differentiator

Priorities the user cares about most: **modern UX, ease of operation, cross-scale
visualization, performance.** Do the solver's data plumbing first, but design the UI
to these targets:

- **Layout:** left scene tree, center wgpu viewport, right property/inspector panel,
  bottom time-line scrubber. Dark engineering theme.
- **wgpu viewport** embedded in Iced via a custom wgpu primitive sharing Iced's device.
- **Cross-scale "regime overlay" (killer feature):** toggleable semi-transparent
  overlay coloring cells by local Kn regime (continuum / slip / transition / free-
  molecular, 4 bands) and a wave-vs-particle-fraction heatmap. Lets users *see* where
  cross-scale physics happens — nothing else does this.
- **Real-time streaming:** solver pushes a frame every N steps; progress bar,
  pause/resume/continue, scrub back through the time series.
- **Field rendering:** scalar colormaps (viridis/turbo/custom), isolines, streamlines,
  vector glyphs, particle point cloud.
- **Interactive probe:** hover a cell → read ρ, u, T, Kn, wave/particle ratio.
- **Case setup UX:** geometry + boundary conditions via forms/wizard (pick wall type,
  accommodation coeff, inlet/outlet), NOT a hand-written config syntax like Elmer's `.sif`.
- **Export:** PNG sequence (baseline), `.jvtk`, legacy-VTK for ParaView.

---

## 10. Milestones

- **M1 (foundation + 2D wave):** workspace scaffold; `janus-core` (Grid2D, MacroFields,
  Distribution, config); `janus-io` (.jvtk writer/reader + mmap + legacy-VTK export);
  `janus-kinetic` 2D **pure-wave** DUGKS + Shakhov; `janus-cli` headless runner.
  **Validation: Sod shock tube (recovers Euler/NS), Couette flow (recovers viscous
  NS profile).** Output `.jvtk` time series. Unit tests for conservation + BCs.
- **M2 (particles → full UGKWP 2D):** particle layer, wave/particle split by local Kn,
  conservation tests, transition-regime validation (e.g. lid-driven cavity / flat plate
  at Kn≈1). dtact integration + load-balance microbench.
- **M3 (visualization + UI):** janus-viz wgpu backend, janus-ui Iced app, regime
  overlay, real-time streaming, probe, case-setup wizard, export.
- **M4:** 3D extension; **full UGKWP flux path** (retire any DUGKS-simplification
  PHYSICS-DEBT; DUGKS kept only as an optional selectable kernel); optional implicit
  stepping (faer); parallelization hardened and scaled (dtact across all cores/NUMA,
  measured strong/weak scaling, no false sharing, halo double-buffering verified).
- **M5 (advanced physics — required for v1-complete):** fast-spectral full Boltzmann
  collision behind the `Collision` trait; **advanced / real-gas properties**: virial
  expansion equation of state and user-definable custom gas property models (transport
  coefficients, EOS) via a `GasModel` trait; polyatomic/internal-DOF where applicable.
  The cross-scale physics model must remain **accurate and un-simplified** throughout.

**v1 is NOT complete at M4.** v1-complete requires M1–M5: full UGKWP across the whole
Kn range, accurate (non-simplified) cross-scale physics, robust parallelization, and
the advanced gas-property system. No permanent physics shortcuts survive into v1.

---

## 10b. Engineering-quality principle (overrides any "keep it simple" instruction)

"Simple" here means *clean, well-structured, minimal-surface* — NOT
*algorithmically weak or incomplete*. Do **not** choose a simpler-but-inaccurate or
simpler-but-inefficient algorithm to save effort. Because system quality is bounded
by its weakest component (barrel principle), every module must meet the highest
correct-and-efficient standard for its job: proper units system, streaming I/O,
efficient quadrature, full-fidelity physics kernels. If a correct approach is more
work than a crude one, do the correct one (or leave a clearly-marked debt item with a
concrete plan — never silently ship the crude version as final). Prefer the type
system over dynamic dispatch in hot/structural paths. When in doubt, optimize for
correctness and completeness first, then performance, then brevity.

## 11. Coding standards

- `cargo fmt` + `cargo clippy` clean (deny warnings in CI later).
- Every `unsafe` block has a `// SAFETY:` justification.
- Public items documented; module headers state the physics/algorithm they implement
  with a reference (e.g. "DUGKS: Guo, Xu & Wang 2013"; "UGKWP: Liu, Zhu & Xu 2020";
  "Shakhov 1968").
- Tests colocated; validation cases as integration tests producing `.jvtk` compared
  against analytic/reference profiles with a tolerance.
- Keep crate boundaries clean; no `janus-kinetic → janus-ui` dependency, ever.
