//! janus-sched: dtact-backed mesh-block scheduler for the UGKWP solver.
//!
//! Partitions a `Grid2D` into rectangular blocks and advances each block on
//! a dtact fiber per timestep. Wave-dominated (continuum) blocks are spawned
//! with `.kind(Compute).affinity(SameCCX).priority(Normal)` (deflectable, so
//! idle cores can steal cheap regular work); particle-dominated (rarefied)
//! blocks are spawned with `.kind(Compute).affinity(SameNUMA).priority(High)`
//! to keep their (heavier, more irregular) working set NUMA-local. Each
//! step, blocks are reweighted by measured cost (particle count is used as
//! the proxy per ENGINEERING_SPEC.md §7) which changes which affinity/
//! priority combination is chosen for the *next* step's spawn.
//!
//! `janus-sched` depends on `janus-kinetic` (the solver) and `dtact`
//! (scheduling); the reverse is never true (ENGINEERING_SPEC.md §3 hard
//! rule) — `janus-kinetic` has no knowledge of blocks, fibers, or dtact.
//!
//! References: UGKWP wave/particle split — Liu, Zhu & Xu, J. Comput. Phys.
//! 401, 108977 (2020); DUGKS — Guo, Xu & Wang, Phys. Rev. E 88, 033305
//! (2013).
//!
//! ## dtact runtime initialization
//!
//! dtact's `GLOBAL_RUNTIME` (a `OnceLock<Runtime>`) must be initialized and
//! its worker threads started (`Runtime::start`) before any `dtact::spawn`/
//! `spawn_with` call, or those calls panic ("Dtact Runtime not
//! initialized"). Per `dtact/src/lib.rs`, the supported initialization path
//! is the `#[dtact::dtact_init]` attribute macro (re-exported from
//! `dtact_macros`), which is applied to a user `fn main()` (or an
//! equivalent entry point) and performs the `OnceLock::set` + `Runtime::
//! start()` dance internally. `janus-sched` is a library crate (no `main`),
//! so it cannot itself apply `#[dtact::dtact_init]` to anything — this
//! attribute must be applied by the *binary* crate that uses `janus-sched`
//! (e.g. a future `janus-cli` mode, or a dedicated benchmark binary — see
//! `examples/load_balance_bench.rs`). The contract is simply: apply
//! `#[dtact::dtact_init]` to that binary's entry point before constructing
//! or stepping any `SchedRunner`; `SchedRunner` itself does not attempt to
//! self-initialize the runtime (dtact does not expose a safe public
//! "initialize now, no macro" function in `src/api.rs`/`src/lib.rs` as
//! read — only the proc-macro path and the internal `Runtime::start`
//! method, which requires a `'static` `Runtime` value that only the macro
//! constructs). DESIGN: if a non-macro programmatic init API turns out to
//! be needed for `janus-cli`/benchmark binaries, that is a dtact-side gap,
//! not something to work around by reaching into dtact's private fields.

pub mod block;
pub mod block3d;
pub mod runner;
pub mod runner3d;

pub use block::{Block, BlockKind, PaddedAccumulator};
pub use block3d::{Block3D, BlockKind3D, PaddedAccumulator3D};
pub use runner::SchedRunner;
pub use runner3d::SchedRunner3D;

// `SchedRunner3D` (dtact fiber-per-block runner over `Block3D`/`Grid3D`,
// mirroring `runner::SchedRunner`) is implemented in `runner3d.rs`: same
// fiber-per-block spawn/join pattern, same `.kind(Compute)` +
// `.affinity(SameCCX|SameNUMA)` + `.priority(Normal|High)` scheduling hints,
// generalized to 6 block faces (west/east/south/north/down/up) for the
// double-buffered halo exchange. See `runner3d.rs`'s module doc for the
// same "block-local physics is a same-sized independent sub-case" honesty
// note that applies to the 2D runner (the actual cross-block ghost-coupled
// DUGKS/particle kernel is a follow-on solver-internals refactor; the
// fiber-parallel scheduling machinery itself, which is what the
// load-balance microbenchmark measures, is fully real).
