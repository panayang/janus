//! janus-io: the `.jvtk` binary format (mmap + bytemuck zero-copy field I/O)
//! plus an ASCII legacy-VTK export fallback for ParaView.
//!
//! Format defined in ENGINEERING_SPEC.md §4. Layout:
//!
//! ```text
//! [8 bytes]  magic = b"JVTK\x01\x00\x00\x00"
//! [u64 LE]   header_len
//! [header_len bytes] JSON header (UTF-8)
//! [padding to 64-byte alignment]
//! [raw binary blocks]  // f64/f32 arrays, C order, 64-byte aligned each
//! ```

pub mod header;
pub mod legacy_vtk;
pub mod reader;
pub mod writer;

pub use header::{FieldDesc, JvtkHeader, ParticleDesc};
pub use reader::JvtkReader;
pub use writer::JvtkWriter;

/// Format magic bytes: "JVTK" + version 1 encoded as 3 zero-padded bytes +
/// 0x00, i.e. `b"JVTK\x01\x00\x00\x00"` (8 bytes total).
pub const MAGIC: [u8; 8] = *b"JVTK\x01\x00\x00\x00";

/// All binary blocks (and the start of the block region as a whole) are
/// aligned to this many bytes (cache line size, good for SIMD too).
pub const BLOCK_ALIGN: usize = 64;

/// Round `n` up to the next multiple of `BLOCK_ALIGN`.
#[inline]
pub fn align_up(n: usize) -> usize {
    (n + BLOCK_ALIGN - 1) / BLOCK_ALIGN * BLOCK_ALIGN
}
