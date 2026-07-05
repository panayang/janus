//! JSON header schema for `.jvtk` files.

use serde::{Deserialize, Serialize};

/// Description of one field block (cell- or point-centered).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldDesc {
    pub name: String,
    pub comps: usize,
    pub dtype: String, // "f64" or "f32"
    pub offset: u64,   // byte offset from start of file
    pub len: u64,      // number of scalar elements (comps * ncells/npoints)
}

/// Description of the particle block, if present.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticleDesc {
    pub count: u64,
    pub offset: u64,
    pub stride: u64,
    pub layout: Vec<String>, // e.g. ["pos3","vel3","weight"]
}

/// The JSON header of a `.jvtk` file, exactly matching ENGINEERING_SPEC.md §4.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JvtkHeader {
    pub dims: [usize; 3],
    pub spacing: [f64; 3],
    pub origin: [f64; 3],
    pub time: f64,
    pub step: u64,
    pub kn_range: [f64; 2],
    pub cell_fields: Vec<FieldDesc>,
    pub point_fields: Vec<FieldDesc>,
    pub particles: Option<ParticleDesc>,
}

impl JvtkHeader {
    /// Serialize to JSON bytes (compact, UTF-8), used to compute `header_len`
    /// and to write the header block.
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("JvtkHeader must always be JSON-serializable")
    }

    pub fn from_json_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}
