//! janus-cli: headless batch runner for the janus-kinetic DUGKS solver.
//!
//! Builds a hardcoded case (grid + BCs + initial condition), runs the
//! time-stepper for N steps, and writes a `.jvtk` time series every M steps.
//! M1 scope: no file-based config parsing — the case is a Rust struct built
//! in `main()`, per the milestone instructions.

use janus_core::config::{BoundaryAssignment, BoundaryKind, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_io::writer::{FieldData, NamedField};
use janus_io::JvtkWriter;
use janus_kinetic::maxwellian::gh_equilibrium;
use janus_kinetic::solver::DugksSolver;
use janus_kinetic::velocity_grid::VelocityGrid2D;

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "janus_output".to_string());
    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

    // Hardcoded lid-off Couette-like shear case: periodic in x, diffuse
    // walls (different tangential velocity) north/south. Small grid for a
    // fast headless smoke run; validation tests use their own configs.
    let nx = 20;
    let ny = 20;
    let gas = GasProperties::monatomic_default();
    let grid = Grid2D::new(nx, ny, 1.0e-3 / nx as f64, 1.0e-3 / ny as f64, [0.0, 0.0]);
    let bcs = BoundaryAssignment {
        west: BoundaryKind::Periodic,
        east: BoundaryKind::Periodic,
        south: BoundaryKind::DiffuseWall { temperature: 300.0, wall_velocity: [0.0, 0.0] },
        north: BoundaryKind::DiffuseWall { temperature: 300.0, wall_velocity: [50.0, 0.0] },
    };
    let config = CaseConfig { grid, bcs, gas };

    let (vgrid, vw) = VelocityGrid2D::simpson(1800.0, 25);
    let mut dist = Distribution::zeros(config.grid.ncells(), vgrid.clone(), vw.clone());

    // Initial condition: quiescent gas at rest, uniform density/temperature.
    let rho0 = 1.0;
    let t0 = 300.0;
    let ncells = config.grid.ncells();
    for c in 0..ncells {
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(rho0, [0.0, 0.0], t0, config.gas.r_gas, *v);
            dist.f[c * dist.nv + k] = g;
            dist.h[c * dist.nv + k] = h;
        }
    }

    let mut solver = DugksSolver::new(&config, dist);
    solver.update_moments();

    let n_steps: u32 = 2000;
    let write_every: u32 = 200;
    let dt = solver.cfl_dt(0.4);

    println!("janus-cli: running {n_steps} steps, dt={dt:.6e}, writing every {write_every} steps to {out_dir}");

    let mut frame_index: u32 = 0;
    write_frame(&out_dir, frame_index, &solver, 0.0, 0);

    for step in 1..=n_steps {
        solver.step(dt, &config.bcs);
        if step % write_every == 0 {
            frame_index += 1;
            write_frame(&out_dir, frame_index, &solver, dt * step as f64, step as u64);
        }
    }

    println!("janus-cli: done, wrote {} frames", frame_index + 1);
}

fn write_frame(out_dir: &str, frame_index: u32, solver: &DugksSolver, time: f64, step: u64) {
    let rho = solver.fields.rho.clone();
    let mom_x = solver.fields.mom[0].clone();
    let mom_y = solver.fields.mom[1].clone();
    let energy = solver.fields.energy.clone();

    let fields = vec![
        NamedField { name: "rho".into(), comps: 1, data: FieldData::F64(&rho) },
        NamedField { name: "mom_x".into(), comps: 1, data: FieldData::F64(&mom_x) },
        NamedField { name: "mom_y".into(), comps: 1, data: FieldData::F64(&mom_y) },
        NamedField { name: "energy".into(), comps: 1, data: FieldData::F64(&energy) },
    ];

    JvtkWriter::write_series_step(
        out_dir,
        "case",
        frame_index,
        [solver.grid.nx, solver.grid.ny, 1],
        [solver.grid.dx, solver.grid.dy, 1.0],
        [solver.grid.origin[0], solver.grid.origin[1], 0.0],
        time,
        step,
        [0.0, 0.0],
        &fields,
        &[],
        None,
    )
    .expect("failed to write jvtk frame");
}
