//! JSON persistence: writes/reads `run.json` and `traj.jsonl` for a run.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fluid::{FrameData, RunRecord};
use crate::system::{BoxConfig, System, Vec2};

/// One saved frame, as written to `traj.jsonl` (one object per line).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FrameRecord {
    pub frame: usize,
    pub t: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub k: f64,
    pub u: f64,
    pub e: f64,
    pub tkin: f64,
}

/// Run metadata plus per-frame histories, as written to `run.json`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunJson {
    pub atoms: usize,
    pub rho: f64,
    pub temperature: f64,
    pub seed: u64,
    pub dt: f64,
    pub cutoff: f64,
    #[serde(rename = "box")]
    pub box_len: f64,
    pub eq: usize,
    pub steps: usize,
    pub save_every: usize,
    pub frames: usize,
    pub e0: f64,
    pub history: History,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct History {
    pub t: Vec<f64>,
    pub k: Vec<f64>,
    pub u: Vec<f64>,
    pub e: Vec<f64>,
    pub tkin: Vec<f64>,
}

fn frame_record(frame: usize, f: &FrameData, box_cfg: BoxConfig) -> FrameRecord {
    let sys = System::with_box(f.positions.clone(), f.velocities.clone(), box_cfg);
    let e = crate::system::total_energy(&sys);
    let ke: f64 = f.velocities.iter().map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1])).sum();
    let n = f.positions.len() as f64;
    FrameRecord {
        frame,
        t: f.t,
        x: f.positions.iter().map(|p| p[0]).collect(),
        y: f.positions.iter().map(|p| p[1]).collect(),
        vx: f.velocities.iter().map(|v| v[0]).collect(),
        vy: f.velocities.iter().map(|v| v[1]).collect(),
        k: ke,
        u: e - ke,
        e,
        tkin: ke / n,
    }
}

/// Write `run.json` and `traj.jsonl` for a run into `dir` (created if missing).
pub fn write_output(dir: &Path, rec: &RunRecord) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let box_cfg = BoxConfig {
        length: rec.box_len,
        cutoff: rec.cfg.cutoff,
    };
    let frames: Vec<FrameRecord> = rec
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| frame_record(i, f, box_cfg))
        .collect();

    let run = RunJson {
        atoms: rec.cfg.atoms,
        rho: rec.cfg.rho,
        temperature: rec.cfg.temperature,
        seed: rec.cfg.seed,
        dt: rec.cfg.dt,
        cutoff: rec.cfg.cutoff,
        box_len: rec.box_len,
        eq: rec.cfg.eq,
        steps: rec.cfg.steps,
        save_every: rec.cfg.save_every,
        frames: frames.len(),
        e0: rec.e0,
        history: History {
            t: frames.iter().map(|f| f.t).collect(),
            k: frames.iter().map(|f| f.k).collect(),
            u: frames.iter().map(|f| f.u).collect(),
            e: frames.iter().map(|f| f.e).collect(),
            tkin: frames.iter().map(|f| f.tkin).collect(),
        },
    };
    fs::write(
        dir.join("run.json"),
        serde_json::to_string_pretty(&run).unwrap(),
    )?;

    let mut lines = String::new();
    for f in &frames {
        lines.push_str(&serde_json::to_string(f).unwrap());
        lines.push('\n');
    }
    fs::write(dir.join("traj.jsonl"), lines)?;
    Ok(())
}

/// Read a run written by [`write_output`]: `(RunJson, Vec<FrameRecord>)`.
pub fn read_run(dir: &Path) -> std::io::Result<(RunJson, Vec<FrameRecord>)> {
    let run: RunJson = serde_json::from_str(&fs::read_to_string(dir.join("run.json"))?)?;
    let traj = fs::read_to_string(dir.join("traj.jsonl"))?;
    let mut frames = Vec::new();
    for line in traj.lines() {
        if !line.trim().is_empty() {
            frames.push(serde_json::from_str::<FrameRecord>(line)?);
        }
    }
    Ok((run, frames))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::{run_fluid, RunConfig};

    #[test]
    fn write_then_read_roundtrip() {
        let dir = std::env::temp_dir().join(format!("md_store_{}", std::process::id()));
        let c = RunConfig {
            atoms: 36,
            rho: 0.8,
            eq: 50,
            steps: 100,
            save_every: 25,
            ..RunConfig::default()
        };
        let rec = run_fluid(&c);
        write_output(&dir, &rec).unwrap();
        assert!(dir.join("run.json").exists());
        assert!(dir.join("traj.jsonl").exists());

        let (meta, frames) = read_run(&dir).unwrap();
        assert_eq!(frames.len(), rec.frames.len());
        assert_eq!(meta.atoms, 36);
        // Stored energies are finite and the file lines parse to full frames.
        for f in &frames {
            assert_eq!(f.x.len(), 36);
            assert!(f.e.is_finite());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
