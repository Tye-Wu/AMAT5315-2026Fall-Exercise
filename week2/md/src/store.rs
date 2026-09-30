//! JSON persistence for the exact Week 2 run contract.

use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fluid::RunRecord;
use crate::system::Vec2;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FrameRecord {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RunJson {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_lengths: Vec2,
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    pub cutoff: f64,
    pub force: String,
    pub ramp_to: Option<f64>,
}

pub fn write_output(dir: &Path, record: &RunRecord) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let run = RunJson {
        n: record.cfg.n,
        rho: record.cfg.rho,
        box_lengths: record.box_lengths,
        dt: record.cfg.dt,
        temperature: record.cfg.temperature,
        eq_steps: record.cfg.eq_steps,
        steps: record.cfg.steps,
        sample_every: record.cfg.sample_every,
        seed: record.cfg.seed,
        integrator: "velocity-verlet".to_string(),
        cutoff: record.cfg.cutoff,
        force: record.cfg.force_method.as_str().to_string(),
        ramp_to: record.cfg.ramp_to,
    };
    fs::write(
        dir.join("run.json"),
        serde_json::to_string_pretty(&run).expect("serialize run metadata"),
    )?;

    let mut trajectory = BufWriter::new(fs::File::create(dir.join("traj.jsonl"))?);
    for frame in &record.frames {
        let stored = FrameRecord {
            step: frame.step,
            t: frame.t,
            pos: frame.positions.clone(),
            vel: frame.velocities.clone(),
            e_pot: frame.e_pot,
            e_kin: frame.e_kin,
        };
        serde_json::to_writer(&mut trajectory, &stored)?;
        trajectory.write_all(b"\n")?;
    }
    trajectory.flush()?;
    Ok(())
}

pub fn read_run(dir: &Path) -> io::Result<(RunJson, Vec<FrameRecord>)> {
    let run = serde_json::from_str(&fs::read_to_string(dir.join("run.json"))?)?;
    let mut frames = Vec::new();
    for line in fs::read_to_string(dir.join("traj.jsonl"))?.lines() {
        if !line.trim().is_empty() {
            frames.push(serde_json::from_str(line)?);
        }
    }
    Ok((run, frames))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid::{RunConfig, run_fluid};

    #[test]
    fn output_roundtrip_uses_required_fields() {
        let dir = std::env::temp_dir().join(format!("md_store_{}", std::process::id()));
        let cfg = RunConfig {
            n: 36,
            eq_steps: 50,
            steps: 100,
            sample_every: 25,
            ..RunConfig::default()
        };
        write_output(&dir, &run_fluid(&cfg)).unwrap();
        let raw = fs::read_to_string(dir.join("run.json")).unwrap();
        for key in [
            "\"n\"",
            "\"rho\"",
            "\"box\"",
            "\"eq_steps\"",
            "\"sample_every\"",
            "\"integrator\"",
            "\"ramp_to\"",
        ] {
            assert!(raw.contains(key), "missing {key}");
        }
        let (meta, frames) = read_run(&dir).unwrap();
        assert_eq!(meta.n, 36);
        assert_eq!(meta.integrator, "velocity-verlet");
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0].pos.len(), 36);
        assert!(frames[0].e_pot.is_finite() && frames[0].e_kin.is_finite());
        let _ = fs::remove_dir_all(dir);
    }
}
