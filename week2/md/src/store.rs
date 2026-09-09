//! JSON persistence: writes/reads `run.json` and `traj.jsonl` for a run.

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
