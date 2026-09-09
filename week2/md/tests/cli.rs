//! End-to-end tests: the built binary writes readable run.json + traj.jsonl.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_md")
}

#[test]
fn run_writes_readable_outputs() {
    let dir = std::env::temp_dir().join(format!("md_cli_{}", std::process::id()));
    let out = Command::new(bin())
        .args([
            "--atoms", "36", "--rho", "0.8", "--eq", "50", "--steps", "100",
            "--save-every", "25", "--seed", "7", "--dir",
        ])
        .arg(&dir)
        .output()
        .expect("run md");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    for f in ["run.json", "traj.jsonl"] {
        let p = dir.join(f);
        assert!(p.exists(), "{f} missing in {}", dir.display());
        let txt = std::fs::read_to_string(&p).unwrap();
        assert!(!txt.trim().is_empty(), "{f} empty");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
