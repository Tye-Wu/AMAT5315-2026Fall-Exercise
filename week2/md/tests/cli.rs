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
            "run",
            "--n",
            "36",
            "--rho",
            "0.8",
            "--eq-steps",
            "50",
            "--steps",
            "100",
            "--sample-every",
            "25",
            "--seed",
            "7",
            "--out",
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
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["n"], 36);
    assert_eq!(run["integrator"], "velocity-verlet");
    let first = std::fs::read_to_string(dir.join("traj.jsonl"))
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    let frame: serde_json::Value = serde_json::from_str(&first).unwrap();
    for key in ["step", "t", "pos", "vel", "E_pot", "E_kin"] {
        assert!(frame.get(key).is_some(), "missing frame field {key}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn heating_run_records_ramp_to() {
    let dir = std::env::temp_dir().join(format!("md_cli_ramp_{}", std::process::id()));
    let out = Command::new(bin())
        .args([
            "run",
            "--n",
            "36",
            "--eq-steps",
            "50",
            "--steps",
            "100",
            "--sample-every",
            "25",
            "--temperature",
            "0.2",
            "--ramp-to",
            "1.2",
            "--out",
        ])
        .arg(&dir)
        .output()
        .expect("run heating md");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["ramp_to"], 1.2);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn contract_run_passes_all_physics_bounds() {
    let dir = std::env::temp_dir().join(format!("md_contract_{}", std::process::id()));
    let run = Command::new(bin())
        .args(["run", "--out"])
        .arg(&dir)
        .output()
        .expect("run contract simulation");
    assert!(
        run.status.success(),
        "contract run failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let check = Command::new(bin())
        .arg("check")
        .arg(&dir)
        .output()
        .expect("check contract simulation");
    assert!(
        check.status.success(),
        "physics check failed:\n{}\n{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(String::from_utf8_lossy(&check.stdout).contains("PASS"));
    let _ = std::fs::remove_dir_all(dir);
}
