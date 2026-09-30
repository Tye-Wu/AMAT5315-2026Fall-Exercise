use std::{env, fs, path::PathBuf};
use week5_seismic::{Experiment, run_forward};

#[cfg(feature = "enzyme")]
unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
}

fn write_npy_f64(
    path: &std::path::Path,
    shape: &[usize],
    values: impl Iterator<Item = f64>,
) -> std::io::Result<()> {
    use std::io::Write;
    let shape_text = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let base = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_text}, }}");
    let pad = (16 - ((10 + base.len() + 1) % 16)) % 16;
    let header = format!("{base}{}\n", " ".repeat(pad));
    let mut f = fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_npy_f32(
    path: &std::path::Path,
    shape: &[usize],
    values: impl Iterator<Item = f32>,
) -> std::io::Result<()> {
    use std::io::Write;
    let shape_text = format!(
        "({})",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    let base = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    let pad = (16 - ((10 + base.len() + 1) % 16)) % 16;
    let header = format!("{base}{}\n", " ".repeat(pad));
    let mut f = fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in values {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_npy_f64(
    path: &std::path::Path,
) -> Result<(Vec<usize>, Vec<f64>), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a NumPy .npy file".into());
    }
    let major = bytes[6];
    let (hlen, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen])?;
    if !(header.contains("'<f8'") || header.contains("\"<f8\"") || header.contains("'|f8'")) {
        return Err("expected little-endian float64 .npy".into());
    }
    let shape_pos = header.find("shape").ok_or("npy header has no shape")?;
    let paren = header[shape_pos..].find('(').ok_or("npy shape missing")? + shape_pos;
    let close = header[paren..].find(')').ok_or("npy shape unclosed")? + paren;
    let shape = header[paren + 1..close]
        .split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    let payload = &bytes[start + hlen..];
    if payload.len() % 8 != 0 {
        return Err("malformed f64 npy payload length".into());
    }
    let values = payload
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    if shape.iter().product::<usize>() != values.len() {
        return Err("npy shape/payload size mismatch".into());
    }
    Ok((shape, values))
}

fn write_recording(
    out: &std::path::Path,
    every: usize,
    steps: &[usize],
    nx: usize,
    nz: usize,
    dt: f64,
    time_unit_s: f64,
    field: &[Vec<f32>],
) -> Result<(), Box<dyn std::error::Error>> {
    write_npy_f32(
        &out.join("wavefield.npy"),
        &[field.len(), nz, nx],
        field.iter().flatten().copied(),
    )?;
    let rec = serde_json::json!({"every":every,"frame_steps":steps,"times":steps.iter().map(|&n|n as f64*dt*time_unit_s).collect::<Vec<_>>()});
    fs::write(out.join("recording.json"), serde_json::to_vec_pretty(&rec)?)?;
    Ok(())
}

fn compact_experiment(exp: &Experiment) -> serde_json::Value {
    let mut value = serde_json::to_value(exp).expect("serialize experiment metadata");
    if let Some(map) = value.as_object_mut() {
        map.remove("background");
        map.remove("perturbation");
    }
    value
}

fn recording_meta(exp: &Experiment, every: usize, steps: &[usize]) -> serde_json::Value {
    serde_json::json!({"every":every,"steps":steps,
        "times":steps.iter().map(|&n| n as f64 * exp.dt * exp.time_unit_s).collect::<Vec<_>>()})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let first = args.next();
    if first.as_deref() == Some("--enzyme-smoke") {
        #[cfg(feature = "enzyme")]
        {
            let mut out = [0.0_f64; 3];
            unsafe {
                enzyme_cube(2.0, out.as_mut_ptr());
            }
            println!("primal={} jvp={} vjp={}", out[0], out[1], out[2]);
            if out != [8.0, 12.0, 12.0] {
                return Err("Enzyme cube smoke test mismatch".into());
            }
            return Ok(());
        }
        #[cfg(not(feature = "enzyme"))]
        {
            return Err("--enzyme-smoke requires cargo --features enzyme".into());
        }
    }
    let mut argument_queue = Vec::new();
    if let Some(v) = first {
        argument_queue.push(v);
    }
    argument_queue.extend(args);
    let mut args = argument_queue.into_iter();
    let mut input = None;
    let mut out = None;
    let mut every = 3_usize;
    let mut mode = "forward".to_string();
    let mut data_path = None;
    let mut storage = "full".to_string();
    let mut checkpoints = 0_usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--experiment" => input = args.next(),
            "--out" => out = args.next(),
            "--every" => every = args.next().ok_or("--every requires a value")?.parse()?,
            "--mode" => mode = args.next().ok_or("--mode requires a value")?,
            "--data" => data_path = args.next(),
            "--storage" => storage = args.next().ok_or("--storage requires a value")?,
            "--checkpoints" => {
                checkpoints = args
                    .next()
                    .ok_or("--checkpoints requires a value")?
                    .parse()?
            }
            _ => return Err(format!("unknown argument: {a}").into()),
        }
    }
    let input = input.ok_or("--experiment is required")?;
    let out = PathBuf::from(out.unwrap_or_else(|| "artifacts/forward".into()));
    fs::create_dir_all(&out)?;
    let experiment: Experiment = serde_json::from_slice(&fs::read(&input)?)?;
    let base = serde_json::json!({"mode":mode,"nx":experiment.nx,"nz":experiment.nz,"dx":experiment.dx,"dt":experiment.dt,"steps":experiment.steps,"shots":experiment.shots,"receivers":experiment.receivers});
    if mode == "forward" {
        let result = run_forward(&experiment, false, every)?;
        let shape = [
            result.traces.len(),
            experiment.steps,
            experiment.receivers.len(),
        ];
        write_npy_f64(
            &out.join("traces.npy"),
            &shape,
            result.traces.iter().flatten().flatten().copied(),
        )?;
        write_npy_f32(
            &out.join("echo.npy"),
            &[result.echo.len(), experiment.nz, experiment.nx],
            result.echo.iter().flatten().copied(),
        )?;
        write_recording(
            &out,
            every,
            &result.frame_steps,
            experiment.nx,
            experiment.nz,
            experiment.dt,
            experiment.time_unit_s,
            &result.wavefield,
        )?;
        let rec = recording_meta(&experiment, every, &result.frame_steps);
        let meta = serde_json::json!({"base":base,"trace_l2":result.trace_l2,"shot_maxima":result.shot_maxima,"max_trace_step":result.max_trace_step,"recording":rec});
        fs::write(out.join("result.json"), serde_json::to_vec_pretty(&meta)?)?;
        fs::write(
            out.join("run.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"experiment_file":input,"experiment":compact_experiment(&experiment),"recording":meta["recording"]}),
            )?,
        )?;
        println!(
            "trace_l2={:.9}\nshot_maxima={:?}\nmax_trace_step={:?}",
            result.trace_l2, result.shot_maxima, result.max_trace_step
        );
    } else if mode == "born" {
        #[cfg(feature = "enzyme")]
        {
            let result = week5_seismic::run_born_enzyme(&experiment)?;
            let shape = [
                result.traces.len(),
                experiment.steps,
                experiment.receivers.len(),
            ];
            write_npy_f64(
                &out.join("born_data.npy"),
                &shape,
                result.traces.iter().flatten().flatten().copied(),
            )?;
            let meta = serde_json::json!({"base":base,"born_l2":result.l2,"storage":"none","reverse_calls":0,"scheduler_forward_calls":0,"peak_saved_states":0,"peak_saved_bytes":0});
            fs::write(out.join("result.json"), serde_json::to_vec_pretty(&meta)?)?;
            fs::write(
                out.join("run.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"experiment_file":input,"experiment":compact_experiment(&experiment)}),
                )?,
            )?;
            println!("born_l2={:.12}", result.l2);
        }
        #[cfg(not(feature = "enzyme"))]
        {
            return Err(
                "born mode requires the course Enzyme toolchain; use --features enzyme on Linux"
                    .into(),
            );
        }
    } else if mode == "adjoint" {
        let data_path = data_path.ok_or("adjoint mode requires --data <born_data.npy>")?;
        let (data_shape, data) = read_npy_f64(std::path::Path::new(&data_path))?;
        if data_shape
            != [
                experiment.shots.len(),
                experiment.steps,
                experiment.receivers.len(),
            ]
        {
            return Err(format!("born-data shape mismatch: {data_shape:?}").into());
        }
        #[cfg(feature = "enzyme")]
        {
            let (
                image,
                left,
                right,
                relerr,
                frames,
                frame_steps,
                reverse_calls,
                scheduler_forward_calls,
                peak_saved_states,
                peak_saved_bytes,
                per_shot,
            ) = if storage == "full" {
                let r = week5_seismic::run_adjoint_enzyme(&experiment, &data, every)?;
                let states = experiment.steps + 1;
                let each = serde_json::json!({"reverse_calls":experiment.steps,
                        "scheduler_forward_calls":experiment.steps,"peak_saved_states":states});
                (
                    r.image,
                    r.transpose_left,
                    r.transpose_right,
                    r.transpose_relative_error,
                    r.wavefield,
                    r.frame_steps,
                    experiment.steps * experiment.shots.len(),
                    experiment.steps * experiment.shots.len(),
                    states,
                    states * 2 * experiment.nx * experiment.nz * 8,
                    vec![each; experiment.shots.len()],
                )
            } else if storage == "treeverse" {
                let r =
                    week5_seismic::run_adjoint_treeverse(&experiment, &data, checkpoints, every)?;
                for (shot, actions) in r.actions.iter().enumerate() {
                    fs::write(
                        out.join(format!("actions-{shot}.json")),
                        serde_json::to_vec_pretty(actions)?,
                    )?;
                }
                let per_shot = r.actions.iter().enumerate().map(|(shot, actions)| {
                    serde_json::json!({"actions_file":format!("actions-{shot}.json"),
                        "reverse_calls":actions.iter().filter(|a|a.action=="grad").count(),
                        "scheduler_forward_calls":actions.iter().filter(|a|a.action=="call").count(),
                        "peak_saved_states":actions.iter().map(|a|a.saved_states).max().unwrap_or(1)})
                }).collect::<Vec<_>>();
                (
                    r.image,
                    r.transpose_left,
                    r.transpose_right,
                    r.transpose_relative_error,
                    r.wavefield,
                    r.frame_steps,
                    r.reverse_calls,
                    r.scheduler_forward_calls,
                    r.peak_saved_states,
                    r.peak_saved_bytes,
                    per_shot,
                )
            } else {
                return Err(format!("unsupported storage mode: {storage}").into());
            };
            write_npy_f64(
                &out.join("image.npy"),
                &[experiment.nz, experiment.nx],
                image.iter().flatten().copied(),
            )?;
            write_recording(
                &out,
                every,
                &frame_steps,
                experiment.nx,
                experiment.nz,
                experiment.dt,
                experiment.time_unit_s,
                &frames,
            )?;
            let rec = recording_meta(&experiment, every, &frame_steps);
            let meta = serde_json::json!({"base":base,"storage":storage,
                "checkpoints":if storage=="full"{serde_json::Value::Null}else{serde_json::json!(checkpoints)},
                "reverse_calls":reverse_calls,"scheduler_forward_calls":scheduler_forward_calls,
                "peak_saved_states":peak_saved_states,"peak_saved_bytes":peak_saved_bytes,
                "per_shot":per_shot,"transpose_left":left,"transpose_right":right,
                "transpose_relative_error":relerr,
                "recording":rec});
            fs::write(out.join("result.json"), serde_json::to_vec_pretty(&meta)?)?;
            fs::write(
                out.join("run.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"experiment_file":input,"experiment":compact_experiment(&experiment),"data_file":data_path,"recording":meta["recording"]}),
                )?,
            )?;
            println!(
                "transpose_left={:.12}\ntranspose_right={:.12}\ntranspose_relative_error={:.3e}",
                left, right, relerr
            );
        }
        #[cfg(not(feature = "enzyme"))]
        {
            let _ = (data, storage, checkpoints);
            return Err(
                "adjoint mode requires the course Enzyme toolchain; use --features enzyme on Linux"
                    .into(),
            );
        }
    } else {
        return Err(format!("unsupported mode: {mode}").into());
    }
    Ok(())
}
