//! Golden outcomes for every example and a construct corpus, under both engines.
//!
//! The interpreter is the reference: its stdout, last stderr line and exit
//! code for each program under `engine_goldens/programs/` are pinned in
//! `engine_goldens/expected/<name>.bio.txt`, and this test fails on any change.
//! Titan's outcome for the same program is compared to the interpreter's and
//! counted as matched, refused (a `titan:` compile error) or diverged; the two
//! failure counts are ratchets that may only fall.
//!
//! `UPDATE_GOLDENS=1 cargo test -p aether-cli --test engine_goldens` rewrites
//! the goldens instead of asserting them.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Programs Titan refuses today. Lower it when Titan learns a construct.
const REFUSED_CEILING: usize = 3;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/engine_goldens")
}

/// Stdout without the banner (it names the file and the engine), then the last
/// stderr line and the exit code. A run past the deadline is `TIMEOUT`.
fn outcome(program: &Path, titan: bool) -> String {
    let rel = program.strip_prefix(root()).unwrap_or(program);
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_aether"));
    cmd.current_dir(root()).arg("run").arg(rel);
    if titan {
        cmd.args(["--mode", "titan"]);
    }
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("aether runs");
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait") {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return String::from("TIMEOUT\n");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let (mut out, mut err) = (String::new(), String::new());
    child.stdout.take().unwrap().read_to_string(&mut out).ok();
    child.stderr.take().unwrap().read_to_string(&mut err).ok();
    let out = out.replace("\r\n", "\n");
    let lines: Vec<&str> = out.lines().collect();
    let rules: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].starts_with('═'))
        .collect();
    let body = if rules.len() >= 2 {
        &lines[rules[1] + 1..]
    } else {
        &lines[..]
    };
    let err = err.replace("\r\n", "\n");
    let last_err = err.lines().last().unwrap_or("");
    format!(
        "{}\n--- stderr ---\n{}\n--- exit ---\n{}\n",
        body.join("\n").trim_end(),
        last_err,
        status
            .code()
            .map_or(String::from("signal"), |c| c.to_string())
    )
}

#[test]
fn the_interpreter_reproduces_every_golden_and_titan_stays_within_its_ratchets() {
    let update = std::env::var_os("UPDATE_GOLDENS").is_some();
    let expected = dir().join("expected");
    fs::create_dir_all(&expected).unwrap();

    let mut programs: Vec<PathBuf> = fs::read_dir(dir().join("programs"))
        .expect("programs")
        .map(|e| e.unwrap().path())
        .collect();
    programs.sort();
    assert!(programs.len() >= 60, "only {} programs", programs.len());

    let (mut matched, mut refused, mut diverged) = (Vec::new(), Vec::new(), Vec::new());
    let mut drift = Vec::new();
    for program in &programs {
        let name = program.file_stem().unwrap().to_string_lossy().to_string();
        let bio = outcome(program, false);
        let titan = outcome(program, true);
        let golden = expected.join(format!("{name}.bio.txt"));
        if update {
            fs::write(&golden, &bio).unwrap();
            fs::write(expected.join(format!("{name}.titan.txt")), &titan).unwrap();
        } else if fs::read_to_string(&golden)
            .map(|g| g.replace("\r\n", "\n"))
            .ok()
            .as_deref()
            != Some(bio.as_str())
        {
            drift.push(name.clone());
        }
        if titan == bio {
            matched.push(name);
        } else if titan.contains("\ntitan: ") {
            refused.push(name);
        } else {
            diverged.push(name);
        }
    }

    println!(
        "  parity over {} programs: {} matched, {} refused, {} diverged",
        programs.len(),
        matched.len(),
        refused.len(),
        diverged.len()
    );
    println!("  refused: {refused:?}");
    println!("  diverged: {diverged:?}");

    assert!(
        drift.is_empty(),
        "the interpreter no longer reproduces these goldens: {drift:?}"
    );
    assert!(
        refused.len() <= REFUSED_CEILING,
        "Titan refuses {} programs, above the ratchet of {REFUSED_CEILING}: {refused:?}",
        refused.len()
    );
    // A divergence is a wrong answer from one engine, so there is no ceiling
    // above zero to ratchet down from.
    assert!(
        diverged.is_empty(),
        "Titan diverges from the interpreter on {diverged:?}"
    );
}
