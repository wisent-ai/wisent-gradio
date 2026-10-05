//! `wisent-gradio-release baseline [--stdout]`: rewrite (or print)
//! `released-surface.json` from the artifact PyPI serves for the latest
//! wisent-gradio — the sdist when the release has one, else the pure-Python
//! wheel — read with this repository's own `release/surface.py --tolerant`.
//! The first token of `source` is the tier marker (`pypi-sdist:<file>` or
//! `pypi-wheel:<file>`) the version-check workflow asserts against PyPI.
//! Exit 1 is a refusal, exit 2 an invocation the command does not take.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use serde_json::{json, Value};

const PROJECT: &str = "wisent-gradio";
const USAGE: &str = "usage: wisent-gradio-release baseline [--stdout]";

fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let response = ureq::get(url).call().map_err(|error| format!("{url}: {error}"))?;
    let mut body = Vec::new();
    response.into_reader().read_to_end(&mut body).map_err(|error| format!("{url}: {error}"))?;
    Ok(body)
}

/// The latest version PyPI serves, its tier marker, and the artifact for it.
fn published() -> Result<(String, &'static str, Value), String> {
    let index = format!("https://pypi.org/pypi/{PROJECT}/json");
    let data: Value = serde_json::from_slice(&fetch(&index)?).map_err(|error| format!("{index} is not JSON: {error}"))?;
    let version = data["info"]["version"].as_str().ok_or_else(|| format!("{index} names no info.version"))?.to_string();
    let files: Vec<Value> = data["releases"][&version]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|file| file["yanked"].as_bool() != Some(true))
        .collect();
    if let Some(sdist) = files.iter().find(|file| file["packagetype"].as_str() == Some("sdist")) {
        return Ok((version, "pypi-sdist", sdist.clone()));
    }
    if let Some(wheel) = files
        .iter()
        .find(|file| file["filename"].as_str().is_some_and(|name| name.ends_with("-py3-none-any.whl")))
    {
        return Ok((version, "pypi-wheel", wheel.clone()));
    }
    Err(format!("{PROJECT} {version} is published but offers neither an sdist nor a pure-Python wheel"))
}

/// Unpack the artifact into `scratch` and return the directory that holds `wisent/`.
fn unpack(marker: &str, artifact: &Value, scratch: &Path) -> Result<PathBuf, String> {
    let filename = artifact["filename"].as_str().unwrap_or_default();
    let payload = fetch(artifact["url"].as_str().ok_or_else(|| format!("{filename} names no url"))?)?;
    if marker == "pypi-sdist" {
        tar::Archive::new(flate2::read::GzDecoder::new(payload.as_slice()))
            .unpack(scratch)
            .map_err(|error| format!("{filename} does not unpack: {error}"))?;
    } else {
        zip::ZipArchive::new(std::io::Cursor::new(&payload))
            .and_then(|mut archive| archive.extract(scratch))
            .map_err(|error| format!("{filename} does not unpack: {error}"))?;
    }
    if scratch.join("wisent").is_dir() {
        return Ok(scratch.to_path_buf());
    }
    let inner: Vec<PathBuf> = std::fs::read_dir(scratch)
        .map_err(|error| format!("{}: {error}", scratch.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|child| child.join("wisent").is_dir())
        .collect();
    match inner.as_slice() {
        [root] => Ok(root.clone()),
        _ => Err(format!("{filename}: expected exactly one tree containing `wisent`, found {}", inner.len())),
    }
}

/// The surface `release/surface.py` reads from `root`, tolerating modules the
/// published artifact itself could not import.
fn surface(repository: &Path, root: &Path) -> Result<Value, String> {
    let reader = repository.join("release").join("surface.py");
    let output = Command::new("python3")
        .arg(&reader)
        .arg(root)
        .arg("--tolerant")
        .output()
        .map_err(|error| format!("python3 {} could not start: {error}", reader.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} refused the published artifact, so its surface is unknown: {}",
            reader.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| format!("{} printed no JSON: {error}", reader.display()))
}

fn baseline(repository: &Path, scratch: &Path, to_stdout: bool) -> Result<(), String> {
    let (version, marker, artifact) = published()?;
    let filename = artifact["filename"].as_str().unwrap_or_default().to_string();
    if scratch.exists() {
        std::fs::remove_dir_all(scratch).map_err(|error| format!("{}: {error}", scratch.display()))?;
    }
    std::fs::create_dir_all(scratch).map_err(|error| format!("{}: {error}", scratch.display()))?;
    let read = unpack(marker, &artifact, scratch).and_then(|root| surface(repository, &root));
    let cleaned = std::fs::remove_dir_all(scratch);
    let read = read?;
    cleaned.map_err(|error| format!("{} was not removed: {error}", scratch.display()))?;
    let mut document = json!({
        "version": version,
        "source": format!(
            "{marker}:{filename} the artifact PyPI serves for {PROJECT} {version}, unpacked and read with release/surface.py"
        ),
        "surface": read["surface"],
    });
    if let Some(unparseable) = read.get("unparseable") {
        document["unparseable"] = unparseable.clone();
    }
    let rendered = serde_json::to_string_pretty(&document).map_err(|error| error.to_string())? + "\n";
    if to_stdout {
        print!("{rendered}");
        return Ok(());
    }
    let path = repository.join("released-surface.json");
    std::fs::write(&path, rendered).map_err(|error| format!("{}: {error}", path.display()))?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

fn main() -> ExitCode {
    let release = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository = release.parent().unwrap_or(release);
    let scratch = release.join("target").join("baseline-artifact");
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let to_stdout = match arguments.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["baseline"] => false,
        ["baseline", "--stdout"] => true,
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match baseline(repository, &scratch, to_stdout) {
        Ok(()) => ExitCode::SUCCESS,
        Err(refusal) => {
            eprintln!("{refusal}");
            ExitCode::FAILURE
        }
    }
}
