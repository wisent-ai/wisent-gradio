<!-- wisent-banner:start -->
<p align="center">
  <img src="assets/readme-banner.webp" alt="wisent-gradio by Wisent" width="100%">
</p>
<!-- wisent-banner:end -->

<!-- wisent-readme-signals:start -->
[![Source](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/wisent-ai/wisent-gradio) [![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github)](https://github.com/wisent-ai/wisent-gradio/issues) [![Wisent](https://img.shields.io/badge/Wisent-Website-0B0B0B)](https://wisent.com) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/wisent-ai/) [![X](https://img.shields.io/badge/X-Follow-000000?logo=x&logoColor=white)](https://x.com/wisentai) [![Enterprise](https://img.shields.io/badge/Enterprise-Book%20a%20call-0B0B0B?logo=calendly)](https://calendly.com/lbartoszcze)
<!-- wisent-readme-signals:end -->

# wisent-gradio

Monitor and Control Your AI Agent Brain.

You look at what your model says. But what was it actually thinking? Wisent shows
you how to use information from AI activations, intermediate steps within its
layers, to your advantage. Wisent is a full toolkit for representation
engineering, activation steering and mechanistic interpretability. Cut
hallucination rates, decensor your model or stop it from being detected by
AI-generated text detectors. Your Models — Yours to Control. Better than
fine-tuning. Better than analysing the outputs directly.

Deploy the latest research in your stack. This is the Gradio interface for it.

## Install

```
pip install wisent-gradio
```

Pulls `wisent` core and `gradio` transitively. Launch with:

```
python -m wisent.app.launch
```

## Release surface

The version check (`.github/workflows/version-check.yml`) compares the
package's public surface with the one the latest PyPI release had. The surface
is read by `release/` (Rust, tree-sitter), never by importing the package:

- `api:<module>:<name>`: every `__all__` entry;
- `tab:<label>`: every `gr.Tab` with a literal label and every `CommandGroup` label;
- `command:<name>`: every `CommandInfo`, including the ones made through a
  package function that returns one (the `_ci` shorthand in `core/groups.py`).

```
cargo run --locked --manifest-path release/Cargo.toml -- surface .
cargo run --locked --manifest-path release/Cargo.toml -- surface UNPACKED_ARTIFACT --tolerant
cargo run --locked --manifest-path release/Cargo.toml -- baseline [--stdout]
```

`surface ROOT` prints `{"surface": [...]}`. It refuses (exit 1) a ROOT without
`wisent/`, a module that does not parse, a computed `__all__`, a literal with
an escape sequence it cannot decode, a kind with no names at all (its
declarations moved), and a tree that declares console scripts (`entry_points`
in setup.py, `[console_scripts]` in a wheel's dist-info), which no kind counts.
`--tolerant`, used for already-published artifacts, skips modules that do not
parse or cannot be read and names them under `"unparseable"`. `baseline`
rewrites `released-surface.json` from the artifact PyPI serves. An invocation
the command does not take exits 2 with the usage.

The real test runs the built binary against this repository and against
package trees it writes under `release/target/real-tests/`:

```
cargo build --locked --manifest-path release/Cargo.toml
WISENT_GRADIO_RELEASE=release/target/debug/wisent-gradio-release tests/release/surface.sh
```
