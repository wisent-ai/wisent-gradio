//! `wisent-gradio-release surface [ROOT] [--tolerant]`: the public surface of
//! wisent-gradio, what a user of the Wisent UI can reach.
//!
//! This package is a Gradio front end, so its contract is not mainly a set of
//! Python symbols. Three things are promised to somebody who installs it, and
//! all three are declared as literal strings in the package:
//!
//! - `api:<module>:<name>`: the importable entry points, taken from `__all__`.
//!   A HuggingFace Space's `app.py` calls `wisent.app.launch`, so dropping an
//!   export breaks a deployment, not just a caller's import.
//! - `tab:<label>`: the tabs a user clicks. `interface.py` writes two out
//!   (`Wizard`, `Benchmark Debug`) and renders one per `CommandGroup` label in
//!   `core/groups.py`; sub-tabs with a literal label (`Inspect`, `Macro Check`)
//!   count too.
//! - `command:<name>`: the CLI commands wired into the UI, one inner tab each,
//!   declared as `CommandInfo(...)`.
//!
//! A function of the package that returns such a declaration (`_ci`, the
//! positional shorthand in `core/groups.py`) declares the same kind through
//! its own calls; it is found from the code, not listed here. Tabs whose label
//! is computed (`gr.Tab(label=group.label)`) are not guessed at: the
//! `CommandGroup` label they are built from is already counted.
//!
//! Read with a parser, never by importing: importing pulls in gradio and the
//! whole wisent core, and a release decision must not depend on a machine
//! having them. The same reader runs unchanged against an unpacked published
//! artifact, so the surface of a version already on PyPI is recovered exactly.

mod scripts;
mod syntax;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tree_sitter::{Node, Tree};

use syntax::{called_name, every_node, literal_argument, parse, string_literal, text};

/// Gradio's tab container; a call of it with a literal label is a tab.
const TAB: &str = "Tab";
/// The package's top-level tab declaration: one `gr.Tab` per group.
const GROUP: &str = "CommandGroup";
/// The package's command declaration: one inner tab per command.
const COMMAND: &str = "CommandInfo";
const PACKAGE: &str = "wisent";
const ALL: &str = "__all__";
const INIT: &str = "__init__";

/// What a declaring call promises: the surface kind and the argument naming it.
#[derive(Clone, Copy)]
struct Declares {
    kind: &'static str,
    argument: &'static str,
}

const TABS: Declares = Declares { kind: "tab:", argument: "label" };
const COMMANDS: Declares = Declares { kind: "command:", argument: "name" };

/// Each kind, and where its declarations live when the kind comes back empty.
const KINDS: &[(&str, &str)] = &[
    ("api:", "__all__ in the package modules"),
    ("tab:", "literal gr.Tab labels and CommandGroup labels"),
    ("command:", "CommandInfo declarations in wisent/app/core/groups.py"),
];

/// `path` relative to `root`, which every module read here lies under.
fn relative<'p>(path: &'p Path, root: &Path) -> Result<&'p Path, String> {
    path.strip_prefix(root)
        .map_err(|_| format!("{} is outside {}, the tree whose surface is read", path.display(), root.display()))
}

/// The dotted module path a caller would import, from the file's location.
fn module_name(path: &Path, root: &Path) -> Result<String, String> {
    let mut parts: Vec<String> =
        relative(path, root)?.with_extension("").iter().map(|part| part.to_string_lossy().into_owned()).collect();
    if parts.last().is_some_and(|last| last == INIT) {
        parts.pop();
    }
    Ok(parts.join("."))
}

/// The `__all__` entries of one module's top-level assignments (chained
/// `a = __all__ = [...]` and annotated ones included), qualified by its
/// dotted path.
fn exported_names(module: &Module, root: &Path) -> Result<Vec<String>, String> {
    let (source, path) = (module.source.as_str(), module.path.as_path());
    let prefix = module_name(path, root)?;
    let top = module.tree.root_node();
    let mut found = Vec::new();
    let mut statements = top.walk();
    for statement in top.named_children(&mut statements).filter(|node| node.kind() == "expression_statement") {
        let mut inner = statement.walk();
        for assignment in statement.named_children(&mut inner).filter(|node| node.kind() == "assignment") {
            let mut targets = Vec::new();
            let mut value = Some(assignment);
            while let Some(link) = value.filter(|node| node.kind() == "assignment") {
                targets.extend(link.child_by_field_name("left"));
                value = link.child_by_field_name("right");
            }
            if !targets.iter().any(|target| target.kind() == "identifier" && text(*target, source) == ALL) {
                continue;
            }
            let value = value.filter(|value| matches!(value.kind(), "list" | "tuple" | "expression_list")).ok_or_else(|| {
                format!(
                    "{}: __all__ is not a literal list or tuple, so the exports cannot be read without importing. \
                     Refusing rather than reporting this module as exporting nothing",
                    path.display()
                )
            })?;
            let mut elements = value.walk();
            for element in value.named_children(&mut elements).filter(|element| element.kind() != "comment") {
                let exported = string_literal(element, source, path)?.ok_or_else(|| {
                    format!(
                        "{}: __all__ holds a computed entry, so the exports cannot be read without importing. \
                         Refusing rather than reporting a partial list",
                        path.display()
                    )
                })?;
                found.push(format!("api:{prefix}:{exported}"));
            }
        }
    }
    Ok(found)
}

/// One parsed module of the package.
struct Module {
    path: PathBuf,
    source: String,
    tree: Tree,
}

/// The value a `return` statement returns, when it returns one.
fn returned(statement: Node) -> Option<Node> {
    let mut cursor = statement.walk();
    let value = statement.named_children(&mut cursor).find(|node| node.kind() != "comment");
    value
}

/// The callables that declare a surface kind: Gradio's tab and the package's
/// group and command types, and every function of the package whose return
/// value is a call of one of those, found until no new one appears.
fn declarers(modules: &[Module]) -> BTreeMap<String, Declares> {
    let mut known = BTreeMap::from([(TAB.to_string(), TABS), (GROUP.to_string(), TABS), (COMMAND.to_string(), COMMANDS)]);
    loop {
        let mut learned = Vec::new();
        for module in modules {
            let source = module.source.as_str();
            for function in every_node(module.tree.root_node()).into_iter().filter(|node| node.kind() == "function_definition") {
                let Some(name) = function.child_by_field_name("name").map(|name| text(name, source)) else { continue };
                if known.contains_key(name) {
                    continue;
                }
                let declares = every_node(function)
                    .into_iter()
                    .filter(|node| node.kind() == "return_statement")
                    .filter_map(returned)
                    .filter(|value| value.kind() == "call")
                    .find_map(|call| called_name(call, source).and_then(|called| known.get(called)).copied());
                if let Some(declares) = declares {
                    learned.push((name.to_string(), declares));
                }
            }
        }
        if learned.is_empty() {
            return known;
        }
        known.extend(learned);
    }
}

/// The tab labels and command names declared anywhere in one module.
fn interface_names(module: &Module, declarers: &BTreeMap<String, Declares>) -> Result<Vec<String>, String> {
    let mut found = Vec::new();
    for call in every_node(module.tree.root_node()).into_iter().filter(|node| node.kind() == "call") {
        let Some(declares) = called_name(call, &module.source).and_then(|called| declarers.get(called)) else { continue };
        if let Some(value) = literal_argument(call, &module.source, declares.argument, &module.path)? {
            found.push(format!("{}{value}", declares.kind));
        }
    }
    Ok(found)
}

fn module_surface(module: &Module, root: &Path, declarers: &BTreeMap<String, Declares>) -> Result<Vec<String>, String> {
    let mut found = exported_names(module, root)?;
    found.extend(interface_names(module, declarers)?);
    Ok(found)
}

fn python_files(directory: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| format!("{}: {error}", directory.display()))?.path();
        if path.is_dir() {
            python_files(&path, found)?;
        } else if path.extension().is_some_and(|extension| extension == "py") {
            found.push(path);
        }
    }
    Ok(())
}

/// The surface of the package under `root` as the document the version check
/// reads: `{"surface": [...]}`, with `"unparseable"` naming skipped modules.
///
/// `tolerant` exists for one job: recovering the surface of an artifact that
/// was already published with a module that does not parse. Such a module
/// cannot be imported by whoever installed it either, so what it declared was
/// never on offer, and leaving it out is the truthful reading. Skipped modules
/// are always reported.
pub fn read(root: &Path, tolerant: bool) -> Result<Value, String> {
    let package = root.join(PACKAGE);
    if !package.is_dir() {
        return Err(format!("{} is not a directory; is {} the repository root?", package.display(), root.display()));
    }
    scripts::refuse_undeclared_console_scripts(root)?;
    let mut files = Vec::new();
    python_files(&package, &mut files)?;
    files.sort();
    let mut modules = Vec::new();
    let mut skipped = BTreeSet::new();
    for path in files {
        match parse(&path) {
            Ok((source, tree)) => modules.push(Module { path, source, tree }),
            Err(_) if tolerant => {
                skipped.insert(relative(&path, root)?.display().to_string());
            }
            Err(refusal) => return Err(refusal),
        }
    }
    let declarers = declarers(&modules);
    let mut names = BTreeSet::new();
    for module in &modules {
        match module_surface(module, root, &declarers) {
            Ok(found) => names.extend(found),
            Err(_) if tolerant => {
                skipped.insert(relative(&module.path, root)?.display().to_string());
            }
            Err(refusal) => return Err(refusal),
        }
    }
    // Each kind is declared in its own shape in its own place. An empty kind
    // means that shape moved, not that the promise was withdrawn, and reporting
    // it as withdrawn would hand the rule a false `breaking`.
    for (kind, place) in KINDS {
        if !names.iter().any(|name| name.starts_with(kind)) {
            return Err(format!(
                "no {kind} names found under {}. Either {place} moved, or they stopped being literals; both change how \
                 this package's promises are declared, so refusing rather than reporting a surface missing a whole kind",
                package.display()
            ));
        }
    }
    let mut document = json!({ "surface": names.into_iter().collect::<Vec<_>>() });
    if !skipped.is_empty() {
        document["unparseable"] = json!(skipped.into_iter().collect::<Vec<_>>());
    }
    Ok(document)
}
