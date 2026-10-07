//! The console-script guard: a tree that offers console scripts is refused,
//! because no surface kind counts them.
//!
//! wisent-gradio ships none: setup.py has no `entry_points` and the published
//! wheel carries no `entry_points.txt`. A console script is the most visible
//! promise a package can make, and if one is added every kind would keep
//! reporting a complete-looking surface that omits it. Both spellings are
//! checked, because a source tree or sdist declares scripts in setup.py while
//! an unpacked wheel declares them in `<dist>-<version>.dist-info/entry_points.txt`.

use std::path::{Path, PathBuf};

use super::syntax::{every_node, falsy_constant, parse, text};

const SETUP: &str = "setup.py";
const ENTRY_POINTS: &str = "entry_points";
const DIST_INFO: &str = "dist-info";
const DECLARED_SCRIPTS: &str = "entry_points.txt";
const CONSOLE_SCRIPTS: &str = "[console_scripts]";

pub fn refuse_undeclared_console_scripts(root: &Path) -> Result<(), String> {
    let setup = root.join(SETUP);
    if setup.is_file() {
        let (source, tree) = parse(&setup)?;
        for argument in every_node(tree.root_node())
            .into_iter()
            .filter(|node| node.kind() == "keyword_argument")
        {
            let named = argument
                .child_by_field_name("name")
                .is_some_and(|name| text(name, &source) == ENTRY_POINTS);
            let Some(value) = argument.child_by_field_name("value").filter(|_| named) else {
                continue;
            };
            if !falsy_constant(value, &source, &setup)? {
                return Err(format!(
                    "{}: declares {ENTRY_POINTS}, so this package now offers console scripts. They are a promise a user \
                     types by name and no kind here counts them, so reporting this surface would silently omit them. \
                     Teach release/src/surface a `script:` kind ({SETUP} in a source tree, \
                     <dist>-<version>.{DIST_INFO}/{DECLARED_SCRIPTS} in a wheel) before trusting another surface from this tree",
                    setup.display()
                ));
            }
        }
    }
    let mut declared: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|error| format!("{}: {error}", root.display()))? {
        let path = entry
            .map_err(|error| format!("{}: {error}", root.display()))?
            .path();
        if path.is_dir()
            && path
                .extension()
                .is_some_and(|extension| extension == DIST_INFO)
        {
            let scripts = path.join(DECLARED_SCRIPTS);
            if scripts.is_file() {
                declared.push(scripts);
            }
        }
    }
    declared.sort();
    for path in declared {
        let content = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if content.contains(CONSOLE_SCRIPTS) {
            return Err(format!(
                "{}: declares {CONSOLE_SCRIPTS}, so this artifact offers console scripts that no kind here counts. \
                 Teach release/src/surface a `script:` kind before trusting another surface from this artifact",
                path.display()
            ));
        }
    }
    Ok(())
}
