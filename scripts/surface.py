"""Print this package's public surface: what a user of the Wisent UI can reach.

This package is a Gradio front end, so its contract is not mainly a set of Python
symbols. Three things are promised to somebody who installs it, and all three are
declared as literal strings in this repository:

  api:<module>:<name>
      The importable entry points, taken from `__all__`. This is a small set but a
      load-bearing one: a HuggingFace Space's `app.py` calls `wisent.app.launch`, so
      dropping an export breaks a deployment, not just a caller's import.

  tab:<label>
      The top-level tabs of the interface. Two are written out in `interface.py`
      (`Wizard`, `Benchmark Debug`) and the rest come from the `label=` of each
      `CommandGroup` in `core/groups.py`, which `interface.py` turns into a tab one
      for one. Sub-tabs declared with a literal label (`Inspect`, `Macro Check`) count
      too — a user clicks them the same way. A tab that disappears is a feature the
      user can no longer find.

  command:<name>
      The CLI commands wired into the UI, one inner tab each, declared as
      `CommandInfo(...)` or through the `_ci(...)` shorthand in `core/groups.py`.
      This is the real substance of the app: every one of these is a wisent command a
      user can drive from the browser, and removing one takes that ability away.

Tabs whose label is computed rather than written (`gr.Tab(label=group.label)`) are
deliberately not guessed at; the `CommandGroup` label they are built from is already
counted, so counting the loop as well would only double the same promise.

Read with `ast`, never by importing. Importing this package pulls in `gradio` and the
whole `wisent` core, and a release decision must not depend on a machine having them.
It also means this runs unchanged against an unpacked published artifact, so the
surface of a version already on PyPI can be recovered exactly rather than assumed.

Usage:
    python3 scripts/surface.py [root]     # root defaults to the repository
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

FIRST = int(False)
LAST = -int(True)
INDENT = int(True) + int(True)

# Gradio containers a user clicks. `Tabs` is the holder, not a tab, and is not here.
TAB_FACTORIES = ("Tab", "TabItem")
# Declares a top-level tab: `interface.py` renders one `gr.Tab` per `CommandGroup`.
GROUP_FACTORIES = ("CommandGroup",)
# Declares one command tab. `_ci` is the positional shorthand used in `core/groups.py`.
COMMAND_FACTORIES = ("CommandInfo", "_ci")

PACKAGE_PARTS = ("wisent",)


def module_name(source: pathlib.Path, root: pathlib.Path) -> str:
    """The dotted module path a caller would import, from the file's location."""
    parts = list(source.relative_to(root).with_suffix("").parts)
    if parts[LAST] == "__init__":
        parts.pop()
    return ".".join(parts)


def called_name(node: ast.Call):
    """The bare callable name of a call, ignoring how it was reached.

    `gr.Tab`, `gradio.Tab` and a bare imported `Tab` all name the same widget, and
    which spelling a module happens to use is not part of the contract.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def literal_str(node):
    """The value of a string literal, or None for anything computed."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def literal_argument(node: ast.Call, keyword: str):
    """A call's string argument, given by keyword or as the first positional."""
    for given in node.keywords:
        if given.arg == keyword:
            return literal_str(given.value)
    if node.args:
        return literal_str(node.args[FIRST])
    return None


def parse(source: pathlib.Path) -> ast.Module:
    """The syntax tree of one module, or a loud failure."""
    try:
        return ast.parse(source.read_text(), filename=str(source))
    except OSError as error:
        raise SystemExit(f"{source}: {error}") from error
    except SyntaxError as error:
        # Refuse rather than skip. A module that does not parse cannot be imported
        # either, so whatever it declares is unreachable at runtime; skipping it would
        # report a smaller surface, and the shared rule would read that as a removed
        # capability. The surface is unknown here, not shrunk.
        raise SystemExit(
            f"{source}: does not parse, so the surface is unknown: {error}"
        ) from error


def exported_names(tree: ast.Module, source: pathlib.Path, root: pathlib.Path) -> list:
    """The `__all__` entries of one module, qualified by its dotted path."""
    prefix = module_name(source, root)
    found = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(target.id == "__all__" for target in targets):
            continue
        if not isinstance(value, (ast.List, ast.Tuple)):
            raise SystemExit(
                f"{source}: __all__ is not a literal list or tuple, so the exports "
                "cannot be read without importing. Refusing rather than reporting "
                "this module as exporting nothing"
            )
        for element in value.elts:
            exported = literal_str(element)
            if exported is None:
                raise SystemExit(
                    f"{source}: __all__ holds a computed entry, so the exports "
                    "cannot be read without importing. Refusing rather than "
                    "reporting a partial list"
                )
            found.append(f"api:{prefix}:{exported}")
    return found


def interface_names(tree: ast.Module) -> list:
    """The tab labels and command names declared anywhere in one module."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = called_name(node)
        if called in TAB_FACTORIES or called in GROUP_FACTORIES:
            label = literal_argument(node, "label")
            if label is not None:
                found.append(f"tab:{label}")
        elif called in COMMAND_FACTORIES:
            name = literal_argument(node, "name")
            if name is not None:
                found.append(f"command:{name}")
    return found


def surface(root: pathlib.Path, tolerant: bool = False) -> tuple:
    """The surface, and the modules that had to be skipped to produce it.

    `tolerant` exists for one job: recovering the surface of an artifact that was
    already published with a module that does not parse. Such a module cannot be
    imported by whoever installed it either, so what it declared was never really on
    offer, and leaving it out is the truthful reading. Skipped modules are always
    reported, never swallowed.
    """
    package = root.joinpath(*PACKAGE_PARTS)
    if not package.is_dir():
        raise SystemExit(
            f"{package} is not a directory; is {root} the repository root?"
        )

    names = set()
    skipped = []
    for source in sorted(package.rglob("*.py")):
        try:
            tree = parse(source)
            found = exported_names(tree, source, root) + interface_names(tree)
        except SystemExit:
            if not tolerant:
                raise
            skipped.append(str(source.relative_to(root)))
            continue
        names.update(found)

    # Each kind is declared in its own shape in its own place. An empty kind means
    # that shape moved, not that the promise was withdrawn, and reporting it as
    # withdrawn would hand the rule a false `breaking`.
    for kind, where in (
        ("api:", "__all__ in the package modules"),
        ("tab:", "literal gr.Tab labels and CommandGroup labels"),
        ("command:", "CommandInfo/_ci declarations in wisent/app/core/groups.py"),
    ):
        if not any(name.startswith(kind) for name in names):
            raise SystemExit(
                f"no {kind} names found under {package}. Either {where} moved, or "
                "they stopped being literals — both change how this package's "
                "promises are declared, so refusing rather than reporting a surface "
                "that is missing a whole kind"
            )
    return sorted(names), skipped


def main(argv: list) -> int:
    tolerant = "--tolerant" in argv
    positional = [arg for arg in argv if not arg.startswith("-")]
    root = (
        pathlib.Path(positional[FIRST])
        if positional
        else pathlib.Path(__file__).resolve().parent.parent
    )
    names, skipped = surface(root, tolerant)
    document = {"surface": names}
    if skipped:
        document["unparseable"] = skipped
    print(json.dumps(document, indent=INDENT))
    return int(False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[int(True) :]))
