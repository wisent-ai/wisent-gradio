//! Reading Python source with tree-sitter: parse a module, walk it, and read
//! the literal strings the surface is declared with.

use std::path::Path;

use tree_sitter::{Node, Parser, Tree};

pub fn text<'s>(node: Node, source: &'s str) -> &'s str {
    &source[node.byte_range()]
}

/// The syntax tree of one module, or a refusal naming it: a module that does
/// not parse cannot be imported either, so the surface is unknown, not shrunk.
pub fn parse(path: &Path) -> Result<(String, Tree), String> {
    let refuse = |reason: String| {
        format!(
            "{}: does not parse, so the surface is unknown: {reason}",
            path.display()
        )
    };
    let source =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|error| refuse(error.to_string()))?;
    let tree = parser
        .parse(&source, None)
        .ok_or_else(|| refuse("no syntax tree".to_string()))?;
    if tree.root_node().has_error() {
        return Err(refuse("invalid Python".to_string()));
    }
    Ok((source, tree))
}

pub fn every_node(root: Node) -> Vec<Node> {
    let mut found = Vec::new();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        found.push(node);
        let mut cursor = node.walk();
        let children: Vec<Node> = node.children(&mut cursor).collect();
        pending.extend(children.into_iter().rev());
    }
    found
}

/// A string literal's value; `None` for bytes, f- and t-strings, which are not
/// literal text. A literal holding an escape sequence is refused rather than
/// read as computed, which would silently drop the name it declares.
pub fn string_literal(node: Node, source: &str, path: &Path) -> Result<Option<String>, String> {
    match node.kind() {
        "string" => {}
        "concatenated_string" => {
            let mut value = String::new();
            let mut cursor = node.walk();
            for part in node
                .named_children(&mut cursor)
                .filter(|part| part.kind() != "comment")
            {
                match string_literal(part, source, path)? {
                    Some(piece) => value.push_str(&piece),
                    None => return Ok(None),
                }
            }
            return Ok(Some(value));
        }
        _ => return Ok(None),
    }
    let mut value = String::new();
    let mut raw = false;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string_start" => {
                let prefix = text(child, source)
                    .trim_end_matches(['"', '\''])
                    .to_ascii_lowercase();
                if prefix.contains('b') || prefix.contains('f') || prefix.contains('t') {
                    return Ok(None);
                }
                raw = prefix.contains('r');
            }
            "string_content" => {
                let mut inner = child.walk();
                let escaped = child
                    .children(&mut inner)
                    .any(|part| part.kind() == "escape_sequence");
                if escaped && !raw {
                    return Err(format!(
                        "{}: the literal {} holds an escape sequence this reader does not decode, so the name it declares is unknown",
                        path.display(),
                        text(node, source)
                    ));
                }
                value.push_str(text(child, source));
            }
            "interpolation" => return Ok(None),
            _ => {}
        }
    }
    Ok(Some(value))
}

/// The bare callable name of a call: `gr.Tab`, `gradio.Tab` and an imported
/// `Tab` name the same widget, and which spelling a module uses is not part of
/// the contract.
pub fn called_name<'s>(call: Node, source: &'s str) -> Option<&'s str> {
    let function = call.child_by_field_name("function")?;
    match function.kind() {
        "attribute" => function
            .child_by_field_name("attribute")
            .map(|name| text(name, source)),
        "identifier" => Some(text(function, source)),
        _ => None,
    }
}

/// A call's string argument, given by `keyword` or as the first positional.
pub fn literal_argument(
    call: Node,
    source: &str,
    keyword: &str,
    path: &Path,
) -> Result<Option<String>, String> {
    let Some(arguments) = call.child_by_field_name("arguments") else {
        return Ok(None);
    };
    let mut cursor = arguments.walk();
    let given: Vec<Node> = arguments
        .named_children(&mut cursor)
        .filter(|node| node.kind() != "comment")
        .collect();
    for argument in given
        .iter()
        .filter(|node| node.kind() == "keyword_argument")
    {
        if argument
            .child_by_field_name("name")
            .is_some_and(|name| text(name, source) == keyword)
        {
            return match argument.child_by_field_name("value") {
                Some(value) => string_literal(value, source, path),
                None => Ok(None),
            };
        }
    }
    match given
        .iter()
        .find(|node| !matches!(node.kind(), "keyword_argument" | "dictionary_splat"))
    {
        Some(first) => string_literal(*first, source, path),
        None => Ok(None),
    }
}

/// Whether `node` is a constant Python reads as false: `None`, `False`, an
/// empty string or a zero integer.
pub fn falsy_constant(node: Node, source: &str, path: &Path) -> Result<bool, String> {
    Ok(match node.kind() {
        "none" | "false" => true,
        "integer" => text(node, source)
            .replace('_', "")
            .trim_start_matches("0")
            .is_empty(),
        "string" | "concatenated_string" => {
            string_literal(node, source, path)?.is_some_and(|value| value.is_empty())
        }
        _ => false,
    })
}
