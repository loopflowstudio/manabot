//! Comment-aware source-census architectural guards across `managym/src/`.
//!
//! Pins exact counts of sensitive single-authority sites in production Rust
//! code so accidental bypasses fail immediately while prose edits in comments
//! never cause false positives and deleted call sites cannot be masked by
//! comments.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Strip Rust line comments (`// ...`) and block comments (`/* ... */`,
/// including nested block comments) outside string, raw-string, and char
/// literals, replacing comment text with spaces while preserving `\n` so line
/// numbers stay aligned.
fn strip_rust_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;

    while i < len {
        // Line comment: `// ...` until newline.
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            out.push_str("  ");
            i += 2;
            while i < len && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }

        // Block comment: `/* ... */` with Rust nesting.
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            let mut depth = 1usize;
            out.push_str("  ");
            i += 2;
            while i < len && depth > 0 {
                if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                } else if bytes[i] == b'\n' {
                    out.push('\n');
                    i += 1;
                } else {
                    out.push(' ');
                    i += 1;
                }
            }
            continue;
        }

        // Raw string literal: `r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`.
        if let Some(end) = raw_string_end(bytes, i) {
            out.push_str(&source[i..end]);
            i = end;
            continue;
        }

        // Standard or byte string literal: `"..."`.
        if bytes[i] == b'"' {
            let end = quoted_string_end(bytes, i);
            out.push_str(&source[i..end]);
            i = end;
            continue;
        }

        // Character literal (not a lifetime like `'a` or `'static`).
        if bytes[i] == b'\'' {
            if let Some(end) = char_literal_end(bytes, i) {
                out.push_str(&source[i..end]);
                i = end;
                continue;
            }
        }

        let ch = source[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

/// If `bytes[start..]` begins a Rust raw string (`r`, `br`, or `cr` followed by
/// zero or more `#` and `"`), return the byte index immediately after its
/// matching closing delimiter.
fn raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    // Must not be part of an identifier (e.g. `foo_r"..."` is not valid Rust,
    // `err` followed by `"` shouldn't match).
    if start > 0 && is_ident_continue(bytes[start - 1]) {
        return None;
    }
    let mut i = start;
    if bytes[i] == b'b' || bytes[i] == b'c' {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'r' {
        return None;
    }
    i += 1;
    let mut hashes = 0usize;
    while i < bytes.len() && bytes[i] == b'#' {
        hashes += 1;
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1; // consume opening `"`
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut matched = 0usize;
            while matched < hashes
                && i + 1 + matched < bytes.len()
                && bytes[i + 1 + matched] == b'#'
            {
                matched += 1;
            }
            if matched == hashes {
                return Some(i + 1 + hashes);
            }
        }
        i += 1;
    }
    Some(bytes.len())
}

fn quoted_string_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i = (i + 2).min(bytes.len());
        } else if bytes[i] == b'"' {
            return i + 1;
        } else {
            i += 1;
        }
    }
    bytes.len()
}

/// Match `'x'` or `'\...'` char literals without misclassifying lifetimes like
/// `'a`, `'static`, or `'_`.
fn char_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    let len = bytes.len();
    if start + 2 >= len || bytes[start] != b'\'' {
        return None;
    }
    if bytes[start + 1] == b'\\' {
        // Escaped char literal: `'\''`, `'\\'`, `'\n'`, `'\x7f'`, `'\u{1234}'`.
        let mut i = start + 2;
        if i < len {
            i += 1; // consume escaped char
        }
        while i < len && bytes[i] != b'\'' && bytes[i] != b'\n' {
            i += 1;
        }
        if i < len && bytes[i] == b'\'' {
            return Some(i + 1);
        }
        return None;
    }
    // Single-character literal: `'<char>'` (non-newline, non-quote).
    let tail = std::str::from_utf8(&bytes[start + 1..]).ok()?;
    let ch = tail.chars().next()?;
    if ch == '\'' || ch == '\n' {
        return None;
    }
    let close = start + 1 + ch.len_utf8();
    (close < len && bytes[close] == b'\'').then_some(close + 1)
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Remove items annotated with `#[cfg(test)]` from comment-stripped Rust code,
/// replacing them with spaces while preserving `\n` so line numbers stay exact.
fn strip_cfg_test_items(code: &str) -> String {
    let bytes = code.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;

    while i < len {
        if let Some(end) = raw_string_end(bytes, i) {
            out.push_str(&code[i..end]);
            i = end;
            continue;
        }
        if bytes[i] == b'"' {
            let end = quoted_string_end(bytes, i);
            out.push_str(&code[i..end]);
            i = end;
            continue;
        }
        if bytes[i] == b'\'' {
            if let Some(end) = char_literal_end(bytes, i) {
                out.push_str(&code[i..end]);
                i = end;
                continue;
            }
        }
        if code[i..].starts_with("#[cfg(test)]") {
            let end = skip_attributed_item(bytes, i);
            for &b in &bytes[i..end] {
                out.push(if b == b'\n' { '\n' } else { ' ' });
            }
            i = end;
            continue;
        }
        let ch = code[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Given `bytes[start..]` starting at `#[cfg(test)]`, advance past any stacked
/// outer attributes and the annotated item (either `{ ... }` or `;`).
fn skip_attributed_item(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut i = start;
    // Consume one or more outer attributes `#[...]`.
    loop {
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i + 1 < len && bytes[i] == b'#' && bytes[i + 1] == b'[' {
            let mut bracket_depth = 0usize;
            i += 1;
            while i < len {
                if let Some(end) = raw_string_end(bytes, i) {
                    i = end;
                    continue;
                }
                if bytes[i] == b'"' {
                    i = quoted_string_end(bytes, i);
                    continue;
                }
                if bytes[i] == b'[' {
                    bracket_depth += 1;
                } else if bytes[i] == b']' {
                    bracket_depth -= 1;
                    i += 1;
                    if bracket_depth == 0 {
                        break;
                    }
                    continue;
                }
                i += 1;
            }
        } else {
            break;
        }
    }

    // Advance until either the item's terminating `;` or the end of its `{ ... }`.
    let mut brace_depth = 0usize;
    let mut saw_brace = false;
    while i < len {
        if let Some(end) = raw_string_end(bytes, i) {
            i = end;
            continue;
        }
        if bytes[i] == b'"' {
            i = quoted_string_end(bytes, i);
            continue;
        }
        if bytes[i] == b'\'' {
            if let Some(end) = char_literal_end(bytes, i) {
                i = end;
                continue;
            }
        }
        match bytes[i] {
            b'{' => {
                brace_depth += 1;
                saw_brace = true;
            }
            b'}' => {
                brace_depth = brace_depth.saturating_sub(1);
                if saw_brace && brace_depth == 0 {
                    return i + 1;
                }
            }
            b';' if !saw_brace => {
                return i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    len
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn is_test_only_file(rel: &str) -> bool {
    rel == "source_census.rs" || rel.ends_with("_tests.rs")
}

/// Production Rust source files under `managym/src/` with comments and
/// `#[cfg(test)]` items stripped, keyed by `/` relative path (e.g. `flow/zones.rs`).
fn production_sources(scope_prefix: &str) -> Vec<(String, String)> {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_root, &mut files);
    let mut result = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(&src_root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if is_test_only_file(&rel) || (!scope_prefix.is_empty() && !rel.starts_with(scope_prefix)) {
            continue;
        }
        let raw = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let stripped = strip_cfg_test_items(&strip_rust_comments(&raw));
        result.push((rel, stripped));
    }
    result
}

/// Count exact substring matches across production files in `scope_prefix`,
/// returning `rel_path -> Vec<(1-based line_number, trimmed_line)>`.
fn census_substring(scope_prefix: &str, needle: &str) -> BTreeMap<String, Vec<(usize, String)>> {
    let mut hits: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
    for (rel, code) in production_sources(scope_prefix) {
        for (line_idx, line) in code.lines().enumerate() {
            let count = line.matches(needle).count();
            for _ in 0..count {
                hits.entry(rel.clone())
                    .or_default()
                    .push((line_idx + 1, line.trim().to_string()));
            }
        }
    }
    hits
}

/// Count whitespace-collapsed matches across production files (handles
/// multiline calls such as `place_object(\n    obj,\n    player,\n    ZoneType::Battlefield,\n)`).
fn census_collapsed_pattern(
    scope_prefix: &str,
    matcher: impl Fn(&str) -> usize,
) -> BTreeMap<String, usize> {
    let mut hits = BTreeMap::new();
    for (rel, code) in production_sources(scope_prefix) {
        let count = matcher(&code);
        if count > 0 {
            hits.insert(rel, count);
        }
    }
    hits
}

/// Count calls to `call_prefix` (e.g. `"place_object("`) whose balanced
/// argument list contains `arg_needle` (e.g. `"ZoneType::Battlefield"`).
fn count_calls_with_arg(code: &str, call_prefix: &str, arg_needle: &str) -> usize {
    let mut count = 0usize;
    let mut search = 0usize;
    while let Some(offset) = code[search..].find(call_prefix) {
        let open_paren = search + offset + call_prefix.len() - 1;
        let bytes = code.as_bytes();
        let mut depth = 0usize;
        let mut i = open_paren;
        while i < bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        let args = &code[open_paren + 1..i];
                        if args.contains(arg_needle) {
                            count += 1;
                        }
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        search = open_paren + 1;
    }
    count
}

/// Count identifier-bounded occurrences of `.field` (e.g. `.statics` without
/// matching `.static_abilities`).
fn count_field_access(code: &str, field: &str) -> usize {
    let needle = format!(".{field}");
    let bytes = code.as_bytes();
    let mut count = 0usize;
    let mut search = 0usize;
    while let Some(offset) = code[search..].find(&needle) {
        let start = search + offset;
        let end = start + needle.len();
        let bounded = end >= bytes.len() || !is_ident_continue(bytes[end]);
        if bounded {
            count += 1;
        }
        search = start + 1;
    }
    count
}

fn assert_census_counts(
    guard: &str,
    actual: BTreeMap<String, Vec<(usize, String)>>,
    expected: &[(&str, usize)],
) {
    let expected_map: BTreeMap<String, usize> = expected
        .iter()
        .map(|&(path, count)| (path.to_string(), count))
        .collect();
    let actual_counts: BTreeMap<String, usize> = actual
        .iter()
        .map(|(path, entries)| (path.clone(), entries.len()))
        .collect();
    if actual_counts != expected_map {
        let mut detail = String::new();
        for (path, entries) in &actual {
            detail.push_str(&format!("  {path} ({}):\n", entries.len()));
            for (line, text) in entries {
                detail.push_str(&format!("    L{line}: {text}\n"));
            }
        }
        panic!(
            "source census guard `{guard}` failed!\nexpected: {expected_map:?}\nactual:   {actual_counts:?}\nmatches:\n{detail}"
        );
    }
}

fn assert_collapsed_counts(
    guard: &str,
    actual: BTreeMap<String, usize>,
    expected: &[(&str, usize)],
) {
    let expected_map: BTreeMap<String, usize> = expected
        .iter()
        .map(|&(path, count)| (path.to_string(), count))
        .collect();
    assert_eq!(
        actual, expected_map,
        "source census guard `{guard}` failed!"
    );
}

#[test]
fn comment_stripper_handles_leading_block_trailing_line_nested_and_literals() {
    let sample = concat!(
        "/* leading block collect_replacements() */ let a = 1; // trailing collect_replacements()\n",
        "/* outer /* nested collect_replacements() */ still outer */ let b = \"// keep /* keep */\";\n",
        "let raw = r#\"/* raw // keep */\"#;\n",
        "let ch: &'a str = \"ok\"; let slash = '/'; let quote = '\"'; let esc = '\\'';\n",
        "/* multiline\n",
        "   collect_replacements()\n",
        "*/ real_call();\n",
    );
    let stripped = strip_rust_comments(sample);
    assert_eq!(
        stripped.lines().count(),
        sample.lines().count(),
        "line numbers must be preserved"
    );
    assert!(
        !stripped.contains("collect_replacements"),
        "comments must never mask or inflate census matches: {stripped}"
    );
    assert!(stripped.contains("let a = 1;"));
    assert!(stripped.contains("\"// keep /* keep */\""));
    assert!(stripped.contains("r#\"/* raw // keep */\"#"));
    assert!(stripped.contains("let ch: &'a str = \"ok\";"));
    assert!(stripped.contains("let slash = '/';"));
    assert!(stripped.contains("real_call();"));
}

#[test]
fn cfg_test_stripper_removes_test_modules_and_preserves_production_lines() {
    let sample = concat!(
        "fn prod_before() { keep_me(); }\n",
        "#[cfg(test)]\n",
        "mod tests {\n",
        "    fn t() { let s = \"}\"; keep_me(); }\n",
        "}\n",
        "#[cfg(test)]\n",
        "#[allow(dead_code)]\n",
        "fn helper_only_in_tests() { keep_me(); }\n",
        "fn prod_after() { keep_me(); }\n",
    );
    let stripped = strip_cfg_test_items(&strip_rust_comments(sample));
    assert_eq!(stripped.lines().count(), sample.lines().count());
    assert_eq!(stripped.matches("keep_me()").count(), 2);
}

#[test]
fn census_replacement_collection_is_single_authority() {
    // Only `flow/proposed_event.rs` defines and calls `collect_replacements(`.
    assert_census_counts(
        "collect_replacements(",
        census_substring("", "collect_replacements("),
        &[("flow/proposed_event.rs", 2)],
    );
}

#[test]
fn census_functioning_abilities_gates_command_zone_statics_and_replacements() {
    // Only `functioning_abilities.rs` reads `zone_objects(ZoneType::Command, ...)`
    // in production code; every other consumer uses `functioning_command_objects*`
    // or `emblems()`.
    assert_collapsed_counts(
        "zone_objects(ZoneType::Command",
        census_collapsed_pattern("", |code| {
            count_calls_with_arg(code, "zone_objects(", "ZoneType::Command")
        }),
        &[("flow/functioning_abilities.rs", 1)],
    );

    // Deleted legacy unphased `static_sources(` helper must stay absent.
    assert_census_counts(
        "legacy static_sources(",
        census_substring("", "static_sources("),
        &[],
    );

    // Ambient `.replacement_effects` gathering lives solely in `functioning_abilities.rs`;
    // `proposed_event.rs` reads self-replacements on entering/stack objects only.
    assert_collapsed_counts(
        "flow/ .replacement_effects",
        census_collapsed_pattern("flow/", |code| {
            count_field_access(code, "replacement_effects")
        }),
        &[
            ("flow/functioning_abilities.rs", 1),
            ("flow/proposed_event.rs", 7),
        ],
    );

    // Direct `.statics` reads in `flow/` are pinned to the authorized CR 113.6
    // zone-gated sites and the layer/index authorities (`layers.rs`, `statics.rs`,
    // `functioning_abilities.rs`). No other `flow/` file may bypass them.
    assert_collapsed_counts(
        "flow/ .statics",
        census_collapsed_pattern("flow/", |code| count_field_access(code, "statics")),
        &[
            ("flow/functioning_abilities.rs", 2),
            ("flow/layers.rs", 17),
            ("flow/opening.rs", 1),
            ("flow/play.rs", 1),
            ("flow/resolution.rs", 1),
            ("flow/statics.rs", 3),
        ],
    );
}

#[test]
fn census_attack_defender_binding_and_combat_targets_are_single_authority() {
    // Attackers enter `combat.attack_targets` only from declaration
    // (`combat_actions.rs`) and CR 508.4 attacking-token entry (`zones.rs`).
    assert_census_counts(
        "attack_targets.push(",
        census_substring("", "attack_targets.push("),
        &[("flow/combat_actions.rs", 1), ("flow/zones.rs", 1)],
    );

    // Within `agent/`, `structured_offer.rs` is the sole offer authority
    // binding `AttackTarget::Permanent`; `observation.rs` and `decision_memory.rs`
    // only project/record it.
    assert_census_counts(
        "agent/ AttackTarget::Permanent(",
        census_substring("agent/", "AttackTarget::Permanent("),
        &[
            ("agent/decision_memory.rs", 1),
            ("agent/observation.rs", 2),
            ("agent/structured_offer.rs", 3),
        ],
    );
}

#[test]
fn census_battlefield_entry_and_token_emblem_creation_are_single_authority() {
    // Only `flow/zones.rs` (real battlefield entry via `ZoneEntry`) and
    // `flow/scenario.rs` (scenario setup helper) place objects onto the battlefield.
    assert_collapsed_counts(
        "place_object(..., ZoneType::Battlefield)",
        census_collapsed_pattern("", |code| {
            count_calls_with_arg(code, "place_object(", "ZoneType::Battlefield")
        }),
        &[("flow/scenario.rs", 1), ("flow/zones.rs", 1)],
    );

    // Only `flow/zones.rs` (`create_emblem_from`) places objects into the command zone.
    assert_collapsed_counts(
        "place_object(..., ZoneType::Command)",
        census_collapsed_pattern("", |code| {
            count_calls_with_arg(code, "place_object(", "ZoneType::Command")
        }),
        &[("flow/zones.rs", 1)],
    );

    // Single `ZoneEntry` battlefield move authority and single emblem/token creation authorities.
    assert_census_counts(
        "fn move_object_with_entry",
        census_substring("", "fn move_object_with_entry"),
        &[("flow/zones.rs", 1)],
    );
    assert_census_counts(
        "fn create_emblem",
        census_substring("", "fn create_emblem"),
        &[("flow/zones.rs", 1)],
    );
    assert_census_counts(
        "fn create_token",
        census_substring("", "fn create_token"),
        &[("flow/zones.rs", 3)],
    );
    assert_census_counts(
        "fn allocate_tokens",
        census_substring("", "fn allocate_tokens"),
        &[("flow/proposed_event.rs", 1)],
    );
}

#[test]
fn census_sba_exemptions_and_static_source_index_are_single_authority() {
    // Layer 6 projection in `layers.rs` is the sole writer of `exempt_sbas`;
    // `sba.rs` is the sole reader (zero-loyalty and legend-rule SBAs).
    assert_census_counts(
        "exempt_sbas.insert(",
        census_substring("", "exempt_sbas.insert("),
        &[("flow/layers.rs", 1)],
    );
    assert_collapsed_counts(
        "flow/ .exempt_sbas",
        census_collapsed_pattern("flow/", |code| count_field_access(code, "exempt_sbas")),
        &[("flow/layers.rs", 1), ("flow/sba.rs", 2)],
    );

    // `StaticSourceIndex::collect` runs solely at the top of `Context::evaluate`.
    assert_census_counts(
        "StaticSourceIndex::collect(",
        census_substring("", "StaticSourceIndex::collect("),
        &[("flow/layers.rs", 1)],
    );
}
