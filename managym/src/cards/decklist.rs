//! Decklist files: source-preserving records, parsing, name resolution and import.
//!
//! A [`Decklist`] is one submitted document in the `counted-lines-v1` dialect:
//! its exact source text (BOM and newlines included), filename, parsed entries
//! and line-specific issues. Importing resolves each entry's name against a
//! content pool and admits the resulting [`Deck`] through the same native
//! construction admission game creation uses, so a decklist never carries a
//! second legality policy. A match retains each seat's admitted [`Decklist`]
//! privately beside that seat's deck, as provenance only; a retained source is
//! re-admitted against the installed world when its record loads
//! ([`Decklist::validate_deck`]).
//!
//! Entry points: [`parse_decklist`], [`read_decklist`], [`admit_decklists`]
//! and [`Decklist::validate_deck`].

use std::{collections::BTreeMap, io::Read, path::Path};

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

use super::{
    content_pool,
    deck_admission::{admit_in_pool, DeckIssue, DeckSection},
    pool_content, ContentError, ContentPool, PoolDefinition,
};
use crate::{cards::content_pack::ContentPack, state::deck::Deck};

pub const DIALECT: &str = "counted-lines-v1";
/// Operational bound on one source document, not a construction rule.
pub const MAX_SOURCE_BYTES: usize = 65_536;

/// One counted line. Resolution replaces `name` with the pool's registry name
/// and fills the definition identity; parsing alone leaves them empty.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct DeckEntry {
    pub line: usize,
    pub section: DeckSection,
    pub count: usize,
    pub name: String,
    pub card_def_id: Option<u32>,
    pub key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct Decklist {
    pub dialect: String,
    pub source: String,
    pub filename: String,
    pub entries: Vec<DeckEntry>,
    pub issues: Vec<DeckIssue>,
}

/// One document's admission: its resolved source record and the admitted deck.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(frozen, eq, module = "manabot.managym")
)]
#[serde(deny_unknown_fields)]
pub struct DeckImport {
    pub decklist: Decklist,
    pub deck: Deck,
}

/// Decklists that failed admission, keyed by the caller's label (a seat, or a
/// position in a supply), each carrying its line diagnostics.
#[derive(Clone, Debug)]
pub enum DecklistError {
    Pool(ContentError),
    Rejected(BTreeMap<String, Decklist>),
}

impl std::fmt::Display for DecklistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pool(error) => write!(f, "{error}"),
            Self::Rejected(decklists) => {
                let mut lines = Vec::new();
                for (label, decklist) in decklists {
                    for issue in &decklist.issues {
                        let line = issue.line.map_or("-".to_owned(), |line| line.to_string());
                        lines.push(format!(
                            "{label} {}:{line}: {}",
                            decklist.filename, issue.message
                        ));
                    }
                }
                write!(f, "{}", lines.join("; "))
            }
        }
    }
}

impl From<ContentError> for DecklistError {
    fn from(error: ContentError) -> Self {
        Self::Pool(error)
    }
}

// ---------------------------------------------------------------------------
// Parsing

/// Parse counts/names, Main/Deck/Sideboard headings and full-line `#` comments.
pub fn parse_decklist(source: &str, filename: &str) -> Decklist {
    let mut decklist = Decklist {
        dialect: DIALECT.into(),
        source: source.into(),
        filename: filename.into(),
        entries: Vec::new(),
        issues: Vec::new(),
    };
    if source.len() > MAX_SOURCE_BYTES {
        decklist.issues.push(DeckIssue {
            code: "size".into(),
            ..DeckIssue::format("Decklist exceeds the 64 KiB import limit.")
        });
        return decklist;
    }
    let mut section = DeckSection::Main;
    for (index, raw) in source_lines(source).enumerate() {
        let number = index + 1;
        let line = if number == 1 {
            raw.trim().trim_start_matches('\u{feff}').trim()
        } else {
            raw.trim()
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let heading = line.to_lowercase();
        match heading.strip_suffix(':').unwrap_or(&heading) {
            "main" | "deck" => {
                section = DeckSection::Main;
                continue;
            }
            "sideboard" => {
                section = DeckSection::Sideboard;
                continue;
            }
            _ => {}
        }
        let issue = |code: &str, card_name: Option<&str>, message: &str| DeckIssue {
            line: Some(number),
            section: Some(section),
            card_name: card_name.map(str::to_owned),
            code: code.into(),
            message: message.into(),
        };
        let Some((digits, name)) = counted_line(line) else {
            decklist.issues.push(issue(
                "syntax",
                None,
                "Expected COUNT Card Name, or a Main/Sideboard heading.",
            ));
            continue;
        };
        // Bound the magnitude before conversion so arbitrary numerals stay actionable.
        let magnitude = digits
            .trim_start_matches(['+', '-'])
            .trim_start_matches('0');
        if digits.starts_with('-') || magnitude.is_empty() || magnitude.len() > 9 {
            decklist.issues.push(issue(
                "count",
                Some(name),
                "Quantity must be a positive supported integer.",
            ));
            continue;
        }
        decklist.entries.push(DeckEntry {
            line: number,
            section,
            count: magnitude.parse().expect("nine ASCII digits"),
            name: name.into(),
            card_def_id: None,
            key: None,
        });
    }
    decklist
}

/// Lines split at `\r\n`, `\n` or `\r`, so CRLF and classic-Mac files number alike.
fn source_lines(source: &str) -> impl Iterator<Item = &str> {
    let mut rest = Some(source);
    std::iter::from_fn(move || {
        let text = rest?;
        match text.find(['\r', '\n']) {
            Some(end) => {
                let skip = if text[end..].starts_with("\r\n") {
                    2
                } else {
                    1
                };
                rest = Some(&text[end + skip..]).filter(|tail| !tail.is_empty());
                Some(&text[..end])
            }
            None => {
                rest = None;
                Some(text)
            }
        }
    })
}

/// `[+-]?[0-9]+`, whitespace, then a nonempty name. Lines are already trimmed.
fn counted_line(line: &str) -> Option<(&str, &str)> {
    let unsigned = line.strip_prefix(['+', '-']).unwrap_or(line);
    let digits = unsigned.len()
        - unsigned
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .len();
    let numeral_end = line.len() - unsigned.len() + digits;
    let rest = &line[numeral_end..];
    let name = rest.trim_start();
    (digits > 0 && name.len() < rest.len() && !name.is_empty())
        .then(|| (&line[..numeral_end], name))
}

/// Bounded UTF-8 read that keeps BOM and newline bytes.
pub fn read_decklist(path: &Path, filename: Option<&str>) -> Result<Decklist, String> {
    let display = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut data = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| {
            file.take(MAX_SOURCE_BYTES as u64 + 1)
                .read_to_end(&mut data)
        })
        .map_err(|error| format!("{display}: {error}"))?;
    if data.len() > MAX_SOURCE_BYTES {
        return Err(format!(
            "{display}: decklist exceeds the 64 KiB import limit"
        ));
    }
    let source =
        String::from_utf8(data).map_err(|_| format!("{display}: decklist is not UTF-8"))?;
    Ok(parse_decklist(&source, filename.unwrap_or(&display)))
}

// ---------------------------------------------------------------------------
// Resolution and admission

fn name_key(name: &str) -> String {
    name.nfc()
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Resolve front or full double-faced names; reverse-only, ambiguous and
/// unknown names are rejected without substitution. Then admit natively,
/// attributing each admission issue to every source line that named the card.
fn admit_with_pool(
    decklist: &Decklist,
    pool: &ContentPool,
    content: &ContentPack,
) -> Result<DeckImport, Decklist> {
    let mut names: BTreeMap<String, Vec<&PoolDefinition>> = BTreeMap::new();
    let mut reverse_faces: BTreeMap<String, &str> = BTreeMap::new();
    for definition in &pool.definitions {
        names
            .entry(name_key(&definition.registry_name))
            .or_default()
            .push(definition);
        if let Some((front, reverse)) = definition.registry_name.split_once(" // ") {
            names.entry(name_key(front)).or_default().push(definition);
            reverse_faces.insert(name_key(reverse), &definition.registry_name);
        }
    }
    let mut issues = decklist.issues.clone();
    let mut entries = Vec::new();
    let mut source_lines: BTreeMap<(DeckSection, String), Vec<usize>> = BTreeMap::new();
    let mut deck = Deck::default();
    for entry in &decklist.entries {
        let key = name_key(&entry.name);
        let candidates = names.get(&key).map(Vec::as_slice).unwrap_or_default();
        let issue = |code: &str, message: String| DeckIssue {
            line: Some(entry.line),
            section: Some(entry.section),
            card_name: Some(entry.name.clone()),
            code: code.into(),
            message,
        };
        let definition = match candidates {
            [definition] => *definition,
            [] if reverse_faces.contains_key(&key) => {
                issues.push(issue(
                    "face",
                    format!(
                        "Reverse faces cannot be deck entries; use {}.",
                        reverse_faces[&key]
                    ),
                ));
                continue;
            }
            [] => {
                issues.push(issue(
                    "unknown",
                    format!("Unknown card name: {}.", entry.name),
                ));
                continue;
            }
            _ => {
                issues.push(issue(
                    "ambiguous",
                    format!("Ambiguous card name: {}.", entry.name),
                ));
                continue;
            }
        };
        let name = definition.registry_name.clone();
        source_lines
            .entry((entry.section, name.clone()))
            .or_default()
            .push(entry.line);
        let cards = match entry.section {
            DeckSection::Main => &mut deck.main,
            DeckSection::Sideboard => &mut deck.sideboard,
        };
        *cards.entry(name.clone()).or_default() += entry.count;
        entries.push(DeckEntry {
            name,
            card_def_id: Some(definition.card_def_id.0),
            key: definition.key.clone(),
            ..entry.clone()
        });
    }
    for issue in admit_in_pool(pool, content, &deck) {
        let lines = match (issue.section, &issue.card_name) {
            (Some(section), Some(name)) => source_lines
                .get(&(section, name.clone()))
                .cloned()
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        if lines.is_empty() {
            issues.push(issue);
        } else {
            issues.extend(lines.into_iter().map(|line| DeckIssue {
                line: Some(line),
                ..issue.clone()
            }));
        }
    }
    let decklist = Decklist {
        entries,
        issues,
        ..decklist.clone()
    };
    if decklist.issues.is_empty() {
        Ok(DeckImport { decklist, deck })
    } else {
        Err(decklist)
    }
}

/// Admit labelled decklists against one pool, rejecting the whole input when
/// any document has an issue; nothing is silently filtered.
pub fn admit_decklists(
    decklists: &[(String, Decklist)],
    pack_key: &str,
) -> Result<Vec<DeckImport>, DecklistError> {
    let pool = content_pool(pack_key)?;
    let content = pool_content(pack_key)?;
    let mut admitted = Vec::new();
    let mut rejected = BTreeMap::new();
    for (label, decklist) in decklists {
        match admit_with_pool(decklist, &pool, &content) {
            Ok(imported) => admitted.push(imported),
            Err(decklist) => {
                rejected.insert(label.clone(), decklist);
            }
        }
    }
    if rejected.is_empty() {
        Ok(admitted)
    } else {
        Err(DecklistError::Rejected(rejected))
    }
}

impl Decklist {
    /// Re-import this retained source against the installed world; it must
    /// admit to exactly `deck`, so a tampered source or deck, or a world that
    /// no longer admits the list, rejects the record that retained it.
    pub fn validate_deck(&self, pack_key: &str, deck: &Deck) -> Result<(), String> {
        let source = parse_decklist(&self.source, &self.filename);
        match admit_decklists(&[(self.filename.clone(), source)], pack_key) {
            Ok(imports) if imports[0].deck == *deck => Ok(()),
            Ok(_) => Err(format!(
                "{}: recorded deck differs from its retained source",
                self.filename
            )),
            Err(error) => Err(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PACK: &str = "world";
    const HERO: &str = "# Novel development deck\nMain\n10 Forest\n10 Mountain\n2 Avatar Enthusiasts\nSideboard\n1 Island\n";
    const VILLAIN: &str = "20 Island\n";

    fn admit(decklist: &Decklist, pool: &ContentPool) -> Result<DeckImport, Decklist> {
        admit_with_pool(decklist, pool, &pool_content(PACK).unwrap())
    }

    fn import_match(hero: &str, villain: &str) -> Result<Vec<DeckImport>, DecklistError> {
        admit_decklists(
            &[
                ("hero".into(), parse_decklist(hero, "hero.txt")),
                ("villain".into(), parse_decklist(villain, "villain.txt")),
            ],
            PACK,
        )
    }

    fn lines(decklist: &Decklist) -> Vec<(Option<usize>, &str)> {
        decklist
            .issues
            .iter()
            .map(|issue| (issue.line, issue.code.as_str()))
            .collect()
    }

    #[test]
    fn dialect_keeps_original_lines_and_reports_bad_counts() {
        let source = "\u{feff}Deck\r\n# comment\r\n2 Forest\r\nSideboard:\r\n0 Island\r\n-1 Mountain\r\n2x Forest\r\n99999999999999999999 Island\r\n+3 Plains";
        let decklist = parse_decklist(source, "input.txt");
        assert_eq!(decklist.source, source);
        assert_eq!(decklist.filename, "input.txt");
        let entries: Vec<_> = decklist
            .entries
            .iter()
            .map(|e| (e.line, e.section, e.name.as_str(), e.count))
            .collect();
        assert_eq!(
            entries,
            [
                (3, DeckSection::Main, "Forest", 2),
                (9, DeckSection::Sideboard, "Plains", 3)
            ]
        );
        assert_eq!(
            lines(&decklist),
            [
                (Some(5), "count"),
                (Some(6), "count"),
                (Some(7), "syntax"),
                (Some(8), "count")
            ]
        );
        let oversized = parse_decklist(&"1 Forest\n".repeat(8000), "big.txt");
        assert_eq!(lines(&oversized), [(None, "size")]);
    }

    #[test]
    fn names_resolve_by_normalized_key_and_sum_counts() {
        let pool = content_pool(PACK).unwrap();
        let reformatted = parse_decklist(
            "10 mountain\n5 Forest\n5  FOREST\n2 Avatar Enthusiasts\nSideboard\n1 Island",
            "x",
        );
        let imported = admit(&reformatted, &pool).unwrap();
        let original = admit(&parse_decklist(HERO, "hero.txt"), &pool).unwrap();
        assert_eq!(imported.deck, original.deck);
        assert_eq!(
            imported.deck.main,
            BTreeMap::from([
                ("Avatar Enthusiasts".into(), 2),
                ("Forest".into(), 10),
                ("Mountain".into(), 10)
            ])
        );
        assert!(imported
            .decklist
            .entries
            .iter()
            .all(|e| e.card_def_id.is_some()));
    }

    #[test]
    fn unknown_unsupported_and_ambiguous_entries_name_their_lines() {
        let mut pool = content_pool(PACK).unwrap();
        let source = "10 Forest\n1 Definitely Not A Card\nSideboard\n1 Accumulate Wisdom\n2 Accumulate Wisdom\nMain\n1 Accumulate Wisdom\n";
        let rejected = admit(&parse_decklist(source, "x"), &pool).unwrap_err();
        assert_eq!(rejected.source, source);
        for expected in [
            (Some(2), "unknown"),
            (Some(4), "unsupported"),
            (Some(5), "unsupported"),
            (Some(7), "unsupported"),
        ] {
            assert!(lines(&rejected).contains(&expected), "{expected:?}");
        }
        let forest = pool
            .definitions
            .iter()
            .find(|row| row.registry_name == "Forest")
            .unwrap()
            .clone();
        pool.definitions.push(PoolDefinition {
            registry_name: "FOREST".into(),
            ..forest
        });
        let ambiguous = admit(&parse_decklist("10 Forest", "x"), &pool).unwrap_err();
        assert!(lines(&ambiguous).contains(&(Some(1), "ambiguous")));
    }

    /// The pool with Forest renamed to a two-face registry name.
    fn double_faced_pool() -> ContentPool {
        let mut pool = content_pool(PACK).unwrap();
        for row in &mut pool.definitions {
            if row.registry_name == "Forest" {
                row.registry_name = "Front Face // Back Face".into();
            }
        }
        pool
    }

    #[test]
    fn double_faced_entries_resolve_by_front_or_full_name_only() {
        let pool = double_faced_pool();
        for name in ["Front Face", "Front Face // Back Face"] {
            let result = admit(&parse_decklist(&format!("10 Island\n1 {name}"), "x"), &pool);
            let decklist = match result {
                Ok(imported) => imported.decklist,
                Err(decklist) => decklist,
            };
            let names: Vec<_> = decklist.entries.iter().map(|e| e.name.as_str()).collect();
            assert_eq!(names, ["Island", "Front Face // Back Face"]);
            assert!(!decklist
                .issues
                .iter()
                .any(|issue| issue.code == "ambiguous" || issue.code == "face"));
        }
        let reverse = admit(&parse_decklist("10 Island\n1 Back Face", "x"), &pool).unwrap_err();
        assert!(lines(&reverse).contains(&(Some(2), "face")));
    }

    #[test]
    fn construction_errors_reject_the_whole_match() {
        for source in [
            "Sideboard\n1 Island",
            "1 Forest\nSideboard\n0 Island",
            "1001 Forest",
        ] {
            let error = import_match(source, VILLAIN).unwrap_err();
            let DecklistError::Rejected(rejected) = &error else {
                panic!("{error}");
            };
            assert_eq!(rejected.keys().collect::<Vec<_>>(), ["hero"]);
            assert!(error.to_string().starts_with("hero hero.txt:"));
        }
    }

    #[test]
    fn retained_sources_reject_tampering_and_ignore_presentation() {
        let hero = import_match(&HERO.replace('\n', "\r\n"), VILLAIN).unwrap()[0].clone();
        hero.decklist.validate_deck(PACK, &hero.deck).unwrap();
        let restored: DeckImport =
            serde_json::from_str(&serde_json::to_string(&hero).unwrap()).unwrap();
        assert_eq!(restored, hero);
        let reformatted = import_match(
            "10 mountain\n5 Forest\n5 forest\n2 Avatar Enthusiasts\nSideboard\n1 Island",
            VILLAIN,
        )
        .unwrap()[0]
            .clone();
        assert_eq!(reformatted.deck, hero.deck);
        reformatted
            .decklist
            .validate_deck(PACK, &hero.deck)
            .unwrap();

        let mut tampered = hero.decklist.clone();
        tampered.source = "10 Island".into();
        let error = tampered.validate_deck(PACK, &hero.deck).unwrap_err();
        assert!(error.contains("retained source"), "{error}");
        let mut unadmitted = hero.decklist.clone();
        unadmitted.source = "1 Definitely Not A Card".into();
        let error = unadmitted.validate_deck(PACK, &hero.deck).unwrap_err();
        assert!(error.contains("Unknown card name"), "{error}");
    }
}
