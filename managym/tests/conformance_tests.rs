// Card-shell conformance tripwire: every registered real card must match
// its Scryfall snapshot in tests/fixtures/scryfall_cards.json (refresh with
// `uv run scripts/refresh_card_fixture.py`).
//
// What is compared per card: mana cost, colors, type line (supertypes,
// types, subtypes — subtypes verbatim even when mechanically inert),
// printed power/toughness ("*" = characteristic-defining, i.e. `power_cda`),
// engine-supported keyword flags, ward/kicker presence, and the text box
// against oracle text.
//
// Text normalization (deliberate, documented):
// - Reminder text — any "(...)" span — is stripped from BOTH sides, so a
//   registration may include or omit reminder text freely. Exception: when
//   the oracle text is nothing but reminder text (basic lands), only the
//   parentheses are removed, since the registration prints the rules text.
// - Whitespace: lines are trimmed, internal runs of spaces collapsed, empty
//   lines dropped. Newlines between ability paragraphs are preserved.
//
// Sanctioned shell deviations (documented in
// wave/rules/01-two-deck-slice.md) are allowlisted explicitly below.
//
// Cards that are NOT real Magic cards are either registered with
// `is_token: true` (token definitions) or listed in NOT_REAL_CARDS and
// marked `// not a real card` at their registration site; everything else
// without a fixture entry fails this test.

use std::collections::{BTreeSet, HashMap};

use managym::cardsets::alpha::CardRegistry;
use managym::state::card::{CardDefinition, CardType};
use managym::state::mana::{Color, ManaCost};

/// Registered names that are deliberately not real Magic cards (marked
/// `// not a real card` at the registration). Tokens are excluded via
/// `is_token` instead.
const NOT_REAL_CARDS: &[&str] = &["Crossroads of Destiny"];

/// name -> engine mana-cost rendering accepted in place of the Scryfall
/// cost. Suki: hybrid {G/W} pips are registered as {G}{W} (no hybrid-mana
/// support — sanctioned, see wave doc Stage-3 deviations).
fn sanctioned_cost_deviations() -> HashMap<&'static str, &'static str> {
    HashMap::from([("Suki, Kyoshi Warrior", "{2}{G/W}{G/W}")])
}

#[derive(serde::Deserialize)]
struct FixtureCard {
    name: String,
    mana_cost: Option<String>,
    type_line: String,
    power: Option<String>,
    toughness: Option<String>,
    keywords: Vec<String>,
    oracle_text: Option<String>,
    colors: Vec<String>,
}

fn fixture() -> HashMap<String, FixtureCard> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/scryfall_cards.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read Scryfall fixture {path}: {e}"));
    serde_json::from_str(&text).expect("fixture JSON is malformed")
}

fn render_mana_cost(cost: &ManaCost) -> String {
    let mut out = String::new();
    let generic = cost.cost[Color::Generic as usize];
    if generic > 0 {
        out.push_str(&format!("{{{generic}}}"));
    }
    for color in [
        Color::White,
        Color::Blue,
        Color::Black,
        Color::Red,
        Color::Green,
        Color::Colorless,
    ] {
        for _ in 0..cost.cost[color as usize] {
            out.push_str(&format!("{{{}}}", color.symbol()));
        }
    }
    out
}

/// Sort the braced pip tokens (generic-count tokens first) so that pip
/// ordering differences between Scryfall and the engine don't matter.
fn canonical_cost(cost: &str) -> Vec<String> {
    let mut tokens: Vec<String> = cost
        .trim_matches(['{', '}'])
        .split("}{")
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    tokens.sort_by_key(|t| (t.parse::<u32>().is_err(), t.clone()));
    tokens
}

fn render_type_line(def: &CardDefinition) -> String {
    let mut words: Vec<String> = def
        .supertypes
        .iter()
        .map(|s| {
            // Engine stores supertypes lowercase; Scryfall capitalizes.
            let mut chars = s.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect();
    // BTreeSet order is the enum order; every registered card is
    // single-typed today, so ordering nuances don't arise.
    for card_type in &def.types.types {
        words.push(
            match card_type {
                CardType::Creature => "Creature",
                CardType::Instant => "Instant",
                CardType::Sorcery => "Sorcery",
                CardType::Planeswalker => "Planeswalker",
                CardType::Land => "Land",
                CardType::Enchantment => "Enchantment",
                CardType::Artifact => "Artifact",
                CardType::Kindred => "Kindred",
                CardType::Battle => "Battle",
            }
            .to_string(),
        );
    }
    let mut line = words.join(" ");
    if !def.subtypes.is_empty() {
        line.push_str(" \u{2014} ");
        line.push_str(&def.subtypes.join(" "));
    }
    line
}

/// Strip reminder text and normalize whitespace; see module docs.
fn normalize_text(text: &str) -> String {
    let stripped = strip_parentheticals(text);
    let candidate = normalize_whitespace(&stripped);
    if candidate.is_empty() && !normalize_whitespace(text).is_empty() {
        // All-reminder oracle text (basic lands): keep the content, drop
        // only the parentheses.
        return normalize_whitespace(&text.replace(['(', ')'], ""));
    }
    candidate
}

fn strip_parentheticals(text: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

fn normalize_whitespace(text: &str) -> String {
    text.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn scryfall_pt(value: Option<&String>) -> Option<String> {
    value.map(|v| v.to_string())
}

fn engine_pt(printed: Option<i32>, cda: bool) -> Option<String> {
    if cda {
        Some("*".to_string())
    } else {
        printed.map(|v| v.to_string())
    }
}

#[test]
fn every_registered_real_card_matches_scryfall() {
    let registry = CardRegistry::default();
    let fixture = fixture();
    let cost_exceptions = sanctioned_cost_deviations();
    let mut failures: Vec<String> = Vec::new();
    let mut seen: BTreeSet<&str> = BTreeSet::new();

    for def in registry.definitions() {
        let name = def.name.as_str();
        if def.is_token {
            continue;
        }
        if NOT_REAL_CARDS.contains(&name) {
            assert!(
                !fixture.contains_key(name),
                "{name} is marked not-a-real-card but has a fixture entry"
            );
            continue;
        }
        seen.insert(name);
        let Some(fx) = fixture.get(name) else {
            failures.push(format!(
                "{name}: no Scryfall fixture entry. New card registrations must be \
                 snapshotted: run `uv run scripts/refresh_card_fixture.py`. If the card \
                 is invented, mark it `// not a real card` and add it to NOT_REAL_CARDS \
                 in conformance_tests.rs; token definitions must set `is_token: true`."
            ));
            continue;
        };
        assert_eq!(fx.name, name, "fixture keyed by a different name");

        // Mana cost. Compared as a canonicalized pip multiset — Scryfall
        // prints pips in color-wheel order ({G}{W}), the engine renders
        // WUBRG; ordering is cosmetic.
        let engine_cost = def
            .mana_cost
            .as_ref()
            .map(render_mana_cost)
            .unwrap_or_default();
        let scryfall_cost = fx.mana_cost.clone().unwrap_or_default();
        let cost_ok = canonical_cost(&engine_cost) == canonical_cost(&scryfall_cost)
            || cost_exceptions.get(name).is_some_and(|allowed| {
                *allowed == scryfall_cost
            });
        if !cost_ok {
            failures.push(format!(
                "{name}: mana cost {engine_cost:?} != Scryfall {scryfall_cost:?}"
            ));
        }

        // Colors.
        let engine_colors: BTreeSet<String> =
            def.colors().iter().map(|c| c.symbol().to_string()).collect();
        let scryfall_colors: BTreeSet<String> = fx.colors.iter().cloned().collect();
        if engine_colors != scryfall_colors {
            failures.push(format!(
                "{name}: colors {engine_colors:?} != Scryfall {scryfall_colors:?}"
            ));
        }

        // Type line (types, supertypes, subtypes — subtypes verbatim).
        let engine_type_line = render_type_line(def);
        if engine_type_line != fx.type_line {
            failures.push(format!(
                "{name}: type line {engine_type_line:?} != Scryfall {:?}",
                fx.type_line
            ));
        }

        // Printed power/toughness ("*" = CDA).
        let power = engine_pt(def.power, def.power_cda.is_some());
        if power != scryfall_pt(fx.power.as_ref()) {
            failures.push(format!(
                "{name}: power {power:?} != Scryfall {:?}",
                fx.power
            ));
        }
        let toughness = engine_pt(def.toughness, false);
        if toughness != scryfall_pt(fx.toughness.as_ref()) {
            failures.push(format!(
                "{name}: toughness {toughness:?} != Scryfall {:?}",
                fx.toughness
            ));
        }

        // Engine-supported keyword flags.
        let kw = |k: &str| fx.keywords.iter().any(|f| f == k);
        let checks = [
            ("Flying", def.keywords.flying),
            ("Reach", def.keywords.reach),
            ("Haste", def.keywords.haste),
            ("Flash", def.keywords.flash),
            ("Vigilance", def.keywords.vigilance),
            ("Trample", def.keywords.trample),
            ("First strike", def.keywords.first_strike),
            ("Double strike", def.keywords.double_strike),
            ("Deathtouch", def.keywords.deathtouch),
            ("Lifelink", def.keywords.lifelink),
            ("Defender", def.keywords.defender),
            ("Menace", def.keywords.menace),
            ("Hexproof", def.keywords.hexproof),
            ("Ward", def.ward.is_some()),
            ("Kicker", def.kicker.is_some()),
        ];
        for (keyword, has) in checks {
            if kw(keyword) != has {
                failures.push(format!(
                    "{name}: keyword {keyword}: engine={has}, Scryfall={}",
                    kw(keyword)
                ));
            }
        }

        // Text box vs oracle text (normalized; see module docs).
        let engine_text = normalize_text(&def.text_box);
        let oracle_text = normalize_text(fx.oracle_text.as_deref().unwrap_or(""));
        if engine_text != oracle_text {
            failures.push(format!(
                "{name}: text box does not match oracle text\n  engine: {engine_text:?}\n  oracle: {oracle_text:?}"
            ));
        }
    }

    // Fixture entries for cards that are no longer registered are stale.
    for name in fixture.keys() {
        if !seen.contains(name.as_str()) {
            failures.push(format!(
                "{name}: fixture entry has no matching registration (stale? \
                 rerun `uv run scripts/refresh_card_fixture.py`)"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} conformance failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// NOT_REAL_CARDS must track actual registrations, so the list can't rot.
#[test]
fn not_real_cards_are_registered() {
    let registry = CardRegistry::default();
    let names: BTreeSet<String> = registry.definitions().map(|d| d.name.clone()).collect();
    for name in NOT_REAL_CARDS {
        assert!(
            names.contains(*name),
            "{name} is in NOT_REAL_CARDS but not registered"
        );
    }
}
