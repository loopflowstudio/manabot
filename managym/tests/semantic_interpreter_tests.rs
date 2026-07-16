//! W2-223 interpreter proof: the generic interpreter consumes the checked-in
//! two-deck typed IR by opcode, never by card name. These tests fail closed on
//! a dropped admission or a re-introduced name-based branch, and pin the
//! resolved effect traces for representative happy-path and interaction cases.

use managym::cardsets::alpha::ContentPack;
use managym::semantic::{Condition, IrError, Predicate, Program, SemanticPack, Step, TraceEvent};

/// Test context whose runtime facts are configured per scenario. Defaults make
/// every branch fall through so structural coverage is deterministic.
#[derive(Default)]
struct Fixture {
    kicked: bool,
    nth_resolution: Option<u64>,
    graveyard_true: bool,
    target_matches: bool,
    role_targets: usize,
}

impl managym::semantic::InterpreterContext for Fixture {
    fn kicked(&self) -> bool {
        self.kicked
    }

    fn nth_resolution(&self) -> u64 {
        self.nth_resolution.unwrap_or(1)
    }

    fn graveyard_at_least(&self, _predicate: &Predicate, _count: u64) -> bool {
        self.graveyard_true
    }

    fn target_matches(&self, _role: &str, _predicate: &Predicate) -> bool {
        self.target_matches
    }

    fn role_target_count(&self, _role: &str) -> usize {
        self.role_targets.max(1)
    }
}

fn pack() -> SemanticPack {
    SemanticPack::two_deck().expect("checked-in two-deck IR parses")
}

fn program<'a>(pack: &'a SemanticPack, semantic_key: &str) -> &'a Program {
    pack.programs
        .iter()
        .find(|program| program.semantic_key == semantic_key)
        .unwrap_or_else(|| panic!("no admitted program named {semantic_key}"))
}

// --- Source checks ------------------------------------------------------------

#[test]
fn source_admits_every_deck_definition() {
    let pack = pack();
    let content = ContentPack::default();
    let bound = pack
        .bind(&content)
        .expect("every acceptance-slice definition is admitted");

    // Every definition an admitted deck references must resolve to a real card.
    assert!(!pack.deck_definition_indexes.is_empty());
    for &index in &pack.deck_definition_indexes {
        assert!(
            bound.definition_id(index).is_some(),
            "deck definition index {index} is not admitted"
        );
    }

    // Full-slice closure: all 31 reviewed definitions bind, none duplicated.
    assert_eq!(pack.definitions.len(), 31);
    for definition in &pack.definitions {
        assert!(
            bound.definition_id(definition.semantic_index).is_some(),
            "{} did not bind",
            definition.semantic_key
        );
    }
}

#[test]
fn unadmitted_definition_fails_binding_closed() {
    let doc = r#"{
        "schema_version": 1,
        "pack_key": "test",
        "ir_hash": "0",
        "source_hash": "0",
        "definitions": [
            {
                "semantic_index": 0,
                "semantic_key": "test.ghost",
                "content_pack_binding": {"kind": "legacy_registry_name", "value": "Card That Does Not Exist"}
            }
        ],
        "programs": [],
        "decks": [{"cards": [{"definition_index": 0, "count": 1}]}]
    }"#;
    let pack = SemanticPack::from_json(doc).expect("document parses");
    let content = ContentPack::default();
    match pack.bind(&content) {
        Err(IrError::UnadmittedDefinition { semantic_key, .. }) => {
            assert_eq!(semantic_key, "test.ghost");
        }
        other => panic!("expected an unadmitted-definition error, got {other:?}"),
    }
}

#[test]
fn name_based_instruction_is_rejected() {
    // An instruction that dispatches on card identity must not parse.
    let doc = r#"{
        "schema_version": 1,
        "pack_key": "test",
        "ir_hash": "0",
        "source_hash": "0",
        "definitions": [],
        "programs": [
            {
                "semantic_key": "test.name_dispatch",
                "kind_name": "spell",
                "definition_index": 0,
                "instructions": [
                    {"opcode": 2, "count": 1, "card_name": "Firebending Lesson"}
                ]
            }
        ],
        "decks": []
    }"#;
    match SemanticPack::from_json(doc) {
        Err(IrError::NameBasedDispatch { field }) => assert_eq!(field, "card_name"),
        other => panic!("expected a name-based-dispatch error, got {other:?}"),
    }
}

#[test]
fn checked_in_ir_carries_only_typed_numeric_opcodes() {
    // Parsing the checked-in IR is itself the source check: every instruction
    // is lowered to a typed Step keyed on a numeric opcode, and no name-bearing
    // field survives (parse_step would have failed). We also confirm the branch
    // and for-each arms are typed rather than opaque.
    let pack = pack();
    assert_eq!(pack.schema_version, 1);
    assert_eq!(pack.pack_key, "ur-lessons-vs-gw-allies");
    assert_eq!(pack.programs.len(), 37);

    let mut branch_seen = false;
    let mut for_each_seen = false;
    fn walk(steps: &[Step], branch_seen: &mut bool, for_each_seen: &mut bool) {
        for step in steps {
            match step {
                Step::Branch {
                    then, otherwise, ..
                } => {
                    *branch_seen = true;
                    walk(then, branch_seen, for_each_seen);
                    walk(otherwise, branch_seen, for_each_seen);
                }
                Step::ForEachTarget { body, .. } => {
                    *for_each_seen = true;
                    walk(body, branch_seen, for_each_seen);
                }
                _ => {}
            }
        }
    }
    for program in &pack.programs {
        walk(&program.steps, &mut branch_seen, &mut for_each_seen);
    }
    assert!(branch_seen, "expected at least one typed branch");
    assert!(for_each_seen, "expected at least one typed for-each");
}

// --- Coverage: every admitted program is consumable ---------------------------

#[test]
fn every_admitted_program_runs_by_opcode() {
    let pack = pack();
    let content = ContentPack::default();
    let bound = pack.bind(&content).expect("bind");
    let fixture = Fixture {
        role_targets: 1,
        ..Fixture::default()
    };

    for program in &pack.programs {
        let trace = bound
            .run(program, &fixture)
            .unwrap_or_else(|error| panic!("{} failed to run: {error}", program.semantic_key));
        // A mana/spell/triggered program always resolves to at least one
        // observable effect unless its only content is a branch whose taken arm
        // is empty (e.g. a graveyard-gated draw that does not fire). Those still
        // resolve to an empty trace without error, which is the point: the
        // interpreter consumed them without a card-name fallback.
        let _ = trace;
    }
}

// --- Happy-path traces --------------------------------------------------------

#[test]
fn basic_land_taps_for_its_color() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let trace = bound
        .run(program(&pack, "basic.island.mana"), &Fixture::default())
        .expect("run");
    assert_eq!(
        trace,
        vec![TraceEvent::AddMana {
            mana: "U".to_owned(),
            until: None
        }]
    );
}

#[test]
fn divide_by_zero_bounces_then_learns() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let trace = bound
        .run(
            program(&pack, "stx.divide_by_zero.spell"),
            &Fixture::default(),
        )
        .expect("run");
    assert_eq!(
        trace,
        vec![
            TraceEvent::ReturnToHand("target".to_owned()),
            TraceEvent::Learn
        ]
    );
}

#[test]
fn forecasting_fortune_teller_makes_the_bound_clue_token() {
    let pack = pack();
    let content = ContentPack::default();
    let bound = pack.bind(&content).expect("bind");
    let clue = content
        .definition_id("Clue")
        .expect("Clue token admitted by content pack");
    let trace = bound
        .run(
            program(&pack, "tla.forecasting_fortune_teller.create_clue"),
            &Fixture::default(),
        )
        .expect("run");
    assert_eq!(
        trace,
        vec![TraceEvent::CreateToken {
            definition: clue,
            count: 1,
            tapped_and_attacking: false,
        }]
    );
}

#[test]
fn firebending_lesson_branch_respects_the_kicker() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let prog = program(&pack, "tla.firebending_lesson.spell");

    let unkicked = bound.run(prog, &Fixture::default()).expect("run");
    assert_eq!(
        unkicked,
        vec![TraceEvent::DealDamage {
            target: "victim".to_owned(),
            amount: 2
        }]
    );

    let kicked = bound
        .run(
            prog,
            &Fixture {
                kicked: true,
                ..Fixture::default()
            },
        )
        .expect("run");
    assert_eq!(
        kicked,
        vec![TraceEvent::DealDamage {
            target: "victim".to_owned(),
            amount: 5
        }]
    );
}

#[test]
fn accumulate_wisdom_branch_selects_by_graveyard() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let prog = program(&pack, "tla.accumulate_wisdom.spell");

    let full_yard = bound
        .run(
            prog,
            &Fixture {
                graveyard_true: true,
                ..Fixture::default()
            },
        )
        .expect("run");
    assert_eq!(full_yard, vec![TraceEvent::PutTopCardsInHand(3)]);

    let empty_yard = bound.run(prog, &Fixture::default()).expect("run");
    assert_eq!(
        empty_yard,
        vec![TraceEvent::LookAndSelect {
            look: 3,
            min_select: 1,
            max_select: 1,
            destination: "hand".to_owned(),
        }]
    );
}

#[test]
fn fancy_footwork_expands_once_per_bound_target() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let prog = program(&pack, "tla.fancy_footwork.spell");

    let two_targets = bound
        .run(
            prog,
            &Fixture {
                role_targets: 2,
                ..Fixture::default()
            },
        )
        .expect("run");
    let one_expansion = [
        TraceEvent::Untap("current_target".to_owned()),
        TraceEvent::ModifyPt {
            target: "current_target".to_owned(),
            power: 2,
            toughness: 2,
            duration: "end_of_turn".to_owned(),
        },
    ];
    let expected: Vec<TraceEvent> = one_expansion
        .iter()
        .chain(one_expansion.iter())
        .cloned()
        .collect();
    assert_eq!(two_targets, expected);
}

// --- Interaction trace --------------------------------------------------------

#[test]
fn rallier_waterbend_feeds_badgermole_cub_mana() {
    // The wave's canonical composition test: Water Tribe Rallier's waterbend
    // taps a creature to help pay, and Badgermole Cub's tapped-for-mana trigger
    // adds {G}. Both programs are executed generically by opcode; the combined
    // trace is the interaction proof.
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let fixture = Fixture::default();

    let mut trace = bound
        .run(
            program(&pack, "tla.water_tribe_rallier.waterbend"),
            &fixture,
        )
        .expect("run rallier");
    trace.extend(
        bound
            .run(
                program(&pack, "tla.badgermole_cub.waterbend_bonus"),
                &fixture,
            )
            .expect("run cub"),
    );

    assert_eq!(
        trace,
        vec![
            TraceEvent::LookAndSelect {
                look: 4,
                min_select: 0,
                max_select: 1,
                destination: "hand".to_owned(),
            },
            TraceEvent::AddMana {
                mana: "G".to_owned(),
                until: None,
            },
        ]
    );
}

#[test]
fn south_pole_voyager_draws_only_on_the_second_arrival() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let prog = program(&pack, "tla.south_pole_voyager.ally_arrival");

    let first = bound.run(prog, &Fixture::default()).expect("run");
    assert_eq!(first, vec![TraceEvent::GainLife(1)]);

    let second = bound
        .run(
            prog,
            &Fixture {
                nth_resolution: Some(2),
                ..Fixture::default()
            },
        )
        .expect("run");
    assert_eq!(
        second,
        vec![TraceEvent::GainLife(1), TraceEvent::DrawCards(1)]
    );
}

#[test]
fn yip_yip_grants_flying_only_to_an_ally() {
    let pack = pack();
    let bound = pack.bind(&ContentPack::default()).expect("bind");
    let prog = program(&pack, "tla.yip_yip.spell");

    let non_ally = bound.run(prog, &Fixture::default()).expect("run");
    assert_eq!(
        non_ally,
        vec![TraceEvent::ModifyPt {
            target: "ally".to_owned(),
            power: 2,
            toughness: 2,
            duration: "end_of_turn".to_owned(),
        }]
    );

    let ally = bound
        .run(
            prog,
            &Fixture {
                target_matches: true,
                ..Fixture::default()
            },
        )
        .expect("run");
    assert_eq!(
        ally,
        vec![
            TraceEvent::ModifyPt {
                target: "ally".to_owned(),
                power: 2,
                toughness: 2,
                duration: "end_of_turn".to_owned(),
            },
            TraceEvent::GrantKeywords {
                target: "ally".to_owned(),
                keywords: vec!["flying".to_owned()],
                duration: "end_of_turn".to_owned(),
            },
        ]
    );
}

// Keep the typed condition surface referenced so an accidental removal of a
// public variant is caught by the interpreter's own consumers.
#[allow(dead_code)]
fn _condition_surface_is_public(condition: &Condition) -> bool {
    matches!(
        condition,
        Condition::Kicked
            | Condition::NthResolution(_)
            | Condition::GraveyardAtLeast { .. }
            | Condition::TargetMatches { .. }
    )
}
