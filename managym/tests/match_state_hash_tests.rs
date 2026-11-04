use std::collections::BTreeMap;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use managym::{
    cards::content_pack::{default_content_pack, ContentPack},
    state::{
        card::{CardDefId, CardDefinition, Keyword, Keywords},
        game_object::{CardId, ObjectId, PlayerId},
        permanent::{CounterKind, Counters},
        stack_object::{SpellOnStack, StackObject},
        zone::ZoneType,
    },
    Game, MATCH_STATE_HASH_VERSION,
};

#[test]
fn sparse_counters_preserve_ordered_serialization_across_spill_and_removal() {
    let mut counters = Counters::default();
    // Insert out of order and exceed the two inline entries. Canonical bytes
    // must match the previous dense representation, independently of storage.
    counters.change(CounterKind::Fire, 3);
    counters.change(CounterKind::Stun, 2);
    counters.change(CounterKind::PlusOnePlusOne, 1);
    assert_eq!(
        serde_json::to_string(&counters).unwrap(),
        r#"{"plus_one_plus_one":1,"stun":2,"fire":3}"#
    );
    let retained = counters.clone();
    counters.change(CounterKind::Stun, -9);
    assert_eq!(retained.get(CounterKind::Stun), 2);
    assert_eq!(
        serde_json::to_string(&counters).unwrap(),
        r#"{"plus_one_plus_one":1,"fire":3}"#
    );
    counters.change(CounterKind::Fire, i32::MAX);
    assert_eq!(counters.get(CounterKind::Fire), i32::MAX);
    counters.change(CounterKind::Fire, i32::MIN);
    counters.change(CounterKind::PlusOnePlusOne, -1);
    assert!(counters.is_empty());
    assert_eq!(serde_json::to_string(&counters).unwrap(), "{}");
}

#[test]
fn keyword_serialization_contains_only_present_kinds_in_enum_order() {
    let keywords = Keywords::of(&[Keyword::Haste, Keyword::Flying, Keyword::Haste]);
    assert_eq!(
        serde_json::to_string(&keywords).unwrap(),
        r#"["flying","haste"]"#
    );
    assert_eq!(serde_json::to_string(&Keywords::default()).unwrap(), "[]");
    let existing_keywords = Keywords::of(&[
        Keyword::Indestructible,
        Keyword::Hexproof,
        Keyword::Menace,
        Keyword::Defender,
        Keyword::Lifelink,
        Keyword::Deathtouch,
        Keyword::DoubleStrike,
        Keyword::FirstStrike,
        Keyword::Trample,
        Keyword::Vigilance,
        Keyword::Flash,
        Keyword::Haste,
        Keyword::Reach,
        Keyword::Flying,
    ]);
    assert_eq!(
        serde_json::to_string(&existing_keywords).unwrap(),
        r#"["flying","reach","haste","flash","vigilance","trample","first_strike","double_strike","deathtouch","lifelink","defender","menace","hexproof","indestructible"]"#
    );
}

fn interactive_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Island".to_string(), 12),
        ("Mountain".to_string(), 12),
        ("Gray Ogre".to_string(), 6),
        ("Wind Drake".to_string(), 6),
        ("Man-o'-War".to_string(), 4),
        ("Raging Goblin".to_string(), 4),
        ("Lightning Bolt".to_string(), 6),
        ("Counterspell".to_string(), 4),
        ("Ancestral Recall".to_string(), 3),
        ("Pyroclasm".to_string(), 3),
    ])
}

fn make_game(seed: u64) -> Game {
    Game::new(
        vec![
            managym::PlayerConfig::new("hero", interactive_deck()),
            managym::PlayerConfig::new("villain", interactive_deck()),
        ],
        seed,
        true,
    )
}

#[test]
fn independently_built_seeded_traces_have_the_same_hash() {
    let mut left = make_game(0x5eed);
    let mut right = make_game(0x5eed);

    for step in 0..256_usize {
        assert_eq!(
            left.state.deterministic_hash(),
            right.state.deterministic_hash(),
            "state hash diverged at fixed-action step {step}"
        );
        assert_eq!(left.is_game_over(), right.is_game_over(), "step {step}");
        if left.is_game_over() {
            return;
        }

        let left_count = left.action_space().unwrap().actions.len();
        let right_count = right.action_space().unwrap().actions.len();
        assert_eq!(left_count, right_count, "step {step}");
        let action = (step.wrapping_mul(37).wrapping_add(11)) % left_count;
        assert_eq!(
            left.step(action).expect("left fixed-action step"),
            right.step(action).expect("right fixed-action step"),
            "step {step}"
        );
    }
}

#[test]
fn clone_and_allocation_addresses_do_not_change_the_hash() {
    let game = make_game(11);
    let clone = game.clone();

    assert_ne!(game.state.cards.as_ptr(), clone.state.cards.as_ptr());
    assert_eq!(
        game.state.deterministic_hash(),
        clone.state.deterministic_hash()
    );

    let reallocated = ContentPack::default();
    assert_eq!(
        game.state.deterministic_hash(),
        game.state.deterministic_hash_over(&reallocated)
    );
}

#[test]
fn hashing_does_not_advance_rng() {
    let game = make_game(13);
    let mut expected_rng = game.state.rng.clone();
    let expected_next = expected_rng.next_u64();

    let first = game.state.deterministic_hash();
    let second = game.state.deterministic_hash();
    let mut actual_rng = game.state.rng.clone();

    assert_eq!(first.version(), MATCH_STATE_HASH_VERSION);
    assert_eq!(first, second);
    assert_eq!(actual_rng.next_u64(), expected_next);
}

#[test]
fn meaningful_mutable_facts_change_the_hash() {
    let game = make_game(17);
    let baseline = game.state.deterministic_hash();

    let mut life = game.clone();
    life.state.players[0].life -= 1;
    assert_ne!(baseline, life.state.deterministic_hash());

    let mut library_order = game.clone();
    library_order
        .state
        .zones
        .zone_cards_mut(ZoneType::Library, PlayerId(0))
        .swap(0, 1);
    assert_ne!(baseline, library_order.state.deterministic_hash());

    let mut definition = game.clone();
    let card = CardId(0);
    let original_id = definition.state.cards[card].definition_id;
    definition.state.cards[card].definition_id =
        CardDefId((original_id.0 + 1) % default_content_pack().len() as u32);
    assert_ne!(baseline, definition.state.deterministic_hash());

    let mut rng = game.clone();
    rng.state.rng = ChaCha8Rng::seed_from_u64(0xd1ff_e2e1);
    assert_ne!(baseline, rng.state.deterministic_hash());

    let mut allocation_watermark = game.clone();
    allocation_watermark.state.id_gen.next_id();
    assert_ne!(baseline, allocation_watermark.state.deterministic_hash());
}

#[test]
fn content_schema_and_content_change_the_hash() {
    let game = make_game(19);
    let baseline = game.state.deterministic_hash();

    let mut schema_pack = default_content_pack().clone();
    schema_pack.schema_version += 1;
    assert_ne!(baseline, game.state.deterministic_hash_over(&schema_pack));

    let mut content_pack = default_content_pack().clone();
    content_pack.register_card(CardDefinition {
        name: "State Hash Test Definition".to_string(),
        ..Default::default()
    });
    assert_ne!(baseline, game.state.deterministic_hash_over(&content_pack));
}

#[test]
fn card_names_are_compatibility_data_not_state_identity() {
    let game = make_game(23);
    let mut renamed = game.clone();
    renamed.state.cards[CardId(0)].name = "Compatibility-only rename".to_string();

    assert_eq!(
        game.state.deterministic_hash(),
        renamed.state.deterministic_hash()
    );
}

#[test]
fn stack_identity_is_typed_while_legacy_serialization_is_preserved() {
    let spell = SpellOnStack {
        cast_from: managym::state::zone::ZoneType::Hand,
        flashback: false,
        id: ObjectId(101),
        card: CardId(3),
        controller: PlayerId(0),
        source_definition_id: CardDefId(7),
        targets: Vec::new(),
        target_req_indices: Vec::new(),
        kicked: false,
        x: 0,
        mode: None,
    };
    let stack_object = StackObject::Spell(spell.clone());

    let typed: CardDefId = spell.source_definition_id;
    assert_eq!(typed, CardDefId(7));
    assert_eq!(stack_object.source_definition_id(), CardDefId(7));
    assert_eq!(stack_object.source_card_registry_key(), ObjectId(7));

    let serialized = serde_json::to_value(spell).expect("stack spell serializes");
    assert_eq!(serialized["source_card_registry_key"], 7);
    assert!(serialized.get("source_definition_id").is_none());
}

#[test]
fn canonical_hash_source_has_no_compatibility_or_formatting_dispatch() {
    let source = include_str!("../src/state/hash.rs");
    for forbidden in [".name", ".registry_key", "as_ptr", "{:?}"] {
        assert!(
            !source.contains(forbidden),
            "canonical hash source must not contain {forbidden}"
        );
    }
    assert!(source.contains("definition_id: CardDefId"));
    assert!(source.contains("let GameState {"));
    assert!(!source.contains(".. } = state"));
}

#[test]
fn pending_turn_and_combat_schedules_are_part_of_match_identity() {
    let baseline = make_game(17);
    let hash = baseline.state.deterministic_hash();
    let mut first = baseline.clone();
    first.state.turn.extra_turns = vec![PlayerId(0), PlayerId(1)];
    assert_ne!(first.state.deterministic_hash(), hash);
    let mut reordered = first.clone();
    reordered.state.turn.extra_turns.reverse();
    assert_ne!(
        first.state.deterministic_hash(),
        reordered.state.deterministic_hash()
    );
    let mut resumed = first.clone();
    resumed.state.turn.resume_with = Some(PlayerId(1));
    assert_ne!(
        resumed.state.deterministic_hash(),
        first.state.deterministic_hash()
    );
    let mut combat = baseline.clone();
    combat
        .state
        .turn
        .additional_combats
        .push(managym::flow::turn::AdditionalCombat { attackers: None });
    assert_ne!(combat.state.deterministic_hash(), hash);
    combat.state.turn.additional_combats[0].attackers =
        Some(managym::state::predicate::CardPredicate::creature());
    let restricted = combat.state.deterministic_hash();
    combat.state.turn.combat_attackers = Some(managym::state::predicate::CardPredicate::creature());
    assert_ne!(combat.state.deterministic_hash(), restricted);
}
