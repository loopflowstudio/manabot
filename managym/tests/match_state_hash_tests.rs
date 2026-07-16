use std::{collections::BTreeMap, sync::Arc};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use managym::{
    cardsets::alpha::ContentPack,
    state::{
        card::{CardDefId, CardDefinition},
        game_object::{CardId, ObjectId, PlayerId},
        stack_object::{SpellOnStack, StackObject},
        zone::ZoneType,
    },
    Game, MATCH_STATE_HASH_VERSION,
};

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
    let mut clone = game.clone();

    assert_ne!(game.state.cards.as_ptr(), clone.state.cards.as_ptr());
    assert_eq!(
        game.state.deterministic_hash(),
        clone.state.deterministic_hash()
    );

    clone.state.content = Arc::new(ContentPack::default());
    assert!(!Arc::ptr_eq(&game.state.content, &clone.state.content));
    assert_eq!(
        game.state.deterministic_hash(),
        clone.state.deterministic_hash()
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
        CardDefId((original_id.0 + 1) % definition.state.content.len() as u32);
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

    let mut schema = game.clone();
    let mut schema_pack = (*schema.state.content).clone();
    schema_pack.schema_version += 1;
    schema.state.content = Arc::new(schema_pack);
    assert_ne!(baseline, schema.state.deterministic_hash());

    let mut content = game.clone();
    let mut content_pack = (*content.state.content).clone();
    content_pack.register_card(CardDefinition {
        name: "State Hash Test Definition".to_string(),
        ..Default::default()
    });
    content.state.content = Arc::new(content_pack);
    assert_ne!(baseline, content.state.deterministic_hash());
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
        id: ObjectId(101),
        card: CardId(3),
        controller: PlayerId(0),
        source_definition_id: CardDefId(7),
        targets: Vec::new(),
        target_req_indices: Vec::new(),
        kicked: false,
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
