use std::{collections::BTreeMap, sync::Arc};

use managym::{
    cardsets::alpha::{default_content_pack, ContentPack, CONTENT_PACK_SCHEMA_VERSION},
    state::{
        game_object::{CardId, PlayerId},
        player::PlayerConfig,
        zone::{ZoneManager, ZoneType},
    },
    Game,
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
            PlayerConfig::new("hero", interactive_deck()),
            PlayerConfig::new("villain", interactive_deck()),
        ],
        seed,
        true,
    )
}

fn zone_trace(zones: &ZoneManager) -> Vec<(ZoneType, PlayerId, Vec<usize>)> {
    let mut out = Vec::new();
    for zone in [
        ZoneType::Library,
        ZoneType::Hand,
        ZoneType::Battlefield,
        ZoneType::Graveyard,
        ZoneType::Stack,
        ZoneType::Exile,
        ZoneType::Command,
    ] {
        for player in [PlayerId(0), PlayerId(1)] {
            out.push((
                zone,
                player,
                zones
                    .zone_cards(zone, player)
                    .iter()
                    .map(|card| card.0)
                    .collect(),
            ));
        }
    }
    out
}

fn trace_point(game: &Game) -> String {
    let card_definitions: Vec<_> = game
        .state
        .cards
        .iter()
        .map(|card| (card.id, card.definition_id, card.registry_key, card.owner))
        .collect();
    format!(
        "turn={:?};priority={:?};players={:?};cards={card_definitions:?};zones={:?};\
         permanents={:?};mapping={:?};stack={:?};combat={:?};events={:?};pending_events={:?};\
         triggers={:?};delayed={:?};exile_links={:?};decision={:?};choice={:?};actions={:?};rng={:?}",
        game.state.turn,
        game.state.priority,
        game.state.players,
        zone_trace(&game.state.zones),
        game.state.permanents,
        game.state.card_to_permanent,
        game.state.stack_objects,
        game.state.combat,
        game.state.events,
        game.state.pending_events,
        game.state.pending_triggers,
        game.state.delayed_triggers,
        game.state.exile_links,
        game.state.suspended_decision,
        game.pending_choice,
        game.current_action_space,
        game.state.rng,
    )
}

#[test]
fn stable_card_def_ids_preserve_legacy_registry_values() {
    let first = ContentPack::default();
    let second = ContentPack::default();
    assert_eq!(first.schema_version, CONTENT_PACK_SCHEMA_VERSION);
    assert_eq!(second.schema_version, CONTENT_PACK_SCHEMA_VERSION);
    assert!(!first.is_empty());
    assert_eq!(first.len(), first.definition_entries().count());
    for (expected, (definition_id, definition)) in first.definition_entries().enumerate() {
        assert_eq!(definition_id.0 as usize, expected);
        assert_eq!(first.definition(definition_id), Some(definition));
    }

    for name in [
        "Plains",
        "Lightning Bolt",
        "Counterspell",
        "Kyoshi Warriors",
        "Clue",
    ] {
        let first_id = first.definition_id(name).expect("definition in first pack");
        let second_id = second
            .definition_id(name)
            .expect("definition in second pack");
        assert_eq!(first_id, second_id, "unstable CardDefId for {name}");
        assert_eq!(first.definition(first_id).unwrap().name, name);
    }
    for (name, legacy_value) in [("Plains", 0), ("Lightning Bolt", 7), ("Counterspell", 9)] {
        assert_eq!(first.definition_id(name).unwrap().0, legacy_value);
    }

    let game = make_game(7);
    for card in game.state.cards.iter() {
        assert_eq!(card.registry_key.0, card.definition_id.0);
        assert_eq!(
            game.state
                .content
                .definition(card.definition_id)
                .unwrap()
                .name,
            card.name
        );
    }
}

#[test]
fn matches_and_search_clones_share_one_immutable_content_pack() {
    let game = make_game(11);
    let other_match = make_game(12);
    let admitted = default_content_pack();
    let expected_digest = admitted.content_digest();
    let mut branch = game.clone();
    let sibling = game.clone();
    let first_card = CardId(0);

    assert!(Arc::ptr_eq(&game.state.content, &admitted));
    assert!(Arc::ptr_eq(&game.state.content, &other_match.state.content));
    assert!(Arc::ptr_eq(&game.state.content, &branch.state.content));
    assert!(Arc::ptr_eq(&game.state.content, &sibling.state.content));
    for content in [
        &game.state.content,
        &other_match.state.content,
        &branch.state.content,
        &sibling.state.content,
    ] {
        assert_eq!(content.schema_version, CONTENT_PACK_SCHEMA_VERSION);
        assert_eq!(content.content_digest(), expected_digest);
    }

    for ((original, branched), sibling_card) in game
        .state
        .cards
        .iter()
        .zip(branch.state.cards.iter())
        .zip(sibling.state.cards.iter())
    {
        assert_eq!(original.definition_id, branched.definition_id);
        assert_eq!(original.definition_id, sibling_card.definition_id);
        assert!(original.shares_definition_with(branched));
        assert!(original.shares_definition_with(sibling_card));
    }

    let original_hash = game.state.deterministic_hash();
    let original_life = game.state.players[0].life;
    branch.state.players[0].life -= 3;
    assert_eq!(game.state.deterministic_hash(), original_hash);
    assert_eq!(sibling.state.deterministic_hash(), original_hash);
    assert_ne!(branch.state.deterministic_hash(), original_hash);
    assert_eq!(game.state.players[0].life, original_life);
    assert_eq!(sibling.state.players[0].life, original_life);
    assert_eq!(branch.state.players[0].life, original_life - 3);
    assert!(Arc::ptr_eq(&game.state.content, &branch.state.content));

    // The legacy scenario mutation seam is copy-on-write and cannot mutate a
    // definition in either the source branch or the process-wide pack.
    branch.state.cards[first_card]
        .subtypes
        .push("scenario-only".to_string());
    assert!(!game.state.cards[first_card].has_subtype("scenario-only"));
    assert!(!branch.state.cards[first_card].shares_definition_with(&game.state.cards[first_card]));
    assert!(!game
        .state
        .content
        .definition(game.state.cards[first_card].definition_id)
        .unwrap()
        .subtypes
        .iter()
        .any(|subtype| subtype == "scenario-only"));
}

#[test]
fn seeded_fixed_action_trace_is_deterministic() {
    let mut left = make_game(0x5eed);
    let mut right = make_game(0x5eed);

    for step in 0..512_usize {
        assert_eq!(trace_point(&left), trace_point(&right), "step {step}");
        assert_eq!(left.is_game_over(), right.is_game_over(), "step {step}");
        if left.is_game_over() {
            return;
        }

        let left_count = left.action_space().unwrap().actions.len();
        let right_count = right.action_space().unwrap().actions.len();
        assert_eq!(left_count, right_count, "step {step}");
        let action = (step.wrapping_mul(37).wrapping_add(11)) % left_count;
        let left_done = left.step(action).expect("left trace step");
        let right_done = right.step(action).expect("right trace step");
        assert_eq!(left_done, right_done, "step {step}");
    }

    panic!("seeded trace did not terminate within 512 fixed-ABI actions");
}
