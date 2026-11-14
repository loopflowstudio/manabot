use std::collections::BTreeMap;

use managym::{
    cards::content_pack::{default_content_pack, ContentPack, CONTENT_PACK_SCHEMA_VERSION},
    state::{
        card::{PrintedCharacteristics, TokenDefinition},
        game_object::{ObjectKind, PlayerId, PublicId},
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
                    .zone_objects(zone, player)
                    .iter()
                    .map(|object| object.slot())
                    .collect(),
            ));
        }
    }
    out
}

fn trace_point(game: &Game) -> String {
    let card_definitions: Vec<_> = game
        .state
        .objects
        .iter()
        .map(|object| {
            (
                object.id,
                object.definition_id,
                object.registry_key,
                object.owner,
            )
        })
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
        game.state.object_to_permanent,
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
fn stable_card_def_ids_are_deterministic() {
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
    ] {
        let first_id = first.definition_id(name).expect("definition in first pack");
        let second_id = second
            .definition_id(name)
            .expect("definition in second pack");
        assert_eq!(first_id, second_id, "unstable CardDefId for {name}");
        assert_eq!(first.definition(first_id).unwrap().name, name);
    }

    let clue = first.token_id("token.clue").expect("token in first pack");
    assert_eq!(
        second.token_id("token.clue"),
        Some(clue),
        "unstable token CardDefId"
    );
    assert_eq!(first.definition(clue).unwrap().name, "Clue");
    assert_eq!(
        first.definition_id("Clue"),
        None,
        "tokens are not found by name"
    );

    let game = make_game(7);
    for object in game.state.objects.iter() {
        assert_eq!(object.registry_key.0, object.definition_id.0);
        assert_eq!(
            default_content_pack()
                .definition(object.definition_id)
                .unwrap()
                .name,
            object.name
        );
    }
}

/// Tokens resolve by key, never by printed name, so a plain Spirit can sit
/// beside TLA's `token.spirit`.
#[test]
fn tokens_sharing_a_printed_name_stay_distinct_by_key() {
    let mut pack = ContentPack::default();
    let printed = pack.token_id("token.spirit").unwrap();
    pack.register_token(&TokenDefinition {
        key: "token.fixture_spirit",
        scryfall_id: None,
        printed: PrintedCharacteristics {
            name: "Spirit".into(),
            power: Some(1),
            toughness: Some(1),
            ..Default::default()
        },
    });
    let plain = pack.token_id("token.fixture_spirit").unwrap();
    assert_ne!(plain, printed);
    assert_eq!(pack.definition_id("Spirit"), None);
    for (key, row) in [("token.spirit", printed), ("token.fixture_spirit", plain)] {
        let token = pack
            .instantiate_token(key, PlayerId(0), PublicId(1))
            .unwrap();
        assert_eq!(token.definition_id, row);
        assert!(
            token.kind() == ObjectKind::Token && token.name == "Spirit",
            "{key}"
        );
    }
    assert_eq!(pack.definition(printed).unwrap().statics.len(), 2);
    assert!(pack.definition(plain).unwrap().statics.is_empty());
    assert_eq!(pack.definition(plain).unwrap().scryfall_id, None);
}

#[test]
fn matches_and_search_clones_share_one_immutable_content_pack() {
    let game = make_game(11);
    let other_match = make_game(12);
    let admitted = default_content_pack();
    let expected_digest = admitted.content_digest();
    let mut branch = game.clone();
    let sibling = game.clone();
    let first_card = game.state.zones.latest().next().unwrap();

    assert!(std::ptr::eq(admitted, default_content_pack()));
    assert_eq!(admitted.schema_version, CONTENT_PACK_SCHEMA_VERSION);
    assert_eq!(admitted.content_digest(), expected_digest);

    for (((original, branched), sibling_card), other_card) in game
        .state
        .objects
        .iter()
        .zip(branch.state.objects.iter())
        .zip(sibling.state.objects.iter())
        .zip(other_match.state.objects.iter())
    {
        assert_eq!(original.definition_id, branched.definition_id);
        assert_eq!(original.definition_id, sibling_card.definition_id);
        assert!(original.shares_definition_with(branched));
        assert!(original.shares_definition_with(sibling_card));
        assert!(original.shares_definition_with(other_card));
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

    // The legacy scenario mutation seam is copy-on-write and cannot mutate a
    // definition in either the source branch or the process-wide pack.
    branch.state.objects[first_card]
        .subtypes
        .push("scenario-only".to_string());
    assert!(!game.state.objects[first_card].has_subtype("scenario-only"));
    assert!(
        !branch.state.objects[first_card].shares_definition_with(&game.state.objects[first_card])
    );
    assert!(!admitted
        .definition(game.state.objects[first_card].definition_id)
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
