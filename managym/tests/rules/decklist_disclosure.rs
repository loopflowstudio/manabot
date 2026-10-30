// decklist_disclosure.rs
// Open and closed decklists: what each viewer's observation carries, and
// that hidden-information sampling refuses a closed opponent.

use std::collections::BTreeMap;

use managym::{
    agent::{
        observation::{Observation, PlayerData},
        observation_encoder::{encode, ObservationEncoderConfig, DECKLIST_DIM},
    },
    cards::setups::authored_pack,
    possible_worlds::HiddenHandConstraints,
    state::{game_object::PlayerId, player::DecklistDisclosure, zone::ZoneType},
    Env, Game, PlayerConfig,
};

fn configs(disclosure: DecklistDisclosure) -> Vec<PlayerConfig> {
    let pack = authored_pack("two_deck").unwrap();
    ["ur_lessons", "gw_allies"]
        .iter()
        .map(|key| {
            pack.player_config(key, key)
                .unwrap()
                .with_disclosure(disclosure)
        })
        .collect()
}

fn by_name(game: &Game, counts: &BTreeMap<u32, u32>) -> BTreeMap<String, u32> {
    counts
        .iter()
        .map(|(definition, count)| {
            let card = game
                .state
                .cards
                .iter()
                .find(|card| card.definition_id.0 == *definition)
                .expect("definition has a card");
            (card.name.clone(), *count)
        })
        .collect()
}

fn assert_known(player: &PlayerData, config: &PlayerConfig, unseen: u32) {
    let main = player.main_counts.as_ref().expect("main deck known");
    assert_eq!(
        main.values().sum::<u32>() as usize,
        config.decklist.values().sum::<usize>()
    );
    let sideboard = player.sideboard_counts.as_ref().expect("sideboard known");
    assert_eq!(
        sideboard.values().sum::<u32>() as usize,
        config.sideboard.values().sum::<usize>()
    );
    assert_eq!(player.remaining_sideboard_counts.as_ref(), Some(sideboard));
    let unseen_counts = player.unseen_counts.as_ref().expect("unseen known");
    assert_eq!(unseen_counts.values().sum::<u32>(), unseen);
    for (definition, count) in &player.known_hand {
        assert!(unseen_counts.get(definition).copied().unwrap_or(0) >= *count);
    }
}

#[test]
fn open_decklists_show_both_lists_to_both_viewers() {
    let configs = configs(DecklistDisclosure::Open);
    let game = Game::new(configs.clone(), 7, true);
    for viewer in [PlayerId(0), PlayerId(1)] {
        let opponent = PlayerId(1 - viewer.0);
        let obs = Observation::for_player(&game, viewer);
        let zones = &game.state.zones;
        assert_known(
            &obs.agent,
            &configs[viewer.0],
            zones.size(ZoneType::Library, viewer) as u32,
        );
        assert_known(
            &obs.opponent,
            &configs[opponent.0],
            (zones.size(ZoneType::Library, opponent) + zones.size(ZoneType::Hand, opponent)) as u32,
        );
        // The opponent's unseen counts are exactly the possible-world pool.
        assert_eq!(
            by_name(&game, obs.opponent.unseen_counts.as_ref().unwrap()),
            HiddenHandConstraints::for_viewer(&game, viewer).pool
        );
    }
}

#[test]
fn closed_decklists_hide_the_opponents_lists_but_never_ones_own() {
    let configs = configs(DecklistDisclosure::Closed);
    let game = Game::new(configs.clone(), 7, true);
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(&game, viewer);
        assert_known(
            &obs.agent,
            &configs[viewer.0],
            game.state.zones.size(ZoneType::Library, viewer) as u32,
        );
        assert_eq!(obs.opponent.main_counts, None);
        assert_eq!(obs.opponent.unseen_counts, None);
        assert_eq!(obs.opponent.sideboard_counts, None);
        assert_eq!(obs.opponent.remaining_sideboard_counts, None);
        let json = obs.to_json().to_string();
        assert_eq!(json.matches("\"main_counts\"").count(), 1);
        assert_eq!(json.matches("\"sideboard_counts\"").count(), 1);
    }
}

#[test]
fn decklist_rows_encode_known_lists_and_zero_for_a_closed_opponent() {
    let config = ObservationEncoderConfig::default();
    let width = config.max_cards_per_player * DECKLIST_DIM;
    for disclosure in [DecklistDisclosure::Open, DecklistDisclosure::Closed] {
        let game = Game::new(configs(disclosure), 7, true);
        let obs = Observation::for_player(&game, PlayerId(0));
        let encoded = encode(&obs, &config).unwrap();
        let (agent, opponent) = encoded.decklists.split_at(width);
        let main_total =
            |rows: &[f32]| -> f32 { rows.chunks(DECKLIST_DIM).map(|row| row[1]).sum() };
        assert_eq!(main_total(agent), 41.0);
        assert!(agent
            .chunks(DECKLIST_DIM)
            .take_while(|row| row[0] > 0.0)
            .collect::<Vec<_>>()
            .windows(2)
            .all(|pair| pair[0][0] < pair[1][0]));
        match disclosure {
            DecklistDisclosure::Open => assert_eq!(main_total(opponent), 40.0),
            DecklistDisclosure::Closed => assert!(opponent.iter().all(|value| *value == 0.0)),
        }
    }
    let legacy = ObservationEncoderConfig {
        policy_input_version: 4,
        ..config
    };
    let game = Game::new(configs(DecklistDisclosure::Open), 7, true);
    let obs = Observation::for_player(&game, PlayerId(0));
    assert!(encode(&obs, &legacy).unwrap().decklists.is_empty());
}

#[test]
fn closed_opponent_refuses_hidden_information_sampling() {
    let mut env = Env::new(7, true, false, false);
    env.reset(configs(DecklistDisclosure::Closed)).unwrap();
    for viewer in [0, 1] {
        for error in [
            env.possible_world_space(viewer).err(),
            env.hidden_hand_constraints(viewer).err(),
        ] {
            assert!(error.expect("refused").0.starts_with("decklist closed"));
        }
    }
    assert!(env
        .determinize(0, 3)
        .unwrap_err()
        .0
        .starts_with("decklist closed"));
    assert!(env
        .flat_mc_scores(1, 1, 3, 10)
        .unwrap_err()
        .0
        .starts_with("decklist closed"));

    let mut open = Env::new(7, true, false, false);
    open.reset(configs(DecklistDisclosure::Open)).unwrap();
    open.possible_world_space(0).unwrap();
    open.determinize(0, 3).unwrap();
}

#[test]
fn open_disclosure_is_omitted_from_serialized_setup() {
    let open = serde_json::to_string(&configs(DecklistDisclosure::Open)).unwrap();
    assert!(!open.contains("disclosure"));
    let closed = serde_json::to_string(&configs(DecklistDisclosure::Closed)).unwrap();
    assert!(closed.contains("\"disclosure\":\"closed\""));
}
