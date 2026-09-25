// learn_setup_tests.rs
// Sideboards are admitted setup and owned outside-game copies, not game zones.

use managym::{
    semantic::SemanticPack,
    state::{game_object::PlayerId, zone::ZoneType},
    Game, PlayerConfig,
};
use std::collections::BTreeMap;

fn counts(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, count)| (name.to_string(), *count))
        .collect()
}

fn configs(sideboard: BTreeMap<String, usize>) -> Vec<PlayerConfig> {
    vec![
        PlayerConfig::new("left", counts(&[("Island", 40)])).with_sideboard(sideboard),
        PlayerConfig::new("right", counts(&[("Plains", 40)])),
    ]
}

#[test]
fn learn_authored_setup_keeps_main_decks_and_sideboards_together_in_both_seats() {
    let pack = SemanticPack::two_deck().unwrap();
    assert_eq!(
        pack.decklist("ur_lessons").unwrap().values().sum::<usize>(),
        41
    );
    assert_eq!(
        pack.decklist("gw_allies").unwrap().values().sum::<usize>(),
        40
    );
    assert_eq!(
        pack.player_config("UR", "ur_lessons").unwrap().sideboard,
        counts(&[
            ("Firebending Lesson", 1),
            ("It'll Quench Ya!", 1),
            ("Accumulate Wisdom", 1)
        ])
    );
    assert_eq!(
        pack.player_config("GW", "gw_allies").unwrap().sideboard,
        counts(&[("Yip Yip!", 1), ("Fancy Footwork", 1)])
    );
    for keys in [["ur_lessons", "gw_allies"], ["gw_allies", "ur_lessons"]] {
        let configs: Vec<_> = keys
            .iter()
            .map(|key| pack.player_config(key, key).unwrap())
            .collect();
        let game = Game::new(configs.clone(), 9, false);
        assert_eq!(
            game.state.content.compiled_semantics().unwrap().ir_hash,
            pack.ir_hash
        );
        for (player, config) in configs.iter().enumerate() {
            let roster = &game.state.players[player].sideboard;
            assert_eq!(roster.len(), config.sideboard.values().sum::<usize>());
            for &card in roster {
                assert_eq!(game.state.cards[card].owner, PlayerId(player));
                assert_eq!(game.state.zones.zone_of(card), None);
                assert!(!game.state.players[player].deck.contains(&card));
            }
        }
    }
}

#[test]
fn learn_sideboard_does_not_change_opening_deal_or_library_order() {
    let empty = Game::new(configs(BTreeMap::new()), 9, false);
    let sideboard = Game::new(
        configs(counts(&[("Firebending Lesson", 2), ("Island", 1)])),
        9,
        false,
    );
    for player in [PlayerId(0), PlayerId(1)] {
        for zone in [ZoneType::Hand, ZoneType::Library] {
            assert_eq!(
                empty.state.zones.zone_cards(zone, player),
                sideboard.state.zones.zone_cards(zone, player)
            );
        }
    }
    assert_ne!(
        empty.state.deterministic_hash(),
        sideboard.state.deterministic_hash()
    );
    assert_eq!(
        sideboard.state.players[0].sideboard.len(),
        3,
        "ordinary non-Lessons are valid sideboard cards"
    );
}

#[test]
fn learn_sideboard_unknown_token_and_zero_counts_fail_admission() {
    for entries in [
        vec![("Misspelled Lesson", 1)],
        vec![("Clue", 1)],
        vec![("Firebending Lesson", 0)],
    ] {
        assert!(
            std::panic::catch_unwind(|| Game::new(configs(counts(&entries)), 9, false)).is_err()
        );
    }
}
