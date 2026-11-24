//! Dragon's Maze card interactions.
use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, force_new_card_in_hand},
    inspect::{card_of, library_card, tapped, zone_of},
    play::{activate, cast_card, command, pay_rest, play_land, resolve_top, tap_for_mana},
};
use managym::{
    agent::action::Action,
    flow::turn::StepKind,
    state::{
        mana::Mana,
        player::{ManaFloats, ManaPayment},
        zone::ZoneType,
    },
};

#[test]
fn mazes_end_enters_tapped_and_produces_colorless_after_untapping() {
    let mut s = five_color_unfunded_setup(&["Maze's End"]);
    s.game_mut().mana_floats[0] = ManaFloats::All;
    let maze = play_land(&mut s, "Maze's End");
    assert!(tapped(&s, maze));
    s.advance_to_active_step(0, StepKind::Upkeep);
    tap_for_mana(&mut s, maze);
    assert!(tapped(&s, maze));
    assert_eq!(
        s.game().state.players[0].mana_pool.mana(),
        &Mana::parse("C")
    );
}

#[test]
fn mazes_end_counts_distinct_gate_names_after_the_search() {
    let gates = [
        "Azorius Guildgate",
        "Boros Guildgate",
        "Dimir Guildgate",
        "Golgari Guildgate",
        "Gruul Guildgate",
        "Izzet Guildgate",
        "Orzhov Guildgate",
        "Rakdos Guildgate",
        "Selesnya Guildgate",
        "Simic Guildgate",
    ];
    let mut cards = gates.to_vec();
    cards.extend(["Maze's End", "Herald of Eternal Dawn"]);
    for (distinct, find, protected, wins) in [
        (9, false, false, false),
        (9, true, false, true),
        (10, false, false, true),
        (10, false, true, false),
    ] {
        let mut s = five_color_setup(&cards);
        for name in gates[..distinct].iter().chain([&gates[0]]) {
            s.force_permanent_on_battlefield(0, name);
        }
        let herald = protected.then(|| force_new_card_in_hand(&mut s, 1, "Herald of Eternal Dawn"));
        let maze = s.force_permanent_on_battlefield(0, "Maze's End");
        let source = card_of(&s, maze);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        s.set_player_mana_pool(0, "3");
        activate(&mut s, maze, 1);
        if s.game().state.stack_objects.is_empty() {
            pay_rest(&mut s);
        }
        assert_eq!(zone_of(&s, source), Some(ZoneType::Hand));
        if let Some(herald) = herald {
            s.pass_priority();
            cast_card(&mut s, herald);
            if s.game().state.stack_objects.len() == 1 {
                pay_rest(&mut s);
            }
            assert_eq!(s.game().state.stack_objects.len(), 2);
            resolve_top(&mut s);
            assert_eq!(zone_of(&s, herald), Some(ZoneType::Battlefield));
        }
        resolve_top(&mut s);
        if find {
            let selected = library_card(&s, 0, gates[9]);
            command(
                &mut s,
                |action| matches!(action, Action::SelectCard { object, .. } if *object == selected),
            );
            assert_eq!(zone_of(&s, selected), Some(ZoneType::Battlefield));
        } else {
            command(&mut s, |action| matches!(action, Action::Decline { .. }));
        }
        assert_eq!(s.game().is_game_over(), wins);
        assert_eq!(s.game().winner_index(), wins.then_some(0));
    }
}
