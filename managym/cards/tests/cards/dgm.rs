//! Dragon's Maze card interactions.
use crate::support::{
    game::five_color_setup,
    inspect::{card_of, zone_of},
    play::{activate, command, pay_rest},
};
use managym::{
    agent::action::Action,
    state::{player::ManaPayment, zone::ZoneType},
};

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
        if protected {
            s.force_permanent_on_battlefield(1, "Herald of Eternal Dawn");
        }
        let maze = s.force_permanent_on_battlefield(0, "Maze's End");
        let source = card_of(&s, maze);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        s.set_player_mana_pool(0, "3");
        activate(&mut s, maze, 1);
        if s.game().state.stack_objects.is_empty() {
            pay_rest(&mut s);
        }
        assert_eq!(zone_of(&s, source), Some(ZoneType::Hand));
        s.pass_priority();
        s.pass_priority();
        if find {
            let selected = s
                .action_space()
                .actions
                .iter()
                .find_map(|action| match action {
                    Action::SelectCard { object, .. }
                        if s.game().state[*object].name == gates[9] =>
                    {
                        Some(*object)
                    }
                    _ => None,
                })
                .unwrap();
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
