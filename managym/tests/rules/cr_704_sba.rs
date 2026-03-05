use managym::{agent::action::ActionType, flow::turn::StepKind, state::zone::ZoneType};

use super::helpers::*;

/// CR 704.5a — A player with 0 or less life loses the game as a state-based action.
#[test]
fn cr_704_5a_player_loses_at_zero_life() {
    let mut s = Scenario::new(mountain_deck(), mountain_deck(), 121);

    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().state.players[0].life = 0;

    s.pass_priority();
    s.pass_priority();

    s.assert_game_over();
    s.assert_winner(1);
}

/// CR 704.5b — A player who attempted to draw from an empty library loses.
#[test]
fn cr_704_5b_empty_library_draw_loses_game() {
    let mut s = Scenario::new(empty_deck(), mountain_deck(), 122);

    for _ in 0..200 {
        if s.game().is_game_over() {
            break;
        }
        s.advance_default_action();
    }

    s.assert_game_over();
    s.assert_winner(1);
}

/// CR 704.5g — Creature with lethal damage is destroyed.
#[test]
fn cr_704_5g_lethal_damage_destroys_creature() {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), 123);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    let elf = s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .into_iter()
        .next()
        .expect("elf should be on battlefield");
    s.game_mut().state.permanents[elf.0]
        .as_mut()
        .expect("permanent should exist")
        .damage = 1;

    s.pass_priority();
    s.pass_priority();

    assert!(s
        .battlefield_permanents_named(0, "Llanowar Elves")
        .is_empty());
    assert!(s.zone_size(0, ZoneType::Graveyard) >= 1);
}
