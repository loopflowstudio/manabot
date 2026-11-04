//! Wilds of Eldraine mana filters.
use crate::support::{game::five_color_unfunded_setup, play::command};
use managym::{agent::action::Action, state::mana::Mana};

#[test]
fn scarecrow_guide_filters_without_a_tap_once_each_turn() {
    let mut s = five_color_unfunded_setup(&["Scarecrow Guide"]);
    let guide = s.force_permanent_on_battlefield(0, "Scarecrow Guide");
    s.set_player_mana_pool(0, "CC");
    s.game_mut().scenario_refresh_priority().unwrap();
    command(
        &mut s,
        |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == guide),
    );
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("C"));
    command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 2, .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("BC"));
    assert!(!s.game().state.permanents[guide].as_ref().unwrap().tapped);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == guide)));
}
