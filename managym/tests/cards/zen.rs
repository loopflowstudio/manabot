//! Tests for cards filed in `src/cardsets/zen.rs`, grouped by the helper family they
//! drive play with (`tests/support/`).

use crate::support::elemental::{card, counters, permanent, settle, setup};
use crate::support::sacrifice::{activate, choose, enter};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::zone::ZoneType,
};

#[test]
fn landfall_creates_resources_and_counters() {
    let mut s = setup(&[
        "Sazh's Chocobo",
        "Tireless Tracker",
        "Lotus Cobra",
        "Tireless Provisioner",
    ]);
    for name in [
        "Sazh's Chocobo",
        "Tireless Tracker",
        "Lotus Cobra",
        "Tireless Provisioner",
    ] {
        enter(&mut s, name);
    }
    // The Cobra's color and Provisioner's token are separate resolution choices.
    let land = card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::Modal {
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
        } else if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        } else if s.game().state.stack_objects.is_empty()
            && s.action_space().kind == ActionSpaceKind::Priority
        {
            break;
        } else {
            s.pass_priority();
        }
    }
    assert_eq!(counters(&s, permanent(&s, "Sazh's Chocobo")), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 1);
    let clue = permanent(&s, "Clue");
    activate(&mut s, clue);
    settle(&mut s);
    assert_eq!(counters(&s, permanent(&s, "Tireless Tracker")), 1);
}
