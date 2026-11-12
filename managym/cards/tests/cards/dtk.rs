//! Tests for cards filed in `src/dtk.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::offered_targets,
    play::{cast, choose, settle},
};
use managym::{
    agent::action::Action,
    state::game_object::{PlayerId, Target},
};

#[test]
fn rending_volley_targets_white_or_blue_creatures_and_cannot_be_countered() {
    let mut s = five_color_setup(&[
        "Rending Volley",
        "Gray Ogre",
        "Agent of Treachery",
        "Counterspell",
    ]);
    let red = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let blue = s.force_permanent_on_battlefield(1, "Agent of Treachery");
    s.force_card_in_hand(1, "Counterspell");
    let volley = cast(&mut s, "Rending Volley");
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(blue)));
    assert!(!targets.contains(&Target::Permanent(red)));
    assert!(s.choose_target(Target::Permanent(blue)));
    s.pass_priority();
    choose(&mut s, |action| {
        matches!(
            action,
            Action::CastSpell {
                player: PlayerId(1),
                ..
            }
        )
    });
    assert!(s.choose_target(Target::StackSpell(volley)));
    settle(&mut s);
    assert!(s.game().state.permanents[blue].is_none());
    assert!(s.game().state.permanents[red].is_some());
}
