//! Tests for cards filed in `src/dtk.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::offered_targets,
    play::{cast, choose, settle, target_opponent},
};
use managym::{
    agent::action::Action,
    state::game_object::{PlayerId, Target},
};

/// One token per creature the target player controls, each a copy under the
/// caster's control; the originals stay with their controller.
#[test]
fn clone_legion_copies_each_creature_target_player_controls() {
    let mut s = five_color_setup(&["Clone Legion", "Gray Ogre", "Agent of Treachery"]);
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Agent of Treachery");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Clone Legion");
    target_opponent(&mut s);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 2);
    let copies = s.battlefield_permanents_named(0, "Agent of Treachery");
    assert_eq!(copies.len(), 1);
    let copy = s.game().state.permanents[copies[0]]
        .as_ref()
        .unwrap()
        .object;
    assert!(!s.game().state.objects[copy].is_card());
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
    assert_eq!(
        s.battlefield_permanents_named(1, "Agent of Treachery")
            .len(),
        1
    );
}

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
