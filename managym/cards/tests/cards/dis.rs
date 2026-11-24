//! Tests for cards filed in `src/dis.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::offered_targets,
    play::{attack_with, pass_until, settle},
};
use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{game_object::Target, zone::ZoneType},
};

/// Trygon Predator's combat damage trigger targets an artifact or enchantment
/// the damaged player controls, never its controller's own, and destroys it
/// only if its controller chooses to.
#[test]
fn trygon_predator_may_destroy_an_artifact_the_damaged_player_controls() {
    let mut s = five_color_setup(&["Trygon Predator", "Sol Ring"]);
    let trygon = s.force_permanent_on_battlefield(0, "Trygon Predator");
    let own = s.force_permanent_on_battlefield(0, "Sol Ring");
    let theirs = s.force_permanent_on_battlefield(1, "Sol Ring");
    clear_summoning_sickness(&mut s, trygon);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, trygon);
    s.advance_until(
        |s| !s.game().state.stack_objects.is_empty(),
        "Trygon's trigger".into(),
    );
    s.assert_life(1, 18);
    // The only legal target is chosen without a decision.
    assert!(!offered_targets(&s).contains(&Target::Permanent(own)));
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    let pending = s.game().clone();
    for destroy in [false, true] {
        *s.game_mut() = pending.clone();
        assert!(s.take_action_by_type(if destroy {
            ActionType::PayCost
        } else {
            ActionType::DeclineChoice
        }));
        settle(&mut s);
        assert_eq!(s.game().state.permanents[theirs].is_none(), destroy);
        assert!(s.game().state.permanents[own].is_some());
    }
    assert_eq!(
        s.zone_size(1, ZoneType::Graveyard),
        1,
        "the destroyed Sol Ring"
    );

    // With no artifact or enchantment to target, the trigger leaves the
    // stack (CR 603.3d) and the game moves on.
    let mut s = five_color_setup(&["Trygon Predator", "Sol Ring"]);
    let trygon = s.force_permanent_on_battlefield(0, "Trygon Predator");
    let own = s.force_permanent_on_battlefield(0, "Sol Ring");
    clear_summoning_sickness(&mut s, trygon);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, trygon);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(1, 18);
    assert!(s.game().state.permanents[own].is_some());
}
