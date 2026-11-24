//! Tests for cards filed in `src/dis.rs`, in card order.
use crate::support::game::{clear_summoning_sickness, five_color_setup};
use crate::support::inspect::offered_targets;
use crate::support::play::{attack_with, pass_until, settle};
use managym::agent::action::{ActionSpaceKind, ActionType};
use managym::flow::turn::StepKind;
use managym::state::game_object::Target;
use managym::state::zone::ZoneType;

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

#[test]
fn foundations_5d_macabre_waltz() {
    use crate::support::game::{five_color_setup, move_from_library};
    use crate::support::inspect::{offered_cards, offered_targets, zone_of};
    use crate::support::play::{cast, choose, pass_until, select_card, settle, spend_colorless};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Macabre Waltz", "Gray Ogre", "Hill Giant"]);
    let one = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    let two = move_from_library(&mut s, 0, "Hill Giant", ZoneType::Graveyard);
    let enemy = move_from_library(&mut s, 1, "Gray Ogre", ZoneType::Graveyard);
    let land = move_from_library(&mut s, 0, "Forest", ZoneType::Hand);
    cast(&mut s, "Macabre Waltz");
    assert!(!offered_targets(&s).contains(&Target::GraveyardCard(enemy)));
    for n in 0..=2 {
        let mut branch = s.clone();
        for id in [one, two].into_iter().take(n) {
            assert!(branch.choose_target(Target::GraveyardCard(id)));
        }
        if n < 2 {
            choose(&mut branch, |a| matches!(a, Action::Decline { .. }));
        }
        spend_colorless(&mut branch);
        pass_until(&mut branch, ActionSpaceKind::Discard);
        assert_eq!(branch.action_space().player, Some(PlayerId(0)));
        assert_eq!(offered_cards(&branch).len(), 1 + n);
        select_card(&mut branch, land);
        settle(&mut branch);
        for (i, id) in [one, two].into_iter().enumerate() {
            assert_eq!(
                zone_of(&branch, id),
                Some(if i < n {
                    ZoneType::Hand
                } else {
                    ZoneType::Graveyard
                })
            );
        }
        assert_eq!(zone_of(&branch, land), Some(ZoneType::Graveyard));
    }
}
