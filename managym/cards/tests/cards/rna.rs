//! Tests for cards filed in `src/rna.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::hand,
    play::{attack_with, enter_and_settle, pass_until},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
};

/// Gateway Sneak's two triggers are separate: a Gate entering under its
/// controller's control makes it unblockable this turn, and combat damage to
/// a player draws a card whether or not a Gate entered. A blocked Sneak
/// deals no damage to a player and draws nothing.
#[test]
fn gateway_sneak_slips_past_blockers_after_a_gate_and_draws_on_combat_damage() {
    for (gate, block) in [(false, true), (false, false), (true, true)] {
        let mut s = five_color_setup(&["Gateway Sneak", "Azorius Guildgate", "Hill Giant"]);
        let sneak = s.force_permanent_on_battlefield(0, "Gateway Sneak");
        let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
        clear_summoning_sickness(&mut s, sneak);
        if gate {
            enter_and_settle(&mut s, "Azorius Guildgate");
        }
        let cards = hand(&s, 0).len();
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, sneak);
        let mut offered = false;
        s.advance_until(
            |s| {
                s.action_space().kind == ActionSpaceKind::DeclareBlocker
                    || s.current_step() == StepKind::PostcombatMain
            },
            "blocks or the second main phase".into(),
        );
        if s.action_space().kind == ActionSpaceKind::DeclareBlocker {
            offered = s.action_space().actions.iter().any(|action| {
                matches!(action, Action::DeclareBlocker { blocker, attacker: Some(attacker), .. }
                    if *blocker == giant && *attacker == sneak)
            });
            if offered && block {
                let index = s
                    .action_space()
                    .actions
                    .iter()
                    .position(|action| {
                        matches!(
                            action,
                            Action::DeclareBlocker {
                                attacker: Some(_),
                                ..
                            }
                        )
                    })
                    .unwrap();
                s.step_action(index);
            }
        }
        assert_eq!(offered, !gate, "gate {gate}");
        s.advance_to_active_step(0, StepKind::PostcombatMain);
        let unblocked = gate || !block;
        s.assert_life(1, if unblocked { 19 } else { 20 });
        assert_eq!(
            hand(&s, 0).len(),
            cards + usize::from(unblocked),
            "gate {gate}, block {block}"
        );
    }
}
