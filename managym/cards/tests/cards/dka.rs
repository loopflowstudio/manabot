//! Dark Ascension card behavior.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_card},
    inspect::{card_of, hand},
    play::{attack_with, command, resolve_top},
};
use managym::{agent::action::Action, flow::turn::StepKind, state::zone::ZoneType};

#[test]
fn drogskol_reaver_draws_once_in_each_damage_step() {
    let mut s = five_color_setup(&["Drogskol Reaver"]);
    let reaver = s.force_permanent_on_battlefield(0, "Drogskol Reaver");
    clear_summoning_sickness(&mut s, reaver);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let opening_hand = hand(&s, 0).len();
    attack_with(&mut s, reaver);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    for hit in 1..=2 {
        assert_eq!(s.current_step(), StepKind::CombatDamage);
        assert_eq!(s.life(0), 20 + 3 * hit);
        assert_eq!(s.life(1), 20 - 3 * hit);
        assert_eq!(s.game().state.stack_objects.len(), 1);
        assert_eq!(hand(&s, 0).len(), opening_hand + hit as usize - 1);
        resolve_top(&mut s);
        assert!(s.game().state.stack_objects.is_empty());
        assert_eq!(hand(&s, 0).len(), opening_hand + hit as usize);
        if hit == 1 {
            let mut removed = command(&mut s, |a| matches!(a, Action::PassPriority { .. }));
            let object = card_of(&removed, reaver);
            move_card(&mut removed, object, ZoneType::Graveyard);
            removed.advance_to_active_step(0, StepKind::EndOfCombat);
            assert_eq!((removed.life(0), removed.life(1)), (23, 17));
            assert_eq!(hand(&removed, 0).len(), opening_hand + 1);
        } else {
            s.pass_priority();
        }
        s.pass_priority();
    }
    assert_eq!(s.current_step(), StepKind::EndOfCombat);
}
