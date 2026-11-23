//! Dark Ascension through ordinary combat and spell resolution.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{attack_with, cast, pass_until, settle},
};
use managym::{agent::action::ActionSpaceKind, flow::turn::StepKind};

#[test]
fn predator_ooze_grows_on_attack_and_when_its_surviving_blocker_dies() {
    let mut s = five_color_setup(&["Predator Ooze", "Hill Giant", "Fumigate"]);
    let ooze = s.force_permanent_on_battlefield(0, "Predator Ooze");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    clear_summoning_sickness(&mut s, ooze);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, ooze);
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    assert_eq!(s.game().characteristics(ooze).pt(), (2, 2));
    s.declare_block();
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.permanents[giant].as_ref().unwrap().damage, 2);
    s.set_player_mana_pool(0, "3WW");
    cast(&mut s, "Fumigate");
    settle(&mut s);
    assert!(s.game().state.permanents[giant].is_none());
    assert_eq!(s.game().characteristics(ooze).pt(), (3, 3));
    s.assert_life(0, 21);
}
