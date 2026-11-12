//! Tests for cards filed in `src/dom.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_to_hand},
    inspect::{keywords, offered_targets, pt_modifications},
    play::{attack_with, cast_card, pass_until, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    flow::turn::StepKind,
    state::{card::Keyword, game_object::Target},
};

#[test]
fn run_amok_pumps_only_an_attacking_creature() {
    let mut s = five_color_setup(&["Run Amok", "Gray Ogre"]);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let idle = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, attacker);
    let run_amok = move_to_hand(&mut s, "Run Amok");
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, attacker);
    s.advance_to_step(StepKind::DeclareAttackers);
    while s.action_space().kind != ActionSpaceKind::Priority {
        s.advance_default_action();
    }
    s.set_player_mana_pool(0, "1R");
    cast_card(&mut s, run_amok);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(attacker)));
    assert!(!targets.contains(&Target::Permanent(idle)));
    assert!(s.choose_target(Target::Permanent(attacker)));
    settle(&mut s);
    assert_eq!(pt_modifications(&s, attacker), (3, 3));
    assert!(keywords(&s, attacker).has(Keyword::Trample));
    s.advance_to_step(StepKind::EndOfCombat);
    s.assert_life(1, 15);
}
