//! Urza's Saga card behavior.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, set_counters},
    play::{attack_with, command, pass_until},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{game_object::PlayerId, permanent::CounterKind},
};

#[test]
fn fog_bank_blocks_a_lifelinker_in_both_replacement_orders() {
    let mut s = five_color_setup(&["Fog Bank", "Healer's Hawk", "Twinflame Tyrant"]);
    let hawk = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    s.force_permanent_on_battlefield(0, "Twinflame Tyrant");
    let fog = s.force_permanent_on_battlefield(1, "Fog Bank");
    set_counters(&mut s, fog, CounterKind::PlusOnePlusOne, 1);
    clear_summoning_sickness(&mut s, hawk);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, hawk);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.action_space().actions.len(), 2);
    for mode in 0..2 {
        let mut branch = s.clone();
        let mut replay = command(
            &mut branch,
            |action| matches!(action, Action::ChooseMode {mode: choice, ..} if *choice == mode),
        );
        for game in [&mut branch, &mut replay] {
            game.advance_to_active_step(0, StepKind::EndOfCombat);
            assert_eq!(game.life(0), 20);
            for creature in [hawk, fog] {
                assert_eq!(
                    game.game().state.permanents[creature]
                        .as_ref()
                        .unwrap()
                        .damage,
                    0
                );
            }
        }
    }
}

#[test]
fn foundations_discard_duress() {
    crate::assert_revealed_discard("Duress", &["Lightning Bolt", "Giant Growth"]);
}
