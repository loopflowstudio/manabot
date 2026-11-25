//! Tests for cards filed in `src/khc.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::{can_activate, card_of, tapped},
    play::{activate, attack_with, pass_until, select_card, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    flow::turn::StepKind,
    state::{game_object::PlayerId, mana::Color},
};

/// Lathril makes one green Elf Warrior for each combat damage it deals to a
/// player, and its drain taps it and ten other untapped Elves: nine others
/// and Lathril itself are not enough.
#[test]
fn lathril_makes_elves_from_combat_damage_and_drains_by_tapping_ten() {
    let mut s = five_color_setup(&["Lathril, Blade of the Elves"]);
    let lathril = s.force_permanent_on_battlefield(0, "Lathril, Blade of the Elves");
    clear_summoning_sickness(&mut s, lathril);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, lathril);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(1, 18);
    let elves = s.battlefield_permanents_named(0, "Elf Warrior");
    assert_eq!(elves.len(), 2);
    for elf in &elves {
        let view = s.game().characteristics(*elf);
        assert_eq!(view.pt(), (1, 1));
        assert_eq!(
            view.colors.iter().copied().collect::<Vec<_>>(),
            [Color::Green]
        );
    }

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    for _ in 0..7 {
        s.game_mut()
            .create_token("token.elf_warrior_1_1_g", PlayerId(0))
            .unwrap();
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, lathril), "nine other Elves");
    s.game_mut()
        .create_token("token.elf_warrior_1_1_g", PlayerId(0))
        .unwrap();
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate(&s, lathril));
    let elves = s.battlefield_permanents_named(0, "Elf Warrior");
    assert_eq!(elves.len(), 10);
    activate(&mut s, lathril, 0);
    for elf in &elves {
        if s.game().state.stack_objects.is_empty() {
            let object = card_of(&s, *elf);
            select_card(&mut s, object);
        }
    }
    settle(&mut s);
    assert!(tapped(&s, lathril));
    assert!(elves.iter().all(|elf| tapped(&s, *elf)));
    s.assert_life(1, 8);
    s.assert_life(0, 30);
}
