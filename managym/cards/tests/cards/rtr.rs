//! Tests for cards filed in `src/rtr.rs`, in card order.

use crate::support::play::tapped_dual_land_pays_either_color;
use managym::agent::action::ActionSpaceKind;

/// Return to Ravnica's five Guildgates enter tapped, do nothing else as they
/// enter, are Gates and pay either of their two colors, and no other.
#[test]
fn guildgates_enter_tapped_and_supply_either_color() {
    for (name, colors) in [
        ("Azorius Guildgate", "WU"),
        ("Golgari Guildgate", "BG"),
        ("Izzet Guildgate", "UR"),
        ("Rakdos Guildgate", "BR"),
        ("Selesnya Guildgate", "GW"),
    ] {
        tapped_dual_land_pays_either_color(name, colors, |s, land| {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
            assert!(s.game().state.stack_objects.is_empty());
            assert_eq!(s.game().characteristics(land).subtypes, ["Gate"]);
        });
    }
}

#[test]
fn foundations_permanent_guttersnipe() {
    use crate::support::game::five_color_setup;
    use crate::support::play::play_and_settle;
    let mut s = five_color_setup(&["Guttersnipe", "Gray Ogre", "Sol Ring", "Scout's Warning"]);
    s.force_permanent_on_battlefield(0, "Guttersnipe");
    play_and_settle(&mut s, "Gray Ogre");
    s.assert_life(1, 20);
    play_and_settle(&mut s, "Sol Ring");
    s.assert_life(1, 20);
    play_and_settle(&mut s, "Scout's Warning");
    s.assert_life(1, 18);
    s.assert_life(0, 20);
}

#[test]
fn foundations_permanent_rogues_passage() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{activate, settle};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Rogue's Passage", "Gray Ogre", "Sol Ring"]);
    let source = s.force_permanent_on_battlefield(0, "Rogue's Passage");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let mut mana = s.clone();
    mana.set_player_mana_pool(0, "");
    crate::support::play::play_and_settle(&mut mana, "Sol Ring");
    assert!(crate::support::inspect::tapped(&mana, source));
    assert_eq!(mana.battlefield_permanents_named(0, "Sol Ring").len(), 1);
    s.set_player_mana_pool(0, "4");
    activate(&mut s, source, 0);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(s.game().cant_be_blocked(ogre));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!s.game().cant_be_blocked(ogre));
}
