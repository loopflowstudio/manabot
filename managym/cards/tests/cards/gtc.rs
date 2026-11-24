//! Tests for cards filed in `src/gtc.rs`, in card order.

use crate::support::play::tapped_dual_land_pays_either_color;
use managym::agent::action::ActionSpaceKind;

/// Gatecrash's five Guildgates enter tapped, do nothing else as they
/// enter, are Gates and pay either of their two colors, and no other.
#[test]
fn guildgates_enter_tapped_and_supply_either_color() {
    for (name, colors) in [
        ("Boros Guildgate", "RW"),
        ("Dimir Guildgate", "UB"),
        ("Gruul Guildgate", "RG"),
        ("Orzhov Guildgate", "WB"),
        ("Simic Guildgate", "GU"),
    ] {
        tapped_dual_land_pays_either_color(name, colors, |s, land| {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
            assert!(s.game().state.stack_objects.is_empty());
            assert_eq!(s.game().characteristics(land).subtypes, ["Gate"]);
        });
    }
}

#[test]
fn foundations_spell_angelic_edict() {
    use managym::state::zone::ZoneType;
    crate::check_targeted_removal("Angelic Edict", "Gray Ogre", "Sol Ring", ZoneType::Exile);
}

#[test]
fn foundations_spell_aetherize() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{attack_with, cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Aetherize", "Gray Ogre", "Hill Giant"]);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let stays = s.force_permanent_on_battlefield(0, "Hill Giant");
    let object = card_of(&s, attacker);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, attacker);
    s.set_player_mana_pool(0, "3U");
    cast(&mut s, "Aetherize");
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Hand));
    assert_eq!(s.game().characteristics(stays).pt(), (3, 3));
}
