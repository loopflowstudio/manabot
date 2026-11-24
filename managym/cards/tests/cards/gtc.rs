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

#[test]
fn foundations_5e_boros_charm() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{keywords, mode_labels, offered_targets};
    use crate::support::play::{cast, choose_mode, settle};
    use managym::state::card::Keyword;
    use managym::state::game_object::{PlayerId, Target};
    let mut s = five_color_setup(&["Boros Charm", "Gray Ogre", "Hill Giant"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Hill Giant");
    let walker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    crate::make_planeswalker(&mut s, walker);
    cast(&mut s, "Boros Charm");
    let labels = mode_labels(&s);
    assert_eq!(labels.len(), 3);
    let mut damage = s.clone();
    choose_mode(&mut damage, 0);
    let offers = offered_targets(&damage);
    assert!(offers.contains(&Target::Player(PlayerId(0))));
    assert!(offers.contains(&Target::Player(PlayerId(1))));
    assert!(offers.contains(&Target::Permanent(walker)));
    assert!(!offers.contains(&Target::Permanent(theirs)));
    assert!(!offers.contains(&Target::Permanent(ogre)));
    assert!(damage.choose_target(Target::Player(PlayerId(1))));
    settle(&mut damage);
    assert_eq!(damage.life(1), 16);
    let mut shield = s.clone();
    choose_mode(&mut shield, 1);
    settle(&mut shield);
    assert!(keywords(&shield, ogre).has(Keyword::Indestructible));
    assert!(!keywords(&shield, theirs).has(Keyword::Indestructible));
    choose_mode(&mut s, 2);
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert!(keywords(&s, theirs).has(Keyword::DoubleStrike));
    assert!(!keywords(&s, ogre).has(Keyword::DoubleStrike));
}
