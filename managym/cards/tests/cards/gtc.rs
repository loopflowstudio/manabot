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

/// Consuming Aberration is as large as its opponents' graveyards, in every
/// zone. Each spell its controller casts has each opponent reveal until a
/// land and put every revealed card into their graveyard at once.
#[test]
fn consuming_aberration_mills_opponents_until_a_land_and_grows() {
    use crate::support::{
        game::{five_color_setup, force_new_card_in_hand, pin_top, plant_in_graveyard},
        inspect::zone_of,
        play::{cast_card, settle},
    };
    use managym::{
        flow::turn::StepKind,
        state::{game_object::PlayerId, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Consuming Aberration", "Gray Ogre"]);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "3R");
    let aberration = s.force_permanent_on_battlefield(1, "Consuming Aberration");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    plant_in_graveyard(&mut s, 1, "Gray Ogre");
    assert_eq!(s.game().characteristics(aberration).pt(), (1, 1));
    let revealed = pin_top(&mut s, &["Gray Ogre", "Gray Ogre", "Plains", "Island"]);
    let graveyard = s.game().state.zones.size(ZoneType::Graveyard, PlayerId(0));
    let spell = force_new_card_in_hand(&mut s, 1, "Gray Ogre");
    cast_card(&mut s, spell);
    settle(&mut s);
    for object in &revealed[..3] {
        assert_eq!(zone_of(&s, *object), Some(ZoneType::Graveyard));
    }
    assert_eq!(zone_of(&s, revealed[3]), Some(ZoneType::Library));
    let graveyard = graveyard + 3;
    assert_eq!(
        s.game().state.zones.size(ZoneType::Graveyard, PlayerId(0)),
        graveyard
    );
    let side = graveyard as i32;
    assert_eq!(s.game().characteristics(aberration).pt(), (side, side));
    let in_hand = force_new_card_in_hand(&mut s, 1, "Consuming Aberration");
    assert_eq!(s.game().object_pt(in_hand), (side, side));
}

#[test]
fn aurelia_first_declaration_tracks_each_appearance() {
    use crate::fdn::assert_raid;
    use crate::support::game::five_color_setup;
    use crate::support::game::move_card;
    use crate::support::inspect::card_of;
    use crate::support::inspect::permanent_of;
    use crate::support::play::attack_with;
    use crate::support::play::settle;
    use crate::support::{game::clear_summoning_sickness, inspect::tapped};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::PlayerId;
    use managym::state::zone::ZoneType;
    use managym::{
        agent::{action::ActionSpaceKind, observation::Observation},
        flow::turn::AdditionalCombat,
    };
    let name = "Aurelia, the Warleader";
    let mut s = five_color_setup(&[name, "Gray Ogre", "Honest Work"]);
    let aurelia = s.force_permanent_on_battlefield(0, name);
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let original = card_of(&s, aurelia);
    clear_summoning_sickness(&mut s, other);
    let main = s.game().clone();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    let partial = s.game().clone();
    for viewer in 0..2 {
        let obs = Observation::for_player(s.game(), PlayerId(viewer));
        assert!(!obs
            .agent_permanents
            .iter()
            .chain(&obs.opponent_permanents)
            .any(|p| p.attacked_this_turn));
    }
    s.decline_attack();
    *s.game_mut() = partial;
    s.declare_attack();
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert!(tapped(&s, other));
    settle(&mut s);
    assert!(!tapped(&s, other));
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    let first = s.game().clone();
    for blink in [false, true] {
        *s.game_mut() = first.clone();
        let attacker = if blink {
            let away = move_card(&mut s, original, ZoneType::Exile);
            let back = move_card(&mut s, away, ZoneType::Battlefield);
            assert_eq!(s.game().state.turn.attack_count(back), 0);
            permanent_of(&s, back).unwrap()
        } else {
            aurelia
        };
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        attack_with(&mut s, attacker);
        assert_eq!(s.game().state.stack_objects.len(), usize::from(blink));
        assert_raid(&s, Some(0));
    }
    *s.game_mut() = main;
    clear_summoning_sickness(&mut s, aurelia);
    let aura = s.force_permanent_on_battlefield(1, "Honest Work");
    s.game_mut().state.permanents[aura]
        .as_mut()
        .unwrap()
        .attached_to = Some(original);
    assert!(s.game().characteristics(aurelia).abilities_lost);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, aurelia);
    assert!(s.game().state.stack_objects.is_empty());
    let aura_object = card_of(&s, aura);
    move_card(&mut s, aura_object, ZoneType::Graveyard);
    assert!(!s.game().characteristics(aurelia).abilities_lost);
    s.game_mut().state.permanents[aurelia]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(AdditionalCombat { attackers: None });
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, aurelia);
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(s.game().state.turn.attack_count(original), 2);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().state.turn.attack_count(original), 0);
    assert_raid(&s, None);
}

#[test]
fn aurelia_that_entered_attacking_still_gets_a_first_declaration() {
    use crate::fdn::assert_raid;
    use crate::support::game::five_color_setup;
    use crate::support::inspect::card_of;
    use crate::support::inspect::library_card;
    use crate::support::play::attack_with;
    use managym::flow::turn::AdditionalCombat;
    use managym::flow::turn::StepKind;
    use managym::state::game_object::PlayerId;
    let name = "Aurelia, the Warleader";
    let mut s = five_color_setup(&[name, "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    crate::support::game::clear_summoning_sickness(&mut s, ogre);
    let printed = s.game().state[library_card(&s, 0, name)].clone();
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    let token = s
        .game_mut()
        .create_token("token.soldier", PlayerId(0), true)
        .unwrap();
    let object = card_of(&s, token);
    s.game_mut().state[object].become_copy_of(&printed, &[]);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.decline_attack();
    assert_raid(&s, None);
    assert_eq!(s.game().state.turn.attack_count(object), 0);
    assert!(s.game().state.stack_objects.is_empty());
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(AdditionalCombat { attackers: None });
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.game_mut().state.permanents[token]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, token);
    assert_eq!(s.game().state.turn.attack_count(object), 1);
    assert_eq!(s.game().state.stack_objects.len(), 1);
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
