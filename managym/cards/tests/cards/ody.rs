//! Tests for cards filed in `src/ody.rs`, in card order.

use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, move_to_hand},
    inspect::{has_action, library_card, offered_cards, zone_of},
    play::{activate, cast, cast_card, cast_owned, choose, choose_mode, select_card, settle},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        effect::{Affected, EffectDuration, Modification},
        game_object::PlayerId,
        mana::Mana,
        zone::ZoneType,
    },
};

#[test]
fn entomb_puts_any_chosen_library_card_into_the_graveyard() {
    let mut s = five_color_setup(&["Entomb", "Gray Ogre"]);
    let ogre = library_card(&s, 0, "Gray Ogre");
    cast_owned(&mut s, "Entomb");
    for _ in 0..4 {
        if offered_cards(&s).contains(&ogre) {
            break;
        }
        s.pass_priority();
    }
    assert!(offered_cards(&s).contains(&ogre));
    let library = s.zone_size(0, ZoneType::Library);
    select_card(&mut s, ogre);
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));
    s.assert_zone_size(0, ZoneType::Library, library - 1);
}

/// The caster's opponent draws: the controller when the opponent casts, the
/// opponent when the controller casts.
#[test]
fn standstill_sacrifices_and_the_casters_opponent_draws_three() {
    for caster in [0usize, 1] {
        let mut s = five_color_setup(&["Standstill", "Lightning Bolt"]);
        let standstill = s.force_permanent_on_battlefield(0, "Standstill");
        if caster == 1 {
            s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
            s.set_player_mana_pool(1, "R");
        }
        let bolt = crate::support::game::force_new_card_in_hand(&mut s, caster, "Lightning Bolt");
        s.game_mut().scenario_refresh_priority().unwrap();
        let hands = [
            s.zone_size(0, ZoneType::Hand),
            s.zone_size(1, ZoneType::Hand),
        ];
        crate::support::play::cast_card(&mut s, bolt);
        crate::support::play::target_opponent(&mut s);
        settle(&mut s);
        assert!(s.game().state.permanents[standstill].is_none());
        let opponent = 1 - caster;
        s.assert_zone_size(opponent, ZoneType::Hand, hands[opponent] + 3);
        s.assert_zone_size(caster, ZoneType::Hand, hands[caster] - 1);
    }
}

/// Its {C} ability is an ordinary fixed mana ability that deals no damage;
/// the any-color ability deals 3 damage to its controller as part of the same
/// mana ability, once, from the land itself, so protection prevents it.
#[test]
fn tarnished_citadel_deals_three_damage_only_with_colored_mana() {
    let mut s = five_color_unfunded_setup(&["Tarnished Citadel", "Sol Ring"]);
    s.force_permanent_on_battlefield(0, "Tarnished Citadel");
    let ring = move_to_hand(&mut s, "Sol Ring");
    cast_card(&mut s, ring);
    assert!(
        has_action(&s, |a| matches!(a, Action::ActivateAbility { .. })),
        "the colored ability is offered only as an explicit choice"
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Sol Ring").len(), 1);
    assert_eq!(s.life(0), 20, "{{C}} paid Sol Ring without damage");

    for protected in [false, true] {
        let mut s = five_color_unfunded_setup(&["Tarnished Citadel"]);
        let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
        if protected {
            s.game_mut().apply_continuous(
                None,
                PlayerId(0),
                [Affected::Player(PlayerId(0))],
                Modification::ProtectionFromEverything,
                EffectDuration::UntilYourNextTurn,
            );
        }
        activate(&mut s, citadel, 0);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
        assert_eq!(s.action_space().actions.len(), 5);
        choose_mode(&mut s, 3);
        assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("R"));
        assert!(s.game().state.stack_objects.is_empty(), "no stack");
        assert_eq!(s.life(0), if protected { 20 } else { 17 });
        settle(&mut s);
        assert_eq!(s.life(0), if protected { 20 } else { 17 }, "damage once");
    }
}

/// Under Teferi's Protection the colored activation still adds its mana and
/// its damage is prevented.
#[test]
fn tarnished_citadel_damage_is_prevented_under_teferis_protection() {
    let mut s =
        five_color_unfunded_setup(&["Tarnished Citadel", "Teferi's Protection", "Sol Ring"]);
    s.set_player_mana_pool(0, "2W");
    cast(&mut s, "Teferi's Protection");
    settle(&mut s);
    let citadel = s.force_permanent_on_battlefield(0, "Tarnished Citadel");
    assert!(s.game().has_protection_from_everything(PlayerId(0)));
    activate(&mut s, citadel, 0);
    choose_mode(&mut s, 0);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("W"));
    assert_eq!(s.life(0), 20);
}
