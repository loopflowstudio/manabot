//! Tests for cards filed in `src/mid.rs`, in card order.

use crate::support::inspect::zone_of;
use crate::support::{
    game::five_color_setup,
    play::{cast, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn consider_surveils_one_then_draws() {
    let mut s = five_color_setup(&["Consider"]);
    cast(&mut s, "Consider");
    let hand = s.zone_size(0, ZoneType::Hand);
    let library = s.zone_size(0, ZoneType::Library);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
    let milled = s.zone_size(0, ZoneType::Graveyard) - 1;
    assert!(milled <= 1);
    s.assert_zone_size(0, ZoneType::Library, library - 1 - milled);
}

#[test]
fn join_the_dance_makes_two_humans_and_flashes_back_for_two_more() {
    let mut s = five_color_setup(&["Join the Dance"]);
    let dance = cast(&mut s, "Join the Dance");
    settle(&mut s);
    let humans = s.battlefield_permanents_named(0, "Human");
    assert_eq!(humans.len(), 2);
    let view = s.game().characteristics(humans[0]);
    assert_eq!((view.power, view.toughness), (1, 1));
    assert_eq!(zone_of(&s, dance), Some(ZoneType::Graveyard));
    crate::support::play::cast_card(&mut s, dance);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Human").len(), 4);
    assert_eq!(zone_of(&s, dance), Some(ZoneType::Exile));
}

/// Cathar Commando pays {1} and itself to destroy an artifact or enchantment,
/// never a creature, and has flash.
#[test]
fn cathar_commando_sacrifices_itself_to_destroy_an_artifact_and_has_flash() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::{can_cast, card_of, has_action, zone_of},
        play::{activate, cast_card, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{game_object::Target, player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_unfunded_setup(&["Cathar Commando", "Sol Ring", "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "Cathar Commando");
    let source_card = card_of(&s, source);
    force_lands(&mut s, 0, "Plains", 1);
    let target = s.force_permanent_on_battlefield(1, "Sol Ring");
    let target_card = card_of(&s, target);
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    activate(&mut s, source, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ChooseTarget { target: Target::Permanent(id), .. } if *id == other)
    }));
    assert!(s.choose_target(Target::Permanent(target)));
    assert_eq!(zone_of(&s, source_card), Some(ZoneType::Graveyard));
    assert_eq!(s.untapped_permanents_named(0, "Plains"), 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, target_card), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[other].is_some());

    // Flash: castable while a spell is on the stack; the Ogre is not.
    let mut s = five_color_unfunded_setup(&["Cathar Commando", "Sol Ring", "Gray Ogre"]);
    force_lands(&mut s, 0, "Mountain", 8);
    force_lands(&mut s, 0, "Plains", 8);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let first = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let second = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let flashed = force_new_card_in_hand(&mut s, 0, "Cathar Commando");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(&mut s, first);
    assert!(!s.game().state.stack_objects.is_empty());
    assert!(can_cast(&s, flashed));
    assert!(!can_cast(&s, second));
}

/// Eaten Alive is cast by sacrificing a creature or by paying {3}{B} more; it
/// exiles its target either way.
#[test]
fn eaten_alive_takes_a_sacrificed_creature_or_three_and_a_black_more() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::{card_of, has_action, top_spell, zone_of},
        play::{cast_card, choose, select_card, settle},
    };
    use managym::{
        agent::action::Action,
        state::{game_object::Target, player::ManaPayment, zone::ZoneType},
    };
    for sacrifice in [false, true] {
        let mut s = five_color_unfunded_setup(&["Eaten Alive", "Gray Ogre", "Hill Giant"]);
        force_lands(&mut s, 0, "Swamp", if sacrifice { 1 } else { 5 });
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let ogre_card = card_of(&s, ogre);
        let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
        let giant_card = card_of(&s, giant);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        let eaten = force_new_card_in_hand(&mut s, 0, "Eaten Alive");
        s.game_mut().scenario_refresh_priority().unwrap();
        cast_card(&mut s, eaten);
        assert!(s.choose_target(Target::Permanent(giant)));
        // One Swamp cannot pay {3}{B}{B}: only the sacrifice is offered.
        assert_eq!(
            has_action(&s, |a| matches!(a, Action::PayCost { .. })),
            !sacrifice
        );
        if sacrifice {
            select_card(&mut s, ogre_card);
        } else {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        assert_eq!(top_spell(&s).spent.total(), if sacrifice { 1 } else { 5 });
        settle(&mut s);
        assert_eq!(zone_of(&s, giant_card), Some(ZoneType::Exile));
        assert_eq!(
            zone_of(&s, ogre_card),
            Some(if sacrifice {
                ZoneType::Graveyard
            } else {
                ZoneType::Battlefield
            })
        );
    }
}
