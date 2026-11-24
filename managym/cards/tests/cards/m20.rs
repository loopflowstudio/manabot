//! Tests for cards filed in `src/m20.rs`, in card order.

use crate::support::game::move_card;
use crate::support::inspect::{now, permanent_of, zone_of};
use crate::support::{
    game::five_color_setup,
    play::{cast, settle, spend_colorless},
};
use managym::{
    agent::action::Action,
    state::{game_object::Target, zone::ZoneType},
};

#[test]
fn spectral_sailor_draw_ability_survives_the_sources_destruction() {
    let mut s = five_color_setup(&["Spectral Sailor", "Lightning Strike"]);
    let sailor = s.force_permanent_on_battlefield(0, "Spectral Sailor");
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==sailor))
        .unwrap();
    s.step_action(index);
    spend_colorless(&mut s);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(sailor)));
    settle(&mut s);
    assert!(s.game().state.permanents[sailor].is_none());
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
}

#[test]
fn agent_of_treachery_steals_and_draws_three_with_three_unowned_permanents() {
    use crate::support::{inspect::permanent, play::resolve_everything};
    use managym::{flow::turn::StepKind, state::game_object::PlayerId};
    let mut s = five_color_setup(&["Agent of Treachery", "Gray Ogre"]);
    s.set_player_mana_pool(0, "30UUUUUUUU");
    let ogres: Vec<_> = (0..3)
        .map(|_| s.force_permanent_on_battlefield(1, "Gray Ogre"))
        .collect();
    for (stolen, ogre) in ogres.iter().enumerate() {
        cast(&mut s, "Agent of Treachery");
        spend_colorless(&mut s);
        settle_until_target(&mut s, Target::Permanent(*ogre));
        assert_eq!(permanent(&s, *ogre).controller, PlayerId(0));
        assert_eq!(
            s.battlefield_permanents_named(0, "Gray Ogre").len(),
            stolen + 1
        );
    }
    let hand = s.zone_size(0, ZoneType::Hand);
    s.advance_to_step(StepKind::End);
    resolve_everything(&mut s);
    // Each of the three Agents triggers.
    s.assert_zone_size(0, ZoneType::Hand, hand + 9);
}

#[test]
fn agent_of_treachery_draws_nothing_with_two_unowned_permanents() {
    use crate::support::play::resolve_everything;
    use managym::flow::turn::StepKind;
    let mut s = five_color_setup(&["Agent of Treachery", "Gray Ogre"]);
    s.set_player_mana_pool(0, "30UUUUUUUU");
    for _ in 0..2 {
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        cast(&mut s, "Agent of Treachery");
        spend_colorless(&mut s);
        settle_until_target(&mut s, Target::Permanent(ogre));
    }
    let hand = s.zone_size(0, ZoneType::Hand);
    s.advance_to_step(StepKind::End);
    resolve_everything(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand);
}

/// Resolve the cast creature, aiming its entry trigger at `target`.
fn settle_until_target(s: &mut crate::support::game::Scenario, target: Target) {
    for _ in 0..20 {
        if s.action_space().kind == managym::agent::action::ActionSpaceKind::ChooseTarget {
            assert!(s.choose_target(target));
        } else if s.action_space().kind == managym::agent::action::ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.pending_triggers.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("entry trigger did not settle");
}

#[test]
fn brought_back_returns_only_cards_put_into_the_graveyard_from_the_battlefield_this_turn() {
    use crate::support::{
        game::{move_from_library, set_counters},
        inspect::{card_of, offered_targets, permanent},
    };
    use managym::state::permanent::CounterKind;
    let mut s = five_color_setup(&["Brought Back", "Gray Ogre", "Lightning Strike"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 1);
    let died = card_of(&s, ogre);
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let left = card_of(&s, other);
    let milled = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    // The second Ogre leaves the battlefield and comes back to its owner's
    // graveyard by another route: graveyard to exile to graveyard is a new
    // object, so it no longer qualifies.
    let left = move_card(&mut s, left, ZoneType::Graveyard);
    let left = move_card(&mut s, left, ZoneType::Exile);
    let left = move_card(&mut s, left, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    cast(&mut s, "Brought Back");
    let offered = offered_targets(&s);
    let graveyard = |object| {
        s.game()
            .state
            .live(now(&s, object))
            .map(Target::GraveyardCard)
    };
    assert!(offered.contains(&graveyard(died).unwrap()));
    assert!(!offered.contains(&graveyard(milled).unwrap()));
    assert!(!offered.contains(&graveyard(left).unwrap()));
    assert!(s.choose_target(graveyard(died).unwrap()));
    settle(&mut s);
    let returned = permanent_of(&s, died).expect("returned");
    assert!(permanent(&s, returned).tapped);
    // A new object: no counters and no damage carried over.
    assert_eq!(permanent(&s, returned).counters.iter().count(), 0);
    for object in [milled, left] {
        assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
    }
}

/// Brineborn Cutthroat has flash and grows only from spells its controller
/// casts during an opponent's turn.
#[test]
fn brineborn_cutthroat_grows_only_from_spells_cast_on_an_opponents_turn() {
    use crate::support::{
        game::{force_new_card_in_hand, Scenario},
        inspect::counters_of,
        play::{cast_card, castable},
    };
    use managym::{
        flow::turn::StepKind,
        state::{game_object::PlayerId, permanent::CounterKind, player::ManaPayment},
    };
    let bolt = |s: &mut Scenario| {
        let card = force_new_card_in_hand(s, 0, "Lightning Bolt");
        cast_card(s, card);
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        settle(s);
    };
    let mut s = five_color_setup(&["Brineborn Cutthroat", "Lightning Bolt", "Gray Ogre"]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let flashed = force_new_card_in_hand(&mut s, 0, "Brineborn Cutthroat");
    cast_card(&mut s, ogre);
    assert!(castable(&mut s, flashed), "flash");
    settle(&mut s);
    let cutthroat = s.force_permanent_on_battlefield(0, "Brineborn Cutthroat");
    bolt(&mut s);
    assert_eq!(counters_of(&s, cutthroat, CounterKind::PlusOnePlusOne), 0);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(0, "R");
    s.pass_priority();
    bolt(&mut s);
    assert_eq!(counters_of(&s, cutthroat, CounterKind::PlusOnePlusOne), 1);
    s.assert_life(1, 14);
}

/// Gnarlback Rhino draws when its controller casts a spell targeting it;
/// not for a spell aimed elsewhere, nor for an opponent's spell targeting it.
#[test]
fn gnarlback_rhino_draws_when_you_cast_a_spell_targeting_it() {
    use crate::support::{game::force_new_card_in_hand, inspect::hand, play::cast_card};
    use managym::state::{game_object::PlayerId, player::ManaPayment};
    let mut s = five_color_setup(&[
        "Gnarlback Rhino",
        "Lightning Bolt",
        "Gray Ogre",
        "Fancy Footwork",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let rhino = s.force_permanent_on_battlefield(0, "Gnarlback Rhino");
    let bolt_at = |s: &mut crate::support::game::Scenario, player: usize, target: Target| {
        let card = force_new_card_in_hand(s, player, "Lightning Bolt");
        cast_card(s, card);
        assert!(s.choose_target(target));
        settle(s);
    };
    bolt_at(&mut s, 0, Target::Player(PlayerId(1)));
    assert!(hand(&s, 0).is_empty());
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let footwork = force_new_card_in_hand(&mut s, 0, "Fancy Footwork");
    cast_card(&mut s, footwork);
    assert!(s.choose_target(Target::Permanent(other)));
    assert!(s.choose_target(Target::Permanent(rhino)));
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    cast_card(&mut s, ogre);
    s.pass_priority();
    bolt_at(&mut s, 1, Target::Permanent(rhino));
    assert_eq!(hand(&s, 0).len(), 1);
    assert!(hand(&s, 1).is_empty());
}
