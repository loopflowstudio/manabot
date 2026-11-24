//! Theros card behavior, including all four categories of "any target".

use crate::support::{
    game::{five_color_setup, set_counters, until_end_of_turn},
    inspect::{assert_choice_contract, card_of, counters_of, offered_targets, zone_of},
    play::{cast, settle},
};
use managym::flow::turn::StepKind;
use managym::state::{
    card::{CardType, CardTypes, Keyword, Keywords},
    effect::Modification,
    game_object::{PermanentId, PlayerId, Target},
    permanent::CounterKind,
    zone::ZoneType,
};

#[test]
fn lightning_strike_any_target_deals_damage_to_creatures_players_planeswalkers_and_battles() {
    for card_type in [CardType::Creature, CardType::Planeswalker, CardType::Battle] {
        for initial in [3, 5] {
            let mut s = five_color_setup(&["Lightning Strike", "Wall of Stone", "Prophetic Prism"]);
            let victim = s.force_permanent_on_battlefield(1, "Wall of Stone");
            let object = card_of(&s, victim);
            // Controlled shell: no invented catalog definition or unsupported
            // battle/planeswalker ability is needed to test CR 120.3.
            s.game_mut().state[object].types = CardTypes::new([card_type]);
            let counter = if card_type == CardType::Battle {
                CounterKind::Defense
            } else {
                CounterKind::Loyalty
            };
            set_counters(&mut s, victim, counter, initial);
            let land = s.force_permanent_on_battlefield(1, "Island");
            let artifact = s.force_permanent_on_battlefield(1, "Prophetic Prism");
            cast(&mut s, "Lightning Strike");
            let targets = offered_targets(&s);
            assert!(targets.contains(&Target::Permanent(victim)));
            assert!(targets.contains(&Target::Player(PlayerId(0))));
            assert!(targets.contains(&Target::Player(PlayerId(1))));
            assert!(!targets.contains(&Target::Permanent(land)));
            assert!(!targets.contains(&Target::Permanent(artifact)));
            assert!(targets
                .iter()
                .all(|t| matches!(t, Target::Permanent(_) | Target::Player(_))));
            assert_choice_contract(&s);
            assert!(s.choose_target(Target::Permanent(victim)));
            settle(&mut s);
            if card_type != CardType::Creature && initial == 3 {
                assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
            } else {
                let permanent = s.game().state.permanents[victim].as_ref().unwrap();
                assert!(permanent.dealt_damage_this_turn);
                assert_eq!(
                    permanent.damage,
                    if card_type == CardType::Creature {
                        3
                    } else {
                        0
                    }
                );
                assert_eq!(
                    counters_of(&s, victim, counter),
                    if card_type == CardType::Creature {
                        initial
                    } else {
                        initial - 3
                    }
                );
            }
        }
    }
    let mut s = five_color_setup(&["Lightning Strike"]);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.life(1), 17);
}

#[test]
fn lightning_strike_rechecks_hexproof_and_damage_uses_every_current_type() {
    for hexproof in [false, true] {
        let mut s = five_color_setup(&["Lightning Strike", "Wall of Stone"]);
        let victim = s.force_permanent_on_battlefield(1, "Wall of Stone");
        let object = card_of(&s, victim);
        s.game_mut().state[object].types =
            CardTypes::new([CardType::Creature, CardType::Planeswalker, CardType::Battle]);
        set_counters(&mut s, victim, CounterKind::Loyalty, 5);
        set_counters(&mut s, victim, CounterKind::Defense, 5);
        cast(&mut s, "Lightning Strike");
        assert!(s.choose_target(Target::Permanent(victim)));
        if hexproof {
            until_end_of_turn(
                &mut s,
                victim,
                Modification::AddKeywords(Keywords::of(&[Keyword::Hexproof])),
            );
        }
        settle(&mut s);
        assert_eq!(
            s.game().state.permanents[victim].as_ref().unwrap().damage,
            if hexproof { 0 } else { 3 }
        );
        for counter in [CounterKind::Loyalty, CounterKind::Defense] {
            assert_eq!(
                counters_of(&s, victim, counter),
                if hexproof { 5 } else { 2 }
            );
        }
    }
}

#[test]
fn lightning_strike_excludes_opposing_hexproof_planeswalkers_and_battles() {
    for (card_type, counter) in [
        (CardType::Planeswalker, CounterKind::Loyalty),
        (CardType::Battle, CounterKind::Defense),
    ] {
        let mut s = five_color_setup(&["Lightning Strike", "Wall of Stone"]);
        let own = s.force_permanent_on_battlefield(0, "Wall of Stone");
        let opposing = s.force_permanent_on_battlefield(1, "Wall of Stone");
        for id in [own, opposing] {
            let object = card_of(&s, id);
            s.game_mut().state[object].types = CardTypes::new([card_type]);
            set_counters(&mut s, id, counter, 5);
            until_end_of_turn(
                &mut s,
                id,
                Modification::AddKeywords(Keywords::of(&[Keyword::Hexproof])),
            );
        }
        cast(&mut s, "Lightning Strike");
        assert!(offered_targets(&s).contains(&Target::Permanent(own)));
        assert!(!offered_targets(&s).contains(&Target::Permanent(opposing)));
        assert!(s.choose_target(Target::Permanent(own)));
        settle(&mut s);
        assert_eq!(counters_of(&s, own, counter), 2);
        assert_eq!(counters_of(&s, opposing, counter), 5);
    }
}

/// Theros's five Temples enter tapped, scry 1 as they enter (the
/// top card stays or goes to the bottom, as chosen) and pay either of their
/// two colors, and no other.
#[test]
fn temples_enter_tapped_scry_one_and_supply_either_color() {
    for (name, colors) in [
        ("Temple of Abandon", "RG"),
        ("Temple of Deceit", "UB"),
        ("Temple of Mystery", "GU"),
        ("Temple of Silence", "WB"),
        ("Temple of Triumph", "RW"),
    ] {
        crate::support::play::temple_scries_one_and_pays_either_color(name, colors);
    }
}

/// Burnished Hart pays {3} and itself for up to two basic lands, both
/// entering tapped.
#[test]
fn burnished_hart_sacrifices_itself_for_two_tapped_basic_lands() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands},
        inspect::{card_of, library_card, tapped, zone_of},
        play::{activate, pass_until, select_card, settle},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{player::ManaPayment, zone::ZoneType},
    };
    let mut s = five_color_unfunded_setup(&["Burnished Hart"]);
    let hart = s.force_permanent_on_battlefield(0, "Burnished Hart");
    let hart_card = card_of(&s, hart);
    force_lands(&mut s, 0, "Mountain", 3);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    activate(&mut s, hart, 0);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(zone_of(&s, hart_card), Some(ZoneType::Graveyard));
    assert_eq!(s.untapped_permanents_named(0, "Mountain"), 0);
    for name in ["Forest", "Plains"] {
        let land = library_card(&s, 0, name);
        select_card(&mut s, land);
    }
    settle(&mut s);
    for name in ["Forest", "Plains"] {
        let [entered] = s.battlefield_permanents_named(0, name)[..] else {
            panic!("one {name} entered")
        };
        assert!(tapped(&s, entered), "{name}");
    }
}

/// Ogre enchanted by a resolved Ordeal of Nylea, ready to attack, with
/// `counters` +1/+1 counters from earlier attacks.
fn ordeal_on_ogre(counters: i32) -> (crate::support::game::Scenario, PermanentId, PermanentId) {
    use crate::support::{game::clear_summoning_sickness, inspect::permanent_of};
    let mut s = five_color_setup(&["Ordeal of Nylea", "Gray Ogre", "Hill Giant"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    let aura = cast(&mut s, "Ordeal of Nylea");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let ordeal = permanent_of(&s, aura).expect("Ordeal resolved");
    assert_eq!(
        s.game().state.permanents[ordeal]
            .as_ref()
            .unwrap()
            .attached_to,
        Some(card_of(&s, ogre))
    );
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, counters);
    (s, ogre, ordeal)
}

/// Declare the Ogre's attack and stop with Ordeal's trigger on the stack.
fn attack_with_enchanted(s: &mut crate::support::game::Scenario, ogre: PermanentId) {
    use crate::support::play::{attack_with, pass_until};
    use managym::agent::action::ActionSpaceKind;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(s, ogre);
    pass_until(s, ActionSpaceKind::Priority);
    assert_eq!(
        s.game().state.stack_objects.len(),
        1,
        "Ordeal's attack trigger"
    );
}

#[test]
fn ordeal_of_nylea_grows_its_attacker_then_sacrifices_for_two_tapped_basics() {
    use crate::support::{
        inspect::{library_card, permanent_of},
        play::{command, pass_until, resolve_top, select_card},
    };
    use managym::agent::action::{Action, ActionSpaceKind};

    // Below the threshold the counter lands and Ordeal stays.
    let (mut s, ogre, ordeal) = ordeal_on_ogre(1);
    let ordeal_object = card_of(&s, ordeal);
    attack_with_enchanted(&mut s, ogre);
    resolve_top(&mut s);
    assert_eq!(counters_of(&s, ogre, CounterKind::PlusOnePlusOne), 2);
    assert_eq!(zone_of(&s, ordeal_object), Some(ZoneType::Battlefield));
    assert!(s.game().state.stack_objects.is_empty());

    // The third counter sacrifices Ordeal, whose own trigger searches.
    let (mut s, ogre, ordeal) = ordeal_on_ogre(2);
    let ordeal_object = card_of(&s, ordeal);
    attack_with_enchanted(&mut s, ogre);
    resolve_top(&mut s);
    assert_eq!(counters_of(&s, ogre, CounterKind::PlusOnePlusOne), 3);
    assert_eq!(zone_of(&s, ordeal_object), Some(ZoneType::Graveyard));
    assert_eq!(s.game().state.stack_objects.len(), 1, "sacrifice trigger");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let search = s.snapshot();
    let lands_before = s.battlefield_permanents_named(0, "Forest").len()
        + s.battlefield_permanents_named(0, "Island").len();

    let forest = library_card(&s, 0, "Forest");
    let island = library_card(&s, 0, "Island");
    assert!(
        !s.action_space().actions.iter().any(|a| matches!(
            a,
            Action::SelectCard { object, .. } if s.game().state[*object].name == "Hill Giant"
        )),
        "only basic lands are offered"
    );
    select_card(&mut s, forest);
    select_card(&mut s, island);
    settle(&mut s);
    for found in [forest, island] {
        let id = permanent_of(&s, found).expect("searched land entered");
        assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
        assert_eq!(
            s.game().state.permanents[id].as_ref().unwrap().controller,
            PlayerId(0)
        );
    }

    // "Up to two": declining finds nothing and still shuffles.
    s.restore(&search);
    command(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "Forest").len()
            + s.battlefield_permanents_named(0, "Island").len(),
        lands_before
    );
    assert_eq!(zone_of(&s, ordeal_object), Some(ZoneType::Graveyard));
}

/// `Attached` names the enchanted creature, never the event's object, and
/// reads the Aura's last known attachment once it has left.
#[test]
fn ordeal_of_nylea_attached_survives_aura_departure_and_ignores_event_objects() {
    use crate::support::{
        game::{inject_ability, move_card},
        inspect::{permanent_of, tapped},
        play::{enter_and_settle, resolve_top},
    };
    use managym::state::{
        ability::{Ability, Effect, ObjectSelector, TriggerCondition, TriggerSubject},
        predicate::CardPredicate,
    };

    // The Aura leaves in response: the attacker still gets its counter, and
    // Ordeal, no longer on the battlefield, is not sacrificed (no search).
    let (mut s, ogre, ordeal) = ordeal_on_ogre(2);
    let ordeal_object = card_of(&s, ordeal);
    attack_with_enchanted(&mut s, ogre);
    move_card(&mut s, ordeal_object, ZoneType::Graveyard);
    resolve_top(&mut s);
    assert_eq!(counters_of(&s, ogre, CounterKind::PlusOnePlusOne), 3);
    assert!(
        s.game().state.stack_objects.is_empty(),
        "no sacrifice trigger"
    );

    // A trigger whose event names another creature: "that creature" is the
    // Giant, the enchanted creature is the Ogre.
    let (mut s, ogre, ordeal) = ordeal_on_ogre(0);
    let ordeal_object = card_of(&s, ordeal);
    inject_ability(
        &mut s,
        ordeal_object,
        Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::another(CardPredicate::creature()),
            },
            intervening_if: None,
            effects: vec![
                Effect::tap(ObjectSelector::Triggering),
                Effect::put_counters(ObjectSelector::Attached, CounterKind::PlusOnePlusOne, 1),
            ],
        },
    );
    let giant = enter_and_settle(&mut s, "Hill Giant");
    let giant = permanent_of(&s, giant).expect("Giant entered");
    assert!(tapped(&s, giant));
    assert!(!tapped(&s, ogre));
    assert_eq!(counters_of(&s, ogre, CounterKind::PlusOnePlusOne), 1);
    assert_eq!(counters_of(&s, giant, CounterKind::PlusOnePlusOne), 0);
}
