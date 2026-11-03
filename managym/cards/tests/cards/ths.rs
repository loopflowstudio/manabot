//! Theros card behavior, including all four categories of "any target".

use crate::support::{
    game::{five_color_setup, set_counters, until_end_of_turn},
    inspect::{assert_choice_contract, card_of, counters_of, offered_targets, zone_of},
    play::{cast, settle},
};
use managym::state::{
    card::{CardType, CardTypes, Keyword, Keywords},
    effect::Modification,
    game_object::{PlayerId, Target},
    permanent::CounterKind,
    zone::ZoneType,
};

#[test]
fn lightning_strike_any_target_deals_damage_to_creatures_players_planeswalkers_and_battles() {
    for card_type in [CardType::Creature, CardType::Planeswalker, CardType::Battle] {
        for initial in [3, 5] {
            let mut s = five_color_setup(&["Lightning Strike", "Wall of Stone", "Prophetic Prism"]);
            let victim = s.force_permanent_on_battlefield(1, "Wall of Stone");
            let card = card_of(&s, victim);
            // Controlled shell: no invented catalog definition or unsupported
            // battle/planeswalker ability is needed to test CR 120.3.
            s.game_mut().state.cards[card].types = CardTypes::new([card_type]);
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
                assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
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
        let card = card_of(&s, victim);
        s.game_mut().state.cards[card].types =
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
            let card = card_of(&s, id);
            s.game_mut().state.cards[card].types = CardTypes::new([card_type]);
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
