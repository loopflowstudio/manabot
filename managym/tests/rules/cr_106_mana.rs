use managym::{
    agent::action::{Action, ActionType},
    flow::{decision::Decision, trigger::ExileLink, turn::StepKind},
    state::{
        card::ReplacementEffect,
        game_object::PlayerId,
        mana::{Color, Mana},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

use crate::support::game::*;
use crate::support::{inspect::card_of, play::command};

/// A constructed mana source exercises the linked-return cost path without
/// adding a card definition. This covers synchronous entry replacement only;
/// affected-player replacement ordering needs the shared event continuation.
#[test]
fn mana_sacrifice_linked_return_preserves_parent_payment_through_commands() {
    for seat in 0..2 {
        for (mode, output) in ["W", "U", "B", "R", "G"].into_iter().enumerate() {
            let cards = deck(&[("Mountain", 24), ("Prophetic Prism", 4), ("Gray Ogre", 12)]);
            let mut s = Scenario::new(cards.clone(), cards, 106_605);
            s.advance_to_active_step(seat, StepKind::Main);
            let player = PlayerId(seat);
            let prism = s.force_permanent_on_battlefield(seat, "Prophetic Prism");
            let treasure = s
                .game_mut()
                .create_token("Treasure", player, false)
                .unwrap();
            let source_card = card_of(&s, treasure);
            s.game_mut().state.cards[source_card].activated_abilities[0].life_cost = 2;
            let returned = s.force_permanent_on_battlefield(1 - seat, "Gray Ogre");
            let returned_card = card_of(&s, returned);
            s.game_mut().move_card(returned_card, ZoneType::Exile);
            s.game_mut().state.cards[returned_card].replacement_effects =
                vec![ReplacementEffect::EntersWithPlusOneCounters {
                    count: 2,
                    condition: None,
                }];
            let source = s.game().current_object_ref(source_card).unwrap();
            s.game_mut().state.exile_links.push(ExileLink {
                source,
                exiled_card: returned_card,
                returns: true,
            });
            s.set_player_mana_pool(seat, "0");
            s.game_mut().scenario_refresh_priority().unwrap();
            let life = s.life(seat);
            let sacrifices = s.game().state.turn.history[seat].permanents_sacrificed;

            command(
                &mut s,
                |action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == prism),
            );
            assert!(!s.game().state.permanents[prism].as_ref().unwrap().tapped);
            assert!(s.game().state.permanents[treasure].is_some());
            assert_eq!(s.life(seat), life);

            command(
                &mut s,
                |action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == treasure),
            );
            assert!(s.game().state.permanents[treasure].is_none());
            assert!(s.game().state.exile_links.is_empty());
            assert_eq!(s.life(seat), life - 2);
            assert_eq!(s.game().state.players[seat].mana_pool.total(), 0);
            assert!(!s.game().state.permanents[prism].as_ref().unwrap().tapped);
            let returned_ref = s.game().current_object_ref(returned_card);
            let entered = s.game().state.card_to_permanent[returned_card].unwrap();
            let permanent = s.game().state.permanents[entered].as_ref().unwrap();
            assert_eq!(permanent.controller, PlayerId(1 - seat));
            assert_eq!(permanent.counter_count(CounterKind::PlusOnePlusOne), 2);
            assert!(matches!(
                &s.game().state.suspended_decision.as_ref().unwrap().decision,
                Decision::ManaChoice { player: chooser, resume: Some(parent), .. }
                    if *chooser == player && matches!(parent.as_ref(), Decision::ManaUpfront { permanent, .. } if *permanent == prism)
            ));

            command(
                &mut s,
                |action| matches!(action, Action::ChooseMode { mode: chosen, .. } if *chosen == mode),
            );
            assert_eq!(s.game().state.players[seat].mana_pool.total(), 0);
            assert!(s.game().state.permanents[prism].as_ref().unwrap().tapped);
            assert_eq!(s.life(seat), life - 2);
            assert!(matches!(
                &s.game().state.suspended_decision.as_ref().unwrap().decision,
                Decision::ManaChoice { resume: None, .. }
            ));

            command(
                &mut s,
                |action| matches!(action, Action::ChooseMode { mode: chosen, .. } if *chosen == mode),
            );
            assert_eq!(s.game().state.players[seat].mana_pool, Mana::parse(output));
            assert_eq!(s.life(seat), life - 2);
            assert!(s.game().state.suspended_decision.is_none());
            assert!(s.game().state.stack_objects.is_empty());
            assert_eq!(
                s.game().state.turn.history[seat].permanents_sacrificed,
                sacrifices + 1
            );
            assert_eq!(s.game().current_object_ref(returned_card), returned_ref);
            assert_eq!(
                s.game().state.card_to_permanent[returned_card],
                Some(entered)
            );
        }
    }
}

/// CR 106.3 — Mana abilities can be activated to produce mana used for casting.
#[test]
fn cr_106_3_casting_uses_mana_abilities() {
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Mountain", 40)]),
        3,
    );

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    s.assert_action_available(ActionType::PriorityPlayLand);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    s.assert_action_available(ActionType::PriorityCastSpell);
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));

    assert_eq!(s.zone_size(0, ZoneType::Stack), 1);

    s.pass_priority();
    s.pass_priority();

    assert!(
        !s.battlefield_permanents_named(0, "Llanowar Elves")
            .is_empty(),
        "cast creature should resolve to the battlefield"
    );

    let forest = s
        .battlefield_permanents_named(0, "Forest")
        .into_iter()
        .next()
        .expect("forest should be on battlefield");
    let tapped = s
        .game()
        .state
        .permanents
        .get(forest.0)
        .and_then(|perm| perm.as_ref())
        .map(|perm| perm.tapped)
        .unwrap_or(false);
    assert!(tapped, "land used for mana should be tapped");
}

/// CR 106.4 — Unspent mana empties from players' mana pools as steps end.
#[test]
fn cr_106_4_mana_pool_empties_between_steps() {
    let mut s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Mountain", 40)]), 9);

    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "3R");
    assert_eq!(
        s.game().state.players[0].mana_pool.mana[Color::Red as usize],
        1
    );

    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

/// Negative: a spell is not castable without enough producible mana.
#[test]
fn cr_106_negative_spell_not_available_without_mana() {
    let mut s = Scenario::new(deck(&[("Gray Ogre", 40)]), deck(&[("Mountain", 40)]), 15);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Gray Ogre");

    s.assert_action_not_available(ActionType::PriorityCastSpell);
}

/// CR 107.4b/c and 118.3: a generic cost accepts any mana; C requires
/// colorless. Exhaust all 0/1 six-type pools against one fixed pip and 0..2
/// generic. The oracle is integer resource counting, independent of payment.
#[test]
fn audit_generic_and_specific_mana_boundary_matrix() {
    use managym::state::mana::{Mana, ManaCost};

    for bits in 0_u8..64 {
        let pool = Mana {
            mana: std::array::from_fn(|slot| (bits >> slot) & 1),
            ..Default::default()
        };
        for (slot, symbol) in ["W", "U", "B", "R", "G", "C"].iter().enumerate() {
            for generic in 0..=2 {
                let cost = ManaCost::parse(&format!("{generic}{symbol}"));
                let expected = pool.mana[slot] == 1 && bits.count_ones() > generic;
                assert_eq!(
                    pool.can_pay(&cost),
                    expected,
                    "bits={bits}, cost={generic}{symbol}"
                );
                if expected {
                    let mut remaining = pool.clone();
                    remaining.pay(&cost);
                    assert_eq!(remaining.total(), pool.total() - generic as u16 - 1);
                    assert_eq!(remaining.mana[slot], 0);
                    assert!(remaining
                        .mana
                        .iter()
                        .zip(pool.mana)
                        .all(|(left, before)| *left <= before));
                }
            }
        }
    }
}

/// CR 106.6: spending restrictions survive storage and cannot pay an
/// unrelated cost. Clearing a pool loses both restricted and ordinary mana.
#[test]
fn audit_restricted_mana_never_pays_unrelated_costs() {
    use managym::state::mana::{Mana, ManaCost, ManaRestriction, SpendingContext};

    let lesson = ManaRestriction::spell(["Lesson"]);
    for slot in 0..6 {
        for eligible in [false, true] {
            let mut amounts = [0; 6];
            amounts[slot] = 1;
            let mut pool = Mana::restricted_to(lesson.clone(), amounts);
            let mut cost = ManaCost::parse("1");
            if eligible {
                cost.purpose = SpendingContext::spell(["Lesson"]);
            }
            assert_eq!(pool.can_pay(&cost), eligible);
            if eligible {
                pool.pay(&cost);
                assert_eq!(pool.total(), 0);
            }
            pool.clear();
            assert_eq!(pool, Mana::default());
        }
    }
}
