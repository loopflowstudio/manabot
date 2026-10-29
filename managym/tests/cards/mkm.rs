//! Murders at Karlov Manor (`src/cardsets/mkm.rs`).
use crate::support::{card_of, cast_available, counters, put_permanent, resolve, ur_lessons_setup};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

#[test]
fn proft_etb_draw_and_combat_counts_at_resolution() {
    let mut s = ur_lessons_setup();
    put_permanent(&mut s, 0, "Island");
    put_permanent(&mut s, 0, "Island");
    let gran = put_permanent(&mut s, 0, "Gran-Gran");
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 1;
    let card = cast_available(&mut s, "Proft's Eidetic Memory");
    let before = s.zone_size(0, ZoneType::Hand);
    resolve(&mut s); // permanent enters, ETB on stack
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Battlefield)
    );
    resolve(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 1);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(gran)));
    // Further draws in response count when the ability resolves (CR 608.2h).
    s.game_mut().draw_cards(PlayerId(0), 2);
    resolve(&mut s);
    assert_eq!(counters(&s, gran), 3);
}

#[test]
fn proft_combat_requires_two_draws_and_own_turn() {
    for draws in [0, 1] {
        let mut s = ur_lessons_setup();
        put_permanent(&mut s, 0, "Proft's Eidetic Memory");
        put_permanent(&mut s, 0, "Gran-Gran");
        s.pass_priority();
        resolve(&mut s); // ETB
        s.game_mut().state.turn.cards_drawn_this_turn[0] = draws;
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
        assert!(s.game().state.stack_objects.is_empty());
    }
    let mut s = ur_lessons_setup();
    put_permanent(&mut s, 1, "Proft's Eidetic Memory");
    put_permanent(&mut s, 1, "Gran-Gran");
    s.pass_priority();
    resolve(&mut s);
    s.game_mut().state.turn.cards_drawn_this_turn[1] = 3;
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn proft_hand_limit_only_while_controlled_and_legend_rule_for_both_card_types() {
    for name in ["Gran-Gran", "Proft's Eidetic Memory"] {
        let mut s = ur_lessons_setup();
        let first = put_permanent(&mut s, 0, name);
        let second = put_permanent(&mut s, 0, name);
        let first_card = card_of(&s, first);
        let second_card = card_of(&s, second);
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::LegendRule);
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::SelectCard { card, .. } if *card == second_card))
            .unwrap();
        s.step_action(index);
        assert_eq!(
            s.game().state.zones.zone_of(first_card),
            Some(ZoneType::Graveyard)
        );
        assert_eq!(
            s.game().state.zones.zone_of(second_card),
            Some(ZoneType::Battlefield)
        );
    }
    let mut s = ur_lessons_setup();
    let proft = put_permanent(&mut s, 0, "Proft's Eidetic Memory");
    s.pass_priority();
    resolve(&mut s);
    s.game_mut().draw_cards(PlayerId(0), 5);
    let before = s.zone_size(0, ZoneType::Hand);
    assert!(before > 7);
    s.advance_to_active_step(0, StepKind::End);
    resolve(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before);
    let card = card_of(&s, proft);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::End);
    resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
}

#[test]
fn proft_has_no_effect_if_its_only_target_leaves() {
    let mut s = ur_lessons_setup();
    put_permanent(&mut s, 0, "Proft's Eidetic Memory");
    let gran = put_permanent(&mut s, 0, "Gran-Gran");
    s.pass_priority();
    resolve(&mut s);
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 3;
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.choose_target(Target::Permanent(gran)));
    let card = card_of(&s, gran);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let new = s.game().state.card_to_permanent[card].unwrap();
    resolve(&mut s);
    assert_eq!(counters(&s, new), 0);
}
