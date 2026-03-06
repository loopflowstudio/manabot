use managym::{
    agent::action::ActionSpaceKind,
    flow::{trigger::PendingTrigger, turn::StepKind},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

use super::helpers::*;

fn find_owned_card(s: &Scenario, player: usize, name: &str) -> CardId {
    let Some(index) = s
        .game()
        .state
        .cards
        .iter()
        .position(|card| card.owner == PlayerId(player) && card.name == name)
    else {
        panic!("card {name} not found for player {player}");
    };
    CardId(index)
}

fn put_owned_card_on_battlefield(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let card_id = find_owned_card(s, player, name);
    let game = s.game_mut();
    game.move_card(card_id, ZoneType::Battlefield);
    // Drain events so the engine doesn't process ETB triggers from setup
    game.state.pending_events.clear();
    game.state.card_to_permanent[card_id].expect("permanent should exist after move_card")
}

fn count_cards_named(s: &Scenario, player: usize, zone: ZoneType, name: &str) -> usize {
    s.game()
        .state
        .zones
        .zone_cards(zone, PlayerId(player))
        .iter()
        .filter(|card_id| s.game().state.cards[*card_id].name == name)
        .count()
}

fn cast_manowar_and_reach_target_choice(s: &mut Scenario) {
    let islands: Vec<CardId> = s
        .game()
        .state
        .cards
        .iter()
        .enumerate()
        .filter(|(_, card)| card.owner == PlayerId(0) && card.name == "Island")
        .map(|(idx, _)| CardId(idx))
        .take(3)
        .collect();
    assert_eq!(islands.len(), 3, "expected at least three Islands in deck");
    for island in islands {
        s.game_mut()
            .state
            .zones
            .move_card(island, PlayerId(0), ZoneType::Hand);
    }
    let manowar = find_owned_card(s, 0, "Man-o'-War");
    s.game_mut()
        .state
        .zones
        .move_card(manowar, PlayerId(0), ZoneType::Hand);

    for land_drop in 0..3 {
        s.advance_to_active_step(0, StepKind::Main);
        assert!(s.take_action_by_type(managym::agent::action::ActionType::PriorityPlayLand));
        if land_drop < 2 {
            s.pass_priority();
            s.pass_priority();
        }
    }

    assert!(s.take_action_by_type(managym::agent::action::ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
}

/// CR 603.6a — Man-o'-War's ETB trigger returns a targeted creature to hand.
#[test]
fn cr_603_6a_manowar_etb_returns_target_creature() {
    let mut s = Scenario::new(manowar_deck(), island_deck(), 81);
    cast_manowar_and_reach_target_choice(&mut s);

    let hand_before = count_cards_named(&s, 0, ZoneType::Hand, "Man-o'-War");
    s.choose_target_named("Man-o'-War");
    s.pass_priority();
    s.pass_priority();

    assert!(s.battlefield_permanents_named(0, "Man-o'-War").is_empty());
    assert_eq!(
        count_cards_named(&s, 0, ZoneType::Hand, "Man-o'-War"),
        hand_before + 1
    );
}

/// CR 603.3, 117.3d — Trigger goes on the stack and players receive priority before it resolves.
#[test]
fn cr_603_3_trigger_uses_stack_and_response_window() {
    let mut s = Scenario::new(manowar_deck(), island_deck(), 82);
    cast_manowar_and_reach_target_choice(&mut s);

    s.choose_target_named("Man-o'-War");

    assert_eq!(s.game().state.stack.len(), 1);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));

    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.game().state.stack.len(), 1);
}

/// CR 603.3b — Pending triggers are ordered APNAP when put on the stack.
#[test]
fn cr_603_3b_pending_triggers_flush_in_apnap_order() {
    let mut s = Scenario::new(manowar_deck(), manowar_deck(), 83);
    let _target = put_owned_card_on_battlefield(&mut s, 0, "Man-o'-War");

    let p0_source = find_owned_card(&s, 0, "Man-o'-War");
    let p1_source = find_owned_card(&s, 1, "Man-o'-War");
    s.game_mut().state.pending_triggers = vec![
        PendingTrigger {
            source_card: p1_source,
            ability_index: 0,
            controller: PlayerId(1),
            enqueue_order: 0,
        },
        PendingTrigger {
            source_card: p0_source,
            ability_index: 0,
            controller: PlayerId(0),
            enqueue_order: 1,
        },
    ];

    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));

    s.advance_default_action();
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
}

/// CR 608.2b — If a triggered ability's target becomes illegal, it does nothing.
#[test]
fn cr_608_2b_trigger_with_illegal_target_does_nothing() {
    let mut s = Scenario::new(manowar_deck(), island_deck(), 84);
    cast_manowar_and_reach_target_choice(&mut s);

    let hand_before = count_cards_named(&s, 0, ZoneType::Hand, "Man-o'-War");
    s.choose_target_named("Man-o'-War");

    let target = s
        .battlefield_permanents_named(0, "Man-o'-War")
        .into_iter()
        .next()
        .expect("man-o'-war should be on battlefield");
    s.game_mut().state.permanents[target]
        .as_mut()
        .expect("target permanent should still exist")
        .damage = 2;

    s.pass_priority();
    s.pass_priority();

    assert!(s.battlefield_permanents_named(0, "Man-o'-War").is_empty());
    assert_eq!(
        count_cards_named(&s, 0, ZoneType::Hand, "Man-o'-War"),
        hand_before
    );
    assert_eq!(
        count_cards_named(&s, 0, ZoneType::Graveyard, "Man-o'-War"),
        1
    );
}

/// CR 117.5, 704.3, 603.3 — SBAs are checked before pending triggers are flushed.
#[test]
fn cr_603_trigger_flush_happens_after_sba_check() {
    let mut s = Scenario::new(manowar_deck(), island_deck(), 85);

    let dying_permanent = put_owned_card_on_battlefield(&mut s, 0, "Man-o'-War");
    s.game_mut().state.permanents[dying_permanent]
        .as_mut()
        .expect("dying permanent should exist")
        .damage = 2;

    let source = find_owned_card(&s, 0, "Man-o'-War");
    s.game_mut().state.pending_triggers.push(PendingTrigger {
        source_card: source,
        ability_index: 0,
        controller: PlayerId(0),
        enqueue_order: 0,
    });

    s.pass_priority();

    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(s.game().state.stack.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
    assert_eq!(
        count_cards_named(&s, 0, ZoneType::Graveyard, "Man-o'-War"),
        1
    );
}
