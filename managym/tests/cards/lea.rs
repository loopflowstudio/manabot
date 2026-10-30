//! Tests for cards filed in `src/cardsets/lea.rs`, in card order.

use crate::support::{
    game::{deck, deck_of, five_color_setup, ready, Scenario},
    inspect::permanent_named,
    play::{activate_named, enter_and_settle},
};
use managym::{
    agent::action::{Action, ActionType},
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

/// CR 121.1 — Resolving a draw spell moves cards from library to hand.
#[test]
fn ancestral_recall_draws_three() {
    let mut s = Scenario::new(
        deck_of("Island", "Ancestral Recall"),
        deck(&[("Island", 40)]),
        1211,
    );

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Ancestral Recall");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));

    // Spell is on the stack; snapshot zone sizes before resolution.
    s.assert_zone_size(0, ZoneType::Stack, 1);
    let hand_before = s.zone_size(0, ZoneType::Hand);
    let library_before = s.zone_size(0, ZoneType::Library);

    s.pass_priority();
    s.pass_priority();

    s.assert_zone_size(0, ZoneType::Stack, 0);
    s.assert_zone_size(0, ZoneType::Hand, hand_before + 3);
    s.assert_zone_size(0, ZoneType::Library, library_before - 3);
    // The spell itself goes to the graveyard.
    s.assert_zone_size(0, ZoneType::Graveyard, 1);
}

/// CR 704.5c — Drawing from an empty library loses the game, including
/// draws caused by spell effects.
#[test]
fn ancestral_recall_decking_loses_game() {
    let mut s = Scenario::new(
        deck_of("Island", "Ancestral Recall"),
        deck(&[("Island", 40)]),
        1212,
    );

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Ancestral Recall");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));

    // Empty player 0's library so the draw spell decks them.
    let library: Vec<_> = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .to_vec();
    for card in library {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(0), ZoneType::Graveyard);
    }

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    s.pass_priority();
    s.pass_priority();

    s.assert_game_over();
    s.assert_winner(1);
}

/// CR 405 — Counterspell can counter a draw spell on the stack; no cards
/// are drawn and the countered spell goes to the graveyard.
#[test]
fn counterspell_counters_ancestral_recall() {
    let mut s = Scenario::new(
        deck_of("Island", "Ancestral Recall"),
        deck_of("Island", "Counterspell"),
        1214,
    );

    s.advance_to_active_step(0, StepKind::Main);
    s.force_permanent_on_battlefield(0, "Island");
    s.force_permanent_on_battlefield(1, "Island");
    s.force_permanent_on_battlefield(1, "Island");
    s.force_card_in_hand(0, "Ancestral Recall");
    s.force_card_in_hand(1, "Counterspell");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let hand_before = s.zone_size(0, ZoneType::Hand);
    let library_before = s.zone_size(0, ZoneType::Library);
    s.pass_priority();

    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let recall_card = s
        .game()
        .state
        .stack_objects
        .iter()
        .find_map(|obj| match obj {
            managym::state::stack_object::StackObject::Spell(spell) => Some(spell.card),
            _ => None,
        })
        .expect("recall should be on stack");
    assert!(s.choose_target(Target::StackSpell(recall_card)));

    s.pass_priority();
    s.pass_priority();

    // Recall was countered: no cards drawn, recall in the graveyard.
    s.assert_zone_size(0, ZoneType::Stack, 0);
    s.assert_zone_size(0, ZoneType::Hand, hand_before);
    s.assert_zone_size(0, ZoneType::Library, library_before);
    assert_eq!(
        s.game().state.zones.zone_of(recall_card),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn an_announced_tap_cannot_be_used_to_fund_itself() {
    use managym::state::{ability::Effect, card::ActivatedAbilityDefinition, mana::ManaCost};

    for tap_source in [false, true] {
        let mut s = five_color_setup(&["Llanowar Elves"]);
        let elves_card = enter_and_settle(&mut s, "Llanowar Elves");
        ready(&mut s, "Llanowar Elves");
        let elves = permanent_named(&s, "Llanowar Elves");
        s.game_mut().state.cards[elves_card]
            .activated_abilities
            .push(ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("1"),
                tap_source,
                sacrifice_source: true,
                effects: vec![Effect::DrawCards { count: 1 }],
                ..Default::default()
            });
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(
            s.action_space().actions.iter().any(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == elves)
            ),
            !tap_source
        );
        if !tap_source {
            activate_named(&mut s, "Llanowar Elves", 0);
            assert!(s.game().state.permanents[elves].is_none());
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            assert_eq!(
                s.game().state.stack_objects.len(),
                1,
                "the nonmana parent ability uses the stack"
            );
        }
    }
}
