// temur_cards.rs
// Directed Temur payment, graveyard casting, and copied-token rules.
use std::collections::BTreeMap;

use super::{
    elemental_cards::{card, cast, pass_until, permanent, settle, setup},
    helpers::Scenario,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        game_object::{CardId, PlayerId},
        mana::Mana,
        zone::ZoneType,
    },
};

fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let index = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(index);
}
fn enter(s: &mut Scenario, name: &str) -> CardId {
    let id = card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    if !s.game().state.pending_triggers.is_empty() {
        s.pass_priority();
    }
    settle(s);
    id
}
fn activate(s: &mut Scenario, name: &str, index: usize) {
    let source = permanent(s, name);
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |action| matches!(action, Action::ActivateAbility { permanent, ability_index, .. } if *permanent == source && *ability_index == index),
    );
}
fn ready(s: &mut Scenario, name: &str) {
    let source = permanent(s, name);
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.game_mut().scenario_refresh_priority().unwrap();
}
fn can_cast(s: &Scenario, id: CardId) -> bool {
    s.action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { card, .. } if *card == id))
}

#[test]
fn herbalist_restriction_survives_color_selection_and_cannot_pay_abilities() {
    let mut s = setup(&[
        "Hermitic Herbalist",
        "Firebending Lesson",
        "Lightning Strike",
        "Prophetic Prism",
    ]);
    enter(&mut s, "Hermitic Herbalist");
    enter(&mut s, "Prophetic Prism");
    ready(&mut s, "Hermitic Herbalist");
    s.set_player_mana_pool(0, "0");
    activate(&mut s, "Hermitic Herbalist", 1);
    super::elemental_cards::assert_choice_contract(&s);
    // The two-color choice is unordered; red/red is one distinct output.
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 12, .. }));
    assert_eq!(s.game().state.players[0].mana_pool.lesson[3], 2);
    let lesson = card(&s, 0, "Firebending Lesson");
    let strike = card(&s, 0, "Lightning Strike");
    s.game_mut().move_card(lesson, ZoneType::Hand);
    s.game_mut().move_card(strike, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, lesson));
    assert!(!can_cast(&s, strike));
    let prism = permanent(&s, "Prophetic Prism");
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism)));
}

#[test]
fn prism_consumes_generic_before_adding_chosen_color() {
    let mut s = setup(&["Prophetic Prism"]);
    enter(&mut s, "Prophetic Prism");
    s.set_player_mana_pool(0, "1");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn prism_payment_preserves_the_spells_required_green_mana() {
    for lands in [
        ["Forest", "Island", "Island", "Mountain"],
        ["Island", "Island", "Mountain", "Forest"],
    ] {
        let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
        for land in lands {
            enter(&mut s, land);
        }
        enter(&mut s, "Prophetic Prism");
        s.set_player_mana_pool(0, "0");
        cast(&mut s, "Gnarled Professor");
        activate(&mut s, "Prophetic Prism", 0);
        assert!(
            !s.game().state.permanents[permanent(&s, "Forest")]
                .as_ref()
                .unwrap()
                .tapped
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        // Green is the only output that leaves the announced GG spell payable.
        assert_eq!(s.action_space().actions.len(), 1);
        choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        assert!(s.game().state.stack_objects.iter().any(|object| matches!(
            object,
            managym::state::stack_object::StackObject::Spell(spell)
                if s.game().state.cards[spell.card].name == "Gnarled Professor"
        )));
    }
}

#[test]
fn prism_payment_preserves_required_blue_already_in_the_pool() {
    let mut s = setup(&["Prophetic Prism", "Gnarled Professor", "Counterspell"]);
    enter(&mut s, "Forest");
    enter(&mut s, "Prophetic Prism");
    let professor = cast(&mut s, "Gnarled Professor");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    s.set_player_mana_pool(0, "U");
    cast(&mut s, "Counterspell");
    choose(&mut s, |a| {
        matches!(a, Action::ChooseTarget {
        target: managym::state::game_object::Target::StackSpell(id), ..
    } if *id == professor)
    });
    activate(&mut s, "Prophetic Prism", 0);
    // Spend the Forest's G on Prism, retaining the pooled U for Counterspell.
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert_eq!(s.game().state.stack_objects.len(), 2);
}

#[test]
fn guide_grants_tap_mana_with_sickness_and_removes_it_on_departure() {
    let mut s = setup(&["Great Divide Guide", "Hermitic Herbalist"]);
    let guide = enter(&mut s, "Great Divide Guide");
    s.set_player_mana_pool(0, "0");
    s.game_mut().scenario_refresh_priority().unwrap();
    let source = permanent(&s, "Great Divide Guide");
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == source)));
    ready(&mut s, "Great Divide Guide");
    activate(&mut s, "Great Divide Guide", 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
    let forest = enter(&mut s, "Forest");
    let land = s.game().state.card_to_permanent[forest].unwrap();
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == land)));
    s.game_mut().move_card(guide, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == land)));
}

#[test]
fn iroh_casts_lessons_for_one_and_exiles_on_resolution_or_bounce() {
    for bounce in [false, true] {
        let mut s = setup(&["Iroh, Grand Lotus", "Waterbending Lesson"]);
        enter(&mut s, "Iroh, Grand Lotus");
        let lesson = card(&s, 0, "Waterbending Lesson");
        s.game_mut().move_card(lesson, ZoneType::Graveyard);
        s.set_player_mana_pool(0, "1");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(can_cast(&s, lesson));
        super::elemental_cards::assert_choice_contract(&s);
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == lesson),
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        if bounce {
            s.game_mut().move_card(lesson, ZoneType::Hand);
        } else {
            pass_until(&mut s, ActionSpaceKind::Waterbend);
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
            settle(&mut s);
        }
        assert_eq!(s.game().state.zones.zone_of(lesson), Some(ZoneType::Exile));
    }
}

#[test]
fn iroh_preserves_sorcery_timing_and_nonlesson_printed_cost() {
    let mut s = setup(&[
        "Iroh, Grand Lotus",
        "Lightning Strike",
        "Waterbending Lesson",
    ]);
    enter(&mut s, "Iroh, Grand Lotus");
    let strike = card(&s, 0, "Lightning Strike");
    let lesson = card(&s, 0, "Waterbending Lesson");
    s.game_mut().move_card(strike, ZoneType::Graveyard);
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.set_player_mana_pool(0, "1");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, strike));
    s.set_player_mana_pool(0, "1R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, strike));
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    s.game_mut().state.priority.holder = PlayerId(0);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, strike));
    assert!(!can_cast(&s, lesson));
}

#[test]
fn waterbend_lesson_draws_before_optional_payment_and_taps_without_mana() {
    let mut s = setup(&[
        "Waterbending Lesson",
        "Hermitic Herbalist",
        "Prophetic Prism",
    ]);
    enter(&mut s, "Hermitic Herbalist");
    enter(&mut s, "Prophetic Prism");
    let before = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    let spell = cast(&mut s, "Waterbending Lesson");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        before + 3
    );
    super::elemental_cards::assert_choice_contract(&s);
    for name in ["Hermitic Herbalist", "Prophetic Prism"] {
        let source = permanent(&s, name);
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == source),
        );
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        before + 3
    );
}

#[test]
fn ran_and_shaw_copy_is_nonlegendary_uncast_and_has_printed_abilities() {
    let mut s = setup(&["Ran and Shaw", "Waterbending Lesson"]);
    for _ in 0..3 {
        let lesson = card(&s, 0, "Waterbending Lesson");
        s.game_mut().move_card(lesson, ZoneType::Graveyard);
    }
    cast(&mut s, "Ran and Shaw");
    settle(&mut s);
    let dragons = s.battlefield_permanents_named(0, "Ran and Shaw");
    assert_eq!(dragons.len(), 2);
    let token = dragons
        .iter()
        .find_map(|id| {
            let p = s.game().state.permanents[*id].as_ref().unwrap();
            s.game().state.cards[p.card].is_token.then_some(p)
        })
        .unwrap();
    assert!(!token.was_cast);
    let definition = &s.game().state.cards[token.card];
    assert!(definition.supertypes.is_empty());
    assert!(definition.keywords.flying);
    assert_eq!(definition.abilities.len(), 2);
    assert_eq!(definition.activated_abilities.len(), 1);
    assert_eq!(definition.mana_cost.as_ref().unwrap().mana_value, 5);
    activate(&mut s, "Ran and Shaw", 0);
    settle(&mut s);
    for id in dragons {
        assert_eq!(s.game().effective_power(id), 6);
    }
}

#[test]
fn ran_and_shaw_requires_a_cast_and_rechecks_graveyard_at_resolution() {
    for mode in 0..3 {
        let mut s = setup(&["Ran and Shaw", "Waterbending Lesson"]);
        let mut lessons = Vec::new();
        for _ in 0..3 {
            let lesson = card(&s, 0, "Waterbending Lesson");
            s.game_mut().move_card(lesson, ZoneType::Graveyard);
            lessons.push(lesson);
        }
        if mode == 0 {
            enter(&mut s, "Ran and Shaw");
        } else {
            cast(&mut s, "Ran and Shaw");
            // Resolve the creature but leave its entry trigger on the stack.
            for _ in 0..10 {
                if s.battlefield_permanents_named(0, "Ran and Shaw").len() == 1 {
                    break;
                }
                s.pass_priority();
            }
            assert_eq!(s.battlefield_permanents_named(0, "Ran and Shaw").len(), 1);
            super::elemental_cards::assert_choice_contract(&s);
            if mode == 1 {
                s.game_mut().move_card(lessons[0], ZoneType::Exile);
            }
            if mode == 2 {
                let source = permanent(&s, "Ran and Shaw");
                s.game_mut().state.permanents[source]
                    .as_mut()
                    .unwrap()
                    .plus1_counters = 3;
                let source_card = s.game().state.permanents[source].as_ref().unwrap().card;
                s.game_mut().move_card(source_card, ZoneType::Graveyard);
            }
            settle(&mut s);
        }
        let dragons = s.battlefield_permanents_named(0, "Ran and Shaw");
        assert_eq!(dragons.len(), 1);
        let dragon = s.game().state.permanents[dragons[0]].as_ref().unwrap();
        assert_eq!(s.game().state.cards[dragon.card].is_token, mode == 2);
        assert_eq!(s.game().effective_power(dragons[0]), 4);
    }
}

#[test]
fn flashback_is_exiled_when_countered_after_iroh_leaves() {
    let mut s = setup(&["Iroh, Grand Lotus", "Waterbending Lesson", "Counterspell"]);
    let iroh = enter(&mut s, "Iroh, Grand Lotus");
    let lesson = card(&s, 0, "Waterbending Lesson");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == lesson),
    );
    s.game_mut().move_card(iroh, ZoneType::Graveyard);
    let counter = card(&s, 1, "Counterspell");
    s.game_mut().move_card(counter, ZoneType::Hand);
    s.pass_priority();
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == counter),
    );
    choose(
        &mut s,
        |a| matches!(a, Action::ChooseTarget { target: managym::state::game_object::Target::StackSpell(id), .. } if *id == lesson),
    );
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(lesson), Some(ZoneType::Exile));
}

#[test]
fn waterbend_cannot_partially_pay_then_discard() {
    let mut s = setup(&["Waterbending Lesson", "Prophetic Prism"]);
    enter(&mut s, "Prophetic Prism");
    cast(&mut s, "Waterbending Lesson");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    s.set_player_mana_pool(0, "1");
    let prism = permanent(&s, "Prophetic Prism");
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == prism),
    );
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn a_committed_nonlesson_cost_excludes_herbalists_restricted_activation() {
    let mut s = setup(&["Hermitic Herbalist", "Lightning Strike"]);
    enter(&mut s, "Hermitic Herbalist");
    ready(&mut s, "Hermitic Herbalist");
    s.set_player_mana_pool(0, "R");
    cast(&mut s, "Lightning Strike");
    choose(&mut s, |a| {
        matches!(
            a,
            Action::ChooseTarget {
                target: managym::state::game_object::Target::Player(PlayerId(1)),
                ..
            }
        )
    });
    let herbalist = permanent(&s, "Hermitic Herbalist");
    assert!(!s.action_space().actions.iter().any(|a| matches!(a, Action::ActivateAbility { permanent, ability_index: 1, .. } if *permanent == herbalist)));
    super::elemental_cards::assert_choice_contract(&s);
    activate(&mut s, "Hermitic Herbalist", 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

fn lesson_payment(names: &[&str], pool: &str) -> (Scenario, CardId, usize) {
    let mut s = setup(names);
    for name in names.iter().filter(|name| **name != "Waterbending Lesson") {
        enter(&mut s, name);
        if *name == "Great Divide Guide" {
            ready(&mut s, name);
        }
    }
    let before = s.zone_size(0, ZoneType::Hand);
    let spell = cast(&mut s, "Waterbending Lesson");
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    s.set_player_mana_pool(0, pool);
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
    (s, spell, before)
}

#[test]
fn optional_lesson_all_mana_mixed_and_all_taps_keep_the_same_continuation() {
    for taps in 0..=2 {
        let (mut s, spell, before) = lesson_payment(
            &[
                "Waterbending Lesson",
                "Prophetic Prism",
                "Hermitic Herbalist",
            ],
            ["2", "1", "0"][taps],
        );
        super::elemental_cards::assert_choice_contract(&s);
        for name in ["Prophetic Prism", "Hermitic Herbalist"].iter().take(taps) {
            let source = permanent(&s, name);
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == source),
            );
            super::elemental_cards::assert_choice_contract(&s);
        }
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
        assert_eq!(
            s.game().state.zones.zone_of(spell),
            Some(ZoneType::Graveyard)
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn optional_lesson_without_resources_only_declines_then_discards_one_drawn_card() {
    let (mut s, spell, before) = lesson_payment(&["Waterbending Lesson"], "0");
    assert_eq!(
        s.action_space().actions,
        vec![Action::Decline {
            player: PlayerId(0)
        }]
    );
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    super::elemental_cards::assert_choice_contract(&s);
    let other = managym::agent::observation::Observation::for_player(s.game(), PlayerId(1));
    assert!(!other
        .opponent_cards
        .iter()
        .any(|c| c.zone == ZoneType::Hand));
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 2);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn optional_lesson_does_not_offer_a_tap_without_a_complete_payment() {
    let (s, _, _) = lesson_payment(&["Waterbending Lesson", "Prophetic Prism"], "0");
    assert_eq!(
        s.action_space().actions,
        vec![Action::Decline {
            player: PlayerId(0)
        }]
    );
}

#[test]
fn committed_optional_payment_resumes_after_granted_chosen_color_mana() {
    let (mut s, spell, before) = lesson_payment(
        &[
            "Waterbending Lesson",
            "Great Divide Guide",
            "Prophetic Prism",
        ],
        "0",
    );
    let prism = permanent(&s, "Prophetic Prism");
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == prism),
    );
    activate(&mut s, "Great Divide Guide", 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    super::elemental_cards::assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    super::elemental_cards::assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    for name in ["Prophetic Prism", "Great Divide Guide"] {
        assert!(
            s.game().state.permanents[permanent(&s, name)]
                .as_ref()
                .unwrap()
                .tapped
        );
    }
}

#[test]
fn optional_decline_retains_mana_activation_costs_and_chosen_output() {
    let (mut s, _, before) = lesson_payment(&["Waterbending Lesson", "Prophetic Prism"], "1");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    super::elemental_cards::assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 2);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    assert!(
        s.game().state.permanents[permanent(&s, "Prophetic Prism")]
            .as_ref()
            .unwrap()
            .tapped
    );
}

/// CR 605.3a/b, 118.3: one Treasure can pay Prism's activation cost;
/// sacrifice precedes the chosen-color output, and neither uses the stack.
/// Kept as an explicit known-failure probe for the separate payment work.
#[test]
#[ignore = "known payment defect: offered Prism activation rejects Treasure-funded cost"]
fn audit_prism_treasure_empty_pool_activation() {
    let deck = BTreeMap::from([
        ("Mountain".to_string(), 8),
        ("Prophetic Prism".to_string(), 1),
    ]);
    let mut s = Scenario::new(deck.clone(), deck, 177_605);
    s.advance_to_active_step(0, managym::flow::turn::StepKind::Main);
    // Only the activation is under test; do not introduce Prism's ETB draw.
    s.force_permanent_on_battlefield(0, "Prophetic Prism");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, "Prophetic Prism", 0);
    choose(&mut s, |action| {
        matches!(action, Action::ChooseMode { mode: 1, .. })
    });
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
    assert!(
        s.game().state.permanents[permanent(&s, "Prophetic Prism")]
            .as_ref()
            .unwrap()
            .tapped
    );
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn priority_prism_activation_can_use_a_treasure_for_its_upfront_cost() {
    let mut s = setup(&["Prophetic Prism", "Tireless Provisioner"]);
    enter(&mut s, "Prophetic Prism");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    s.game_mut().scenario_refresh_priority().unwrap();
    let prism = permanent(&s, "Prophetic Prism");
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == prism))
        .expect("Prism is offered when Treasure can fund its upfront cost");
    s.step_action(index);
}

#[test]
fn prism_upfront_payment_preserves_professors_second_green() {
    for sources in [
        ["Forest", "Island", "Prophetic Prism", "Island", "Mountain"],
        ["Mountain", "Island", "Prophetic Prism", "Island", "Forest"],
    ] {
        let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
        for name in sources {
            enter(&mut s, name);
        }
        s.set_player_mana_pool(0, "0");
        let professor = cast(&mut s, "Gnarled Professor");
        assert!(s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::ActivateAbility { .. })));
        activate(&mut s, "Prophetic Prism", 0);
        assert!(
            !s.game().state.permanents[permanent(&s, "Forest")]
                .as_ref()
                .unwrap()
                .tapped
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        assert_eq!(s.action_space().actions.len(), 1);
        choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(
            s.game().state.zones.zone_of(professor),
            Some(ZoneType::Stack)
        );
        settle(&mut s);
        assert!(!s
            .battlefield_permanents_named(0, "Gnarled Professor")
            .is_empty());
    }
}

#[test]
fn prism_payment_offers_execute_and_undo_without_mutating_sibling_forks() {
    use managym::{
        search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver},
        Game,
    };

    fn complete_every_offer(game: &mut Game, spell: CardId, depth: usize) {
        if game.state.zones.zone_of(spell) == Some(ZoneType::Stack) {
            return;
        }
        assert!(
            depth > 0,
            "payment must complete within its three decisions"
        );
        let driver = ClonePlusUndoDriver::default();
        let root = witness(game);
        let actions = game.action_space().unwrap().actions.clone();
        assert!(!actions.is_empty());
        for action_index in 0..actions.len() {
            let mark = driver.mark(game);
            driver
                .apply(
                    game,
                    BenchCommand {
                        action_index,
                        expected_action_hash: None,
                        expected_state_hash: None,
                    },
                )
                .unwrap();
            complete_every_offer(game, spell, depth - 1);
            driver.rollback(game, mark);
            assert_eq!(witness(game), root);
        }
    }

    for forests in 0..=2 {
        let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
        for _ in 0..forests {
            enter(&mut s, "Forest");
        }
        for name in ["Island", "Prophetic Prism", "Island", "Mountain"] {
            enter(&mut s, name);
        }
        s.set_player_mana_pool(0, "0");
        let professor = card(&s, 0, "Gnarled Professor");
        s.game_mut().move_card(professor, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&s, professor), forests > 0);
        if forests == 0 {
            continue;
        }
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == professor),
        );
        let root = witness(s.game());
        let driver = ClonePlusUndoDriver::default();
        let mut branch = driver.fork_exact(s.game());
        complete_every_offer(&mut branch, professor, 3);
        assert_eq!(witness(s.game()), root);
        assert_eq!(witness(&branch), root);
    }
}

#[test]
fn prism_upfront_payment_can_preserve_mana_already_in_the_pool() {
    let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
    for name in ["Prophetic Prism", "Island", "Island", "Mountain"] {
        enter(&mut s, name);
    }
    s.set_player_mana_pool(0, "G");
    let professor = cast(&mut s, "Gnarled Professor");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(professor),
        Some(ZoneType::Stack)
    );
}

#[test]
fn prism_upfront_payment_search_preserves_a_required_pooled_white() {
    let mut s = setup(&["Prophetic Prism", "Serra Angel"]);
    enter(&mut s, "Prophetic Prism");
    s.set_player_mana_pool(0, "WUUUU");
    let angel = cast(&mut s, "Serra Angel");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("WUUU"));
    assert_eq!(s.action_space().actions.len(), 1);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.zones.zone_of(angel), Some(ZoneType::Stack));
}

// Exhaustive tiny-board proof: expected accounting below is printed-cost
// arithmetic, not a second call to the feasibility model.
#[test]
fn nested_mana_every_offer_completes_replays_and_rolls_back() {
    use managym::{
        search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver},
        state::game_object::PermanentId,
        Game,
    };

    struct Expected {
        spell: Option<CardId>,
        prism: PermanentId,
        treasure: PermanentId,
    }

    fn explore(
        game: &mut Game,
        root: &Game,
        expected: &Expected,
        prefix: &mut Vec<usize>,
        nodes: &mut usize,
    ) {
        *nodes += 1;
        assert!(*nodes <= 256, "tiny board node bound");
        assert!(prefix.len() <= 8, "one Treasure, one filter, one spell");
        let driver = ClonePlusUndoDriver::default();
        let complete = if let Some(spell) = expected.spell {
            game.state.zones.zone_of(spell) == Some(ZoneType::Stack)
        } else {
            game.state.suspended_decision.is_none()
        };
        if complete {
            // Treasure contributes exactly one; Prism spends one then adds one.
            // Goblin spends the remaining R. Mana abilities add no stack object.
            assert!(game.state.permanents[expected.treasure].is_none());
            let prism_tapped = game.state.permanents[expected.prism]
                .as_ref()
                .unwrap()
                .tapped;
            assert_eq!(
                game.state.players[0].mana_pool.total(),
                u16::from(expected.spell.is_none())
            );
            assert_eq!(
                game.state.stack_objects.len(),
                usize::from(expected.spell.is_some())
            );
            if expected.spell.is_none() {
                assert!(prism_tapped);
            }
            let mut replay = driver.fork_exact(root);
            for action_index in prefix.iter().copied() {
                driver
                    .apply(
                        &mut replay,
                        BenchCommand {
                            action_index,
                            expected_action_hash: None,
                            expected_state_hash: None,
                        },
                    )
                    .unwrap();
            }
            assert_eq!(
                witness(&replay),
                witness(game),
                "complete Command prefix replay"
            );
            return;
        }
        let before = witness(game);
        let actions = game.action_space().unwrap().actions.clone();
        assert!(
            !actions.is_empty(),
            "an offered transition must have a continuation"
        );
        for action_index in 0..actions.len() {
            let mark = driver.mark(game);
            driver
                .apply(
                    game,
                    BenchCommand {
                        action_index,
                        expected_action_hash: None,
                        expected_state_hash: None,
                    },
                )
                .unwrap();
            prefix.push(action_index);
            explore(game, root, expected, prefix, nodes);
            prefix.pop();
            driver.rollback(game, mark);
            assert_eq!(
                witness(game),
                before,
                "exact pools, objects, queues, RNG and offers"
            );
        }
    }

    for spell in [false, true] {
        let mut s = setup(&["Prophetic Prism", "Tireless Provisioner", "Raging Goblin"]);
        enter(&mut s, "Prophetic Prism");
        s.game_mut().create_token("Treasure", PlayerId(0), false);
        let prism = permanent(&s, "Prophetic Prism");
        let treasure = permanent(&s, "Treasure");
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        let spell = if spell {
            Some(cast(&mut s, "Raging Goblin"))
        } else {
            activate(&mut s, "Prophetic Prism", 0);
            // The upfront cost precedes both tap and output.
            assert!(!s.game().state.permanents[prism].as_ref().unwrap().tapped);
            assert!(s.game().state.permanents[treasure].is_some());
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            None
        };
        let root = s.game().clone();
        let original = witness(&root);
        let mut branch = root.clone();
        let mut nodes = 0;
        explore(
            &mut branch,
            &root,
            &Expected {
                spell,
                prism,
                treasure,
            },
            &mut Vec::new(),
            &mut nodes,
        );
        assert!(nodes >= 10, "visible color alternatives were explored");
        assert_eq!(witness(&root), original, "sibling isolation");
    }
}

#[test]
fn filters_require_a_real_seed_and_do_not_reuse_a_reserved_source() {
    for seed in [false, true] {
        let mut s = setup(&["Prophetic Prism", "Tireless Provisioner"]);
        // Use two distinct printed copies; identical output does not make
        // their tap and sacrifice identities interchangeable.
        enter(&mut s, "Prophetic Prism");
        enter(&mut s, "Prophetic Prism");
        if seed {
            s.game_mut().create_token("Treasure", PlayerId(0), false);
        }
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        let filters: Vec<_> = s
            .action_space()
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::ActivateAbility { permanent, .. }
                    if s.game().state.cards
                        [s.game().state.permanents[*permanent].as_ref().unwrap().card]
                        .name
                        == "Prophetic Prism" =>
                {
                    Some(*permanent)
                }
                _ => None,
            })
            .collect();
        assert_eq!(filters.len(), if seed { 2 } else { 0 });
        if seed {
            choose(
                &mut s,
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == filters[0]),
            );
            // The unpaid filter is reserved, and the other seedless filter
            // cannot spend Treasure and leave the original payment unpaid.
            let actions = &s.action_space().actions;
            assert!(!actions.iter().any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == filters[0])));
            activate(&mut s, "Treasure", 0);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
            assert!(s.game().state.stack_objects.is_empty());
        }
    }
}

#[test]
fn prism_pool_alternatives_keep_restrictions_and_expiring_mana_classes() {
    for blue in [2, 3] {
        let mut s = setup(&["Prophetic Prism", "Serra Angel"]);
        enter(&mut s, "Prophetic Prism");
        s.set_player_mana_pool(0, &format!("W{}", "U".repeat(blue)));
        s.game_mut().state.players[0].combat_mana_pool = Mana::parse("U");
        let angel = card(&s, 0, "Serra Angel");
        s.game_mut().move_card(angel, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        // Five total mana is necessary: the filter conserves total mana.
        assert_eq!(can_cast(&s, angel), blue == 3);
        if blue == 3 {
            choose(
                &mut s,
                |a| matches!(a, Action::CastSpell { card, .. } if *card == angel),
            );
            activate(&mut s, "Prophetic Prism", 0);
            assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("WUUU"));
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        }
    }
}

#[test]
fn filtering_cannot_launder_restricted_mana_from_a_mixed_pool() {
    for unrestricted in [false, true] {
        let mut s = setup(&["Prophetic Prism"]);
        enter(&mut s, "Prophetic Prism");
        s.set_player_mana_pool(0, if unrestricted { "U" } else { "0" });
        s.game_mut().state.players[0].mana_pool.lesson[1] = 1;
        s.game_mut().scenario_refresh_priority().unwrap();
        let prism = permanent(&s, "Prophetic Prism");
        assert_eq!(s.action_space().actions.iter().any(|action|
            matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == prism)
        ), unrestricted);
        if unrestricted {
            activate(&mut s, "Prophetic Prism", 0);
            assert_eq!(s.game().state.players[0].mana_pool.mana[1], 0);
            assert_eq!(s.game().state.players[0].mana_pool.lesson[1], 1);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            assert_eq!(s.game().state.players[0].mana_pool.mana[4], 1);
            assert_eq!(s.game().state.players[0].mana_pool.lesson[1], 1);
        }
    }
}

#[test]
fn waterbend_and_mana_taps_have_distinct_bonus_accounting() {
    use managym::state::{card::TriggeredManaAbility, predicate::CardPredicate};

    for with_artifact in [false, true] {
        let names = if with_artifact {
            vec![
                "Waterbending Lesson",
                "Hermitic Herbalist",
                "Prophetic Prism",
            ]
        } else {
            vec!["Waterbending Lesson", "Hermitic Herbalist"]
        };
        let mut s = setup(&names);
        for name in names.iter().filter(|name| **name != "Waterbending Lesson") {
            enter(&mut s, name);
        }
        ready(&mut s, "Hermitic Herbalist");
        let herbalist = permanent(&s, "Hermitic Herbalist");
        let source_card = s.game().state.permanents[herbalist].as_ref().unwrap().card;
        s.game_mut().state.cards[source_card]
            .triggered_mana_abilities
            .push(TriggeredManaAbility {
                predicate: CardPredicate::creature(),
                mana: Mana::parse("G"),
            });
        cast(&mut s, "Waterbending Lesson");
        if s.action_space().kind == ActionSpaceKind::PayOrNot {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        s.set_player_mana_pool(0, "0");
        pass_until(&mut s, ActionSpaceKind::Waterbend);
        assert_eq!(
            s.action_space().actions.iter().any(
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == herbalist)
            ),
            with_artifact
        );
        if with_artifact {
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == herbalist),
            );
            assert_eq!(
                s.game().state.players[0].mana_pool.total(),
                0,
                "waterbend never fires a mana bonus"
            );
            let prism = permanent(&s, "Prophetic Prism");
            assert!(!s.action_space().actions.iter().any(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism)
            ));
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == prism),
            );
        } else {
            activate(&mut s, "Hermitic Herbalist", 0);
            assert_eq!(
                s.game().state.players[0].mana_pool,
                Mana::parse("G"),
                "exactly one immediate tap bonus"
            );
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("GU"));
        }
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn nested_mana_nonmana_life_cost_is_reserved_and_paid_once() {
    for life in [1, 3] {
        let mut s = setup(&["Prophetic Prism", "Tireless Provisioner"]);
        enter(&mut s, "Prophetic Prism");
        s.game_mut().create_token("Treasure", PlayerId(0), false);
        let treasure = permanent(&s, "Treasure");
        let source_card = s.game().state.permanents[treasure].as_ref().unwrap().card;
        s.game_mut().state.cards[source_card].activated_abilities[0].life_cost = 2;
        s.game_mut().state.players[0].life = life;
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        let prism = permanent(&s, "Prophetic Prism");
        assert_eq!(
            s.action_space().actions.iter().any(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism)
            ),
            life >= 2
        );
        if life >= 2 {
            activate(&mut s, "Prophetic Prism", 0);
            activate(&mut s, "Treasure", 0);
            assert_eq!(s.game().state.players[0].life, 1);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            assert_eq!(s.game().state.players[0].life, 1);
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
        }
    }
}

#[test]
fn committed_optional_payment_resumes_after_two_nested_mana_decisions() {
    let (mut s, lesson, drawn) = lesson_payment(
        &[
            "Waterbending Lesson",
            "Prophetic Prism",
            "Callous Inspector",
            "Tireless Provisioner",
        ],
        "0",
    );
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    // Inject the extra resource before payment begins. The initial waterbend
    // offer is already legal from the creatures and Prism; its normal tap
    // transition publishes the offers that include this Treasure.
    let creature = permanent(&s, "Callous Inspector");
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == creature),
    );
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    activate(&mut s, "Prophetic Prism", 0);
    activate(&mut s, "Treasure", 0);
    assert_eq!(
        s.game().state.stack_objects.len(),
        1,
        "only the resolving Lesson is on the stack"
    );
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), drawn + 3);
    assert_eq!(
        s.game().state.zones.zone_of(lesson),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn selected_sacrifice_cannot_also_fund_the_announced_spell() {
    for mode in 0..3 {
        let mut s = setup(&[
            "Deadly Dispute",
            "Tireless Provisioner",
            "Callous Inspector",
        ]);
        if mode == 2 {
            enter(&mut s, "Callous Inspector");
        }
        s.game_mut().create_token("Treasure", PlayerId(0), false);
        let treasure = permanent(&s, "Treasure");
        let treasure_card = s.game().state.permanents[treasure].as_ref().unwrap().card;
        s.set_player_mana_pool(0, if mode == 1 { "BB" } else { "B" });
        let dispute = card(&s, 0, "Deadly Dispute");
        s.game_mut().move_card(dispute, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(
            can_cast(&s, dispute),
            mode > 0,
            "one Treasure cannot be both mana and sacrifice"
        );
        if mode == 0 {
            continue;
        }
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == dispute),
        );
        let offers_treasure = s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == treasure_card));
        assert_eq!(offers_treasure, mode == 1);
        if mode == 1 {
            choose(
                &mut s,
                |a| matches!(a, Action::SelectCard { card, .. } if *card == treasure_card),
            );
        } else {
            let inspector = permanent(&s, "Callous Inspector");
            let inspector_card = s.game().state.permanents[inspector].as_ref().unwrap().card;
            choose(
                &mut s,
                |a| matches!(a, Action::SelectCard { card, .. } if *card == inspector_card),
            );
            activate(&mut s, "Treasure", 0);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        assert!(s.game().state.permanents[treasure].is_none());
        assert_eq!(s.game().state.zones.zone_of(dispute), Some(ZoneType::Stack));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn a_mana_tap_can_precede_a_distinct_sacrifice_cost() {
    for pool in ["0", "B"] {
        let mut s = setup(&["Deadly Dispute", "Llanowar Elves"]);
        let elves_card = enter(&mut s, "Llanowar Elves");
        ready(&mut s, "Llanowar Elves");
        let elves = permanent(&s, "Llanowar Elves");
        s.set_player_mana_pool(0, pool);
        let dispute = card(&s, 0, "Deadly Dispute");
        s.game_mut().move_card(dispute, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&s, dispute), pool == "B");
        if pool == "B" {
            choose(
                &mut s,
                |a| matches!(a, Action::CastSpell { card, .. } if *card == dispute),
            );
            choose(
                &mut s,
                |a| matches!(a, Action::SelectCard { card, .. } if *card == elves_card),
            );
            // Tap and sacrifice are different costs: G plus pooled B pays 1B,
            // then the still-present tapped creature pays the sacrifice.
            assert!(s.game().state.permanents[elves].is_none());
            assert_eq!(
                s.game().state.zones.zone_of(elves_card),
                Some(ZoneType::Graveyard)
            );
            assert_eq!(s.game().state.zones.zone_of(dispute), Some(ZoneType::Stack));
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        }
    }
}

#[test]
fn an_announced_tap_cannot_be_used_to_fund_itself() {
    use managym::state::{ability::Effect, card::ActivatedAbilityDefinition, mana::ManaCost};

    for tap_source in [false, true] {
        let mut s = setup(&["Llanowar Elves"]);
        let elves_card = enter(&mut s, "Llanowar Elves");
        ready(&mut s, "Llanowar Elves");
        let elves = permanent(&s, "Llanowar Elves");
        s.game_mut().state.cards[elves_card]
            .activated_abilities
            .push(ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("1"),
                tap_source,
                sacrifice_source: true,
                effect: Effect::DrawCards { count: 1 },
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
            activate(&mut s, "Llanowar Elves", 0);
            assert!(s.game().state.permanents[elves].is_none());
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            assert_eq!(
                s.game().state.stack_objects.len(),
                1,
                "the nonmana parent ability uses the stack"
            );
        }
    }
