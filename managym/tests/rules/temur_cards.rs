// temur_cards.rs
// Directed Temur payment, graveyard casting, and copied-token rules.
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
