//! Tests for cards filed in `src/lea.rs`, in card order.

use crate::support::{
    game::{deck, deck_of, five_color_setup, move_card, ready, Scenario},
    inspect::{card_of, current_object, library_card, now, permanent_named, permanent_of, zone_of},
    play::{
        activate_named, cast, cast_card, cast_clone_as, choose, enter_and_settle, resolve_top,
        settle, spend_colorless,
    },
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        card::EntryScope,
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
        .zone_objects(ZoneType::Library, PlayerId(0))
        .to_vec();
    for object in library {
        s.game_mut()
            .state
            .place_object(object, PlayerId(0), ZoneType::Graveyard);
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
            managym::state::stack_object::StackObject::Spell(spell) => Some(spell.object),
            _ => None,
        })
        .expect("recall should be on stack");
    assert!(s.choose_target(Target::StackSpell(now(&s, recall_card))));

    s.pass_priority();
    s.pass_priority();

    // Recall was countered: no cards drawn, recall in the graveyard.
    s.assert_zone_size(0, ZoneType::Stack, 0);
    s.assert_zone_size(0, ZoneType::Hand, hand_before);
    s.assert_zone_size(0, ZoneType::Library, library_before);
    assert_eq!(zone_of(&s, recall_card), Some(ZoneType::Graveyard));
}

#[test]
fn an_announced_tap_cannot_be_used_to_fund_itself() {
    use managym::state::{
        ability::{Effect, EffectValue},
        card::ActivatedAbilityDefinition,
        mana::ManaCost,
    };

    for tap_source in [false, true] {
        let mut s = five_color_setup(&["Llanowar Elves"]);
        let elves_card = enter_and_settle(&mut s, "Llanowar Elves");
        ready(&mut s, "Llanowar Elves");
        let elves = permanent_named(&s, "Llanowar Elves");
        s.game_mut().state[elves_card]
            .activated_abilities
            .push(ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("1"),
                tap_source,
                sacrifice_source: true,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
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

#[test]
fn sol_ring_taps_for_two_colorless() {
    let mut s = crate::support::game::five_color_unfunded_setup(&["Sol Ring", "Prophetic Prism"]);
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let prism = crate::support::game::move_to_hand(&mut s, "Prophetic Prism");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(crate::support::inspect::can_cast(&s, prism));
    crate::support::play::cast_card(&mut s, prism);
    crate::support::play::settle(&mut s);
    assert!(crate::support::inspect::tapped(&s, ring));
    assert_eq!(
        s.battlefield_permanents_named(0, "Prophetic Prism").len(),
        1
    );
}

/// Clone offers every creature on the battlefield, enters as the chosen one
/// (a card, not a token, still its caster's), and is Clone again once it
/// leaves the battlefield.
#[test]
fn clone_enters_as_a_copy_and_leaves_as_clone() {
    let mut s = five_color_setup(&["Clone", "Gray Ogre", "Wolf Cove Villager"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let villager = s.force_permanent_on_battlefield(0, "Wolf Cove Villager");
    let clone = cast(&mut s, "Clone");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let offered: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::ChooseTarget {
                target: Target::Permanent(id),
                ..
            } => Some(*id),
            _ => None,
        })
        .collect();
    assert_eq!(offered.len(), 2);
    assert!(offered.contains(&ogre) && offered.contains(&villager));
    // Project the pending choice too: its entering card and copy candidates
    // must have registered execution labels before a policy can choose.
    let pending = Observation::for_player(s.game(), PlayerId(0));
    assert!(pending
        .continuation
        .iter()
        .any(|frame| !frame.effect_execution.is_empty()));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);

    let copy = permanent_of(&s, clone).expect("entered");
    let object = current_object(&s, clone);
    assert_eq!(object.name, "Gray Ogre");
    assert!(object.is_card());
    assert_eq!(s.game().characteristics(copy).pt(), (2, 2));
    assert_eq!(
        s.game().state.permanents[copy].as_ref().unwrap().controller,
        PlayerId(0)
    );
    let observation = Observation::for_player(s.game(), PlayerId(1));
    let seen = observation
        .opponent_permanents
        .iter()
        .find_map(|permanent| permanent.copy.as_ref())
        .expect("the copy effect is observed");
    assert_eq!(seen.name, "Gray Ogre");

    let clone = move_card(&mut s, clone, ZoneType::Graveyard);
    assert_eq!(s.game().state[clone].name, "Clone");
    assert!(s.game().state[clone].copy().is_none());
}

/// The copied creature's own entry replacement applies (CR 614.12): a copy
/// of Wolf Cove Villager enters tapped. Declining leaves a 0/0 Clone, which
/// dies.
#[test]
fn clone_takes_the_copied_entry_replacements_or_dies_as_itself() {
    let mut s = five_color_setup(&["Clone", "Wolf Cove Villager"]);
    let villager = s.force_permanent_on_battlefield(1, "Wolf Cove Villager");
    let clone = cast_clone_as(&mut s, villager);
    let copy = permanent_of(&s, clone).expect("entered");
    assert!(s.game().state.permanents[copy].as_ref().unwrap().tapped);

    let second = library_card(&s, 0, "Clone");
    let second = move_card(&mut s, second, ZoneType::Hand);
    cast_card(&mut s, second);
    spend_colorless(&mut s);
    resolve_top(&mut s);
    choose(&mut s, |action| matches!(action, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(zone_of(&s, second), Some(ZoneType::Graveyard));
    assert_eq!(current_object(&s, second).name, "Clone");
}

/// CR 614.12: the prospective copy and its entry terms survive another
/// replacement choice and an exact fork. The original printing remains the
/// base for the next appearance, including changes made by a rules fixture.
#[test]
fn clone_copy_survives_suspended_entry_and_keeps_its_printed_base() {
    use managym::state::{card::ReplacementEffect, permanent::CounterKind};

    let mut s = five_color_setup(&["Clone", "Wolf Cove Villager"]);
    let villager = s.force_permanent_on_battlefield(1, "Wolf Cove Villager");
    let source = s.game().state.permanents[villager].as_ref().unwrap().object;
    s.game_mut().state[source].replacement_effects = vec![
        ReplacementEffect::EntersTapped,
        ReplacementEffect::EntersWithCounters {
            on: EntryScope::This,
            condition: None,
            kind: CounterKind::PlusOnePlusOne,
            count: 2.into(),
        },
    ];
    let clone = crate::support::game::move_to_hand(&mut s, "Clone");
    s.game_mut().state[clone].power = Some(3);
    s.game_mut().state[clone].toughness = Some(3);
    cast_card(&mut s, clone);
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(villager)));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    assert_eq!(current_object(&s, clone).name, "Wolf Cove Villager");
    // The prospective copy is on the suspended entry, and both copied
    // replacements wait as the modes of the order choice.
    let villager_def = s.game().state[source].definition_id;
    assert_eq!(
        current_object(&s, clone).copy().map(|copy| copy.of),
        Some(villager_def)
    );
    assert_eq!(
        current_object(&s, clone).replacement_effects.len(),
        2,
        "the copy carries the source's replacements"
    );
    assert_eq!(
        s.action_space()
            .actions
            .iter()
            .filter(|action| matches!(action, Action::ChooseMode { .. }))
            .count(),
        2
    );
    let pending = Observation::for_player(s.game(), PlayerId(0));
    let mut fork = s.game().clone();
    assert_eq!(
        pending.to_json(),
        Observation::for_player(&fork, PlayerId(0)).to_json()
    );
    // Both the prospective copy and the unanswered replacement belong to
    // the canonical state. Vary each independently within the same catalog.
    let hash = s.game().state.deterministic_hash();
    assert_eq!(fork.state.deterministic_hash(), hash);
    let entering = now(&s, clone);
    let form = fork.state[entering].form.take().expect("prospective copy");
    assert_ne!(fork.state.deterministic_hash(), hash);
    fork.state[entering].form = Some(form);
    assert_eq!(fork.state.deterministic_hash(), hash);
    fork.state
        .suspended_decision
        .take()
        .expect("pending replacement");
    assert_ne!(fork.state.deterministic_hash(), hash);

    crate::support::play::command(&mut s, |action| {
        matches!(action, Action::ChooseMode { mode: 0, .. })
    });
    settle(&mut s);
    let copy = permanent_of(&s, clone).expect("entered");
    let permanent = s.game().state.permanents[copy].as_ref().unwrap();
    assert!(permanent.tapped);
    assert_eq!(permanent.counter_count(CounterKind::PlusOnePlusOne), 2);
    assert_eq!(current_object(&s, clone).name, "Wolf Cove Villager");

    let clone = move_card(&mut s, clone, ZoneType::Hand);
    assert_eq!(s.game().state[clone].name, "Clone");
    assert_eq!(s.game().state[clone].power, Some(3));
    assert_eq!(s.game().state[clone].toughness, Some(3));
    assert!(s.game().state[clone].copy().is_none());
}

#[test]
fn departed_clone_creates_joo_dee_before_the_sacrifice_choice() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{activate, pass_until, select_card},
    };

    let name = "Joo Dee, One of Many";
    let mut s = five_color_setup(&["Clone", name]);
    let source = s.force_permanent_on_battlefield(1, name);
    let clone = cast_clone_as(&mut s, source);
    let permanent = permanent_of(&s, clone).unwrap();
    clear_summoning_sickness(&mut s, permanent);
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, permanent, 0);
    move_card(&mut s, clone, ZoneType::Hand);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    let tokens = s.battlefield_permanents_named(0, name);
    assert_eq!(tokens.len(), 1);
    let token = card_of(&s, tokens[0]);
    assert!(!s.game().state[token].is_card());
    select_card(&mut s, token);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, name).is_empty());
}

#[test]
fn departed_clone_horde_trigger_survives_a_new_form_and_snapshot() {
    let horde = "Homunculus Horde";
    let mut s = five_color_setup(&["Clone", horde, "Gray Ogre", "Scout's Warning"]);
    let original = s.force_permanent_on_battlefield(1, horde);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let clone = cast_clone_as(&mut s, original);
    s.game_mut().state.turn.history[0].cards_drawn = 1;
    cast(&mut s, "Scout's Warning");
    resolve_top(&mut s);
    assert_eq!(s.game().state.stack_objects.len(), 1);

    let parent = s.game().clone();
    let pending = parent.state.deterministic_hash();
    let departed = now(&s, clone);
    let returned = move_card(&mut s, clone, ZoneType::Hand);
    cast_card(&mut s, returned);
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert_eq!(s.game().state.stack_objects.len(), 1);

    s.restore(&s.snapshot());
    resolve_top(&mut s);
    assert_eq!(current_object(&s, clone).name, "Gray Ogre");
    let copies = s.battlefield_permanents_named(0, horde);
    assert_eq!(copies.len(), 1);
    let token = card_of(&s, copies[0]);
    assert!(!s.game().state[token].is_card());
    let values = s.game().state[token].copiable_definition();
    assert!(s.game().object_lki(departed).unwrap().copiable.is_none());
    assert!(s.game().state.stack_objects.is_empty());

    assert_eq!(parent.state.deterministic_hash(), pending);
    *s.game_mut() = parent;
    resolve_top(&mut s);
    let copies = s.battlefield_permanents_named(0, horde);
    assert_eq!(copies.len(), 2);
    let live_token = copies
        .into_iter()
        .map(|id| card_of(&s, id))
        .find(|id| !s.game().state[*id].is_card())
        .unwrap();
    assert_eq!(s.game().state[live_token].copiable_definition(), values);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn clone_legion_offers_the_entering_clone_token_as_focus() {
    use crate::support::{game::set_counters, play::target_opponent};
    use managym::state::permanent::CounterKind;

    let mut s = five_color_setup(&["Clone Legion", "Clone", "Gray Ogre"]);
    let clone = s.force_permanent_on_battlefield(1, "Clone");
    set_counters(&mut s, clone, CounterKind::PlusOnePlusOne, 1);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Clone Legion");
    target_opponent(&mut s);
    spend_colorless(&mut s);
    resolve_top(&mut s);
    let focus = s.action_space().focus.clone();
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    let copies = s.battlefield_permanents_named(0, "Gray Ogre");
    assert_eq!(copies.len(), 2);
    let token = card_of(&s, *copies.iter().find(|id| **id != ogre).unwrap());
    assert!(!s.game().state[token].is_card());
    assert_eq!(focus, [s.game().state[token].id]);
}
