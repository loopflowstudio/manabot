use managym::{
    agent::action::ActionSpaceKind,
    flow::{
        trigger::{PendingTrigger, TriggerId},
        turn::StepKind,
    },
    state::{
        game_object::{AbilitySource, ObjectId, ObjectLookupError, PermanentId, PlayerId},
        zone::ZoneType,
    },
};

use crate::support::game::*;
use crate::support::inspect::{now, owned_card, permanent_of, zone_of};

fn put_owned_card_on_battlefield(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let object_id = owned_card(s, player, name);
    let game = s.game_mut();
    let object_id = game.move_object(object_id, ZoneType::Battlefield);
    // Drain events and triggers so the engine doesn't process ETB triggers from setup
    game.state.pending_events.clear();
    game.state.pending_triggers.clear();
    game.state.observation_events.clear();
    game.state
        .permanent_of(object_id)
        .expect("permanent should exist after move_object")
}

fn count_cards_named(s: &Scenario, player: usize, zone: ZoneType, name: &str) -> usize {
    s.game()
        .state
        .zones
        .zone_objects(zone, PlayerId(player))
        .iter()
        .filter(|object_id| s.game().state[*object_id].name == name)
        .count()
}

fn cast_manowar_and_reach_target_choice(s: &mut Scenario) {
    let islands: Vec<ObjectId> = s
        .game()
        .state
        .latest_objects()
        .filter(|(_, object)| object.owner == PlayerId(0) && object.name == "Island")
        .map(|(idx, _)| idx)
        .take(3)
        .collect();
    assert_eq!(islands.len(), 3, "expected at least three Islands in deck");
    for island in islands {
        s.game_mut()
            .state
            .place_object(island, PlayerId(0), ZoneType::Hand);
    }
    let manowar = owned_card(s, 0, "Man-o'-War");
    s.game_mut()
        .state
        .place_object(manowar, PlayerId(0), ZoneType::Hand);

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
    let mut s = Scenario::new(deck_of("Island", "Man-o'-War"), deck(&[("Island", 40)]), 81);
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
    let mut s = Scenario::new(deck_of("Island", "Man-o'-War"), deck(&[("Island", 40)]), 82);
    cast_manowar_and_reach_target_choice(&mut s);

    s.choose_target_named("Man-o'-War");

    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));

    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.game().state.stack_objects.len(), 1);
}

/// CR 603.3b — Pending triggers are ordered APNAP when put on the stack.
#[test]
fn cr_603_3b_pending_triggers_flush_in_apnap_order() {
    let mut s = Scenario::new(
        deck_of("Island", "Man-o'-War"),
        deck_of("Island", "Man-o'-War"),
        83,
    );
    let _target = put_owned_card_on_battlefield(&mut s, 0, "Man-o'-War");

    let p0_source = owned_card(&s, 0, "Man-o'-War");
    let p1_source = owned_card(&s, 1, "Man-o'-War");
    s.game_mut().state.pending_triggers = vec![
        PendingTrigger::new(
            TriggerId(0),
            PlayerId(1),
            AbilitySource::new(p1_source, None),
            0,
        ),
        PendingTrigger::new(
            TriggerId(1),
            PlayerId(0),
            AbilitySource::new(p0_source, None),
            0,
        ),
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
    let mut s = Scenario::new(deck_of("Island", "Man-o'-War"), deck(&[("Island", 40)]), 84);
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

/// CR 400.7, 608.2b — leaving and returning does not make the new object the
/// target of an ability that selected the old object.
#[test]
fn cr_400_7_manowar_target_does_not_follow_reentered_card() {
    let mut s = Scenario::new(
        deck_of("Island", "Man-o'-War"),
        deck(&[("Island", 40)]),
        400_704,
    );
    cast_manowar_and_reach_target_choice(&mut s);
    s.choose_target_named("Man-o'-War");

    let old_permanent = s.battlefield_permanents_named(0, "Man-o'-War")[0];
    let object = s.game().state.permanents[old_permanent]
        .as_ref()
        .expect("old permanent")
        .object;
    let old_ref = s
        .game()
        .state
        .live(now(&s, object))
        .expect("old object ref");

    let object = move_card(&mut s, object, ZoneType::Hand);
    let object = move_card(&mut s, object, ZoneType::Battlefield);
    // Ignore the new object's own ETB trigger; this scenario is specifically
    // resolving the already-stacked trigger that targeted the old object.
    s.game_mut().state.pending_triggers.clear();
    let new_ref = s
        .game()
        .state
        .live(now(&s, object))
        .expect("new object ref");
    let new_permanent = permanent_of(&s, object).expect("new permanent");

    assert_ne!(old_ref, new_ref);
    assert_eq!(
        s.game().state.card_id(old_ref),
        s.game().state.card_id(new_ref)
    );
    assert_eq!(
        s.game().lookup_current_permanent(old_ref),
        Err(ObjectLookupError::Stale)
    );

    s.pass_priority();
    s.pass_priority();

    assert_eq!(
        permanent_of(&s, object),
        Some(new_permanent),
        "the new incarnation must remain on the battlefield"
    );
    assert_eq!(zone_of(&s, object), Some(ZoneType::Battlefield));
}

/// CR 117.5, 704.3, 603.3 — SBAs are checked before pending triggers are flushed.
#[test]
fn cr_603_trigger_flush_happens_after_sba_check() {
    let mut s = Scenario::new(deck_of("Island", "Man-o'-War"), deck(&[("Island", 40)]), 85);

    let dying_permanent = put_owned_card_on_battlefield(&mut s, 0, "Man-o'-War");
    s.game_mut().state.permanents[dying_permanent]
        .as_mut()
        .expect("dying permanent should exist")
        .damage = 2;

    let source = owned_card(&s, 0, "Man-o'-War");
    s.game_mut()
        .state
        .pending_triggers
        .push(PendingTrigger::new(
            TriggerId(0),
            PlayerId(0),
            AbilitySource::new(source, None),
            0,
        ));

    s.pass_priority();

    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(s.game().state.stack_objects.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
    assert_eq!(
        count_cards_named(&s, 0, ZoneType::Graveyard, "Man-o'-War"),
        1
    );
}

/// CR 700.2a, 603.3c: a triggered ability's mode that needs two targets is
/// not offered while only one legal target exists.
#[test]
fn trigger_mode_needs_as_many_legal_targets_as_its_clause_minimum() {
    use crate::support::{inspect::card_of, play::cast};
    use managym::{
        agent::action::Action,
        state::{
            ability::{Ability, Actor, Effect, TargetRequirement, TargetSpec, TriggerCondition},
            player::ManaPayment,
            predicate::CardPredicate,
        },
    };
    let modes = |s: &Scenario| {
        let mut s = s.clone();
        cast(&mut s, "Llanowar Elves");
        let offered: Vec<usize> = s
            .action_space()
            .actions
            .iter()
            .filter_map(|a| match a {
                Action::ChooseMode { mode, .. } => Some(*mode),
                _ => None,
            })
            .collect();
        offered
    };
    let mut s = five_color_setup(&["Hill Giant", "Gray Ogre", "Llanowar Elves"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let object = card_of(&s, giant);
    s.game_mut().state[object].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::casts(Actor::You, CardPredicate::creature()),
        intervening_if: None,
        effects: vec![Effect::TriggerModes {
            modes: vec![
                vec![Effect::TargetClause {
                    requirement: TargetRequirement {
                        min: 2,
                        max: 2,
                        ..TargetRequirement::one(TargetSpec::CreatureOpponentControls)
                    },
                    effects: Vec::new(),
                }],
                vec![Effect::gain_life(1)],
            ],
            min: 0,
            up_to: None,
            unchosen: false,
        }],
    }];
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(modes(&s), [1], "one opposing creature, two targets");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(modes(&s), [0, 1]);
}

/// CR 700.2b, 603.4: a printed modal trigger resolves its chosen mode as that
/// printed ability. It keeps its intervening "if", rechecked on resolution,
/// and its stack row describes the printed ability, not a delayed one.
#[test]
fn chosen_mode_trigger_keeps_its_printed_ability_and_intervening_if() {
    use crate::support::{
        game::move_card,
        inspect::card_of,
        play::{cast, choose_mode, resolve_top},
    };
    use managym::{
        agent::observation::Observation,
        state::{
            ability::{Ability, Actor, Condition, Effect, TriggerCondition},
            player::ManaPayment,
            predicate::CardPredicate,
        },
    };
    let mut s = five_color_setup(&["Hill Giant", "Gray Ogre", "Llanowar Elves"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let object = card_of(&s, giant);
    s.game_mut().state[object].abilities = vec![Ability::Triggered {
        condition: TriggerCondition::casts(Actor::You, CardPredicate::creature()),
        intervening_if: Some(Condition::ControlsAtLeast {
            distinct_names: false,
            count: 2,
            predicate: CardPredicate::creature(),
        }),
        effects: vec![Effect::TriggerModes {
            modes: vec![vec![Effect::gain_life(1)], vec![Effect::gain_life(2)]],
            min: 0,
            up_to: None,
            unchosen: false,
        }],
    }];
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Llanowar Elves");
    choose_mode(&mut s, 1);
    assert_eq!(
        s.game().state.stack_objects.len(),
        2,
        "Elves and the trigger"
    );
    let descriptions: Vec<String> = Observation::for_player(s.game(), PlayerId(0))
        .stack_objects
        .iter()
        .filter_map(|row| row.description.clone())
        .collect();
    let [description] = descriptions.as_slice() else {
        panic!("one described ability: {descriptions:?}");
    };
    assert!(description.contains("Hill Giant"), "{description}");
    assert!(!description.contains("delayed ability"), "{description}");
    assert!(description.contains("gain 2 life"), "{description}");
    assert!(!description.contains("gain 1 life"), "{description}");
    let modes: Vec<Option<usize>> = Observation::for_player(s.game(), PlayerId(0))
        .stack_objects
        .iter()
        .filter(|row| row.description.is_some())
        .map(|row| row.chosen_mode)
        .collect();
    assert_eq!(modes, vec![Some(1)], "the stack row names the chosen mode");
    let life = s.life(0);

    let mut kept = s.clone();
    resolve_top(&mut kept);
    assert_eq!(kept.life(0), life + 2, "the chosen mode");

    move_card(&mut s, ogre_card, ZoneType::Graveyard);
    resolve_top(&mut s);
    assert_eq!(
        s.life(0),
        life,
        "one creature left: the intervening if fails"
    );
}
