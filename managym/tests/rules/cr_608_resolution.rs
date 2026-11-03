//! Directed resolution boundaries, with explicit proposals for known gaps.
//! Synthetic effects isolate rules obligations without changing card admission.

use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::{
        trigger::{PendingTrigger, TriggerId, TriggerOrderingClass},
        turn::StepKind,
    },
    state::{
        ability::{Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec},
        card::Keywords,
        effect::Modification,
        game_object::{CardId, PlayerId},
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

use crate::support::{game::*, play::select_card};

fn queue_effects(s: &mut Scenario, source: CardId, effects: Vec<Effect>) {
    let source_ref = s.game().current_object_ref(source);
    s.game_mut().state.pending_triggers.push(PendingTrigger {
        source_card: source,
        source_ref,
        source_lki: None,
        ability_index: 0,
        controller: PlayerId(0),
        targets: Vec::new(),
        target_req_indices: Vec::new(),
        requirement_index: 0,
        id: TriggerId(0),
        ordering_class: TriggerOrderingClass::Ordinary,
        context: None,
        player: None,
        chosen_modes: Vec::new(),
        attacker: None,
        inline_effects: Some(effects),
        x: 0,
    });
}

/// CR 608.2c/d, 704.4: no priority or life-loss SBA between instructions,
/// even while the resolving frame waits for the controller's choice.
#[test]
fn cr_704_4_sba_waits_for_suspended_resolution_to_finish() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Mountain", 40)]),
        608_704,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = s.game().state.permanents[source].as_ref().unwrap().card;
    s.game_mut().state.players[0].life = 1;
    queue_effects(
        &mut s,
        card,
        vec![
            Effect::DealDamageToPlayer {
                player: PlayerSelector::You,
                amount: EffectValue::Fixed(1),
            },
            Effect::Modal {
                modes: vec![vec![Effect::gain_life(2)], vec![Effect::gain_life(3)]],
            },
        ],
    );
    s.pass_priority(); // place the trigger
    s.pass_priority();
    s.pass_priority(); // resolve to the choice
    assert_eq!(s.game().state.players[0].life, 0);
    assert!(s.game().state.players[0].alive);
    assert_ne!(s.action_space().kind, ActionSpaceKind::Priority);
    let choice = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseMode { mode: 0, .. }))
        .expect("resolution choice");
    s.step_action(choice);
    assert_eq!(s.game().state.players[0].life, 2);
    assert!(s.game().state.players[0].alive);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert!(s.game().state.stack_objects.is_empty());
}

/// CR 115.10b, 603.5: "you may" is a choice on resolution, not a target
/// or a reason to discard the trigger before the response window.
#[test]
fn cr_603_5_optional_draw_is_chosen_at_resolution() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        deck(&[("Mountain", 40)]),
        608_603,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = s.game().state.permanents[source].as_ref().unwrap().card;
    queue_effects(
        &mut s,
        card,
        vec![Effect::OptionalDiscardDraw { count: 1, up_to: 1 }],
    );
    let hand = s.zone_size(0, ZoneType::Hand);
    s.pass_priority();
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    s.pass_priority();
    s.pass_priority();
    let decline = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::Decline { .. }))
        .expect("may decline at resolution");
    s.step_action(decline);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand);
    assert!(s.game().state.stack_objects.is_empty());
}

/// A verb over a `Chosen` selector suspends on the choice and then runs over
/// the chosen permanent alone: "tap a creature you control" taps the one
/// selected and leaves the other untapped.
#[test]
fn chosen_selector_offers_the_choice_for_any_verb() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        608_120,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert_ne!(first, second);
    let permanents = &s.game().state.permanents;
    let source = permanents[first].as_ref().unwrap().card;
    let chosen = permanents[second].as_ref().unwrap().card;
    queue_effects(
        &mut s,
        source,
        vec![Effect::tap(ObjectSelector::chosen(
            CardPredicate::creature(),
        ))],
    );
    s.pass_priority();
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert!(
        !s.action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })),
        "the choice is not optional"
    );
    select_card(&mut s, chosen);
    let permanents = &s.game().state.permanents;
    assert!(permanents[second].as_ref().unwrap().tapped);
    assert!(!permanents[first].as_ref().unwrap().tapped);
    assert!(s.game().state.stack_objects.is_empty());
}

/// Stopping a nested counter choice ends that instruction's object binding.
#[test]
fn chosen_selector_is_released_after_stopping_counter_removal() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        608_123,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let source = s.game().state.permanents[first].as_ref().unwrap().card;
    let other = s.game().state.permanents[second].as_ref().unwrap().card;
    queue_effects(
        &mut s,
        source,
        vec![
            Effect::put_counters(
                ObjectSelector::This,
                managym::state::permanent::CounterKind::PlusOnePlusOne,
                1,
            ),
            Effect::RemoveCounters {
                from: ObjectSelector::chosen(CardPredicate::creature()),
                up_to: 1,
                chosen: None,
            },
            Effect::tap(ObjectSelector::chosen(CardPredicate::creature())),
        ],
    );
    s.pass_priority();
    s.pass_priority();
    s.pass_priority();
    select_card(&mut s, source);
    let stop = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::ChooseMode { mode: 1, .. }))
        .expect("stop removing counters");
    s.step_action(stop);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    select_card(&mut s, other);
    assert!(!s.game().state.permanents[first].as_ref().unwrap().tapped);
    assert!(s.game().state.permanents[second].as_ref().unwrap().tapped);
    assert!(s.game().state.stack_objects.is_empty());
}

/// An instruction with no eligible choice does nothing, including clearing
/// the prior move result before a following "if you do" reads it.
#[test]
fn chosen_selector_without_candidates_does_not_reuse_a_previous_move() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        608_121,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = s.game().state.permanents[source].as_ref().unwrap().card;
    let life = s.game().state.players[0].life;
    queue_effects(
        &mut s,
        card,
        vec![
            Effect::exile(ObjectSelector::This),
            Effect::exile(ObjectSelector::chosen(CardPredicate::creature())),
            Effect::if_you_do(vec![Effect::gain_life(5)]),
        ],
    );
    s.pass_priority();
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
    assert_eq!(s.game().state.players[0].life, life);
    assert!(s.game().state.stack_objects.is_empty());
}

/// Every land selected by one earthbend instruction becomes a creature;
/// the selector's cardinality must not silently turn the instruction off.
#[test]
fn each_selector_earthbends_every_selected_land() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        608_122,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let first = s.force_permanent_on_battlefield(0, "Mountain");
    let second = s.force_permanent_on_battlefield(0, "Mountain");
    assert_ne!(first, second);
    let source = s.game().state.permanents[first].as_ref().unwrap().card;
    queue_effects(
        &mut s,
        source,
        vec![Effect::Earthbend {
            count: EffectValue::Fixed(2),
            what: ObjectSelector::each_you_control(CardPredicate {
                card_type: Some(managym::state::card::CardType::Land),
                ..Default::default()
            }),
        }],
    );
    s.pass_priority();
    s.pass_priority();
    s.pass_priority();
    for id in [first, second] {
        let view = s.game().characteristics(id);
        assert!(view.is_creature());
        assert_eq!(view.power, 2);
        assert_eq!(view.toughness, 2);
        assert!(view.keywords.haste);
    }
    assert!(s.game().state.stack_objects.is_empty());
}

/// CR 603.6/603.7c, 400.7e: finding the first destination object is allowed;
/// following it through a second zone change is not. Direct zone movement
/// isolates the obligation; it does not assert an admitted-deck sequence.
#[test]
#[ignore = "audit proposal: earthbend return must retain destination incarnation"]
fn audit_earthbend_return_does_not_follow_a_second_zone_change() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        deck(&[("Mountain", 40)]),
        608_400,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let land = s.force_permanent_on_battlefield(0, "Mountain");
    let card = s.game().state.permanents[land].as_ref().unwrap().card;
    queue_effects(
        &mut s,
        card,
        vec![Effect::Earthbend {
            count: EffectValue::Fixed(1),
            what: ObjectSelector::Target(TargetSpec::LandYouControl),
        }],
    );
    s.pass_priority();
    assert!(s.choose_target(managym::state::game_object::Target::Permanent(land)));
    s.pass_priority();
    s.pass_priority();
    s.game_mut().move_card(card, ZoneType::Graveyard);
    s.pass_priority(); // the return is now on the stack
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.game_mut().move_card(card, ZoneType::Exile);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
}

/// CR 704.5h: surviving a deathtouch SBA check does not keep a destruction
/// obligation alive until cleanup. The keyword removal is a synthetic fixture.
#[test]
#[ignore = "audit proposal: clear deathtouch history after an SBA check"]
fn audit_deathtouch_does_not_reapply_after_surviving_an_sba_check() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Mountain", 40)]),
        608_705,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let permanent = s.game_mut().state.permanents[ogre].as_mut().unwrap();
    let card = permanent.card;
    permanent.damage = 1;
    permanent.deathtouch_damage = true;
    let indestructible = Modification::AddKeywords(Keywords {
        indestructible: true,
        ..Default::default()
    });
    until_end_of_turn(&mut s, ogre, indestructible.clone());
    s.pass_priority();
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Battlefield)
    );
    s.game_mut()
        .state
        .effects
        .retain(|effect| effect.modification != indestructible);
    s.pass_priority();
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Battlefield)
    );
}

/// CR 608.2b: target legality is checked as resolution starts, not again
/// after each instruction changes the target's characteristics.
#[test]
fn audit_target_gaining_hexproof_during_resolution_is_still_affected() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        608_115,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let source = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = s.game().state.permanents[source].as_ref().unwrap().card;
    queue_effects(
        &mut s,
        card,
        vec![
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords {
                    hexproof: true,
                    ..Default::default()
                }),
            ),
            Effect::DealDamage {
                amount: EffectValue::Fixed(1),
                what: ObjectSelector::Target(TargetSpec::Creature),
            },
        ],
    );
    s.pass_priority();
    assert!(s.choose_target(managym::state::game_object::Target::Permanent(victim)));
    s.pass_priority();
    s.pass_priority();
    assert!(s.game().characteristics(victim).keywords.hexproof);
    assert_eq!(
        s.game().state.permanents[victim].as_ref().unwrap().damage,
        1
    );
}
