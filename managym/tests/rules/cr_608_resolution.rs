//! Directed resolution boundaries, with explicit proposals for known gaps.
//! Synthetic effects isolate rules obligations without changing card admission.

use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::{
        trigger::{PendingTrigger, TriggerId, TriggerOrderingClass},
        turn::StepKind,
    },
    state::{
        ability::{AffectedSpec, Effect, EffectValue, TargetSpec},
        card::Keywords,
        effect::Modification,
        game_object::{CardId, PlayerId},
        zone::ZoneType,
    },
};

use crate::support::game::*;

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
        inline_effects: Some(effects),
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
            Effect::DealDamageToController { amount: 1 },
            Effect::Modal {
                modes: vec![
                    vec![Effect::GainLife { amount: 2 }],
                    vec![Effect::GainLife { amount: 3 }],
                ],
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
            target: TargetSpec::LandYouControl,
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
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords {
                    hexproof: true,
                    ..Default::default()
                }),
            ),
            Effect::DealDamage {
                amount: 1,
                target: TargetSpec::Creature,
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
