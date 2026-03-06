use managym::{
    agent::{
        action::ActionType,
        observation::{Observation, StackObjectKindData},
    },
    flow::turn::StepKind,
    state::zone::ZoneType,
};

use super::helpers::*;

fn setup_shivan_with_mountains(seed: u64, mountain_count: usize) -> Scenario {
    let mut s = Scenario::new(shivan_deck(), mountain_deck(), seed);
    let shivan = s.force_permanent_on_battlefield(0, "Shivan Dragon");
    s.game_mut().state.permanents[shivan]
        .as_mut()
        .expect("shivan should exist")
        .summoning_sick = false;
    for _ in 0..mountain_count {
        let _ = s.force_permanent_on_battlefield(0, "Mountain");
    }
    s.advance_to_active_step(0, StepKind::Main);
    s
}

/// CR 602 — Activated ability appears as a priority action when costs are payable.
#[test]
fn cr_602_can_activate_shivan_when_red_mana_is_available() {
    let s = setup_shivan_with_mountains(60201, 1);
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|action| action.action_type() == ActionType::PriorityActivateAbility));
}

/// Negative: without payable mana, activation action is not offered.
#[test]
fn cr_602_cannot_activate_without_available_red_mana() {
    let s = setup_shivan_with_mountains(60202, 0);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| action.action_type() == ActionType::PriorityActivateAbility));
}

/// Activation uses the stack and resolves only after both players pass priority.
#[test]
fn cr_602_activation_uses_stack_and_resolves_after_passes() {
    let mut s = setup_shivan_with_mountains(60203, 1);
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.zone_size(0, ZoneType::Stack), 0);

    let shivan = s.battlefield_permanents_named(0, "Shivan Dragon")[0];
    let base_temp_power = s.game().state.permanents[shivan]
        .as_ref()
        .expect("shivan should exist")
        .temp_power;
    s.pass_priority();
    s.pass_priority();

    let boosted_temp_power = s.game().state.permanents[shivan]
        .as_ref()
        .expect("shivan should exist")
        .temp_power;
    assert_eq!(boosted_temp_power, base_temp_power + 1);
}

/// Multiple activations stack and resolve in LIFO order, accumulating +N/+0.
#[test]
fn cr_602_multiple_activations_stack_and_accumulate() {
    let mut s = setup_shivan_with_mountains(60204, 2);
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));

    let observation = Observation::new(s.game());
    assert_eq!(observation.stack_objects.len(), 2);
    assert_eq!(
        observation.stack_objects[0].kind,
        StackObjectKindData::ActivatedAbility
    );
    assert_eq!(
        observation.stack_objects[1].kind,
        StackObjectKindData::ActivatedAbility
    );

    let shivan = s.battlefield_permanents_named(0, "Shivan Dragon")[0];
    s.pass_priority();
    s.pass_priority();
    s.pass_priority();
    s.pass_priority();

    let temp_power = s.game().state.permanents[shivan]
        .as_ref()
        .expect("shivan should exist")
        .temp_power;
    assert_eq!(temp_power, 2);
}

/// Until-EOT buff expires in cleanup.
#[test]
fn cr_602_shivan_buff_expires_in_cleanup() {
    let mut s = setup_shivan_with_mountains(60205, 1);
    let shivan = s.battlefield_permanents_named(0, "Shivan Dragon")[0];

    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(
        s.game().state.permanents[shivan]
            .as_ref()
            .expect("shivan should exist")
            .temp_power,
        1
    );

    s.advance_to_active_step(0, StepKind::Cleanup);
    assert_eq!(
        s.game().state.permanents[shivan]
            .as_ref()
            .expect("shivan should exist")
            .temp_power,
        0
    );
}

/// If the source leaves before resolution, the activation resolves with no effect.
#[test]
fn cr_602_source_leaves_before_resolution_no_effect() {
    let mut s = setup_shivan_with_mountains(60206, 1);
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));

    let shivan = s.battlefield_permanents_named(0, "Shivan Dragon")[0];
    let shivan_card = s.game().state.permanents[shivan]
        .as_ref()
        .expect("shivan should exist")
        .card;
    s.game_mut().state.zones.move_card(
        shivan_card,
        managym::state::game_object::PlayerId(0),
        ZoneType::Graveyard,
    );
    s.game_mut().state.card_to_permanent[shivan_card] = None;
    s.game_mut().state.permanents[shivan] = None;

    s.pass_priority();
    s.pass_priority();
    assert!(s
        .battlefield_permanents_named(0, "Shivan Dragon")
        .is_empty());
}

/// Observation exposes activated abilities on stack with metadata and stable semantics.
#[test]
fn cr_602_observation_exposes_stack_object_metadata_for_ability() {
    let mut s = setup_shivan_with_mountains(60207, 1);
    let shivan_perm = s.battlefield_permanents_named(0, "Shivan Dragon")[0];
    let shivan_obj_id = s.game().state.permanents[shivan_perm]
        .as_ref()
        .expect("shivan should exist")
        .id
        .0 as i32;
    let shivan_card = s.game().state.permanents[shivan_perm]
        .as_ref()
        .expect("shivan should exist")
        .card;
    let shivan_registry_key = s.game().state.cards[shivan_card].registry_key.0 as i32;
    let controller_id = s.game().state.players[0].id.0 as i32;

    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    let observation = Observation::new(s.game());

    assert_eq!(observation.stack_objects.len(), 1);
    let stack_object = &observation.stack_objects[0];
    assert_eq!(stack_object.kind, StackObjectKindData::ActivatedAbility);
    assert_eq!(stack_object.controller_id, controller_id);
    assert_eq!(stack_object.source_card_registry_key, shivan_registry_key);
    assert_eq!(stack_object.source_permanent_id, Some(shivan_obj_id));
    assert_eq!(stack_object.ability_index, Some(0));
    assert!(stack_object.targets.is_empty());
}
