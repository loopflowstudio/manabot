//! Explanations preserve the legal offer set and the viewer boundary.
use crate::support::game::{deck, Scenario};
use managym::{
    agent::{action::ActionSpaceKind, guidance::UnavailableReason, observation::Observation},
    flow::turn::StepKind,
    state::game_object::PlayerId,
};

#[test]
fn decision_guidance_cast_refusals_preserve_offers_and_hidden_information() {
    let mut s = Scenario::new(
        deck(&[("Island", 20), ("Counterspell", 10), ("Gray Ogre", 10)]),
        deck(&[("Forest", 40)]),
        368,
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Counterspell");
    s.force_card_in_hand(0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    let actions = s.action_space().actions.clone();
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let refusals = obs.unavailable_actions.as_ref().unwrap();
    for (name, reason) in [
        ("Counterspell", UnavailableReason::NoLegalTarget),
        ("Gray Ogre", UnavailableReason::TotalCost),
    ] {
        let card = obs
            .agent_cards
            .iter()
            .find(|card| card.characteristics.name == name)
            .unwrap();
        assert!(refusals
            .iter()
            .any(|row| row.object == card.id && row.reason == reason));
    }
    assert_eq!(s.action_space().actions, actions);
    let other = Observation::for_player(s.game(), PlayerId(1));
    assert!(other.unavailable_actions.unwrap().is_empty());
    let decoded: Observation = serde_json::from_str(&obs.to_json()).unwrap();
    assert_eq!(decoded.unavailable_actions, obs.unavailable_actions);
}

#[test]
fn decision_guidance_timing_is_specific_to_the_current_priority() {
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Gray Ogre", 16)]),
        deck(&[("Forest", 40)]),
        368,
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Gray Ogre");
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let card = obs
        .agent_cards
        .iter()
        .find(|card| card.characteristics.name == "Gray Ogre")
        .unwrap();
    assert!(obs
        .unavailable_actions
        .as_ref()
        .unwrap()
        .iter()
        .any(|row| row.object == card.id && row.reason == UnavailableReason::Timing));
}

#[test]
fn decision_guidance_attack_restrictions_do_not_hide_eligible_attackers() {
    let mut s = Scenario::new(
        deck(&[("Forest", 10), ("Gray Ogre", 20), ("Wall of Stone", 10)]),
        deck(&[("Forest", 40)]),
        368,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let ready = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[ready]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let tapped = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[tapped]
        .as_mut()
        .unwrap()
        .tapped = true;
    let sick = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let wall = s.force_permanent_on_battlefield(0, "Wall of Stone");
    s.game_mut().state.permanents[wall]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let refusals = obs.unavailable_actions.as_ref().unwrap();
    for (permanent, reason) in [
        (tapped, UnavailableReason::Tapped),
        (sick, UnavailableReason::SummoningSick),
        (wall, UnavailableReason::Defender),
    ] {
        let object = s.game().state.permanents[permanent].as_ref().unwrap().id.0 as i32;
        assert!(
            refusals
                .iter()
                .any(|row| row.object == object && row.reason == reason),
            "{reason:?}: {refusals:?}"
        );
    }
    let ready_id = s.game().state.permanents[ready].as_ref().unwrap().id.0 as i32;
    assert!(!refusals.iter().any(|row| row.object == ready_id));
    assert!(obs
        .action_space
        .actions
        .iter()
        .any(|action| action.focus.contains(&ready_id)));
}

#[test]
fn decision_guidance_payment_uses_the_native_payment_gate() {
    use managym::agent::action::{Action, ActionType};
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 16)]),
        deck(&[("Forest", 40)]),
        368,
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut()
        .create_token("token.treasure", PlayerId(0), false)
        .unwrap();
    s.game_mut()
        .create_token("token.treasure", PlayerId(0), false)
        .unwrap();
    s.force_card_in_hand(0, "Llanowar Elves");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert!(obs
        .unavailable_actions
        .as_ref()
        .unwrap()
        .iter()
        .any(|row| row.reason == UnavailableReason::PaymentNeedsActivation));
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::PayCost { .. })));
    s.step_action(0);
    s.step_action(4); // Deliberately choose green; this spends the first Treasure.
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert!(!obs
        .unavailable_actions
        .as_ref()
        .unwrap()
        .iter()
        .any(|row| row.reason == UnavailableReason::PaymentNeedsActivation));
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::PayCost { .. })));
}
