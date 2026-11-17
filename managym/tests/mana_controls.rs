//! Live native control offers beside a normal priority decision.
use managym::{
    agent::structured_offer::{ChoiceAnswer, OfferVerb, RoleId},
    decision::Command,
    flow::mana::controls::ControlOffer,
    state::player::PlayerConfig,
    Env,
};
use std::collections::BTreeMap;

fn command(id: &str, revision: u64, offer_id: u32) -> Command {
    Command {
        command_id: id.into(),
        expected_revision: revision,
        offer_id,
        answers: vec![],
        object_preconditions: vec![],
    }
}

#[test]
fn float_a_forest_then_restore_the_decision() {
    let deck = BTreeMap::from([("Forest".into(), 20), ("Llanowar Elves".into(), 20)]);
    let mut env = Env::new(11, true, false, false);
    env.reset(vec![
        PlayerConfig::new("p0", deck.clone()),
        PlayerConfig::new("p1", deck),
    ])
    .unwrap();
    let frame = env.semantic_decision_frame().unwrap();
    let land = frame
        .offers
        .iter()
        .find(|offer| offer.verb == OfferVerb::PlayLand)
        .unwrap();
    env.execute_semantic_command(&command("land", frame.revision, land.id.0))
        .unwrap();
    let actor = usize::from(frame.actor);
    let before = env.state_digest().unwrap();
    let original = env.semantic_observation(actor).unwrap().viewer_state;
    assert!(env.mana_controls(actor).is_err());
    env.enable_mana_controls();
    assert!(env.fork().unwrap().mana_controls(actor).is_err());
    assert!(env.mana_controls(1 - actor).unwrap().offers.is_empty());
    let controls = env.mana_controls(actor).unwrap();
    let [ControlOffer::Float {
        id, label, output, ..
    }] = controls.offers.as_slice()
    else {
        panic!("{controls:?}");
    };
    assert_eq!(label, "Tap for mana: {G}");
    assert_eq!(output.get("green"), Some(&1));
    let tap = command("float", controls.revision, *id as u32);
    assert!(env.execute_semantic_command(&tap).is_err());
    assert!(env
        .execute_mana_control(&command("action", controls.revision, 0))
        .is_err());
    let mut answered = tap.clone();
    answered.answers.push(ChoiceAnswer::Candidates {
        role: RoleId(0),
        candidates: vec![],
    });
    assert!(env.execute_mana_control(&answered).is_err());
    assert_eq!(env.state_digest().unwrap(), before);
    let transition = env.execute_mana_control(&tap).unwrap();
    assert_eq!(transition.receipt.before_revision, controls.revision);
    let floated = env.observation_for_player(actor).unwrap();
    assert!(floated.agent_permanents[0].tapped);
    assert_eq!(floated.agent.mana, [0, 0, 0, 0, 1, 0]);
    assert!(env
        .execute_mana_control(&tap)
        .unwrap_err()
        .to_string()
        .contains("stale_revision"));
    let controls = env.mana_controls(actor).unwrap();
    let changes = || {
        let current = env.semantic_observation(actor).unwrap().viewer_state;
        original
            .as_object()
            .unwrap()
            .iter()
            .filter(|(field, value)| current[field.as_str()] != **value)
            .map(|(field, value)| format!("{field}: {value} -> {}", current[field]))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(transition.receipt.after_revision, controls.revision);
    let [ControlOffer::Undo { id, label }] = controls.offers.as_slice() else {
        panic!("{controls:?}\n{}", changes());
    };
    assert_eq!(label, "Undo: Forest tapped for {G}");
    let transition = env
        .execute_mana_control(&command("undo", controls.revision, *id as u32))
        .unwrap();
    assert!(transition.receipt.after_revision > controls.revision);
    assert_eq!(env.state_digest().unwrap(), before);
    assert!(env.mana_controls(actor).unwrap().undo.is_none());
}

#[test]
fn floating_after_a_pass_reopens_priority_for_the_other_player() {
    let deck = BTreeMap::from([("Island".into(), 20), ("Ancestral Recall".into(), 20)]);
    let mut env = Env::new(11, true, false, false);
    env.reset(vec![
        PlayerConfig::new("p0", deck.clone()),
        PlayerConfig::new("p1", deck),
    ])
    .unwrap();
    for seat in [0, 1] {
        env.scenario_force_battlefield(seat, "Island", true)
            .unwrap();
        env.scenario_force_card_in_hand(seat, "Ancestral Recall")
            .unwrap();
    }
    env.scenario_refresh().unwrap();
    env.enable_mana_controls();
    let pass = |env: &mut Env, name| {
        let frame = env.semantic_decision_frame().unwrap();
        let offer = frame
            .offers
            .iter()
            .find(|offer| offer.verb == OfferVerb::PassPriority)
            .unwrap();
        env.execute_semantic_command(&command(name, frame.revision, offer.id.0))
            .unwrap()
    };
    let first = env.current_agent_index().unwrap();
    pass(&mut env, "first-pass");
    let second = 1 - first;
    assert_eq!(env.current_agent_index(), Some(second));
    let before = env.state_digest().unwrap();
    let float = |env: &mut Env, name| {
        let frame = env.mana_controls(second).unwrap();
        let ControlOffer::Float { id, .. } = frame.offers[0] else {
            panic!("{frame:?}")
        };
        env.execute_mana_control(&command(name, frame.revision, id as u32))
            .unwrap()
    };
    float(&mut env, "float");
    let frame = env.mana_controls(second).unwrap();
    let ControlOffer::Undo { id, .. } = frame.offers[0] else {
        panic!("{frame:?}")
    };
    env.execute_mana_control(&command("undo", frame.revision, id as u32))
        .unwrap();
    assert_eq!(env.state_digest().unwrap(), before);
    float(&mut env, "float-again");
    pass(&mut env, "pass-with-mana");
    assert_eq!(env.current_agent_index(), Some(first));
    assert_eq!(env.observation_for_player(second).unwrap().agent.mana[1], 1);
}
