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
    let first = env.current_agent_index().unwrap();
    let second = 1 - first;
    env.enable_mana_controls_for(second).unwrap();
    assert!(env.mana_controls(first).unwrap().offers.is_empty());
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
    pass(&mut env, "first-pass");
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

fn position(board: &[&str], hand: &str) -> Env {
    let mut deck = BTreeMap::from([("Forest".into(), 20), (hand.into(), 12)]);
    for &name in board {
        *deck.entry(name.to_string()).or_default() += 2;
    }
    let mut env = Env::new(11, true, false, false);
    env.reset(vec![
        PlayerConfig::new("p0", deck.clone()),
        PlayerConfig::new("p1", deck),
    ])
    .unwrap();
    let seat = env.current_agent_index().unwrap();
    for name in board {
        env.scenario_force_battlefield(seat, name, true).unwrap();
    }
    env.scenario_force_card_in_hand(seat, hand).unwrap();
    env.scenario_refresh().unwrap();
    env.enable_mana_controls();
    env
}

fn choose(env: &mut Env, verb: OfferVerb, index: usize) {
    let frame = env.semantic_decision_frame().unwrap();
    let offer = frame
        .offers
        .iter()
        .filter(|offer| offer.verb == verb)
        .nth(index)
        .unwrap_or_else(|| panic!("{:?}", frame.offers));
    env.step_semantic_command(&command("choose", frame.revision, offer.id.0))
        .unwrap();
}

fn undo(env: &mut Env, before: &str) {
    let controls = env
        .mana_controls(env.current_agent_index().unwrap())
        .unwrap();
    let id = controls
        .offers
        .iter()
        .find_map(|offer| match offer {
            ControlOffer::Undo { id, .. } => Some(*id),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{controls:?}"));
    env.execute_mana_control(&command("undo", controls.revision, id as u32))
        .unwrap();
    assert_eq!(env.state_digest().unwrap(), before);
}

#[test]
fn any_colour_land_undo_includes_choosing_red() {
    let mut env = position(&["Jasmine Dragon Tea Shop"], "Forest");
    let seat = env.current_agent_index().unwrap();
    let before = env.state_digest().unwrap();
    choose(&mut env, OfferVerb::ActivateAbility, 0);
    assert!(env.mana_controls(seat).unwrap().undo.is_none());
    choose(&mut env, OfferVerb::ChooseMode, 3);
    assert_eq!(
        env.observation_for_player(seat)
            .unwrap()
            .agent
            .restricted_mana[0]
            .mana[3],
        1
    );
    undo(&mut env, &before);
}

#[test]
fn payment_taps_can_be_undone_until_the_spell_is_paid() {
    let mut env = position(
        &["Mountain", "Mountain", "Mountain", "Mountain"],
        "Gray Ogre",
    );
    let seat = env.current_agent_index().unwrap();
    choose(&mut env, OfferVerb::Cast, 0);
    let before = env.state_digest().unwrap();
    choose(&mut env, OfferVerb::TapForMana, 0);
    undo(&mut env, &before);
    for _ in 0..3 {
        choose(&mut env, OfferVerb::TapForMana, 0);
    }
    assert!(env.mana_controls(seat).unwrap().undo.is_none());
}

#[test]
fn casting_commits_floated_mana() {
    let mut env = position(&["Forest"], "Llanowar Elves");
    let seat = env.current_agent_index().unwrap();
    let controls = env.mana_controls(seat).unwrap();
    let ControlOffer::Float { id, .. } = controls.offers[0] else {
        panic!()
    };
    env.execute_mana_control(&command("float", controls.revision, id as u32))
        .unwrap();
    choose(&mut env, OfferVerb::Cast, 0);
    assert!(env.mana_controls(seat).unwrap().undo.is_none());
}

#[test]
fn disabled_controls_reject_both_methods_without_mutation() {
    let mut env = position(&["Forest"], "Llanowar Elves").fork().unwrap();
    let seat = env.current_agent_index().unwrap();
    let frame = env.semantic_decision_frame().unwrap();
    let before = env.state_digest().unwrap();
    assert!(env.mana_controls(seat).is_err());
    assert!(env
        .execute_mana_control(&command("float", frame.revision, frame.offers.len() as u32))
        .is_err());
    assert_eq!(env.state_digest().unwrap(), before);
    assert_eq!(
        env.semantic_decision_frame().unwrap().fingerprint,
        frame.fingerprint
    );
}

fn assert_refusal(env: &Env, code: &str) {
    let controls = env
        .mana_controls(env.current_agent_index().unwrap())
        .unwrap();
    let status = serde_json::to_value(controls.undo.unwrap()).unwrap();
    assert!(status["offer"].is_null());
    assert_eq!(status["reason"]["code"], code);
}

#[test]
fn tapping_a_healer_with_a_granted_mana_ability_triggers_its_ability() {
    let mut env = position(&["Great Divide Guide", "Compassionate Healer"], "Forest");
    activate(&mut env, "Compassionate Healer", 0);
    choose(&mut env, OfferVerb::ChooseMode, 3);
    assert_refusal(&env, "triggered_ability");
}

fn source_id(env: &Env, name: &str) -> i32 {
    env.observation_for_player(env.current_agent_index().unwrap())
        .unwrap()
        .agent_permanents
        .iter()
        .find(|p| p.characteristics.name == name)
        .unwrap()
        .id
}

fn activate(env: &mut Env, name: &str, index: usize) {
    let frame = env.semantic_decision_frame().unwrap();
    let observation = env
        .observation_for_player(usize::from(frame.actor))
        .unwrap();
    let id = source_id(env, name);
    let choice = observation
        .action_space
        .actions
        .iter()
        .position(|a| a.focus == [id] && a.ability_index == Some(index))
        .unwrap_or_else(|| panic!("{name} ability {index} missing"));
    env.execute_semantic_command(&command(
        "activate",
        frame.revision,
        frame.offers[choice].id.0,
    ))
    .unwrap();
}

#[test]
fn treasure_sacrifice_disables_undo_after_the_colour_choice() {
    let mut env = position(&[], "Forest");
    env.scenario_create_token(env.current_agent_index().unwrap(), "token.treasure")
        .unwrap();
    env.scenario_refresh().unwrap();
    activate(&mut env, "Treasure", 0);
    choose(&mut env, OfferVerb::ChooseMode, 3);
    assert_refusal(&env, "sacrificed_source");
}

#[test]
fn every_catalog_mana_output_has_a_proved_undo_or_a_reason() {
    let cards = managym_cards::catalog()
        .iter()
        .flat_map(|card| std::iter::once(&card.printed).chain(card.back_face.as_deref()))
        .map(|printed| (printed, None));
    let tokens = managym_cards::token_catalog()
        .iter()
        .map(|t| (&t.printed, Some(t.key)));
    for (printed, token) in cards.chain(tokens) {
        let abilities: Vec<_> = printed
            .activated_abilities
            .iter()
            .enumerate()
            .filter_map(|(i, a)| Some((i, a.mana_choices()?.len())))
            .collect();
        if printed.mana_abilities.is_empty() && abilities.is_empty() {
            continue;
        }
        let name = printed.name.as_str();
        let mut board = vec!["Mountain", "Gray Ogre", "Gray Ogre"];
        if token.is_none() {
            board.insert(0, name);
        }
        let mut position = position(&board, "Forest");
        let seat = position.current_agent_index().unwrap();
        if let Some(key) = token {
            position.scenario_create_token(seat, key).unwrap();
        }
        if name == "The Cabbage Merchant" {
            for _ in 0..2 {
                position.scenario_create_token(seat, "token.food").unwrap();
            }
        }
        if token.is_some() || name == "The Cabbage Merchant" {
            position.scenario_refresh().unwrap();
            // Drain the injected entries through an ordinary Command boundary.
            choose(&mut position, OfferVerb::PlayLand, 0);
        }
        let id = source_id(&position, name);
        let before = position.state_digest().unwrap();
        let verify = |env: &mut Env| {
            let control = env.mana_controls(seat).unwrap();
            if control
                .offers
                .iter()
                .any(|o| matches!(o, ControlOffer::Undo { .. }))
            {
                assert_ne!(name, "Treasure");
                undo(env, &before);
            } else {
                assert!(
                    !matches!(
                        name,
                        "Forest"
                            | "Mountain"
                            | "Island"
                            | "Swamp"
                            | "Plains"
                            | "Jasmine Dragon Tea Shop"
                    ),
                    "{name}: {control:?}"
                );
                let status = serde_json::to_value(
                    control
                        .undo
                        .unwrap_or_else(|| panic!("missing status for {name}")),
                )
                .unwrap();
                assert!(
                    status["reason"]["explanation"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty()),
                    "{name}"
                );
            }
        };
        if !printed.mana_abilities.is_empty() {
            let mut env = position.fork().unwrap();
            env.enable_mana_controls();
            let controls = env.mana_controls(seat).unwrap();
            let offer = controls
                .offers
                .iter()
                .find_map(|o| match o {
                    ControlOffer::Float {
                        id: offer,
                        source: managym::agent::structured_offer::SubjectRef::Object { id: object },
                        ..
                    } if object.entity == id as u32 => Some(*offer),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("missing float for {name}"));
            env.execute_mana_control(&command("float", controls.revision, offer as u32))
                .unwrap();
            verify(&mut env);
        }
        for (ability, modes) in abilities {
            for mode in 0..modes {
                let mut env = position.fork().unwrap();
                env.enable_mana_controls();
                activate(&mut env, name, ability);
                // Chosen tap costs, then the output. A singleton output is forced.
                while env
                    .semantic_decision_frame()
                    .unwrap()
                    .offers
                    .iter()
                    .any(|o| o.verb == OfferVerb::SelectCard)
                {
                    choose(&mut env, OfferVerb::SelectCard, 0);
                }
                if modes > 1 {
                    choose(&mut env, OfferVerb::ChooseMode, mode);
                }
                let definition = &printed.activated_abilities[ability];
                if definition.life_cost > 0 && !definition.sacrifice_source {
                    assert_refusal(&env, "paid_life");
                } else if !definition.mana_suffix().is_empty() {
                    assert_refusal(&env, "additional_instructions");
                }
                verify(&mut env);
            }
        }
    }
}

#[test]
fn choosing_which_mana_to_spend_restores_the_pool_and_once_each_turn_use() {
    let mut env = position(&["Forest", "Mountain", "Scarecrow Guide"], "Forest");
    let seat = env.current_agent_index().unwrap();
    let original = env.state_digest().unwrap();
    for _ in 0..2 {
        let controls = env.mana_controls(seat).unwrap();
        let ControlOffer::Float { id, .. } = controls.offers[0] else {
            panic!()
        };
        env.execute_mana_control(&command("float", controls.revision, id as u32))
            .unwrap();
    }
    let before = env.state_digest().unwrap();
    activate(&mut env, "Scarecrow Guide", 0);
    choose(&mut env, OfferVerb::ChooseMode, 0);
    assert!(env.mana_controls(seat).unwrap().undo.is_none());
    choose(&mut env, OfferVerb::ChooseMode, 1);
    undo(&mut env, &before);
    // Undoing the paid ability also uncovers the two earlier floats.
    let controls = env.mana_controls(seat).unwrap();
    let ControlOffer::Undo { id, .. } = controls.offers.last().unwrap() else {
        panic!()
    };
    env.execute_mana_control(&command("undo", controls.revision, *id as u32))
        .unwrap();
    undo(&mut env, &original);
}
