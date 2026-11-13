use super::*;
use crate::{
    agent::{env::Env, observation::SideboardCardData},
    state::player::PlayerConfig,
};
use serde_json::json;

fn initial_observation() -> Observation {
    let players =
        ["a", "b"].map(|name| PlayerConfig::new(name, BTreeMap::from([("Forest".into(), 20)])));
    Env::new(42, true, false, false)
        .reset(players.to_vec())
        .unwrap()
        .0
}
fn board(kind: Space) -> Observation {
    let mut obs = initial_observation();
    obs.agent.id = 10;
    obs.opponent.id = 20;
    obs.turn.active_player_id = 10;
    obs.turn.turn_number = 3;
    obs.turn.step = StepKind::Main;
    obs.agent_cards.clear();
    obs.opponent_cards.clear();
    obs.recent_events.clear();
    obs.action_space.action_space_type = kind;
    obs.action_space.actions.clear();
    obs
}
fn offer(kind: Action, focus: &[i32], declared: Option<bool>) -> ActionOption {
    ActionOption {
        action_type: kind,
        focus: focus.to_vec(),
        declared,
        description: None,
        mode_index: None,
        name_definition: None,
        name_index: None,
        name_group: None,
        ability_index: None,
        continuation_index: None,
    }
}
fn card(id: i32, name: &str, mv: u8) -> CardData {
    let mut c = initial_observation().agent_cards.remove(0);
    c.id = id;
    c.registry_key = id;
    c.owner_id = 10;
    c.name = name.into();
    c.mana_cost.mana_value = mv;
    c.card_types.is_land = false;
    c.zone = ZoneType::Hand;
    c
}
fn creature(id: i32, owner: i32, power: i32, toughness: i32) -> PermanentData {
    serde_json::from_value(
        json!({"id":id,"registry_key":id,"controller_id":owner,"tapped":false,
        "damage":0,"is_summoning_sick":false,"plus1_counters":0,"cant_be_blocked_this_turn":false,
        "power":power,"toughness":toughness,"is_animated":true,"has_exile_link":false,
        "exile_if_dies_this_turn":false,"temp_subtypes":[],"keywords":KeywordData::default()}),
    )
    .unwrap()
}
fn lands(obs: &mut Observation, count: i32) {
    for i in 0..count {
        let id = 80 + i;
        let mut c = card(id, "Forest", 0);
        c.card_types.is_land = true;
        c.zone = ZoneType::Battlefield;
        let mut p = creature(id, 10, 0, 0);
        p.is_animated = false;
        obs.agent_cards.push(c);
        obs.agent_permanents.push(p);
    }
}
fn choose(obs: &Observation) -> usize {
    Heuristic::default().select_action(obs, None).unwrap()
}
fn attack(obs: &mut Observation, id: i32) {
    obs.action_space.actions = vec![
        offer(Action::DeclareAttacker, &[id], Some(true)),
        offer(Action::DeclareAttacker, &[id], Some(false)),
    ];
}

#[test]
fn heuristic_priority_timing_and_cast_ties() {
    let mut o = board(Space::Priority);
    o.agent_cards = vec![card(1, "Gray Ogre", 3), card(2, "Gray Ogre", 3)];
    o.action_space.actions = vec![
        offer(Action::PriorityCastSpell, &[1], None),
        offer(Action::PriorityCastSpell, &[2], None),
        offer(Action::PriorityPassPriority, &[], None),
    ];
    assert_eq!(choose(&o), 0);
    o.action_space
        .actions
        .push(offer(Action::PriorityPlayLand, &[3], None));
    assert_eq!(choose(&o), 3);
    o.action_space.actions.pop();
    o.agent_cards[0] = card(1, "Lightning Bolt", 1);
    o.agent_cards.truncate(1);
    assert_eq!(choose(&o), 2);
    o.opponent.life = 3;
    assert_eq!(choose(&o), 0);
    o.turn.step = StepKind::Upkeep;
    assert_eq!(choose(&o), 2);
    o.agent_cards[0] = card(1, "Pop Quiz", 3);
    o.turn.step = StepKind::End;
    assert_eq!(choose(&o), 2);
    o.turn.active_player_id = 20;
    assert_eq!(choose(&o), 0);
}
#[test]
fn heuristic_counters_wait_for_opposing_spells() {
    let mut o = board(Space::Priority);
    o.agent_cards = vec![card(1, "Counterspell", 2)];
    o.action_space.actions = vec![
        offer(Action::PriorityCastSpell, &[1], None),
        offer(Action::PriorityPassPriority, &[], None),
    ];
    assert_eq!(choose(&o), 1);
    o.stack_objects.push(
        serde_json::from_value(json!({
            "stack_object_id": 2, "kind": 0, "controller_id": 20,
            "source_card_registry_key": 2, "targets": []
        }))
        .unwrap(),
    );
    assert_eq!(choose(&o), 0);
    o.stack_objects[0].controller_id = 10;
    assert_eq!(choose(&o), 1);
}
#[test]
fn heuristic_target_polarity_lethal_and_decline() {
    let mut o = board(Space::ChooseTarget);
    o.agent_cards = vec![card(1, "Igneous Inspiration", 3)];
    o.action_space.focus = vec![1];
    o.agent_permanents = vec![creature(30, 10, 5, 3)];
    o.opponent_permanents = vec![creature(31, 20, 3, 3), creature(32, 20, 6, 6)];
    o.action_space.actions = [10, 20, 30, 31, 32]
        .iter()
        .map(|id| offer(Action::ChooseTarget, &[*id], None))
        .collect();
    assert_eq!(choose(&o), 3);
    o.opponent.life = 3;
    assert_eq!(choose(&o), 1);
    o.agent_cards[0] = card(1, "Fancy Footwork", 3);
    o.action_space.actions.drain(..2);
    assert_eq!(choose(&o), 0);
    o.action_space
        .actions
        .push(offer(Action::DeclineChoice, &[], None));
    assert_eq!(choose(&o), 3);
}
#[test]
fn heuristic_kicker_memory_and_optional_costs() {
    let mut o = board(Space::PayOrNot);
    let mut c = card(1, "Firebending Lesson", 1);
    c.kicker_cost = 4;
    o.agent_cards.push(c);
    o.action_space.focus = vec![1];
    o.action_space.actions = vec![
        offer(Action::PayCost, &[], None),
        offer(Action::DeclineChoice, &[], None),
    ];
    let mut h = Heuristic::default();
    assert_eq!(h.select_action(&o, None).unwrap(), 1);
    lands(&mut o, 5);
    assert_eq!(h.select_action(&o, None).unwrap(), 0);
    for p in &mut o.agent_permanents {
        p.tapped = true;
    }
    o.action_space.action_space_type = Space::ChooseTarget;
    o.opponent_permanents = vec![creature(30, 20, 2, 2), creature(31, 20, 4, 4)];
    o.action_space.actions = vec![
        offer(Action::ChooseTarget, &[30], None),
        offer(Action::ChooseTarget, &[31], None),
    ];
    assert_eq!(h.select_action(&o, None).unwrap(), 1);
    h.reset();
    assert_eq!(h.select_action(&o, None).unwrap(), 0);
    o.action_space.action_space_type = Space::PayOrNot;
    o.action_space.focus.clear();
    o.action_space.actions = vec![
        offer(Action::DeclineChoice, &[], None),
        offer(Action::PayCost, &[], None),
    ];
    assert_eq!(choose(&o), 1);
}
#[test]
fn heuristic_learn_discard_and_selection() {
    let mut o = board(Space::Learn);
    o.agent_sideboard = [(1, "Igneous Inspiration"), (2, "Introduction to Prophecy")]
        .map(|(id, name)| SideboardCardData {
            candidate_id: id,
            owner_id: 10,
            registry_key: id,
            name: name.into(),
            text_box: String::new(),
        })
        .to_vec();
    o.action_space.actions = vec![
        offer(Action::LearnTakeLesson, &[1], None),
        offer(Action::LearnTakeLesson, &[2], None),
    ];
    assert_eq!(choose(&o), 1);
    o.opponent_permanents.push(creature(30, 20, 2, 2));
    assert_eq!(choose(&o), 0);
    o.action_space.action_space_type = Space::Discard;
    o.agent_cards = vec![card(1, "Gray Ogre", 3), card(2, "Gray Ogre", 3)];
    o.action_space.actions = vec![
        offer(Action::SelectCard, &[1], None),
        offer(Action::SelectCard, &[2], None),
    ];
    assert_eq!(choose(&o), 1);
    o.action_space
        .actions
        .push(offer(Action::DeclineChoice, &[], None));
    assert_eq!(choose(&o), 2);
    o.agent_cards[0].card_types.is_land = true;
    lands(&mut o, 5);
    assert_eq!(choose(&o), 0);
    o.action_space.action_space_type = Space::LookAndSelect;
    assert_eq!(choose(&o), 1);
    o.agent_permanents.clear();
    assert_eq!(choose(&o), 0);
}
#[test]
fn heuristic_attacks_reserve_defenders_and_one_trick() {
    let mut o = board(Space::DeclareAttacker);
    o.agent_permanents = vec![creature(30, 10, 3, 3), creature(31, 10, 3, 3)];
    o.opponent_permanents = vec![creature(40, 20, 3, 4)];
    attack(&mut o, 30);
    assert_eq!(choose(&o), 1);
    o.agent_cards.push(card(1, "Fancy Footwork", 3));
    lands(&mut o, 3);
    let mut h = Heuristic::default();
    assert_eq!(h.select_action(&o, None).unwrap(), 0);
    attack(&mut o, 31);
    assert_eq!(h.select_action(&o, None).unwrap(), 1);
    o.agent_permanents[0].keywords.flying = true;
    o.opponent.life = 3;
    assert_eq!(choose(&o), 0);
    o.opponent.life = 20;
    o.agent.life = 3;
    o.agent_permanents[0].toughness = 6;
    attack(&mut o, 30);
    assert_eq!(choose(&o), 1);
}
#[test]
fn heuristic_combat_keywords_and_trade() {
    let mut o = board(Space::DeclareAttacker);
    o.agent_permanents = vec![creature(30, 10, 3, 3)];
    o.opponent_permanents = vec![creature(40, 20, 3, 4)];
    attack(&mut o, 30);
    assert_eq!(choose(&o), 1); // Dies without a trade.
    o.opponent_permanents[0].toughness = 3;
    assert_eq!(choose(&o), 0);
    o.opponent_permanents[0].keywords.first_strike = true;
    assert_eq!(choose(&o), 1);
    o.agent_permanents[0].keywords.double_strike = true;
    assert_eq!(choose(&o), 0);
    o.opponent_permanents[0].keywords.first_strike = false;
    o.agent_permanents[0].keywords.double_strike = false;
    o.opponent_permanents[0].power = 1;
    o.opponent_permanents[0].toughness = 4;
    o.opponent_permanents[0].keywords.deathtouch = true;
    assert_eq!(choose(&o), 1);
    o.agent_permanents[0].keywords.flying = true;
    assert_eq!(choose(&o), 0);
    o.opponent_permanents[0].keywords.reach = true;
    assert_eq!(choose(&o), 1);
    o.opponent_permanents[0].tapped = true;
    assert_eq!(choose(&o), 0);
}
#[test]
fn heuristic_blocks_spread_and_chump_for_lethal() {
    let mut o = board(Space::DeclareBlocker);
    o.agent_permanents = vec![creature(30, 10, 2, 2), creature(31, 10, 2, 2)];
    o.opponent_permanents = vec![creature(40, 20, 1, 1), creature(41, 20, 5, 5)];
    o.action_space.actions = vec![
        offer(Action::DeclareBlocker, &[30, 40], Some(true)),
        offer(Action::DeclareBlocker, &[30, 41], Some(true)),
        offer(Action::DeclareBlocker, &[30], Some(false)),
    ];
    // Equal blocks are assigned by descending blocker id; 31 takes the 1/1.
    assert_eq!(choose(&o), 2);
    o.agent.life = 5;
    assert_eq!(choose(&o), 1);
}
#[test]
fn heuristic_pumps_only_when_combat_improves() {
    let mut o = board(Space::Priority);
    o.turn.step = StepKind::DeclareBlockers;
    o.agent_cards.push(card(1, "Fancy Footwork", 3));
    o.agent_permanents.push(creature(30, 10, 2, 2));
    o.opponent_permanents.push(creature(40, 20, 3, 3));
    o.action_space.actions = vec![
        offer(Action::PriorityCastSpell, &[1], None),
        offer(Action::PriorityPassPriority, &[], None),
    ];
    assert_eq!(choose(&o), 1);
    o.recent_events.push(crate::agent::observation::EventData {
        event_type: EventType::BlockersDeclared as i32,
        source_kind: 2,
        source_id: 30,
        target_kind: 2,
        target_id: 40,
        amount: 0,
        controller_id: 10,
        from_zone: -1,
        to_zone: -1,
        source_incarnation: -1,
        target_incarnation: -1,
    });
    assert_eq!(choose(&o), 0);
    o.opponent_permanents[0].power = 8;
    o.opponent_permanents[0].toughness = 8;
    assert_eq!(choose(&o), 1);
}
#[test]
fn heuristic_abilities_and_lesson_damage() {
    let mut o = board(Space::Priority);
    let mut c = card(30, "Fabled Passage", 0);
    c.zone = ZoneType::Battlefield;
    o.agent_cards.push(c);
    o.agent_permanents.push(creature(30, 10, 0, 0));
    o.action_space.actions = vec![
        offer(Action::PriorityPassPriority, &[], None),
        offer(Action::PriorityActivateAbility, &[30], None),
    ];
    assert_eq!(choose(&o), 1);
    o.agent_cards.push(card(1, "Combustion Technique", 2));
    o.opponent_permanents.push(creature(40, 20, 4, 4));
    o.action_space.actions[1] = offer(Action::PriorityCastSpell, &[1], None);
    assert_eq!(choose(&o), 0);
    o.agent.graveyard_lessons = 2;
    assert_eq!(choose(&o), 1);
}
#[test]
fn heuristic_prefix_and_empty_errors() {
    let mut o = board(Space::Modal);
    o.action_space.actions = vec![
        offer(Action::DeclineChoice, &[], None),
        offer(Action::PayCost, &[], None),
    ];
    let mut h = Heuristic::default();
    assert_eq!(h.select_action(&o, None).unwrap(), 1);
    assert_eq!(h.select_action(&o, Some(1)).unwrap(), 0);
    assert!(h.select_action(&o, Some(0)).is_err());
    o.action_space.actions.clear();
    assert!(h.select_action(&o, None).is_err());
}

#[test]
fn heuristic_env_forks_copy_history_and_same_seed_reset_clears_it() {
    use crate::agent::opponent::PlayerPolicy;

    let players = ["a", "b"]
        .map(|name| PlayerConfig::new(name, BTreeMap::from([("Forest".into(), 20)])))
        .to_vec();
    let mut env = Env::new(42, true, false, false);
    env.reset(players.clone()).unwrap();
    // Select through the batch policy boundary. Kicker history must survive
    // a fork even after the lands used to estimate its cost become tapped.
    let policy = PlayerPolicy::Heuristic;
    let mut obs = board(Space::PayOrNot);
    let mut spell = card(1, "Firebending Lesson", 1);
    spell.kicker_cost = 4;
    obs.agent_cards.push(spell);
    obs.action_space.focus = vec![1];
    obs.action_space.actions = vec![
        offer(Action::PayCost, &[], None),
        offer(Action::DeclineChoice, &[], None),
    ];
    lands(&mut obs, 5);
    assert_eq!(policy.select_action(&mut env, &obs).unwrap(), 0);
    let mut fork = env.fork().unwrap();
    let mut selected = env.selected_fork().unwrap();
    for permanent in &mut obs.agent_permanents {
        permanent.tapped = true;
    }
    obs.action_space.action_space_type = Space::ChooseTarget;
    obs.opponent_permanents = vec![creature(30, 20, 2, 2), creature(31, 20, 4, 4)];
    obs.action_space.actions = vec![
        offer(Action::ChooseTarget, &[30], None),
        offer(Action::ChooseTarget, &[31], None),
    ];
    assert_eq!(policy.select_action(&mut env, &obs).unwrap(), 1);
    env.reset(players).unwrap();
    assert_eq!(policy.select_action(&mut env, &obs).unwrap(), 0);
    assert_eq!(policy.select_action(&mut fork, &obs).unwrap(), 1);
    assert_eq!(policy.select_action(&mut selected, &obs).unwrap(), 1);
}

#[test]
fn heuristic_per_seat_memory_is_independent_and_forked() {
    use crate::agent::opponent::PlayerPolicy;
    let players = ["a", "b"]
        .map(|name| PlayerConfig::new(name, BTreeMap::from([("Forest".into(), 20)])))
        .to_vec();
    let mut env = Env::new(42, true, false, false);
    env.reset(players.clone()).unwrap();
    let mut obs = board(Space::PayOrNot);
    let mut spell = card(1, "Firebending Lesson", 1);
    spell.kicker_cost = 4;
    obs.agent_cards.push(spell);
    obs.action_space.focus = vec![1];
    obs.action_space.actions = vec![
        offer(Action::PayCost, &[], None),
        offer(Action::DeclineChoice, &[], None),
    ];
    lands(&mut obs, 5);
    PlayerPolicy::Heuristic
        .select_action(&mut env, &obs)
        .unwrap();
    assert!(env.heuristics[0].is_some());
    assert!(env.heuristics[1].is_none());
    obs.agent.player_index = 1;
    PlayerPolicy::Heuristic
        .select_action(&mut env, &obs)
        .unwrap();
    assert!(env.heuristics[1].is_some());
    let fork = env.fork().unwrap();
    env.reset(players).unwrap();
    assert!(env.heuristics.iter().all(Option::is_none));
    assert!(fork.heuristics.iter().all(Option::is_some));
}

#[test]
fn heuristic_per_seat_scalar_vector_parity_and_terminal_rewards() {
    use crate::agent::{opponent::PlayerPolicy, vector_env::VectorEnv};
    let players = ["a", "b"]
        .map(|name| {
            PlayerConfig::new(
                name,
                BTreeMap::from([("Forest".into(), 8), ("Llanowar Elves".into(), 4)]),
            )
        })
        .to_vec();
    for policies in [
        [PlayerPolicy::Agent, PlayerPolicy::Heuristic],
        [PlayerPolicy::Heuristic, PlayerPolicy::Agent],
        [PlayerPolicy::Random, PlayerPolicy::Heuristic],
        [PlayerPolicy::Heuristic, PlayerPolicy::Heuristic],
    ] {
        let mut scalar = Env::new(213, true, false, false).with_player_policies(policies);
        let mut vector = VectorEnv::new_with_player_policies(1, 213, true, policies);
        let (mut obs, _) = scalar.reset(players.clone()).unwrap();
        let batch = vector.reset_all(players.clone()).unwrap();
        assert_eq!(obs.to_json(), batch[0].0.to_json());
        let mut finished = false;
        for _ in 0..1024 {
            let chooser = obs.agent.player_index as usize;
            let action = if policies[chooser] == PlayerPolicy::Agent {
                0
            } else {
                -1
            };
            let (next, reward, done, truncated, info) = scalar.step(action).unwrap();
            let batch = vector.step(&[action]).unwrap();
            assert_eq!(reward, batch[0].reward);
            assert_eq!(done, batch[0].terminated);
            assert_eq!(truncated, batch[0].truncated);
            if done {
                if let Some(crate::infra::profiler::InfoValue::Int(winner)) =
                    info.get("winner_index")
                {
                    assert_eq!(
                        reward,
                        if *winner as usize == chooser {
                            1.0
                        } else {
                            -1.0
                        }
                    );
                }
                finished = true;
                break;
            }
            assert_eq!(next.to_json(), batch[0].obs.to_json());
            obs = next;
        }
        assert!(finished, "bounded fixture must finish: {policies:?}");
    }
}

#[test]
fn heuristic_builtin_steps_retain_live_event_parity() {
    use crate::agent::opponent::PlayerPolicy;
    let setup = crate::cards::setups::authored_setup("elemental-world-v2").unwrap();
    let players = ["elemental_ur", "elemental_gw"]
        .map(|key| setup.player_config(key, key).unwrap())
        .to_vec();
    let mut builtins =
        Env::new(213, true, false, false).with_player_policies([PlayerPolicy::Heuristic; 2]);
    let mut manual = Env::new(213, true, false, false);
    builtins.reset(players.clone()).unwrap();
    let (mut obs, _) = manual.reset(players).unwrap();
    let mut selectors = [Heuristic::default(), Heuristic::default()];
    for _ in 0..256 {
        let chooser = obs.agent.player_index as usize;
        let index = selectors[chooser].select_action(&obs, None).unwrap();
        let expected = manual.step(index as i64).unwrap();
        let actual = builtins.step(-1).unwrap();
        assert_eq!(actual.0.to_json(), expected.0.to_json());
        assert_eq!(
            (actual.1, actual.2, actual.3),
            (expected.1, expected.2, expected.3)
        );
        obs = expected.0;
        if expected.2 || expected.3 {
            break;
        }
    }
}
