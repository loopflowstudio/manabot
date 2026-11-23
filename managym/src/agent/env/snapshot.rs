//! An immutable decision boundary. Only live branches have projection caches,
//! producer sinks or a generation used to admit append-only memory transport.
use super::*;
use crate::{
    agent::{
        action::{ActionSpace, ActionSpaceKind},
        decision_memory::DecisionLog,
        policy_history::HistoryWindow,
    },
    decision::ObjectCandidateAddress,
    flow::{
        game::{PendingChoice, TrackedState},
        turn::PhaseKind,
    },
    state::{card::CardDefId, game_object::ObjectId},
};
use std::collections::BTreeSet;

mod replay;

/// A detached live position shareable by any number of restoring threads.
pub struct GameSnapshot {
    state: TrackedState,
    decision_memory: [DecisionLog; 2],
    memory_enabled: bool,
    include_own: bool,
    history: HistoryWindow,
    action_space: Option<ActionSpace>,
    pending_choice: Option<PendingChoice>,
    epoch: u64,
    candidates: BTreeMap<ObjectCandidateAddress, ObjectId>,
    skip_trivial: bool,
    skipped: usize,
    priority_passes: usize,
    mana_floats: [ManaFloats; 2],
    mana_payment: [ManaPayment; 2],
    heuristics: [Option<Box<Heuristic>>; 2],
    heuristic_events: Vec<EventData>,
    policies: [PlayerPolicy; 2],
    reset_seed: u64,
    reset_setup: Option<Arc<GameSetup>>,
    provenance: Option<GameProvenance>,
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GameSnapshot>();
};

#[derive(Clone, Copy)]
pub enum BranchMode {
    Exact { reseed: Option<u64> },
    ShuffleLibraries { seed: u64 },
    Determinize { viewer: usize, seed: u64 },
}

#[derive(Clone, Copy)]
pub struct RestoreOptions {
    pub mode: BranchMode,
    pub enable_decision_memory: Option<bool>,
    pub forced_first_action: Option<usize>,
}

impl RestoreOptions {
    pub fn exact(reseed: Option<u64>) -> Self {
        Self {
            mode: BranchMode::Exact { reseed },
            enable_decision_memory: None,
            forced_first_action: None,
        }
    }
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self::exact(None)
    }
}

/// Public board and decision summary at the saved boundary.
pub struct SnapshotMetadata {
    pub turn: u32,
    pub phase: PhaseKind,
    pub life: [i32; 2],
    pub board_counts: [usize; 2],
    pub stack_len: usize,
    pub action_count: usize,
    pub action_kind: Option<ActionSpaceKind>,
    pub acting_player: Option<usize>,
    pub terminal: bool,
    pub active_card_defs: Vec<CardDefId>,
}

impl Env {
    pub fn snapshot(&self) -> Result<GameSnapshot, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("snapshot before reset".into()))?;
        Ok(GameSnapshot {
            state: game.state.clone(),
            decision_memory: game.decision_memory.clone(),
            memory_enabled: game.decision_memory_enabled,
            include_own: game.decision_memory_include_own,
            history: game.policy_history.clone(),
            action_space: game.current_action_space.clone(),
            pending_choice: game.pending_choice.clone(),
            epoch: game.decision_epoch,
            candidates: game.semantic_object_candidates.borrow().clone(),
            skip_trivial: game.skip_trivial,
            skipped: game.skip_trivial_count,
            priority_passes: game.skipped_priority_passes,
            mana_floats: game.mana_floats,
            mana_payment: game.mana_payment,
            heuristics: self.heuristics.clone(),
            heuristic_events: self.heuristic_events.clone(),
            policies: self.player_policies,
            reset_seed: self.reset_seed,
            reset_setup: self.reset_setup.clone(),
            provenance: self.provenance.clone(),
        })
    }
}

impl GameSnapshot {
    /// Materialize a new live game generation, continuing both viewer logs.
    pub fn restore(&self, options: RestoreOptions) -> Result<Env, AgentError> {
        if let BranchMode::Determinize { viewer, .. } = options.mode {
            self.state.sampling_viewer(viewer)?;
        }
        let forced = if let Some(index) = options.forced_first_action {
            let space = self
                .action_space
                .as_ref()
                .filter(|_| !self.state.is_game_over())
                .ok_or_else(|| {
                    AgentError("forced_first_action requires a non-terminal decision".into())
                })?;
            let action = space
                .actions
                .get(index)
                .ok_or_else(|| AgentError(format!("forced_first_action {index} out of bounds")))?;
            Some((index, space.player, action))
        } else {
            None
        };
        let mut env = self.live_env();
        let intervention = match options.mode {
            BranchMode::Exact { reseed } => reseed.map(|seed| Intervention::Reseed { seed }),
            BranchMode::ShuffleLibraries { seed } => Some(Intervention::ShuffleLibraries { seed }),
            BranchMode::Determinize { viewer, seed } => {
                Some(Intervention::Determinize { viewer, seed })
            }
        };
        if let Some(op) = intervention {
            env.apply_intervention(op)?;
        }
        if let Some(enabled) = options
            .enable_decision_memory
            .filter(|&enabled| enabled != self.memory_enabled)
        {
            env.apply_intervention(Intervention::SetDecisionMemory { enabled })?;
        }
        if let Some((index, chooser, action)) = forced {
            let current = env.game.as_ref().and_then(|g| g.action_space());
            let seat = env
                .current_agent_index()
                .ok_or_else(|| AgentError("no active chooser".into()))?;
            if Some(PlayerId(seat)) != chooser
                || current.and_then(|s| s.actions.get(index)) != Some(action)
            {
                return Err(AgentError(format!(
                    "forced_first_action {index} changed after resampling"
                )));
            }
            if env.player_policies[seat] == PlayerPolicy::Agent {
                env.step(index as i64)?;
            } else {
                env.apply_intervention(Intervention::ForceAction { action: index })?;
            }
        }
        Ok(env)
    }

    fn live_env(&self) -> Env {
        let game = Game {
            state: self.state.clone(),
            decision_memory: self.decision_memory.clone(),
            decision_memory_enabled: self.memory_enabled,
            decision_memory_include_own: self.include_own,
            policy_history: self.history.clone(),
            current_action_space: self.action_space.clone(),
            pending_choice: self.pending_choice.clone(),
            decision_epoch: self.epoch,
            semantic_object_candidates: self.candidates.clone().into(),
            skip_trivial: self.skip_trivial,
            skip_trivial_count: self.skipped,
            skipped_priority_passes: self.priority_passes,
            mana_floats: self.mana_floats,
            mana_payment: self.mana_payment,
            derived: Default::default(),
            payment_cache: Default::default(),
            payment_sources: Default::default(),
            statistics: None,
            card_capture: None,
            capture_originals: Default::default(),
            capture_drawing: None,
        };
        Env {
            game: Some(game),
            heuristics: self.heuristics.clone(),
            heuristic_events: self.heuristic_events.clone(),
            player_policies: self.policies,
            provenance: self.provenance.clone(),
            reset_setup: self.reset_setup.clone(),
            decision_memory_enabled: self.memory_enabled,
            decision_memory_include_own: self.include_own,
            capture_generation: GAME_GENERATION.fetch_add(1, Ordering::Relaxed),
            ..Env::new(self.reset_seed, self.skip_trivial, false, false)
        }
    }

    pub fn metadata(&self) -> SnapshotMetadata {
        let mut definitions = BTreeSet::new();
        let mut board_counts = [0; 2];
        for seat in [PlayerId(0), PlayerId(1)] {
            for &object in self.state.zones.battlefield_objects(seat) {
                if let Some(id) = self.state.permanent_of(object) {
                    let permanent = self.state.permanents[id]
                        .as_ref()
                        .expect("live battlefield permanent");
                    board_counts[permanent.controller.0] += 1;
                    definitions.insert(self.state[object].definition_id);
                }
            }
        }
        definitions.extend(
            self.state
                .stack_objects
                .iter()
                .map(|obj| obj.source_definition_id()),
        );
        let terminal = self.state.is_game_over();
        let space = self.action_space.as_ref().filter(|_| !terminal);
        SnapshotMetadata {
            turn: self.state.turn.turn_number,
            phase: self.state.turn.current_phase_kind(),
            life: self.state.players.each_ref().map(|p| p.life),
            board_counts,
            stack_len: self.state.stack_objects.len(),
            action_count: space.map_or(0, |s| s.actions.len()),
            action_kind: space.map(|s| s.kind),
            acting_player: space.and_then(|s| s.player).map(|p| p.0),
            terminal,
            active_card_defs: definitions.into_iter().collect(),
        }
    }
}

pub(super) fn state_digest(
    state: &TrackedState,
    actions: &Option<ActionSpace>,
    pending: &Option<PendingChoice>,
    skip_trivial: bool,
    skipped: usize,
) -> String {
    let surface = (
        state.deterministic_hash_value(),
        actions,
        pending,
        skip_trivial,
        skipped,
    );
    let canonical = serde_json::to_vec(&surface).expect("semantic differential surface serializes");
    blake3::hash(&canonical).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn concurrent_exact_snapshots_preserve_the_witness_and_live_decision_memory() {
        let mut source = Env::new(42, true, false, false).with_decision_memory_include_own(true);
        let players = ["a", "b"].map(|name| {
            PlayerConfig::new(
                name,
                BTreeMap::from([("Mountain".into(), 8), ("Gray Ogre".into(), 4)]),
            )
        });
        source.reset(players.to_vec()).unwrap();
        source.step(0).unwrap();
        source.set_seed(99);
        let snapshot = Arc::new(source.snapshot().unwrap());
        let prefix = source.selected_witness().unwrap();
        let logs = source.game.as_ref().unwrap().decision_memory.clone();
        assert!(logs.iter().all(|log| !log.is_empty()));
        let generation = source.capture_generation;
        source.step(0).unwrap();
        let after = source.selected_witness().unwrap();
        let next_logs = source.game.as_ref().unwrap().decision_memory.clone();
        assert_ne!(logs, next_logs);
        drop(source);

        let cold =
            Arc::new(GameSnapshot::from_replay_json(&snapshot.to_replay_json().unwrap()).unwrap());
        let mut generations = BTreeSet::from([generation]);
        thread::scope(|scope| {
            let handles: Vec<_> = [snapshot.clone(), snapshot.clone(), cold]
                .into_iter()
                .map(|snapshot| {
                    let (prefix, logs, after, next_logs) = (&prefix, &logs, &after, &next_logs);
                    scope.spawn(move || {
                        let mut branch = snapshot.restore(RestoreOptions::default()).unwrap();
                        assert_eq!(branch.seed(), 42);
                        assert_eq!(&branch.selected_witness().unwrap(), prefix);
                        assert_eq!(&branch.game.as_ref().unwrap().decision_memory, logs);
                        branch.step(0).unwrap();
                        assert_eq!(&branch.selected_witness().unwrap(), after);
                        assert_eq!(&branch.game.as_ref().unwrap().decision_memory, next_logs);
                        branch.capture_generation
                    })
                })
                .collect();
            for handle in handles {
                assert!(generations.insert(handle.join().unwrap()));
            }
        });
        assert_eq!(
            snapshot
                .restore(Default::default())
                .unwrap()
                .selected_witness()
                .unwrap(),
            prefix
        );
    }
}

#[cfg(test)]
mod card_history_tests {
    use super::{Env, RestoreOptions};
    use crate::{
        agent::action::Action,
        decision::Command,
        state::{
            game_object::{PlayerId, Target},
            player::PlayerConfig,
            zone::ZoneType,
        },
    };

    #[test]
    fn predator_ooze_remembers_only_the_exact_damage_pair_this_turn() {
        use crate::agent::{
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
            observation_layout::relation,
        };
        use crate::flow::game::Game;
        use crate::flow::proposed_event::ProposedDamageTarget;
        let mut env = Env::new(418, true, false, false);
        let deck = [("Predator Ooze".to_owned(), 12), ("Forest".to_owned(), 28)].into();
        env.reset(vec![
            PlayerConfig::new("p0", deck),
            PlayerConfig::new(
                "p1",
                [("Hill Giant".into(), 12), ("Forest".into(), 28)].into(),
            ),
        ])
        .unwrap();
        let game = env.game.as_mut().unwrap();
        let source = game
            .scenario_force_battlefield(PlayerId(0), "Predator Ooze", true)
            .unwrap();
        let victim = game
            .scenario_force_battlefield(PlayerId(1), "Hill Giant", true)
            .unwrap();
        let recipient = game.permanent_object(victim).unwrap();
        let ooze = game.permanent_object(source).unwrap();
        let hit = |g: &mut Game| {
            assert!(g.commit_damage(
                Some(ooze),
                ProposedDamageTarget::Object(recipient),
                1,
                false
            ))
        };
        let other = game
            .scenario_force_battlefield(PlayerId(0), "Predator Ooze", true)
            .unwrap();
        let other_ooze = game.permanent_object(other).unwrap();
        let mut alternate = game.clone();
        alternate.commit_damage(
            Some(other_ooze),
            ProposedDamageTarget::Object(recipient),
            1,
            false,
        );
        hit(game);
        let config = ObservationEncoderConfig::default();
        for viewer in [PlayerId(0), PlayerId(1)] {
            let encoded_pair = |g: &Game| {
                let obs = Observation::for_player(g, viewer);
                let encoded = encode(&obs, &config).unwrap();
                let rows: Vec<_> = encoded
                    .relations
                    .chunks(relation::WIDTH)
                    .filter(|r| r[relation::KIND] == 11.)
                    .collect();
                assert_eq!(rows.len(), 1);
                (rows[0][relation::SOURCE], rows[0][relation::REFERENCE])
            };
            let pair = encoded_pair(game);
            let other_pair = encoded_pair(&alternate);
            assert_ne!(pair.0, other_pair.0);
            assert_eq!(pair.1, other_pair.1);
            assert!(pair.0 > 0. && pair.1 > 0.);
        }
        let witness = env.selected_witness().unwrap();
        let snapshot = env.snapshot().unwrap();
        let fork = || snapshot.restore(RestoreOptions::default()).unwrap();
        let blink = |g: &mut Game, object| {
            let exiled = g.move_object(object, ZoneType::Exile);
            g.move_object(exiled, ZoneType::Battlefield)
        };
        let die = |g: &mut Game, object| {
            g.move_object(object, ZoneType::Graveyard);
            g.process_game_events();
            g.state.pending_triggers.len()
        };
        let mut branch_env = fork();
        assert_eq!(branch_env.selected_witness().unwrap(), witness);
        let repeated = branch_env.game.as_mut().unwrap();
        hit(repeated);
        assert_eq!(die(repeated, recipient), 1);
        assert!(repeated.flush_triggers().is_none());
        let pending = branch_env.snapshot().unwrap();
        let mut restored = pending.restore(RestoreOptions::default()).unwrap();
        let countered = restored.game.as_mut().unwrap();
        let repeated = branch_env.game.as_mut().unwrap();
        let id = countered.state.stack_objects.last().unwrap().id();
        countered.counter_ability(id, None);
        repeated.resolve_top_of_stack();
        assert_eq!(repeated.characteristics(source).power, 2);
        assert_eq!(countered.characteristics(source).power, 1);

        let mut branch = fork().game.take().unwrap();
        let returned = blink(&mut branch, recipient);
        assert_eq!(die(&mut branch, returned), 0);
        let mut branch = fork().game.take().unwrap();
        blink(&mut branch, ooze);
        assert_eq!(die(&mut branch, recipient), 0);
        let mut both = fork().game.take().unwrap();
        both.commit_damage(
            Some(other_ooze),
            ProposedDamageTarget::Object(recipient),
            1,
            false,
        );
        assert_eq!(die(&mut both, recipient), 2);
        assert_eq!(env.selected_witness().unwrap(), witness);
        let game = env.game.as_mut().unwrap();
        let initial_turn = game.state.turn.turn_number;
        while game.state.turn.turn_number == initial_turn {
            game.state.turn.advance_step();
        }
        assert_eq!(die(game, recipient), 0);
    }

    #[test]
    fn panda_snapshot_replays_source_and_target_reentry() {
        let mut env = Env::new(418, false, false, false);
        let deck = [
            "Fiendish Panda",
            "Gray Ogre",
            "Hill Giant",
            "Serra Angel",
            "Forest",
        ]
        .into_iter()
        .map(|name| (name.to_owned(), 8))
        .collect();
        env.reset(vec![
            PlayerConfig::new("Panda", deck),
            PlayerConfig::new("Opponent", [("Forest".into(), 40)].into()),
        ])
        .unwrap();
        let game = env.game.as_mut().unwrap();
        let player = PlayerId(0);
        let panda = game
            .scenario_force_battlefield(player, "Fiendish Panda", true)
            .unwrap();
        let mut bury = |name| {
            let object = game
                .state
                .zones
                .zone_objects(ZoneType::Library, player)
                .iter()
                .copied()
                .find(|id| game.state[*id].name == name)
                .unwrap();
            game.move_object(object, ZoneType::Graveyard)
        };
        let giant = bury("Hill Giant");
        let ogre = bury("Gray Ogre");
        bury("Fiendish Panda");
        bury("Serra Angel");
        game.gain_life(player, 3);
        assert!(game.stabilize_before_priority().is_none());
        game.resolve_top_of_stack();
        assert_eq!(game.characteristics(panda).power, 4);
        let dead = game.move_object(game.permanent_object(panda).unwrap(), ZoneType::Graveyard);
        let choice = game.stabilize_before_priority().unwrap();
        let targets: Vec<_> = choice
            .actions
            .iter()
            .map(|action| match action {
                Action::ChooseTarget { target, .. } => *target,
                _ => panic!("expected a graveyard target"),
            })
            .collect();
        assert_eq!(targets.len(), 2);
        assert!(targets.contains(&Target::GraveyardCard(ogre)));
        let index = targets
            .iter()
            .position(|target| *target == Target::GraveyardCard(giant))
            .unwrap();
        game.publish_action_space(choice);

        let frame = env.semantic_decision_frame().unwrap();
        let command = Command {
            command_id: "return-giant".into(),
            expected_revision: frame.revision,
            offer_id: frame.offers[index].id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        let snapshot = env.snapshot().unwrap();
        let boundary = |branch: &Env| {
            (
                branch.selected_witness().unwrap(),
                serde_json::to_value(branch.semantic_decision_frame().unwrap()).unwrap(),
                [0, 1].map(|seat| branch.observation_for_player(seat).unwrap().to_json()),
            )
        };
        let original = boundary(&env);
        for move_target in [false, true] {
            let replay = || {
                let mut branch = snapshot.restore(RestoreOptions::default()).unwrap();
                assert_eq!(boundary(&branch), original);
                branch.execute_semantic_command(&command).unwrap();
                let game = branch.game.as_mut().unwrap();
                let card = game.state.card_of(giant).unwrap();
                if move_target {
                    let away = game.move_object(giant, ZoneType::Exile);
                    game.move_object(away, ZoneType::Graveyard);
                } else {
                    let returned = game.move_object(dead, ZoneType::Battlefield);
                    assert_eq!(
                        game.characteristics(game.state.permanent_of(returned).unwrap())
                            .power,
                        3
                    );
                }
                game.resolve_top_of_stack();
                assert_eq!(
                    game.state.zones.zone_of(game.state.object_of(card)),
                    Some(if move_target {
                        ZoneType::Graveyard
                    } else {
                        ZoneType::Battlefield
                    })
                );
                game.scenario_refresh_priority().unwrap();
                boundary(&branch)
            };
            assert_eq!(replay(), replay());
        }
        assert_eq!(boundary(&env), original);
    }
}
