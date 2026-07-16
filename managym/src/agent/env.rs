use crate::{
    agent::{
        action::{Action, AgentError},
        behavior_tracker::BehaviorTracker,
        observation::Observation,
        observation_encoder::{
            encode, encode_into, EncodedObservation, EncodedObservationMut, ObservationEncodeError,
            ObservationEncoderConfig,
        },
    },
    flow::{game::Game, search::mix_seed},
    infra::profiler::{empty_info_dict, insert_info, InfoDict, InfoValue, Profiler},
    state::{game_object::PlayerId, player::PlayerConfig},
};
use rand::Rng;

/// Result of a flat Monte Carlo evaluation of the current action space.
#[derive(Clone, Debug)]
pub struct FlatMcResult {
    /// Mean playout score per legal action (win = 1.0, loss = 0.0,
    /// draw/step-cap = 0.5), for the player holding the current decision.
    pub scores: Vec<f64>,
    /// Total playouts performed (actions x worlds x rollouts).
    pub simulations: u64,
    /// Playouts that hit the step cap without terminating.
    pub cap_hits: u64,
}

#[derive(Debug)]
pub struct Env {
    game: Option<Game>,
    skip_trivial: bool,
    seed: u64,
    pub profiler: Profiler,
    pub hero_tracker: BehaviorTracker,
    pub villain_tracker: BehaviorTracker,
}

impl Env {
    pub fn new(
        seed: u64,
        skip_trivial: bool,
        enable_profiler: bool,
        enable_behavior_tracking: bool,
    ) -> Self {
        Self {
            game: None,
            skip_trivial,
            seed,
            profiler: Profiler::new(enable_profiler, 64),
            hero_tracker: BehaviorTracker::new(enable_behavior_tracking),
            villain_tracker: BehaviorTracker::new(enable_behavior_tracking),
        }
    }

    pub fn reset(
        &mut self,
        player_configs: Vec<PlayerConfig>,
    ) -> Result<(Observation, InfoDict), AgentError> {
        let _scope = self.profiler.track("env_reset");
        let mut game = Game::new(player_configs, self.seed, self.skip_trivial);
        let events = game.take_observation_events();
        let observation = Observation::new(&game, &events);
        self.game = Some(game);
        Ok((observation, empty_info_dict()))
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Number of trivial decision points auto-collapsed by `skip_trivial`
    /// since the current game began. Resets to zero on `reset`.
    pub fn skip_trivial_count(&self) -> usize {
        self.game.as_ref().map_or(0, |game| game.skip_trivial_count)
    }

    pub fn step(
        &mut self,
        action: i64,
    ) -> Result<(Observation, f64, bool, bool, InfoDict), AgentError> {
        let _scope = self.profiler.track("env_step");
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.step called before reset".to_string()))?;

        if game.is_game_over() {
            return Err(AgentError("env.step called after game over".to_string()));
        }

        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        let agent = action_space
            .player
            .ok_or_else(|| AgentError("no agent player in current action space".to_string()))?;
        let action_count = action_space.actions.len();
        let out_of_bounds = || {
            AgentError(format!(
                "Action index {action} out of bounds: {action_count}"
            ))
        };
        let action = match usize::try_from(action) {
            Ok(index) if index < action_count => index,
            _ => return Err(out_of_bounds()),
        };

        let done = game.step(action)?;
        let events = game.take_observation_events();
        let observation = Observation::new(game, &events);

        let mut reward = 0.0;
        let mut info = empty_info_dict();
        if done {
            if let Some(winner) = game.winner_index() {
                reward = if winner == agent.0 { 1.0 } else { -1.0 };
                insert_info(&mut info, "winner_index", InfoValue::Int(winner as i64));
                insert_info(
                    &mut info,
                    "winner_name",
                    InfoValue::String(game.state.players[winner].name.clone()),
                );
            } else {
                insert_info(
                    &mut info,
                    "winner_name",
                    InfoValue::String("draw".to_string()),
                );
            }
            for (i, player) in game.state.players.iter().enumerate() {
                if !player.alive {
                    let reason = if player.drew_when_empty {
                        "deck_empty"
                    } else {
                        "life_total"
                    };
                    insert_info(
                        &mut info,
                        format!("p{i}_loss_reason"),
                        InfoValue::String(reason.to_string()),
                    );
                }
            }
            self.add_profiler_info(&mut info);
            self.add_behavior_info(&mut info);
        }

        Ok((observation, reward, done, false, info))
    }

    pub fn info(&self) -> InfoDict {
        let _scope = self.profiler.track("env_info");
        let mut info = empty_info_dict();
        self.add_profiler_info(&mut info);
        self.add_behavior_info(&mut info);
        info
    }

    pub fn encode_observation(&self, observation: &Observation) -> EncodedObservation {
        encode(observation, &ObservationEncoderConfig::default())
    }

    pub fn encode_observation_into(
        &self,
        observation: &Observation,
        out: EncodedObservationMut<'_>,
    ) -> Result<(), ObservationEncodeError> {
        encode_into(observation, &ObservationEncoderConfig::default(), out)
    }

    pub fn export_profile_baseline(&self) -> String {
        if self.profiler.is_enabled() {
            self.profiler.export_baseline()
        } else {
            String::new()
        }
    }

    pub fn compare_profile(&self, baseline: &str) -> String {
        if self.profiler.is_enabled() {
            self.profiler.compare_to_baseline(baseline)
        } else {
            "Profiler not enabled".to_string()
        }
    }

    pub fn pass_priority_action_index(&self) -> Result<usize, AgentError> {
        let game = self.game.as_ref().ok_or_else(|| {
            AgentError("env.pass_priority_action_index called before reset".to_string())
        })?;
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        if action_space.actions.is_empty() {
            return Err(AgentError("no valid actions available".to_string()));
        }
        Ok(action_space
            .actions
            .iter()
            .position(|action| matches!(action, Action::PassPriority { .. }))
            .unwrap_or(0))
    }

    /// Independent copy of this env's game for search rollouts.
    ///
    /// Profiling and behavior tracking are disabled on the fork; stepping the
    /// fork never mutates the original.
    pub fn fork(&self) -> Result<Env, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.fork called before reset".to_string()))?
            .clone();
        Ok(Env {
            game: Some(game),
            skip_trivial: self.skip_trivial,
            seed: self.seed,
            profiler: Profiler::new(false, 64),
            hero_tracker: BehaviorTracker::new(false),
            villain_tracker: BehaviorTracker::new(false),
        })
    }

    /// Kind of the current action space, if any.
    pub fn action_space_kind(&self) -> Option<crate::agent::action::ActionSpaceKind> {
        self.game
            .as_ref()
            .and_then(|game| game.action_space())
            .map(|space| space.kind)
    }

    /// Number of legal actions in the current action space.
    pub fn action_count(&self) -> Result<usize, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.action_count called before reset".to_string()))?;
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        Ok(action_space.actions.len())
    }

    /// Player index holding the current decision, if any.
    pub fn current_agent_index(&self) -> Option<usize> {
        self.game
            .as_ref()
            .and_then(|game| game.action_space())
            .and_then(|space| space.player)
            .map(|player| player.0)
    }

    pub fn is_game_over(&self) -> bool {
        self.game.as_ref().is_some_and(|game| game.is_game_over())
    }

    pub fn winner_index(&self) -> Option<usize> {
        self.game.as_ref().and_then(|game| game.winner_index())
    }

    // ------------------------------------------------------------------
    // Scenario / state-injection surface (flow/scenario.rs).
    //
    // FOR TEST AND MEASUREMENT HARNESSES ONLY: bypasses the rules engine.
    // Typical use: reset, inject at the first priority decision, then call
    // scenario_refresh to recompute the action space and observation.
    // ------------------------------------------------------------------

    fn scenario_game_mut(&mut self, method: &str) -> Result<&mut Game, AgentError> {
        if self.game.as_ref().is_some_and(|game| game.is_game_over()) {
            return Err(AgentError(format!("env.{method} called after game over")));
        }
        self.game
            .as_mut()
            .ok_or_else(|| AgentError(format!("env.{method} called before reset")))
    }

    fn scenario_player(player: usize) -> Result<PlayerId, AgentError> {
        if player > 1 {
            return Err(AgentError(format!(
                "scenario: player {player} out of range"
            )));
        }
        Ok(PlayerId(player))
    }

    pub fn scenario_set_life(&mut self, player: usize, life: i32) -> Result<(), AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_set_life")?
            .scenario_set_life(player, life);
        Ok(())
    }

    pub fn scenario_clear_hand(&mut self, player: usize) -> Result<(), AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_clear_hand")?
            .scenario_clear_hand(player);
        Ok(())
    }

    pub fn scenario_force_card_in_hand(
        &mut self,
        player: usize,
        name: &str,
    ) -> Result<(), AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_force_card_in_hand")?
            .scenario_force_card_in_hand(player, name)
    }

    pub fn scenario_force_battlefield(
        &mut self,
        player: usize,
        name: &str,
        ready: bool,
    ) -> Result<usize, AgentError> {
        let player = Self::scenario_player(player)?;
        self.scenario_game_mut("scenario_force_battlefield")?
            .scenario_force_battlefield(player, name, ready)
            .map(|permanent| permanent.0)
    }

    /// Recompute the current priority action space after injections and
    /// return a fresh observation of the repaired state.
    pub fn scenario_refresh(&mut self) -> Result<Observation, AgentError> {
        let game = self.scenario_game_mut("scenario_refresh")?;
        game.scenario_refresh_priority()?;
        Ok(Observation::new(game, &[]))
    }

    /// Resample hidden information from `perspective`'s point of view.
    /// See [`Game::determinize`].
    pub fn determinize(&mut self, perspective: usize, seed: u64) -> Result<(), AgentError> {
        if perspective > 1 {
            return Err(AgentError(format!(
                "determinize: perspective {perspective} out of range"
            )));
        }
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.determinize called before reset".to_string()))?;
        game.determinize(PlayerId(perspective), seed);
        Ok(())
    }

    /// Reseed the game RNG and play both sides uniformly-random-legal to
    /// terminal. Returns the winner index, or None on draw / step cap.
    pub fn random_playout(
        &mut self,
        seed: u64,
        max_steps: usize,
    ) -> Result<Option<usize>, AgentError> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.random_playout called before reset".to_string()))?;
        game.reseed(seed);
        game.random_playout(max_steps, None)
    }

    /// Flat determinized Monte Carlo evaluation of the current action space.
    ///
    /// For each of `worlds` determinizations (sampled from the perspective of
    /// the player holding the decision), every legal action is applied to a
    /// clone of the world and scored by `rollouts` uniformly-random playouts
    /// (win 1.0 / loss 0.0 / draw-or-cap 0.5). Worlds are shared across
    /// actions (common random numbers) to reduce comparison variance.
    pub fn flat_mc_scores(
        &self,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> Result<FlatMcResult, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.flat_mc_scores called before reset".to_string()))?;
        if game.is_game_over() {
            return Err(AgentError(
                "env.flat_mc_scores called after game over".to_string(),
            ));
        }
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        let hero = action_space
            .player
            .ok_or_else(|| AgentError("no agent player in current action space".to_string()))?;
        let num_actions = action_space.actions.len();
        if num_actions == 0 {
            return Err(AgentError("no valid actions available".to_string()));
        }

        let mut totals = vec![0.0f64; num_actions];
        let mut simulations = 0u64;
        let mut cap_hits = 0u64;

        for world_index in 0..worlds {
            let world_seed = mix_seed(seed, world_index as u64);
            let mut world = game.clone();
            world.determinize(hero, world_seed);
            #[allow(clippy::needless_range_loop)] // action is a semantic index
            for action in 0..num_actions {
                for rollout in 0..rollouts {
                    let mut sim = world.clone();
                    sim.reseed(mix_seed(
                        world_seed,
                        (action * rollouts + rollout + 1) as u64,
                    ));
                    let done = sim.step(action)?;
                    let outcome = if done {
                        sim.winner_index()
                    } else {
                        let mut hit_cap = false;
                        let result = sim.random_playout(max_steps, Some(&mut hit_cap))?;
                        if hit_cap {
                            cap_hits += 1;
                        }
                        result
                    };
                    simulations += 1;
                    totals[action] += match outcome {
                        Some(winner) if winner == hero.0 => 1.0,
                        Some(_) => 0.0,
                        None => 0.5,
                    };
                }
            }
        }

        let denominator = (worlds * rollouts).max(1) as f64;
        Ok(FlatMcResult {
            scores: totals.into_iter().map(|t| t / denominator).collect(),
            simulations,
            cap_hits,
        })
    }

    /// Build a batched rollout pool from the current decision point.
    /// See [`crate::agent::rollout_pool::RolloutPool`].
    pub fn rollout_pool(
        &self,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> Result<crate::agent::rollout_pool::RolloutPool, AgentError> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| AgentError("env.rollout_pool called before reset".to_string()))?;
        crate::agent::rollout_pool::RolloutPool::from_game(game, worlds, rollouts, seed, max_steps)
    }

    pub fn random_action_index(&mut self) -> Result<usize, AgentError> {
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.random_action_index called before reset".to_string()))?;
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;
        if action_space.actions.is_empty() {
            return Err(AgentError("no valid actions available".to_string()));
        }
        Ok(game.state.rng.gen_range(0..action_space.actions.len()))
    }

    fn add_profiler_info(&self, info: &mut InfoDict) {
        let mut out = empty_info_dict();
        if self.profiler.is_enabled() {
            for (name, stats) in self.profiler.get_stats() {
                let mut scoped = empty_info_dict();
                insert_info(
                    &mut scoped,
                    "total_time",
                    InfoValue::Float(stats.total_time),
                );
                insert_info(&mut scoped, "count", InfoValue::Int(stats.count as i64));
                insert_info(&mut out, name, InfoValue::Map(scoped));
            }
        }
        insert_info(info, "profiler", InfoValue::Map(out));
    }

    fn add_behavior_info(&self, info: &mut InfoDict) {
        let mut behavior = empty_info_dict();
        if self.hero_tracker.is_enabled() || self.villain_tracker.is_enabled() {
            let mut hero = empty_info_dict();
            for (k, v) in self.hero_tracker.get_stats() {
                insert_info(&mut hero, k, InfoValue::String(v));
            }
            let mut villain = empty_info_dict();
            for (k, v) in self.villain_tracker.get_stats() {
                insert_info(&mut villain, k, InfoValue::String(v));
            }
            insert_info(&mut behavior, "hero", InfoValue::Map(hero));
            insert_info(&mut behavior, "villain", InfoValue::Map(villain));
        }
        insert_info(info, "behavior", InfoValue::Map(behavior));
    }
}

#[cfg(test)]
mod content_pack_contract_tests {
    use std::{collections::BTreeMap, sync::Arc};

    use super::Env;
    use crate::{
        cardsets::alpha::{default_content_pack, CONTENT_PACK_SCHEMA_VERSION},
        state::player::PlayerConfig,
    };

    fn interactive_deck() -> BTreeMap<String, usize> {
        BTreeMap::from([
            ("Island".to_string(), 12),
            ("Mountain".to_string(), 12),
            ("Gray Ogre".to_string(), 6),
            ("Wind Drake".to_string(), 6),
            ("Man-o'-War".to_string(), 4),
            ("Raging Goblin".to_string(), 4),
            ("Lightning Bolt".to_string(), 6),
            ("Counterspell".to_string(), 4),
            ("Ancestral Recall".to_string(), 3),
            ("Pyroclasm".to_string(), 3),
        ])
    }

    fn configs() -> Vec<PlayerConfig> {
        vec![
            PlayerConfig::new("hero", interactive_deck()),
            PlayerConfig::new("villain", interactive_deck()),
        ]
    }

    #[test]
    fn content_pack_contract_covers_env_roots_siblings_and_rollout_slots() {
        let absent = Env::new(1, true, false, false);
        let error = absent
            .fork()
            .expect_err("an environment without a match has no content pack");
        assert_eq!(error.0, "env.fork called before reset");

        let mut first = Env::new(11, true, false, false);
        let mut second = Env::new(12, true, false, false);
        first.reset(configs()).expect("first reset");
        second.reset(configs()).expect("second reset");

        let admitted = default_content_pack();
        let expected_digest = admitted.content_digest();
        let first_game = first.game.as_ref().expect("first game");
        let second_game = second.game.as_ref().expect("second game");
        assert_eq!(admitted.schema_version, CONTENT_PACK_SCHEMA_VERSION);
        assert!(Arc::ptr_eq(&first_game.state.content, &admitted));
        assert!(Arc::ptr_eq(&second_game.state.content, &admitted));
        assert_eq!(first_game.state.content.content_digest(), expected_digest);
        assert_eq!(second_game.state.content.content_digest(), expected_digest);

        let root_hash = first_game.state.deterministic_hash();
        let root_life = first_game.state.players[0].life;
        let mut changed = first.fork().expect("changed sibling");
        let untouched = first.fork().expect("untouched sibling");
        let pool = first.rollout_pool(1, 1, 0x208, 2000).expect("rollout pool");

        for content in pool.content_pack_contract_refs() {
            assert!(Arc::ptr_eq(content, &admitted));
            assert_eq!(content.content_digest(), expected_digest);
        }

        let action = changed.action_count().expect("changed action count");
        assert!(action > 0);
        changed.step(0).expect("legal branch action");
        changed.game.as_mut().expect("changed game").state.players[0].life -= 3;

        let root = first.game.as_ref().expect("root game");
        let changed_game = changed.game.as_ref().expect("changed game");
        let untouched_game = untouched.game.as_ref().expect("untouched game");
        assert_eq!(root.state.deterministic_hash(), root_hash);
        assert_eq!(untouched_game.state.deterministic_hash(), root_hash);
        assert_ne!(changed_game.state.deterministic_hash(), root_hash);
        assert_eq!(root.state.players[0].life, root_life);
        assert_eq!(untouched_game.state.players[0].life, root_life);
        assert_eq!(changed_game.state.players[0].life, root_life - 3);

        for branch in [changed_game, untouched_game] {
            assert!(Arc::ptr_eq(&branch.state.content, &admitted));
            assert_eq!(branch.state.content.content_digest(), expected_digest);
            for (root_card, branch_card) in root.state.cards.iter().zip(branch.state.cards.iter()) {
                assert_eq!(root_card.definition_id, branch_card.definition_id);
                assert!(root_card.shares_definition_with(branch_card));
            }
        }
    }
}
