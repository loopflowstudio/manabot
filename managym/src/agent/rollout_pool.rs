// rollout_pool.rs
// Batched policy-rollout support for flat determinized Monte Carlo search.
//
// A RolloutPool holds every simulation of one search decision simultaneously:
// for each of `worlds` determinizations (sampled from the deciding player's
// perspective, shared across actions for common random numbers), each legal
// root action is applied to `rollouts` clones. The pool then exposes the set
// of still-running simulations so a caller can advance *all* of them one
// decision at a time with externally chosen actions — e.g. a policy network
// evaluated in a single batched forward pass per step (wave/search goal 1).
//
// Scoring matches Env::flat_mc_scores exactly: win 1.0 / loss 0.0 /
// draw-or-step-cap 0.5 from the deciding player's perspective, averaged over
// worlds x rollouts per root action.

use crate::{
    agent::{action::AgentError, observation::Observation},
    flow::{game::Game, search::mix_seed},
    state::game_object::PlayerId,
};

#[derive(Debug)]
struct RolloutSlot {
    game: Game,
    root_action: usize,
    steps: usize,
    active: bool,
}

#[derive(Debug)]
pub struct RolloutPool {
    slots: Vec<RolloutSlot>,
    hero: PlayerId,
    num_actions: usize,
    max_steps: usize,
    denominator: f64,
    totals: Vec<f64>,
    simulations: u64,
    cap_hits: u64,
    active_count: usize,
}

impl RolloutPool {
    /// Build the pool from the game's current decision point.
    ///
    /// Slot layout is (world, action, rollout) lexicographic; each slot has
    /// the root action already applied. Slots whose game ended on the root
    /// action are scored immediately and start inactive.
    pub fn from_game(
        game: &Game,
        worlds: usize,
        rollouts: usize,
        seed: u64,
        max_steps: usize,
    ) -> Result<Self, AgentError> {
        if worlds == 0 || rollouts == 0 {
            return Err(AgentError(
                "rollout_pool: worlds and rollouts must be >= 1".to_string(),
            ));
        }
        if game.is_game_over() {
            return Err(AgentError(
                "rollout_pool: called after game over".to_string(),
            ));
        }
        let action_space = game
            .action_space()
            .ok_or_else(|| AgentError("rollout_pool: no active action space".to_string()))?;
        let hero = action_space.player.ok_or_else(|| {
            AgentError("rollout_pool: no agent player in current action space".to_string())
        })?;
        let num_actions = action_space.actions.len();
        if num_actions == 0 {
            return Err(AgentError(
                "rollout_pool: no valid actions available".to_string(),
            ));
        }

        let mut pool = Self {
            slots: Vec::with_capacity(worlds * num_actions * rollouts),
            hero,
            num_actions,
            max_steps,
            denominator: (worlds * rollouts) as f64,
            totals: vec![0.0; num_actions],
            simulations: 0,
            cap_hits: 0,
            active_count: 0,
        };

        for world_index in 0..worlds {
            let world_seed = mix_seed(seed, world_index as u64);
            let mut world = game.clone();
            world.determinize(hero, world_seed);
            for action in 0..num_actions {
                for rollout in 0..rollouts {
                    let mut sim = world.clone();
                    sim.reseed(mix_seed(
                        world_seed,
                        (action * rollouts + rollout + 1) as u64,
                    ));
                    let done = sim.step(action)?;
                    let _ = sim.take_observation_events();
                    pool.simulations += 1;
                    if done {
                        pool.totals[action] += Self::outcome_score(&sim, hero);
                    } else {
                        pool.active_count += 1;
                    }
                    pool.slots.push(RolloutSlot {
                        game: sim,
                        root_action: action,
                        steps: 1,
                        active: !done,
                    });
                }
            }
        }
        Ok(pool)
    }

    fn outcome_score(game: &Game, hero: PlayerId) -> f64 {
        match game.winner_index() {
            Some(winner) if winner == hero.0 => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        }
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Encode the current observation of every active slot via `write`,
    /// returning the active slot indices (ascending).
    pub fn encode_active_into<F>(&mut self, write: F) -> Result<Vec<usize>, AgentError>
    where
        F: Fn(usize, &Observation) -> Result<(), AgentError>,
    {
        let mut active = Vec::with_capacity(self.active_count);
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if !slot.active {
                continue;
            }
            let events = slot.game.take_observation_events();
            let obs = Observation::new(&slot.game, &events);
            write(index, &obs)?;
            active.push(index);
        }
        Ok(active)
    }

    /// Step every active slot with `actions[slot_index]` (inactive entries
    /// are ignored), then encode the observations of slots that remain
    /// active. Terminated slots are scored; slots reaching `max_steps` count
    /// as cap hits and score 0.5. Returns the still-active slot indices.
    pub fn step_active_into<F>(
        &mut self,
        actions: &[i64],
        write: F,
    ) -> Result<Vec<usize>, AgentError>
    where
        F: Fn(usize, &Observation) -> Result<(), AgentError>,
    {
        if actions.len() != self.slots.len() {
            return Err(AgentError(format!(
                "rollout_pool: expected {} actions, got {}",
                self.slots.len(),
                actions.len()
            )));
        }

        let mut still_active = Vec::with_capacity(self.active_count);
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if !slot.active {
                continue;
            }
            let action = actions[index];
            let action = usize::try_from(action).map_err(|_| {
                AgentError(format!(
                    "rollout_pool: negative action {action} for slot {index}"
                ))
            })?;
            let done = slot.game.step(action)?;
            let _ = slot.game.take_observation_events();
            slot.steps += 1;

            if done {
                self.totals[slot.root_action] += Self::outcome_score(&slot.game, self.hero);
                slot.active = false;
                self.active_count -= 1;
            } else if slot.steps >= self.max_steps {
                self.cap_hits += 1;
                self.totals[slot.root_action] += 0.5;
                slot.active = false;
                self.active_count -= 1;
            } else {
                still_active.push(index);
            }
        }

        for &index in &still_active {
            let slot = &mut self.slots[index];
            let events = slot.game.take_observation_events();
            let obs = Observation::new(&slot.game, &events);
            write(index, &obs)?;
        }
        Ok(still_active)
    }

    /// Finish every still-active slot with uniformly-random play to
    /// terminal, entirely engine-side (no observation encoding). Used for
    /// hybrid rollouts: the policy plays the first K plies, the random tail
    /// costs ~0.2 ms/playout. Respects the per-slot step budget; slots that
    /// exhaust it count as cap hits and score 0.5.
    pub fn finish_random(&mut self) -> Result<usize, AgentError> {
        let mut finished = 0usize;
        for slot in self.slots.iter_mut() {
            if !slot.active {
                continue;
            }
            let remaining = self.max_steps.saturating_sub(slot.steps).max(1);
            let mut hit_cap = false;
            let outcome = slot.game.random_playout(remaining, Some(&mut hit_cap))?;
            if hit_cap {
                self.cap_hits += 1;
            }
            self.totals[slot.root_action] += match outcome {
                Some(winner) if winner == self.hero.0 => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
            slot.active = false;
            self.active_count -= 1;
            finished += 1;
        }
        Ok(finished)
    }

    /// Mean playout score per root action, plus (simulations, cap_hits).
    /// Valid once `active_count() == 0`; callable earlier for debugging
    /// (unfinished slots simply have not contributed yet).
    pub fn scores(&self) -> (Vec<f64>, u64, u64) {
        let scores = self
            .totals
            .iter()
            .map(|total| total / self.denominator)
            .collect();
        (scores, self.simulations, self.cap_hits)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::RolloutPool;
    use crate::{flow::game::Game, state::player::PlayerConfig};

    fn sample_game() -> Game {
        let configs = vec![
            PlayerConfig::new(
                "Hero",
                BTreeMap::from([
                    ("Mountain".to_string(), 12usize),
                    ("Forest".to_string(), 12usize),
                    ("Llanowar Elves".to_string(), 18usize),
                    ("Grey Ogre".to_string(), 18usize),
                ]),
            ),
            PlayerConfig::new(
                "Villain",
                BTreeMap::from([
                    ("Mountain".to_string(), 12usize),
                    ("Forest".to_string(), 12usize),
                    ("Llanowar Elves".to_string(), 18usize),
                    ("Grey Ogre".to_string(), 18usize),
                ]),
            ),
        ];
        Game::new(configs, 7, true)
    }

    #[test]
    fn pool_runs_random_rollouts_to_completion() {
        let mut game = sample_game();
        let _ = game.take_observation_events();
        let num_actions = game.action_space().map(|s| s.actions.len()).unwrap();

        let mut pool = RolloutPool::from_game(&game, 2, 2, 99, 2000).expect("pool should build");
        assert_eq!(pool.num_slots(), 2 * 2 * num_actions);
        assert_eq!(pool.num_actions(), num_actions);

        // Drive every rollout with "always action 0" (a legal index in any
        // non-empty action space) until the pool drains.
        let mut active = pool
            .encode_active_into(|_, _| Ok(()))
            .expect("encode should succeed");
        let mut guard = 0;
        while !active.is_empty() {
            guard += 1;
            assert!(guard < 10_000, "rollouts failed to terminate");
            let actions = vec![0i64; pool.num_slots()];
            active = pool
                .step_active_into(&actions, |_, _| Ok(()))
                .expect("step should succeed");
        }

        assert_eq!(pool.active_count(), 0);
        let (scores, simulations, _cap_hits) = pool.scores();
        assert_eq!(scores.len(), num_actions);
        assert_eq!(simulations, (2 * 2 * num_actions) as u64);
        for score in scores {
            assert!((0.0..=1.0).contains(&score), "score out of range: {score}");
        }
    }

    #[test]
    fn pool_rejects_wrong_action_len() {
        let mut game = sample_game();
        let _ = game.take_observation_events();
        let mut pool = RolloutPool::from_game(&game, 1, 1, 3, 2000).expect("pool should build");
        let err = pool
            .step_active_into(&[0], |_, _| Ok(()))
            .expect_err("wrong length must fail");
        assert!(err.0.contains("expected"));
    }
}
