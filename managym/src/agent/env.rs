use crate::{
    agent::{action::AgentError, behavior_tracker::BehaviorTracker, observation::Observation},
    flow::game::Game,
    infra::profiler::{empty_info_dict, insert_info, InfoDict, InfoValue, Profiler},
    state::player::PlayerConfig,
};

#[derive(Debug)]
pub struct Env {
    game: Option<Game>,
    skip_trivial: bool,
    enable_profiler: bool,
    enable_behavior_tracking: bool,
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
            enable_profiler,
            enable_behavior_tracking,
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
        let game = Game::new(player_configs, self.seed, self.skip_trivial);
        let observation = Observation::new(&game);
        self.game = Some(game);
        Ok((observation, empty_info_dict()))
    }

    pub fn step(
        &mut self,
        action: usize,
    ) -> Result<(Observation, f64, bool, bool, InfoDict), AgentError> {
        let _scope = self.profiler.track("env_step");
        let game = self
            .game
            .as_mut()
            .ok_or_else(|| AgentError("env.step called before reset".to_string()))?;

        if game.is_game_over() {
            return Err(AgentError("env.step called after game over".to_string()));
        }

        let agent = game
            .action_space()
            .and_then(|space| space.player)
            .ok_or_else(|| AgentError("no agent player in current action space".to_string()))?;

        let done = game.step(action)?;
        let observation = Observation::new(game);

        let mut reward = 0.0;
        let mut info = empty_info_dict();
        if done {
            if let Some(winner) = game.winner_index() {
                reward = if winner == agent.0 { 1.0 } else { -1.0 };
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

    pub fn export_profile_baseline(&self) -> String {
        if self.enable_profiler {
            self.profiler.export_baseline()
        } else {
            String::new()
        }
    }

    pub fn compare_profile(&self, baseline: &str) -> String {
        if self.enable_profiler {
            self.profiler.compare_to_baseline(baseline)
        } else {
            "Profiler not enabled".to_string()
        }
    }

    fn add_profiler_info(&self, info: &mut InfoDict) {
        let mut out = empty_info_dict();
        if self.enable_profiler {
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
        if self.enable_behavior_tracking {
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
