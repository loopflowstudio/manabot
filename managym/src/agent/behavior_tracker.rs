use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct BehaviorTracker {
    enabled: bool,
    pub games_played: usize,
    pub games_won: usize,
}

impl BehaviorTracker {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            games_played: 0,
            games_won: 0,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn on_game_start(&mut self) {
        if self.enabled {
            self.games_played += 1;
        }
    }

    pub fn on_game_won(&mut self) {
        if self.enabled {
            self.games_won += 1;
        }
    }

    pub fn get_stats(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        if !self.enabled {
            return out;
        }
        out.insert("games_played".to_string(), self.games_played.to_string());
        out.insert("games_won".to_string(), self.games_won.to_string());
        out
    }
}
