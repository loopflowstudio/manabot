// identity.rs
// Game identity helpers: player accessors, game-over checks.

use crate::{agent::action::ActionSpace, flow::game::Game, state::game_object::PlayerId};

impl Game {
    pub fn action_space(&self) -> Option<&ActionSpace> {
        self.current_action_space.as_ref()
    }

    pub fn active_player(&self) -> PlayerId {
        self.state.turn.active_player
    }

    pub fn non_active_player(&self) -> PlayerId {
        PlayerId((self.state.turn.active_player.0 + 1) % 2)
    }

    pub fn agent_player(&self) -> PlayerId {
        self.current_action_space
            .as_ref()
            .and_then(|space| space.player)
            .unwrap_or(PlayerId(0))
    }

    pub fn players_starting_with_active(&self) -> [PlayerId; 2] {
        [self.active_player(), self.non_active_player()]
    }

    pub fn players_starting_with_agent(&self) -> [PlayerId; 2] {
        let agent = self.agent_player();
        [agent, PlayerId((agent.0 + 1) % 2)]
    }

    pub fn is_active_player(&self, player: PlayerId) -> bool {
        player == self.active_player()
    }

    pub fn is_game_over(&self) -> bool {
        self.state.players.iter().filter(|p| p.alive).count() < 2
    }

    pub fn winner_index(&self) -> Option<usize> {
        if !self.is_game_over() {
            return None;
        }
        self.state.players.iter().position(|p| p.alive)
    }
}
