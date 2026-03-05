use crate::state::game_object::PlayerId;

#[derive(Clone, Debug)]
pub struct PriorityState {
    pub holder: PlayerId,
    pub consecutive_passes: usize,
    pub sba_done: bool,
}

impl Default for PriorityState {
    fn default() -> Self {
        Self {
            holder: PlayerId(0),
            consecutive_passes: 0,
            sba_done: false,
        }
    }
}

impl PriorityState {
    pub fn start_round(&mut self, active: PlayerId) {
        self.holder = active;
        self.consecutive_passes = 0;
        self.sba_done = false;
    }

    pub fn on_pass(&mut self, next: PlayerId) {
        self.consecutive_passes += 1;
        self.holder = next;
    }

    pub fn on_non_pass_action(&mut self, active: PlayerId) {
        self.holder = active;
        self.consecutive_passes = 0;
        self.sba_done = false;
    }
}
