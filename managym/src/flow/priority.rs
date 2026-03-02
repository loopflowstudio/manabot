#[derive(Clone, Debug, Default)]
pub struct PriorityState {
    pub pass_count: usize,
    pub sba_done: bool,
}

impl PriorityState {
    pub fn reset(&mut self) {
        self.pass_count = 0;
        self.sba_done = false;
    }

    pub fn pass_priority(&mut self) {
        self.pass_count += 1;
    }
}
