// damage.rs
// Damage application and clearing.

use crate::{flow::game::Game, state::game_object::PermanentId};

impl Game {
    pub(crate) fn apply_permanent_damage(&mut self, permanent_id: PermanentId, amount: i32) {
        if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
            permanent.take_damage(amount);
        }
    }

    pub(crate) fn clear_damage(&mut self) {
        for permanent in self.state.permanents.iter_mut().flatten() {
            permanent.clear_damage();
        }
    }
}
