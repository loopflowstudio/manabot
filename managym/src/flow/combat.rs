use std::collections::BTreeMap;

use crate::state::game_object::PermanentId;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CombatState {
    /// CR 510.4 remembers which combatants had first or double strike
    /// when the first damage step began, even if their abilities change.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub first_strikers: Vec<PermanentId>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub regular_damage_step: bool,
    pub attackers: Vec<PermanentId>,
    pub attacker_to_blockers: BTreeMap<PermanentId, Vec<PermanentId>>,
    /// Prefix already in combat before declaration (CR 508.4).
    pub already_attacking: usize,
    pub attackers_to_declare: Vec<PermanentId>,
    pub blockers_to_declare: Vec<PermanentId>,
}

impl CombatState {
    pub fn in_first_strike_damage_step(&self) -> bool {
        !self.first_strikers.is_empty() && !self.regular_damage_step
    }
}
