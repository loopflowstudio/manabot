use std::collections::BTreeMap;

use crate::state::game_object::PermanentId;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CombatState {
    pub attackers: Vec<PermanentId>,
    pub attacker_to_blockers: BTreeMap<PermanentId, Vec<PermanentId>>,
    /// Prefix already in combat before declaration (CR 508.4).
    pub already_attacking: usize,
    pub attackers_to_declare: Vec<PermanentId>,
    pub blockers_to_declare: Vec<PermanentId>,
}
