use std::collections::BTreeMap;

use crate::state::game_object::PermanentId;

#[derive(Clone, Debug, Default)]
pub struct CombatState {
    pub attackers: Vec<PermanentId>,
    pub attacker_to_blockers: BTreeMap<PermanentId, Vec<PermanentId>>,
    pub attackers_to_declare: Vec<PermanentId>,
    pub blockers_to_declare: Vec<PermanentId>,
}
