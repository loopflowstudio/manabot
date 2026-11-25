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
    /// A blocked attacker remains blocked after its last blocker changes control.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked_without_blockers: Vec<PermanentId>,
    pub attackers_to_declare: Vec<PermanentId>,
    pub blockers_to_declare: Vec<PermanentId>,
}

impl CombatState {
    pub fn is_blocked(&self, attacker: PermanentId) -> bool {
        self.attacker_to_blockers
            .get(&attacker)
            .is_some_and(|blockers| !blockers.is_empty())
            || self.blocked_without_blockers.contains(&attacker)
    }

    pub fn remove_from_combat(&mut self, id: PermanentId) {
        if let Some(index) = self.attackers.iter().position(|&p| p == id) {
            self.attackers.remove(index);
            self.already_attacking -= usize::from(index < self.already_attacking);
        }
        self.attackers_to_declare.retain(|&p| p != id);
        self.blockers_to_declare.retain(|&p| p != id);
        self.attacker_to_blockers.remove(&id);
        self.blocked_without_blockers.retain(|&p| p != id);
        for (&attacker, blockers) in &mut self.attacker_to_blockers {
            if blockers.contains(&id) {
                blockers.retain(|&p| p != id);
                if blockers.is_empty() && !self.blocked_without_blockers.contains(&attacker) {
                    self.blocked_without_blockers.push(attacker);
                }
            }
        }
    }

    pub fn in_first_strike_damage_step(&self) -> bool {
        !self.first_strikers.is_empty() && !self.regular_damage_step
    }
}
