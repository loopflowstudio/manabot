use std::collections::BTreeMap;

use crate::state::game_object::{ObjectId, PermanentId, PlayerId};

/// CR 508.1b. A permanent defender is bound to its battlefield appearance.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize)]
pub enum AttackTarget {
    Player(PlayerId),
    Permanent(#[serde(serialize_with = "crate::state::game_object::exact")] ObjectId),
}

/// Player attacks and declines retain their historical action bytes.
pub(crate) fn serialize_attack<S: serde::Serializer>(
    attack: &Option<AttackTarget>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match attack {
        Some(AttackTarget::Permanent(object)) => {
            crate::state::game_object::exact(object, serializer)
        }
        _ => serializer.serialize_bool(attack.is_some()),
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CombatState {
    /// CR 510.4 remembers which combatants had first or double strike
    /// when the first damage step began, even if their abilities change.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub first_strikers: Vec<PermanentId>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub regular_damage_step: bool,
    /// Attacking creatures and their current defender (`None` once a
    /// non-player defender has left combat, CR 506.4c).
    pub attack_targets: Vec<(PermanentId, Option<AttackTarget>)>,
    pub attacker_to_blockers: BTreeMap<PermanentId, Vec<PermanentId>>,
    /// Prefix already in combat before declaration (CR 508.4).
    pub already_attacking: usize,
    /// Attackers that became blocked in this combat (CR 509.1h).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocked: Vec<PermanentId>,
    pub attackers_to_declare: Vec<PermanentId>,
    pub blockers_to_declare: Vec<PermanentId>,
}

impl CombatState {
    pub fn attackers(&self) -> impl ExactSizeIterator<Item = PermanentId> + '_ {
        self.attack_targets.iter().map(|&(attacker, _)| attacker)
    }

    pub fn remove_defender(&mut self, object: ObjectId) {
        for (_, target) in &mut self.attack_targets {
            if *target == Some(AttackTarget::Permanent(object)) {
                *target = None;
            }
        }
    }

    pub fn defender(&self, attacker: PermanentId) -> Option<AttackTarget> {
        self.attack_targets
            .iter()
            .find(|(id, _)| *id == attacker)
            .and_then(|&(_, target)| target)
    }

    pub fn is_blocked(&self, attacker: PermanentId) -> bool {
        self.blocked.contains(&attacker)
    }

    pub fn remove_from_combat(&mut self, id: PermanentId) {
        if let Some(index) = self.attack_targets.iter().position(|&(p, _)| p == id) {
            self.attack_targets.remove(index);
            self.already_attacking -= usize::from(index < self.already_attacking);
        }
        self.attackers_to_declare.retain(|&p| p != id);
        self.blockers_to_declare.retain(|&p| p != id);
        self.attacker_to_blockers.remove(&id);
        self.blocked.retain(|&p| p != id);
        for blockers in self.attacker_to_blockers.values_mut() {
            blockers.retain(|&p| p != id);
        }
    }
    pub fn in_first_strike_damage_step(&self) -> bool {
        !self.first_strikers.is_empty() && !self.regular_damage_step
    }
}
