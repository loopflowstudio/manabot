// combat_actions.rs
// Combat declaration and resolution methods on Game.

use crate::{
    agent::action::AgentError,
    flow::{
        combat::CombatState,
        game::{CombatDamagePass, Game},
    },
    state::{
        card::Restriction,
        game_object::{PermanentId, PlayerId},
    },
};

impl Game {
    fn combat_condition_met(&self, id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[id].as_ref() else {
            return false;
        };
        if self.has_restriction(id, Restriction::CantAttackOrBlock) {
            return false;
        }
        if self.has_lost_abilities(id) {
            return true;
        }
        self.state.cards[permanent.card]
            .combat_requires_other
            .as_ref()
            .is_none_or(|predicate| {
                self.battlefield_permanents(permanent.controller)
                    .into_iter()
                    .any(|other| other != id && self.permanent_matches_predicate(other, predicate))
            })
    }

    pub(crate) fn declare_attacker(
        &mut self,
        permanent_id: PermanentId,
        attack: bool,
    ) -> Result<(), AgentError> {
        if !attack {
            return Ok(());
        }

        let has_vigilance = {
            if self.state.permanents[permanent_id].is_none() {
                return Err(AgentError("attacker permanent not found".to_string()));
            }
            if !self.permanent_can_attack(permanent_id) {
                return Err(AgentError("permanent cannot attack".to_string()));
            }
            self.effective_keywords(permanent_id).vigilance
        };
        self.journal_permanent(permanent_id);
        if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
            permanent.attacking = true;
            let attacker = permanent.controller;
            self.update_turn_history(attacker, |history| history.attacked = true);
        }
        if !has_vigilance {
            // CR 508.1f — Attackers without vigilance become tapped, which
            // fires "becomes tapped" triggers.
            self.tap_permanent(permanent_id, false);
        }
        self.journal_combat();
        if let Some(combat) = self.state.combat.as_mut() {
            combat.attackers.push(permanent_id);
            combat.attacker_to_blockers.entry(permanent_id).or_default();
        }
        Ok(())
    }

    pub(crate) fn declare_blocker(
        &mut self,
        blocker: PermanentId,
        attacker: Option<PermanentId>,
    ) -> Result<(), AgentError> {
        if let Some(attacker_id) = attacker {
            if !self.blocker_can_block_attacker(blocker, attacker_id) {
                return Err(AgentError(
                    "block declaration is illegal for this attacker/blocker pair".to_string(),
                ));
            }
            self.journal_combat();
            if let Some(combat) = self.state.combat.as_mut() {
                combat
                    .attacker_to_blockers
                    .entry(attacker_id)
                    .or_default()
                    .push(blocker);
            }
        }
        Ok(())
    }

    pub(crate) fn resolve_combat_damage(&mut self) {
        self.journal_combat();
        let Some(combat) = self.state.combat.take() else {
            return;
        };

        let has_first_or_double_strike = self.combat_has_first_or_double_strike(&combat);
        if has_first_or_double_strike {
            self.resolve_combat_damage_pass(&combat, CombatDamagePass::FirstStrike);
            while self.perform_state_based_actions() {}
            if !self.is_game_over() {
                self.resolve_combat_damage_pass(&combat, CombatDamagePass::NormalWithFirstStrike);
            }
        } else {
            self.resolve_combat_damage_pass(&combat, CombatDamagePass::Normal);
        }

        self.state.combat = Some(combat);
    }

    fn combat_has_first_or_double_strike(&self, combat: &CombatState) -> bool {
        for attacker_id in &combat.attackers {
            if self.state.permanents[*attacker_id].is_none() {
                continue;
            }
            let keywords = self.effective_keywords(*attacker_id);
            if keywords.first_strike || keywords.double_strike {
                return true;
            }
        }
        for blockers in combat.attacker_to_blockers.values() {
            for blocker_id in blockers {
                if self.state.permanents[*blocker_id].is_none() {
                    continue;
                }
                let keywords = self.effective_keywords(*blocker_id);
                if keywords.first_strike || keywords.double_strike {
                    return true;
                }
            }
        }
        false
    }

    fn resolve_combat_damage_pass(&mut self, combat: &CombatState, pass: CombatDamagePass) {
        let defender = self.non_active_player();

        for attacker_id in combat.attackers.iter().copied() {
            if self.state.permanents[attacker_id].is_none() {
                continue;
            }
            let attacker_ref = self.permanent_object_ref(attacker_id);
            if !self.creature_deals_damage_in_pass(attacker_id, pass) {
                continue;
            }

            let attacker_power = self.effective_power(attacker_id).max(0);
            let attacker_keywords = self.effective_keywords(attacker_id);
            let attacker_has_trample = attacker_keywords.trample;
            let attacker_has_deathtouch = attacker_keywords.deathtouch;

            let declared_blockers = combat
                .attacker_to_blockers
                .get(&attacker_id)
                .cloned()
                .unwrap_or_default();
            let was_blocked = !declared_blockers.is_empty();
            let blockers: Vec<PermanentId> = declared_blockers
                .iter()
                .copied()
                .filter(|blocker_id| self.state.permanents[*blocker_id].is_some())
                .collect();

            if !was_blocked {
                self.apply_combat_player_damage(attacker_ref, defender, attacker_power);
                continue;
            }

            for blocker_id in &blockers {
                if self.state.permanents[*blocker_id].is_none() {
                    continue;
                }
                let blocker_ref = self.permanent_object_ref(*blocker_id);
                if !self.creature_deals_damage_in_pass(*blocker_id, pass) {
                    continue;
                }
                let blocker_power = self.effective_power(*blocker_id).max(0);
                self.apply_combat_permanent_damage(blocker_ref, attacker_id, blocker_power);
            }

            let mut remaining_damage = attacker_power;
            for blocker_id in blockers {
                if remaining_damage <= 0 {
                    break;
                }
                let Some(blocker) = self.state.permanents[blocker_id].as_ref() else {
                    continue;
                };
                let blocker_toughness = self.effective_toughness(blocker_id);
                let needed_damage = (blocker_toughness - blocker.damage).max(0);
                let lethal = if needed_damage == 0 {
                    0
                } else if attacker_has_deathtouch {
                    1
                } else {
                    needed_damage
                };
                let assigned = remaining_damage.min(lethal);
                self.apply_combat_permanent_damage(attacker_ref, blocker_id, assigned);
                remaining_damage -= assigned;
            }

            if attacker_has_trample && remaining_damage > 0 {
                self.apply_combat_player_damage(attacker_ref, defender, remaining_damage);
            }
        }
    }

    fn creature_deals_damage_in_pass(
        &self,
        permanent_id: PermanentId,
        pass: CombatDamagePass,
    ) -> bool {
        let keywords = self.effective_keywords(permanent_id);
        let has_first = keywords.first_strike;
        let has_double = keywords.double_strike;
        match pass {
            CombatDamagePass::FirstStrike => has_first || has_double,
            CombatDamagePass::NormalWithFirstStrike => !has_first || has_double,
            CombatDamagePass::Normal => true,
        }
    }

    pub(crate) fn eligible_attackers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            if self.permanent_can_attack(permanent_id) {
                out.push(permanent_id);
            }
        }
        out
    }

    /// Whether a permanent can be declared as an attacker (CR 508.1a):
    /// a creature that can tap, without defender unless an effect lets it
    /// attack as though it didn't have defender (Platypus-Bear).
    fn permanent_can_attack(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let card = &self.state.cards[permanent.card];
        let defender_applies = self.effective_keywords(permanent_id).defender
            && card
                .attacks_despite_defender
                .as_ref()
                .is_none_or(|condition| {
                    !self.check_static_condition(condition, permanent.controller)
                });
        self.permanent_is_creature(permanent_id)
            && self.permanent_can_tap(permanent_id)
            && !defender_applies
            && self.combat_condition_met(permanent_id)
    }

    pub(crate) fn eligible_blockers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            if !permanent.tapped
                && self.permanent_is_creature(permanent_id)
                && self.combat_condition_met(permanent_id)
            {
                out.push(permanent_id);
            }
        }
        out
    }

    pub(crate) fn blocker_can_block_attacker(
        &self,
        blocker_id: PermanentId,
        attacker_id: PermanentId,
    ) -> bool {
        let Some(blocker) = self.state.permanents[blocker_id].as_ref() else {
            return false;
        };
        let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
            return false;
        };
        let blocker_card = &self.state.cards[blocker.card];
        let attacker_card = &self.state.cards[attacker.card];

        if blocker.tapped
            || !self.permanent_is_creature(blocker_id)
            || !self.combat_condition_met(blocker_id)
        {
            return false;
        }

        if !self.has_lost_abilities(blocker_id)
            && blocker_card
                .blocking_restriction
                .as_ref()
                .is_some_and(|predicate| self.permanent_matches_predicate(attacker_id, predicate))
        {
            return false;
        }
        if !self.has_lost_abilities(attacker_id)
            && self.state.cards[attacker.card]
                .unblockable_if
                .as_ref()
                .is_some_and(|condition| {
                    self.check_static_condition(condition, attacker.controller)
                })
        {
            return false;
        }
        if self.cant_be_blocked(attacker_id) {
            return false;
        }
        // Blockers are declared one at a time; a second blocker is refused.
        if self.has_restriction(attacker_id, Restriction::CantBeBlockedByMoreThanOne)
            && self
                .state
                .combat
                .as_ref()
                .and_then(|combat| combat.attacker_to_blockers.get(&attacker_id))
                .is_some_and(|blockers| blockers.iter().any(|&other| other != blocker_id))
        {
            return false;
        }

        let attacker_keywords = self.effective_keywords(attacker_id);
        let blocker_keywords = self.effective_keywords(blocker_id);
        if attacker_keywords.flying && !(blocker_keywords.flying || blocker_keywords.reach) {
            return false;
        }

        // "This creature can't be blocked by creatures [matching predicate]."
        if let Some(restriction) = &attacker_card.block_restriction {
            if self.permanent_matches_predicate(blocker_id, restriction) {
                return false;
            }
        }

        true
    }

    pub(crate) fn cleanup_illegal_menace_blocks(&mut self) {
        let attackers = match self.state.combat.as_ref() {
            Some(combat) => combat.attackers.clone(),
            None => return,
        };
        for attacker_id in attackers {
            if self.state.permanents[attacker_id].is_none() {
                continue;
            }
            if !self.effective_keywords(attacker_id).menace {
                continue;
            }
            self.journal_combat();
            let Some(combat) = self.state.combat.as_mut() else {
                return;
            };
            let Some(blockers) = combat.attacker_to_blockers.get_mut(&attacker_id) else {
                continue;
            };
            if blockers.len() == 1 {
                blockers.clear();
            }
        }
    }
}
