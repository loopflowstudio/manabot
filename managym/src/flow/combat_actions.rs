// combat_actions.rs
// Combat declaration and resolution methods on Game.

use crate::{
    agent::action::AgentError,
    flow::{
        combat::CombatState,
        game::{CombatDamagePass, Game},
    },
    state::{
        card::Card,
        game_object::{PermanentId, PlayerId},
    },
};

impl Game {
    pub(crate) fn declare_attacker(
        &mut self,
        permanent_id: PermanentId,
        attack: bool,
    ) -> Result<(), AgentError> {
        if !attack {
            return Ok(());
        }

        let Some(permanent) = self.state.permanents[permanent_id].as_mut() else {
            return Err(AgentError("attacker permanent not found".to_string()));
        };
        let card = &self.state.cards[permanent.card];
        if !permanent.can_attack(card) {
            return Err(AgentError("permanent cannot attack".to_string()));
        }

        let controller = permanent.controller;
        permanent.attack_with_card(card);
        self.invalidate_mana_cache(controller);
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
        let Some(combat) = self.state.combat.take() else {
            return;
        };

        let has_first_or_double_strike = self.combat_has_first_or_double_strike(&combat);
        if has_first_or_double_strike {
            self.resolve_combat_damage_pass(&combat, CombatDamagePass::FirstStrike);
            self.perform_state_based_actions();
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
            let Some(attacker) = self.state.permanents[*attacker_id].as_ref() else {
                continue;
            };
            let keywords = &self.state.cards[attacker.card].keywords;
            if keywords.first_strike || keywords.double_strike {
                return true;
            }
        }
        for blockers in combat.attacker_to_blockers.values() {
            for blocker_id in blockers {
                let Some(blocker) = self.state.permanents[*blocker_id].as_ref() else {
                    continue;
                };
                let keywords = &self.state.cards[blocker.card].keywords;
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
            let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
                continue;
            };
            let attacker_card_id = attacker.card;
            let attacker_card = &self.state.cards[attacker_card_id];
            if !self.creature_deals_damage_in_pass(attacker_card, pass) {
                continue;
            }

            let attacker_power = attacker.effective_power(attacker_card).max(0);
            let attacker_has_trample = attacker_card.keywords.trample;
            let attacker_has_deathtouch = attacker_card.keywords.deathtouch;

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
                self.apply_player_damage(Some(attacker_card_id), defender, attacker_power);
                continue;
            }

            for blocker_id in &blockers {
                let Some(blocker) = self.state.permanents[*blocker_id].as_ref() else {
                    continue;
                };
                let blocker_card = &self.state.cards[blocker.card];
                if !self.creature_deals_damage_in_pass(blocker_card, pass) {
                    continue;
                }
                let blocker_power = blocker.effective_power(blocker_card).max(0);
                self.apply_permanent_damage(Some(blocker.card), attacker_id, blocker_power);
            }

            let mut remaining_damage = attacker_power;
            for blocker_id in blockers {
                if remaining_damage <= 0 {
                    break;
                }
                let Some(blocker) = self.state.permanents[blocker_id].as_ref() else {
                    continue;
                };
                let blocker_card = &self.state.cards[blocker.card];
                let blocker_toughness = blocker.effective_toughness(blocker_card);
                let needed_damage = (blocker_toughness - blocker.damage).max(0);
                let lethal = if needed_damage == 0 {
                    0
                } else if attacker_has_deathtouch {
                    1
                } else {
                    needed_damage
                };
                let assigned = remaining_damage.min(lethal);
                self.apply_permanent_damage(Some(attacker_card_id), blocker_id, assigned);
                remaining_damage -= assigned;
            }

            if attacker_has_trample && remaining_damage > 0 {
                self.apply_player_damage(Some(attacker_card_id), defender, remaining_damage);
            }
        }
    }

    fn creature_deals_damage_in_pass(&self, card: &Card, pass: CombatDamagePass) -> bool {
        let has_first = card.keywords.first_strike;
        let has_double = card.keywords.double_strike;
        match pass {
            CombatDamagePass::FirstStrike => has_first || has_double,
            CombatDamagePass::NormalWithFirstStrike => !has_first || has_double,
            CombatDamagePass::Normal => true,
        }
    }

    pub(crate) fn eligible_attackers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            if permanent.can_attack(card) {
                out.push(permanent_id);
            }
        }
        out
    }

    pub(crate) fn eligible_blockers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            if permanent.can_block(card) {
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

        if !blocker.can_block(blocker_card) {
            return false;
        }

        if attacker_card.keywords.flying
            && !(blocker_card.keywords.flying || blocker_card.keywords.reach)
        {
            return false;
        }

        true
    }

    pub(crate) fn cleanup_illegal_menace_blocks(&mut self) {
        let Some(combat) = self.state.combat.as_mut() else {
            return;
        };
        for attacker_id in combat.attackers.clone() {
            let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
                continue;
            };
            let attacker_card = &self.state.cards[attacker.card];
            if !attacker_card.keywords.menace {
                continue;
            }
            let Some(blockers) = combat.attacker_to_blockers.get_mut(&attacker_id) else {
                continue;
            };
            if blockers.len() == 1 {
                blockers.clear();
            }
        }
    }
}
