// combat_actions.rs
// Combat declaration and resolution methods on Game.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::game_object::{PermanentId, PlayerId},
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
        permanent.attack();
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

        for (attacker_id, blockers) in &combat.attacker_to_blockers {
            let Some(attacker) = self.state.permanents[*attacker_id].as_ref() else {
                continue;
            };
            let attacker_power = self.state.cards[attacker.card].power.unwrap_or(0);

            if blockers.is_empty() {
                // CR 510.1c — Unblocked attackers assign combat damage to defending player.
                let defender = self.non_active_player();
                self.state.players[defender.0].take_damage(attacker_power);
                continue;
            }

            for blocker_id in blockers {
                // CR 510.1a — Combat damage is dealt simultaneously by attacking and blocking creatures.
                let Some(blocker) = self.state.permanents[*blocker_id].as_ref() else {
                    continue;
                };
                let blocker_power = self.state.cards[blocker.card].power.unwrap_or(0);
                self.apply_permanent_damage(*attacker_id, blocker_power);
                self.apply_permanent_damage(*blocker_id, attacker_power);
            }
        }

        self.state.combat = Some(combat);
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
}
