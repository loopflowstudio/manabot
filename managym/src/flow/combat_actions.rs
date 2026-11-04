// combat_actions.rs
// Combat declaration and resolution methods on Game.

use crate::state::card::Keyword;
use crate::{
    agent::action::AgentError,
    flow::{
        combat::CombatState,
        decision::{EffectFrame, FrameCompletion},
        game::{CombatDamagePass, Game},
        proposed_event::{EventBatch, EventBinding, ProposedDamageTarget, ProposedEvent},
    },
    state::{
        card::{BlockRestriction, Restriction, StaticAbility},
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
        if self.characteristics(id).abilities_lost {
            return true;
        }
        self.state.cards[permanent.card]
            .statics
            .iter()
            .all(|ability| match ability {
                StaticAbility::CantAttackOrBlockUnlessYouControlAnother(predicate) => self
                    .battlefield_permanents(permanent.controller)
                    .into_iter()
                    .any(|other| other != id && self.permanent_matches_predicate(other, predicate)),
                _ => true,
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
            self.characteristics(permanent_id)
                .keywords
                .has(Keyword::Vigilance)
        };
        if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
            permanent.attacking = true;
            let attacker = permanent.controller;
            self.state.turn.history[attacker.0].attacked = true;
        }
        if !has_vigilance {
            // CR 508.1f — Attackers without vigilance become tapped, which
            // fires "becomes tapped" triggers.
            self.tap_permanent(permanent_id, false);
        }
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
        let Some(combat) = &self.state.combat else {
            return;
        };
        let first_strike = self.combat_has_first_or_double_strike(combat);
        let pass = if first_strike {
            CombatDamagePass::FirstStrike
        } else {
            CombatDamagePass::Normal
        };
        let mut frame = EffectFrame::new(self.active_player(), Vec::new());
        frame.batch = self.combat_damage_batch(pass);
        frame.completion = if first_strike {
            FrameCompletion::CombatNormal
        } else {
            FrameCompletion::Continue
        };
        self.run_frame(frame);
    }

    fn combat_has_first_or_double_strike(&self, combat: &CombatState) -> bool {
        for attacker_id in &combat.attackers {
            if self.state.permanents[*attacker_id].is_none() {
                continue;
            }
            let keywords = self.characteristics(*attacker_id).keywords;
            if keywords.has(Keyword::FirstStrike) || keywords.has(Keyword::DoubleStrike) {
                return true;
            }
        }
        for blockers in combat.attacker_to_blockers.values() {
            for blocker_id in blockers {
                if self.state.permanents[*blocker_id].is_none() {
                    continue;
                }
                let keywords = self.characteristics(*blocker_id).keywords;
                if keywords.has(Keyword::FirstStrike) || keywords.has(Keyword::DoubleStrike) {
                    return true;
                }
            }
        }
        false
    }

    /// One combat damage step (CR 510.2): every assignment is dealt, then
    /// the creatures that dealt damage to a player are announced together
    /// for "deals combat damage to a player" triggers.
    pub(crate) fn combat_damage_batch(&self, pass: CombatDamagePass) -> Option<EventBatch> {
        let combat = self.state.combat.as_ref()?;
        let defender = self.non_active_player();
        let mut events = Vec::new();

        for attacker_id in combat.attackers.iter().copied() {
            if self.state.permanents[attacker_id].is_none() {
                continue;
            }
            let attacker_ref = self.permanent_object_ref(attacker_id);
            if !self.creature_deals_damage_in_pass(attacker_id, pass) {
                continue;
            }

            let attacker_power = self.characteristics(attacker_id).power.max(0);
            let attacker_keywords = self.characteristics(attacker_id).keywords;
            let attacker_has_trample = attacker_keywords.has(Keyword::Trample);
            let attacker_has_deathtouch = attacker_keywords.has(Keyword::Deathtouch);

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
                events.push(ProposedEvent::Damage {
                    source: attacker_ref,
                    target: ProposedDamageTarget::Player(defender),
                    amount: attacker_power,
                    combat: true,
                });
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
                let blocker_power = self.characteristics(*blocker_id).power.max(0);
                if let Some(target) = attacker_ref {
                    events.push(ProposedEvent::Damage {
                        source: blocker_ref,
                        target: ProposedDamageTarget::Object(target),
                        amount: blocker_power,
                        combat: true,
                    });
                }
            }

            let mut remaining_damage = attacker_power;
            for blocker_id in blockers {
                if remaining_damage <= 0 {
                    break;
                }
                let Some(blocker) = self.state.permanents[blocker_id].as_ref() else {
                    continue;
                };
                let blocker_toughness = self.characteristics(blocker_id).toughness;
                let needed_damage = (blocker_toughness - blocker.damage).max(0);
                let lethal = if needed_damage == 0 {
                    0
                } else if attacker_has_deathtouch {
                    1
                } else {
                    needed_damage
                };
                let assigned = remaining_damage.min(lethal);
                if let Some(target) = self.permanent_object_ref(blocker_id) {
                    events.push(ProposedEvent::Damage {
                        source: attacker_ref,
                        target: ProposedDamageTarget::Object(target),
                        amount: assigned,
                        combat: true,
                    });
                }
                remaining_damage -= assigned;
            }

            if attacker_has_trample && remaining_damage > 0 {
                events.push(ProposedEvent::Damage {
                    source: attacker_ref,
                    target: ProposedDamageTarget::Player(defender),
                    amount: remaining_damage,
                    combat: true,
                });
            }
        }
        let mut batch = self.event_batch(events);
        batch.binding = EventBinding::Combat;
        Some(batch)
    }

    fn creature_deals_damage_in_pass(
        &self,
        permanent_id: PermanentId,
        pass: CombatDamagePass,
    ) -> bool {
        let keywords = self.characteristics(permanent_id).keywords;
        let has_first = keywords.has(Keyword::FirstStrike);
        let has_double = keywords.has(Keyword::DoubleStrike);
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
        let defender_applies = self
            .characteristics(permanent_id)
            .keywords
            .has(Keyword::Defender)
            && card.attacks_despite_defender().is_none_or(|condition| {
                !self.check_static_condition(condition, permanent.controller)
            });
        self.characteristics(permanent_id).is_creature()
            && self.permanent_can_tap(permanent_id)
            && !defender_applies
            && self.combat_condition_met(permanent_id)
            && self
                .state
                .turn
                .combat_attackers
                .as_ref()
                .is_none_or(|attackers| self.permanent_matches_predicate(permanent_id, attackers))
    }

    pub(crate) fn eligible_blockers(&self, player: PlayerId) -> Vec<PermanentId> {
        let mut out = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            if !permanent.tapped
                && self.characteristics(permanent_id).is_creature()
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
            || !self.characteristics(blocker_id).is_creature()
            || !self.combat_condition_met(blocker_id)
            || self.cant_block(blocker_id)
        {
            return false;
        }

        if !self.characteristics(blocker_id).abilities_lost
            && blocker_card.statics.iter().any(|ability| {
                matches!(
                    ability,
                    StaticAbility::BlockRestriction(BlockRestriction::CantBlock(predicate))
                        if self.permanent_matches_predicate(attacker_id, predicate)
                )
            })
        {
            return false;
        }
        if !self.characteristics(attacker_id).abilities_lost
            && attacker_card.statics.iter().any(|ability| match ability {
                StaticAbility::CantBeBlockedIf(condition) => {
                    self.check_static_condition(condition, attacker.controller)
                }
                StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(predicate)) => {
                    self.permanent_matches_predicate(blocker_id, predicate)
                }
                _ => false,
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

        let attacker_keywords = self.characteristics(attacker_id).keywords;
        let blocker_keywords = self.characteristics(blocker_id).keywords;
        if attacker_keywords.has(Keyword::Flying)
            && !(blocker_keywords.has(Keyword::Flying) || blocker_keywords.has(Keyword::Reach))
        {
            return false;
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
            if !self
                .characteristics(attacker_id)
                .keywords
                .has(Keyword::Menace)
            {
                continue;
            }
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
