// combat_actions.rs
// Combat declaration and resolution methods on Game.

use crate::state::card::Keyword;
use crate::{
    agent::action::AgentError,
    flow::{
        decision::{Decision, EffectFrame, FrameCompletion, SuspendedResolution},
        game::{CombatDamagePass, Game},
        proposed_event::{EventBatch, EventBinding, ProposedDamageTarget, ProposedEvent},
        statics::Characteristics,
    },
    state::{
        card::{BlockRestriction, Restriction, StaticAbility},
        game_object::{PermanentId, PlayerId},
        predicate::CardPredicate,
    },
};

impl Game {
    pub(crate) fn cant_attack_or_block(&self, id: PermanentId, view: &Characteristics) -> bool {
        self.has_restriction(id, Restriction::CantAttackOrBlock)
            || view.static_abilities.iter().any(|ability| match ability {
                StaticAbility::CantAttackOrBlockUnlessYouControlAnother(predicate) => {
                    let controller = self.state.permanents[id].as_ref().unwrap().controller;
                    !self.iter_battlefield_permanents(controller).any(|other| {
                        other != id && self.permanent_matches_predicate(other, predicate)
                    })
                }
                _ => false,
            })
    }

    pub(crate) fn attacks_despite_defender(
        &self,
        view: &Characteristics,
        controller: PlayerId,
    ) -> bool {
        view.keywords.has(Keyword::Defender) && view.static_abilities.iter().any(|ability| matches!(ability,
            StaticAbility::AttacksDespiteDefender(condition) if self.check_static_condition(condition, controller)))
    }

    pub(crate) fn cant_block_unconditionally(
        &self,
        id: PermanentId,
        view: &Characteristics,
    ) -> bool {
        self.cant_block(id) || view.static_abilities.iter().any(|ability| matches!(ability,
            StaticAbility::BlockRestriction(BlockRestriction::CantBlock(predicate)) if *predicate == CardPredicate::default()))
    }

    pub(crate) fn cant_be_blocked_unconditionally(
        &self,
        id: PermanentId,
        view: &Characteristics,
        controller: PlayerId,
    ) -> bool {
        self.cant_be_blocked(id) || view.static_abilities.iter().any(|ability| matches!(ability,
            StaticAbility::CantBeBlockedIf(condition) if self.check_static_condition(condition, controller)))
    }

    pub(crate) fn declare_attacker(
        &mut self,
        permanent_id: PermanentId,
        attack: bool,
    ) -> Result<(), AgentError> {
        if !attack {
            if self.permanent_can_attack(permanent_id) && self.must_attack(permanent_id) {
                return Err(AgentError(
                    "creature attacks each combat if able".to_string(),
                ));
            }
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
        }
        if !has_vigilance {
            // CR 508.1f — Attackers without vigilance become tapped, which
            // fires "becomes tapped" triggers.
            self.tap_permanent(permanent_id, false);
        }
        if let Some(combat) = self.state.combat.as_mut() {
            combat.attackers.push(permanent_id);
            combat.attacker_to_blockers.entry(permanent_id).or_default();
            if let Some(exposure) = self.exposure_mut() {
                exposure.combat.attackers.push(permanent_id);
            }
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
                if let Some(exposure) = self.exposure_mut() {
                    exposure.combat.blocks.push((blocker, attacker_id));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn resolve_combat_damage(&mut self) {
        let Some(combat) = &self.state.combat else {
            return;
        };
        let pass = if combat.regular_damage_step {
            CombatDamagePass::NormalWithFirstStrike
        } else {
            let first_strikers: Vec<_> = combat
                .attackers
                .iter()
                .chain(combat.attacker_to_blockers.values().flatten())
                .copied()
                .filter(|id| {
                    self.still_in_combat(*id)
                        && self.creature_deals_damage_in_pass(*id, CombatDamagePass::FirstStrike)
                })
                .collect();
            if first_strikers.is_empty() {
                CombatDamagePass::Normal
            } else {
                self.state.combat.as_mut().unwrap().first_strikers = first_strikers;
                CombatDamagePass::FirstStrike
            }
        };
        let mut frame = EffectFrame::new(self.active_player(), Vec::new());
        frame.completion = FrameCompletion::Continue;
        self.assign_combat_damage(frame, pass, Vec::new());
    }

    pub(crate) fn assign_combat_damage(
        &mut self,
        mut frame: EffectFrame,
        pass: CombatDamagePass,
        assigned: Vec<i32>,
    ) {
        match self.combat_damage_batch(pass, &assigned) {
            Ok(batch) => {
                frame.batch = Some(batch);
                self.run_frame(frame);
            }
            Err(decision) => {
                *self.state.suspended_decision_mut() = Some(SuspendedResolution {
                    frame,
                    decision: *decision,
                });
            }
        }
    }

    /// Whether a creature recorded in combat still takes part: one that
    /// left the battlefield or phased out is removed from combat, and an
    /// attacker it blocked stays blocked (CR 506.4).
    fn still_in_combat(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id].is_some() && !self.is_phased_out(permanent_id)
    }

    /// One combat damage step (CR 510.2): every assignment is dealt, then
    /// the creatures that dealt damage to a player are announced together
    /// for "deals combat damage to a player" triggers.
    fn combat_damage_batch(
        &self,
        pass: CombatDamagePass,
        assignments: &[i32],
    ) -> Result<EventBatch, Box<Decision>> {
        let combat = self.state.combat.as_ref().expect("combat damage step");
        let mut choices = assignments.iter().copied();
        let defender = self.non_active_player();
        let mut events = Vec::new();

        for attacker_id in combat.attackers.iter().copied() {
            if !self.still_in_combat(attacker_id) {
                continue;
            }
            let attacker_ref = self.permanent_object(attacker_id);
            let attacker_deals = self.creature_deals_damage_in_pass(attacker_id, pass);

            let characteristics = self.characteristics(attacker_id);
            let attacker_power = characteristics.power.max(0);
            let attacker_keywords = characteristics.keywords;
            let attacker_has_trample = attacker_keywords.has(Keyword::Trample);
            let attacker_has_deathtouch = attacker_keywords.has(Keyword::Deathtouch);

            let declared_blockers = combat
                .attacker_to_blockers
                .get(&attacker_id)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let was_blocked = !declared_blockers.is_empty();
            let blockers: Vec<PermanentId> = declared_blockers
                .iter()
                .copied()
                .filter(|blocker_id| self.still_in_combat(*blocker_id))
                .collect();

            for blocker_id in &blockers {
                let blocker_ref = self.permanent_object(*blocker_id);
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

            if !attacker_deals {
                continue;
            }
            let mut remaining_damage = attacker_power;
            let mut can_trample = attacker_has_trample;
            for (index, &blocker_id) in blockers.iter().enumerate() {
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
                // CR 510.1c and 702.19b: damage can be divided freely among
                // blockers. Only assigning damage to the defender requires lethal.
                let last = index + 1 == blockers.len();
                let minimum = if !last {
                    0
                } else if can_trample {
                    lethal.min(remaining_damage)
                } else {
                    remaining_damage
                };
                let assigned = if minimum == remaining_damage {
                    minimum
                } else if let Some(amount) = choices.next() {
                    amount
                } else {
                    return Err(Box::new(Decision::AssignCombatDamage {
                        player: self.active_player(),
                        attacker: attacker_ref.unwrap(),
                        blocker: blocker.object,
                        minimum,
                        remaining: remaining_damage,
                        last,
                        pass,
                        assigned: assignments.to_vec(),
                    }));
                };
                can_trample &= assigned >= lethal;
                events.push(ProposedEvent::Damage {
                    source: attacker_ref,
                    target: ProposedDamageTarget::Object(blocker.object),
                    amount: assigned,
                    combat: true,
                });
                remaining_damage -= assigned;
            }

            if !was_blocked || (can_trample && remaining_damage > 0) {
                events.push(ProposedEvent::Damage {
                    source: attacker_ref,
                    target: ProposedDamageTarget::Player(defender),
                    amount: remaining_damage,
                    combat: true,
                });
            }
        }
        Ok(self.event_batch(EventBinding::Combat, events))
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
            CombatDamagePass::NormalWithFirstStrike => {
                has_double
                    || !self
                        .state
                        .combat
                        .as_ref()
                        .unwrap()
                        .first_strikers
                        .contains(&permanent_id)
            }
            CombatDamagePass::Normal => true,
        }
    }

    pub(crate) fn eligible_attackers(&self, player: PlayerId) -> Vec<PermanentId> {
        self.iter_battlefield_permanents(player)
            .filter(|&permanent_id| self.permanent_can_attack(permanent_id))
            .collect()
    }

    /// Whether a permanent can be declared as an attacker (CR 508.1a):
    /// a creature that can tap, without defender unless an effect lets it
    /// attack as though it didn't have defender (Platypus-Bear).
    fn permanent_can_attack(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let view = self.characteristics(permanent_id);
        let defender_applies = view.keywords.has(Keyword::Defender)
            && !self.attacks_despite_defender(&view, permanent.controller);
        view.is_creature()
            && self.permanent_can_tap(permanent_id)
            && !defender_applies
            && !self.cant_attack_or_block(permanent_id, &view)
            && self
                .state
                .turn
                .combat_attackers
                .as_ref()
                .is_none_or(|attackers| self.permanent_matches_predicate(permanent_id, attackers))
    }

    pub(crate) fn eligible_blockers(&self, player: PlayerId) -> Vec<PermanentId> {
        self.iter_battlefield_permanents(player)
            .filter(|&permanent_id| self.can_block(permanent_id))
            .collect()
    }

    /// Whether the permanent may be declared as a blocker at all (CR
    /// 509.1a-b): an untapped creature that no prohibition keeps from
    /// blocking every attacker. Pairwise restrictions are
    /// [`Self::blocker_can_block_attacker`]'s.
    fn can_block(&self, id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[id].as_ref() else {
            return false;
        };
        let view = self.characteristics(id);
        !permanent.tapped
            && view.is_creature()
            && !self.cant_attack_or_block(id, &view)
            && !self.cant_block_unconditionally(id, &view)
    }

    pub(crate) fn blocker_can_block_attacker(
        &self,
        blocker_id: PermanentId,
        attacker_id: PermanentId,
    ) -> bool {
        let Some(attacker) = self.state.permanents[attacker_id].as_ref() else {
            return false;
        };
        if !self.can_block(blocker_id) {
            return false;
        }
        let blocker_view = self.characteristics(blocker_id);
        let attacker_view = self.characteristics(attacker_id);

        if blocker_view.static_abilities.iter().any(|ability| {
            matches!(
                ability,
                StaticAbility::BlockRestriction(BlockRestriction::CantBlock(predicate))
                    if self.permanent_matches_predicate(attacker_id, predicate)
            )
        }) {
            return false;
        }
        if attacker_view
            .static_abilities
            .iter()
            .any(|ability| match ability {
                StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(predicate)) => {
                    self.permanent_matches_predicate(blocker_id, predicate)
                }
                _ => false,
            })
        {
            return false;
        }
        if self.cant_be_blocked_unconditionally(attacker_id, &attacker_view, attacker.controller) {
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
                if let Some(exposure) = self.exposure_mut() {
                    exposure
                        .combat
                        .blocks
                        .retain(|&(_, target)| target != attacker_id);
                }
            }
        }
    }
}
