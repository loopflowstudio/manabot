// play.rs
// Spell casting, land plays, and activated abilities.
//
// Casting is a staged pipeline driven by `PendingChoice`:
//   kicker choice (optional) -> spell mode (if any) -> targets per clause -> payment.
// Waterbend activations run their own payment sub-decision before the
// ability reaches the stack.

use crate::{
    agent::action::AgentError,
    flow::{
        event::GameEvent,
        game::{Game, Payment, PendingChoice},
    },
    state::{
        ability::TargetRequirement,
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::ManaCost,
        stack_object::{ActivatedAbilityOnStack, StackObject},
        zone::ZoneType,
    },
};

#[derive(Default)]
struct TargetSelection {
    requirement_index: usize,
    chosen: Vec<Target>,
    chosen_req_indices: Vec<usize>,
}

impl Game {
    pub fn can_cast_sorceries(&self, player: PlayerId) -> bool {
        // CR 117.1a, 307.1 — Sorcery-speed actions are available only to the active player
        // during a main phase with an empty stack.
        self.is_active_player(player)
            && self.stack_is_empty()
            && self.state.turn.can_cast_sorceries()
    }

    pub fn can_cast_instants(&self, _player: PlayerId) -> bool {
        // CR 117.1a — Any player with priority may cast an instant.
        true
    }

    pub fn can_play_land(&self, player: PlayerId) -> bool {
        // CR 305.1, 305.2 — Land plays use sorcery timing and are limited to one per turn.
        self.can_cast_sorceries(player) && self.state.turn.lands_played < 1
    }

    pub fn can_pay_mana_cost(&self, player: PlayerId, cost: &ManaCost) -> bool {
<<<<<<< HEAD
        self.available_mana(player).can_pay(cost)
=======
        self.can_pay_automatic_mana(player, cost)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    }

    /// Iroh's continuous grant is restricted to its controller's turn.
    pub(crate) fn flashback_cost(&self, player: PlayerId, card: CardId) -> Option<ManaCost> {
        let definition = &self.state.cards[card];
        if self.active_player() != player
            || !definition.types.is_spell()
            || !self.state.zones.contains(card, ZoneType::Graveyard, player)
            || !self.battlefield_permanents(player).iter().any(|id| {
                let source = self.state.permanents[*id].as_ref().unwrap();
                self.state.cards[source.card].grants_flashback
            })
        {
            return None;
        }
        if definition.has_subtype("Lesson") {
            Some(ManaCost::parse("{1}"))
        } else {
            definition.mana_cost.clone()
        }
    }

    /// The cost to cast `card` right now: base (+ kicker if `kicked`),
    /// with affinity-style reductions applied (CR 601.2f).
    pub(crate) fn effective_spell_cost(
        &self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
    ) -> Option<ManaCost> {
        let card_ref = &self.state.cards[card];
        let mut cost = if self.state.zones.zone_of(card) == Some(ZoneType::Graveyard) {
            self.flashback_cost(player, card)?
        } else {
            card_ref.mana_cost.clone()?
        };
        cost.spend_on_lesson = card_ref.has_subtype("Lesson");
        if kicked {
            if let Some(kicker) = &card_ref.kicker {
                cost = cost.plus(kicker);
            }
        }
        if let Some(predicate) = &card_ref.cost_reduction_per {
            let mut count = 0_u8;
            for permanent_id in self.battlefield_permanents(player) {
                if self.permanent_matches_predicate(permanent_id, predicate) {
                    count = count.saturating_add(1);
                }
            }
            cost = cost.reduced_generic(count);
        }
        for permanent_id in self.battlefield_permanents(player) {
            let permanent = self.state.permanents[permanent_id].as_ref()?;
            for reduction in &self.state.cards[permanent.card].spell_cost_reductions {
                if reduction.first_on_own_turn
                    && (self.active_player() != player
                        || self.state.turn.spells_cast[player.0]
                            .iter()
                            .any(|cast| reduction.predicate.matches_card(&self.state.cards[*cast])))
                {
                    continue;
                }
                if reduction.predicate.matches_card(card_ref)
                    && self.check_static_condition(&reduction.condition, player)
                {
                    cost = cost.reduced_generic(reduction.generic);
                }
            }
        }
        Some(cost)
    }

    pub(crate) fn play_land(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let card_ref = &self.state.cards[card];
        if !card_ref.types.is_land() {
            return Err(AgentError("only land cards can be played".to_string()));
        }
        if card_ref.owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        if !self.can_play_land(player) {
            return Err(AgentError("cannot play land now".to_string()));
        }

        // CR 305.2 — Track one normal land play per turn.
        self.journal_turn();
        self.state.turn.lands_played += 1;
        self.move_card(card, ZoneType::Battlefield);
        self.invalidate_mana_cache(player);

        Ok(())
    }

    pub(crate) fn cast_spell_action(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<(), AgentError> {
        if self.pending_choice.is_some() {
            return Err(AgentError("a choice is already pending".to_string()));
        }

        let (is_land, owner, is_instant_speed) = {
            let card_ref = &self.state.cards[card];
            (
                card_ref.types.is_land(),
                card_ref.owner,
                card_ref.is_instant_speed(),
            )
        };

        if !self.state.zones.contains(card, ZoneType::Hand, player)
            && self.flashback_cost(player, card).is_none()
        {
            return Err(AgentError("card is not in a permitted casting zone".into()));
        }
        if is_land {
            return Err(AgentError("land cards cannot be cast".to_string()));
        }
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        if is_instant_speed {
            if !self.can_cast_instants(player) {
                return Err(AgentError("cannot cast instant now".to_string()));
            }
        } else if !self.can_cast_sorceries(player) {
            return Err(AgentError(
                "cannot cast sorcery-speed spell now".to_string(),
            ));
        }

        // CR 601.2b — Optional additional costs (kicker) are chosen before
        // targets. Only offer the choice when the kicked cost is payable.
        if self.state.cards[card].kicker.is_some() {
            let kicked_cost = self.effective_spell_cost(player, card, true);
            if kicked_cost.is_some_and(|cost| self.can_pay_with_mana_abilities(player, &cost, None))
            {
                self.pending_choice = Some(PendingChoice::KickerChoice { player, card });
                return Ok(());
            }
        }

        self.begin_mode_selection(player, card, false)
    }

    /// Resolve the pending kicker choice, then continue the pipeline.
    pub(crate) fn kicker_choice_action(
        &mut self,
        player: PlayerId,
        pay: bool,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::KickerChoice {
            player: chooser,
            card,
        }) = self.pending_choice
        else {
            return Err(AgentError("no kicker choice is pending".to_string()));
        };
        if chooser != player {
            return Err(AgentError("wrong player for kicker choice".to_string()));
        }
        self.pending_choice = None;
        self.begin_mode_selection(player, card, pay)
    }

    pub(crate) fn legal_spell_modes(&self, player: PlayerId, card: CardId) -> Vec<usize> {
        self.state.cards[card]
            .spell_modes()
            .map_or_else(Vec::new, |modes| {
                (0..modes.len())
                    .filter(|mode| {
                        self.state.cards[card]
                            .target_requirements_for_mode(Some(*mode))
                            .iter()
                            .all(|req| {
                                req.min == 0
                                    || self.legal_targets_for_requirement(player, card, req).len()
                                        >= req.min
                            })
                    })
                    .collect()
            })
    }

    fn begin_mode_selection(
        &mut self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
    ) -> Result<(), AgentError> {
        if self.state.cards[card].spell_modes().is_some() {
            if self.legal_spell_modes(player, card).is_empty() {
                return Err(AgentError("no legal spell mode".into()));
            }
            self.pending_choice = Some(PendingChoice::SpellMode {
                player,
                card,
                kicked,
            });
            return Ok(());
        }
        self.advance_target_selection(player, card, kicked, None, TargetSelection::default())
    }

    pub(crate) fn spell_mode_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::SpellMode {
            player: chooser,
            card,
            kicked,
        }) = self.pending_choice
        else {
            return Err(AgentError("no spell mode is pending".into()));
        };
        if chooser != player || !self.legal_spell_modes(player, card).contains(&mode) {
            return Err(AgentError("illegal spell mode".into()));
        }
        self.advance_target_selection(player, card, kicked, Some(mode), TargetSelection::default())
    }

    /// Move the casting pipeline to `requirement_index`, skipping
    /// requirements with no legal targets when they're optional, and
    /// finishing the cast once all requirements are satisfied.
    fn advance_target_selection(
        &mut self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
        mode: Option<usize>,
        selection: TargetSelection,
    ) -> Result<(), AgentError> {
        let TargetSelection {
            mut requirement_index,
            chosen,
            chosen_req_indices,
        } = selection;
        let requirements = self.state.cards[card].target_requirements_for_mode(mode);
        loop {
            let Some(requirement) = requirements.get(requirement_index) else {
                return self.finish_casting(player, card, kicked, mode, chosen, chosen_req_indices);
            };
            let legal = self.remaining_legal_targets(
                player,
                card,
                requirement,
                requirement_index,
                &chosen,
                &chosen_req_indices,
            );
            if legal.is_empty() {
                if requirement.min == 0 {
                    requirement_index += 1;
                    continue;
                }
                return Err(AgentError("no legal targets".to_string()));
            }
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player,
                card,
                kicked,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets: legal,
            });
            return Ok(());
        }
    }

    fn remaining_legal_targets(
        &self,
        player: PlayerId,
        card: CardId,
        requirement: &TargetRequirement,
        requirement_index: usize,
        chosen: &[Target],
        chosen_req_indices: &[usize],
    ) -> Vec<Target> {
        self.legal_targets_for_requirement(player, card, requirement)
            .into_iter()
            // Distinctness is local to the clause; other clauses may reuse a target.
            .filter(|target| {
                !chosen
                    .iter()
                    .zip(chosen_req_indices)
                    .any(|(chosen, index)| *index == requirement_index && chosen == target)
            })
            .collect()
    }

    /// Handle a ChooseTarget action for the pending cast.
    pub(crate) fn choose_target_action(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::ChooseTargets {
            player: chooser,
            card,
            kicked,
            mode,
            requirement_index,
            chosen,
            chosen_req_indices,
            legal_targets,
        }) = self.pending_choice.take()
        else {
            return Err(AgentError("no target choice is pending".to_string()));
        };

        if chooser != player {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player: chooser,
                card,
                kicked,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets,
            });
            return Err(AgentError("wrong player for target choice".to_string()));
        }
        if !legal_targets.contains(&target) {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player: chooser,
                card,
                kicked,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets,
            });
            return Err(AgentError("target is not legal".to_string()));
        }

        let mut chosen = chosen;
        let mut chosen_req_indices = chosen_req_indices;
        chosen.push(target);
        chosen_req_indices.push(requirement_index);

        let requirements = self.state.cards[card].target_requirements_for_mode(mode);
        let requirement = &requirements[requirement_index];
        let chosen_in_req = chosen_req_indices
            .iter()
            .filter(|i| **i == requirement_index)
            .count();

        if chosen_in_req >= requirement.max {
            return self.advance_target_selection(
                player,
                card,
                kicked,
                mode,
                TargetSelection {
                    requirement_index: requirement_index + 1,
                    chosen,
                    chosen_req_indices,
                },
            );
        }

        let legal = self.remaining_legal_targets(
            player,
            card,
            requirement,
            requirement_index,
            &chosen,
            &chosen_req_indices,
        );
        if legal.is_empty() {
            return self.advance_target_selection(
                player,
                card,
                kicked,
                mode,
                TargetSelection {
                    requirement_index: requirement_index + 1,
                    chosen,
                    chosen_req_indices,
                },
            );
        }
        self.pending_choice = Some(PendingChoice::ChooseTargets {
            player,
            card,
            kicked,
            mode,
            requirement_index,
            chosen,
            chosen_req_indices,
            legal_targets: legal,
        });
        Ok(())
    }

    /// Stop choosing targets for an "up to N" requirement (its minimum has
    /// been met) and advance to the next requirement.
    pub(crate) fn finish_targets_action(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(PendingChoice::ChooseTargets {
            player: chooser,
            card,
            kicked,
            mode,
            requirement_index,
            chosen,
            chosen_req_indices,
            legal_targets,
        }) = self.pending_choice.take()
        else {
            return Err(AgentError("no target choice is pending".to_string()));
        };
        if chooser != player {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player: chooser,
                card,
                kicked,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets,
            });
            return Err(AgentError("wrong player for target choice".to_string()));
        }

        let requirements = self.state.cards[card].target_requirements_for_mode(mode);
        let requirement = &requirements[requirement_index];
        let chosen_in_req = chosen_req_indices
            .iter()
            .filter(|i| **i == requirement_index)
            .count();
        if chosen_in_req < requirement.min {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player: chooser,
                card,
                kicked,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets,
            });
            return Err(AgentError(
                "minimum number of targets not chosen".to_string(),
            ));
        }
        self.advance_target_selection(
            player,
            card,
            kicked,
            mode,
            TargetSelection {
                requirement_index: requirement_index + 1,
                chosen,
                chosen_req_indices,
            },
        )
    }

    fn finish_casting(
        &mut self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
        mode: Option<usize>,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
    ) -> Result<(), AgentError> {
        let cost = self.state.cards[card].sacrifice_cost.clone();
        self.begin_payment(
            player,
            Payment::Spell {
                card,
                kicked,
                mode,
                targets,
                target_req_indices,
            },
            cost,
        )
    }

    pub(crate) fn cast_spell(
        &mut self,
        player: PlayerId,
        card: CardId,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
        kicked: bool,
    ) -> Result<(), AgentError> {
        let owner = self.state.cards[card].owner;
        if owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        self.journal_turn();
        self.state.turn.spells_cast[player.0].push(card);
        // CR 601.2i — A cast spell is put onto the stack.
        self.push_spell_to_stack(card, player, targets.clone(), target_req_indices, kicked);
        self.emit(GameEvent::SpellCast {
            card,
            targets: targets.clone(),
        });
        // Ward (CR 702.21) — targeted permanents see the spell that
        // targeted them.
        let mut notified = std::collections::HashSet::new();
        for target in targets {
            if !notified.insert(target) {
                continue;
            }
            if let Target::Permanent(permanent) = target {
                self.emit(GameEvent::PermanentTargeted {
                    permanent,
                    spell: card,
                    spell_controller: player,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn activate_ability_action(
        &mut self,
        player: PlayerId,
        permanent_id: PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        if permanent.controller != player {
            return Err(AgentError(
                "source permanent is not controlled by player".to_string(),
            ));
        }
        if self.state.zones.zone_of(permanent.card) != Some(ZoneType::Battlefield) {
            return Err(AgentError(
                "source permanent must be on battlefield".to_string(),
            ));
        }

        let source_card = permanent.card;
        let Some(ability) = self
            .activated_abilities_for(permanent_id)
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

        if ability.mana_choices().is_some() {
            return self.activate_mana_ability(player, permanent_id, ability_index);
        }
        if ability.own_turn_only && self.active_player() != player {
            return Err(AgentError("activation requires your turn".into()));
        }
        if ability.tap_source && !permanent.can_tap(&self.state.cards[source_card]) {
            return Err(AgentError("source cannot tap".into()));
        }
        if self.state.players[player.0].life < ability.life_cost as i32 {
            return Err(AgentError("cannot pay life cost".into()));
        }
        if ability.effect.target_spec().is_some() {
            if crate::state::ability::effect_target_requirements(std::slice::from_ref(
                &ability.effect,
            ))
            .first()
            .is_some_and(|requirement| requirement.min > 0)
                && self
                    .activated_ability_targets(player, permanent_id, ability_index)
                    .is_empty()
            {
                return Err(AgentError("no legal ability targets".into()));
            }
            self.pending_choice = Some(PendingChoice::ActivatedTarget {
                player,
                permanent: permanent_id,
                ability_index,
            });
            return Ok(());
        }

        // Waterbend costs open a payment sub-decision instead of paying
        // directly (the generic part may be paid by tapping permanents).
        let generic = ability.mana_cost.generic();
        if ability.waterbend && generic > 0 {
            self.pending_choice = Some(PendingChoice::Waterbend {
                player,
                permanent: permanent_id,
                ability_index,
                remaining_generic: generic,
            });
            return Ok(());
        }

        self.begin_activation_payment(player, permanent_id, ability_index, Vec::new())
    }

    pub(crate) fn activated_ability_targets(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Vec<Target> {
        let Some(source) = self.state.permanents[permanent].as_ref() else {
            return Vec::new();
        };
        let Some(ability) = self.state.cards[source.card]
            .activated_abilities
            .get(ability_index)
        else {
            return Vec::new();
        };
        ability.effect.target_spec().map_or_else(Vec::new, |spec| {
            self.legal_targets_for_spec(spec, player)
                .into_iter()
                .filter(|target| *target != Target::Permanent(permanent))
                .collect()
        })
    }

    pub(crate) fn decline_activated_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(PendingChoice::ActivatedTarget {
            player: chooser,
            permanent,
            ability_index,
        }) = self.pending_choice
        else {
            return Err(AgentError("no activated target pending".into()));
        };
        let source = self.state.permanents[permanent]
            .as_ref()
            .ok_or_else(|| AgentError("missing activation source".into()))?;
        let ability = &self.state.cards[source.card].activated_abilities[ability_index];
        if chooser != player
            || !crate::state::ability::effect_target_requirements(std::slice::from_ref(
                &ability.effect,
            ))
            .first()
            .is_some_and(|requirement| requirement.min == 0)
        {
            return Err(AgentError("target required".into()));
        }
        self.begin_activation_payment(player, permanent, ability_index, Vec::new())
    }

    pub(crate) fn choose_activated_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::ActivatedTarget {
            player: chooser,
            permanent,
            ability_index,
        }) = self.pending_choice
        else {
            return Err(AgentError("no activated target pending".into()));
        };
        if player != chooser
            || !self
                .activated_ability_targets(player, permanent, ability_index)
                .contains(&target)
        {
            return Err(AgentError("illegal activated target".into()));
        }
        self.begin_activation_payment(player, permanent, ability_index, vec![target])
    }

    pub(crate) fn payment_candidates(
        &self,
        player: PlayerId,
        payment: &Payment,
        cost: &crate::state::card::SacrificeCost,
    ) -> Vec<PermanentId> {
        self.sacrifice_candidates(player, &cost.predicate)
            .into_iter()
            .filter(|id| {
                !cost.another
                    || !matches!(payment, Payment::Activation { permanent, .. } if permanent == id)
            })
            .collect()
    }

    fn begin_activation_payment(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
        targets: Vec<Target>,
    ) -> Result<(), AgentError> {
        let source = self.state.permanents[permanent]
            .as_ref()
            .ok_or_else(|| AgentError("missing source".into()))?;
        let cost = self.state.cards[source.card].activated_abilities[ability_index]
            .sacrifice
            .clone();
        self.begin_payment(
            player,
            Payment::Activation {
                permanent,
                ability_index,
                targets,
            },
            cost,
        )
    }

    fn begin_payment(
        &mut self,
        player: PlayerId,
        payment: Payment,
        cost: Option<crate::state::card::SacrificeCost>,
    ) -> Result<(), AgentError> {
        if let Some(cost) = cost {
            self.pending_choice = Some(PendingChoice::SacrificePayment {
                player,
                cost: Box::new(cost),
                payment: Box::new(payment),
            });
            Ok(())
        } else {
            self.begin_mana_payment(player, payment, None, None)
        }
    }

    pub(crate) fn sacrifice_payment_action(
        &mut self,
        player: PlayerId,
        card: Option<CardId>,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::SacrificePayment {
            player: chooser,
            payment,
            cost,
        }) = self.pending_choice.clone()
        else {
            return Err(AgentError("no sacrifice payment pending".into()));
        };
        if player != chooser {
            return Err(AgentError("wrong payer".into()));
        }
        let selected = if let Some(card) = card {
            Some(
                self.payment_candidates(player, &payment, &cost)
                    .into_iter()
                    .find(|id| {
                        self.state.permanents[*id]
                            .as_ref()
                            .is_some_and(|p| p.card == card)
                    })
                    .ok_or_else(|| AgentError("illegal sacrifice cost".into()))?,
            )
        } else if cost.or_mana.is_some() {
            None
        } else {
            return Err(AgentError("sacrifice required".into()));
        };
        let alternative = if selected.is_none() {
            cost.or_mana
        } else {
            None
        };
        self.begin_mana_payment(player, *payment, selected, alternative)
    }

    pub(crate) fn payment_mana_cost(
        &self,
        player: PlayerId,
        payment: &Payment,
        alternative: Option<&ManaCost>,
    ) -> ManaCost {
        let base = match payment {
            Payment::Spell { card, kicked, .. } => self
                .effective_spell_cost(player, *card, *kicked)
                .unwrap_or_default(),
            Payment::Activation {
                permanent,
                ability_index,
                ..
            } => {
                let source = self.state.permanents[*permanent]
                    .as_ref()
                    .expect("pending activation source");
                self.state.cards[source.card].activated_abilities[*ability_index]
                    .mana_cost
                    .clone()
            }
        };
        match alternative {
            Some(extra) => base.plus(extra),
            None => base,
        }
    }

<<<<<<< HEAD
    pub(crate) fn reserved_payment_source(&self, payment: &Payment) -> Option<PermanentId> {
        match payment {
            Payment::Activation {
                permanent,
                ability_index,
                ..
            } => {
                let source = self.state.permanents[*permanent].as_ref().unwrap();
                let ability = &self.state.cards[source.card].activated_abilities[*ability_index];
                (ability.tap_source || ability.sacrifice_source).then_some(*permanent)
            }
            Payment::Spell { .. } => None,
        }
    }

    pub(crate) fn pending_mana_source(&self) -> Option<PermanentId> {
        match &self.pending_choice {
            Some(PendingChoice::ManaPayment { payment, .. }) => {
                self.reserved_payment_source(payment)
            }
            _ => None,
        }
    }

=======
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    fn begin_mana_payment(
        &mut self,
        player: PlayerId,
        payment: Payment,
        sacrifice: Option<PermanentId>,
        alternative: Option<ManaCost>,
    ) -> Result<(), AgentError> {
        self.pending_choice = Some(PendingChoice::ManaPayment {
            player,
            payment: Box::new(payment.clone()),
            sacrifice,
            alternative: alternative.clone(),
        });
        if !self.mana_ability_actions(player, sacrifice).is_empty() {
            return Ok(());
        }
        self.commit_payment(player, payment, sacrifice, alternative)
    }

    pub(crate) fn finish_mana_payment(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(PendingChoice::ManaPayment {
            player: chooser,
            payment,
            sacrifice,
            alternative,
        }) = self.pending_choice.clone()
        else {
            return Err(AgentError("no mana payment pending".into()));
        };
        if chooser != player {
            return Err(AgentError("wrong payer".into()));
        }
        self.commit_payment(player, *payment, sacrifice, alternative)
    }

    fn commit_payment(
        &mut self,
        player: PlayerId,
        payment: Payment,
        sacrifice: Option<PermanentId>,
        alternative: Option<ManaCost>,
    ) -> Result<(), AgentError> {
        let cost = self.payment_mana_cost(player, &payment, alternative.as_ref());
<<<<<<< HEAD
        let reserved = self.reserved_payment_source(&payment);
        self.produce_mana_without(player, &cost, reserved)?;
=======
        self.produce_mana(player, &cost)?;
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
        // Commit the complete announced cost once, before sacrifice triggers
        // can observe a departing permanent or a spell reaches the stack.
        self.spend_mana(player, &cost)?;
        if let Some(selected) = sacrifice {
            self.sacrifice_permanent(selected);
        }
        self.pending_choice = None;
        match payment {
            Payment::Spell {
                card,
                kicked,
                mode,
                targets,
                target_req_indices,
            } => {
                self.cast_spell(player, card, targets, target_req_indices, kicked)?;
                if let Some(StackObject::Spell(spell)) = self.state.stack_objects.last_mut() {
                    spell.mode = mode;
                }
                self.state.priority.on_non_pass_action(self.active_player());
                Ok(())
            }
            Payment::Activation {
                permanent,
                ability_index,
                targets,
            } => {
                let source = self.state.permanents[permanent].as_ref().ok_or_else(|| {
                    AgentError("activation source sacrificed as another cost".into())
                })?;
                let life =
                    self.state.cards[source.card].activated_abilities[ability_index].life_cost;
                self.change_life(player, -(life as i32));
                self.push_activated_ability(player, permanent, ability_index, targets)
            }
        }
    }

    /// Untapped artifacts/creatures `player` controls — the permanents that
    /// can each pay {1} of a waterbend cost.
    pub(crate) fn waterbend_candidates(&self, player: PlayerId) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|permanent_id| {
                let Some(permanent) = self.state.permanents[*permanent_id].as_ref() else {
                    return false;
                };
                if permanent.tapped {
                    return false;
                }
                let card = &self.state.cards[permanent.card];
                card.types.is_artifact() || card.types.is_creature()
            })
            .collect()
    }

    /// Can `player` cover `cost` with untapped waterbend candidates
    /// ({1} each, generic only) plus available mana?
    pub(crate) fn can_pay_with_waterbend(&self, player: PlayerId, cost: &ManaCost) -> bool {
<<<<<<< HEAD
        if self.can_pay_with_mana_abilities(player, cost, None) {
            return true;
        }
        if cost.generic() == 0 {
            return false;
        }
        self.waterbend_candidates(player)
            .into_iter()
            .any(|candidate| {
                let mut probe = self.clone();
                probe.tap_permanent(candidate, false);
                probe.can_pay_with_waterbend(player, &cost.reduced_generic(1))
            })
=======
        self.waterbend_payment_possible(player, cost)
>>>>>>> 558e0186b585228dd2130cf14d9cb40de217e66f
    }

    /// Tap `permanent` to pay {1} of the pending waterbend cost.
    pub(crate) fn waterbend_tap_action(
        &mut self,
        player: PlayerId,
        tap_permanent: PermanentId,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::Waterbend {
            player: payer,
            permanent,
            ability_index,
            remaining_generic,
        }) = self.pending_choice
        else {
            return Err(AgentError("no waterbend payment is pending".to_string()));
        };
        if payer != player {
            return Err(AgentError("wrong player for waterbend payment".to_string()));
        }
        if remaining_generic == 0 {
            return Err(AgentError("waterbend cost is already paid".to_string()));
        }
        if !self.waterbend_candidates(player).contains(&tap_permanent) {
            return Err(AgentError(
                "permanent cannot be tapped for waterbend".to_string(),
            ));
        }

        // Waterbend pays a cost without activating a mana ability.
        self.tap_permanent(tap_permanent, false);
        let remaining_generic = remaining_generic - 1;

        if remaining_generic == 0 {
            return self.finish_waterbend(player, permanent, ability_index, remaining_generic);
        }
        self.pending_choice = Some(PendingChoice::Waterbend {
            player,
            permanent,
            ability_index,
            remaining_generic,
        });
        Ok(())
    }

    /// Pay the rest of the pending waterbend cost with mana.
    pub(crate) fn waterbend_pay_remainder_action(
        &mut self,
        player: PlayerId,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::Waterbend {
            player: payer,
            permanent,
            ability_index,
            remaining_generic,
        }) = self.pending_choice
        else {
            return Err(AgentError("no waterbend payment is pending".to_string()));
        };
        if payer != player {
            return Err(AgentError("wrong player for waterbend payment".to_string()));
        }
        self.finish_waterbend(player, permanent, ability_index, remaining_generic)
    }

    fn finish_waterbend(
        &mut self,
        player: PlayerId,
        permanent_id: PermanentId,
        ability_index: usize,
        remaining_generic: u8,
    ) -> Result<(), AgentError> {
        let Some(_) = self.state.permanents[permanent_id].as_ref() else {
            self.pending_choice = None;
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        let Some(ability) = self
            .activated_abilities_for(permanent_id)
            .get(ability_index)
            .cloned()
        else {
            self.pending_choice = None;
            return Err(AgentError("invalid ability index".to_string()));
        };

        // Pay the colored components plus the unpaid generic remainder.
        let remainder = ability.mana_cost.with_generic(remaining_generic);
        if remainder.mana_value > 0 {
            self.produce_mana(player, &remainder)?;
            self.spend_mana(player, &remainder)?;
        }
        self.pending_choice = None;
        self.push_activated_ability(player, permanent_id, ability_index, Vec::new())
    }

    /// Costs are paid: put the activated ability on the stack (handling
    /// sacrifice-as-cost) and reset the priority round.
    fn push_activated_ability(
        &mut self,
        player: PlayerId,
        permanent_id: PermanentId,
        ability_index: usize,
        targets: Vec<Target>,
    ) -> Result<(), AgentError> {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        let source_ref = self
            .permanent_object_ref(permanent_id)
            .ok_or_else(|| AgentError("missing activation source identity".into()))?;
        let source_card = permanent.card;
        let source_definition_id = self.state.cards[source_card].definition_id;
        let Some(ability) = self
            .activated_abilities_for(permanent_id)
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

        if ability.tap_source {
            if !permanent.can_tap(&self.state.cards[source_card]) {
                return Err(AgentError("source cannot pay tap cost".into()));
            }
            self.tap_permanent(permanent_id, false);
        }
        if ability.sacrifice_source {
            // CR 601.2h, 602.2b — Costs are paid on activation; sacrificing
            // moves the source to its owner's graveyard immediately.
            self.sacrifice_permanent(permanent_id);
            self.invalidate_mana_cache(player);
        }

        let id = self.state.id_gen.next_id();
        self.push_to_stack(StackObject::ActivatedAbility(ActivatedAbilityOnStack {
            id,
            controller: player,
            source_definition_id,
            source_card,
            source_ref,
            ability_index,
            targets,
        }));
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
    }

    /// Legal targets for one targeting requirement of `card`, cast by
    /// `player`. The card being cast is excluded from stack-spell targets
    /// (it is not on the stack yet while targets are chosen).
    pub(crate) fn legal_targets_for_requirement(
        &self,
        player: PlayerId,
        card: CardId,
        requirement: &TargetRequirement,
    ) -> Vec<Target> {
        self.legal_targets_for_spec(&requirement.spec, player)
            .into_iter()
            .filter(|target| *target != Target::StackSpell(card))
            .collect()
    }
}
