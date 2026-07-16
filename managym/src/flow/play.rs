// play.rs
// Spell casting, land plays, and activated abilities.
//
// Casting is a staged pipeline driven by `PendingChoice`:
//   kicker choice (optional) -> target selection per requirement -> payment.
// Waterbend activations run their own payment sub-decision before the
// ability reaches the stack.

use crate::{
    agent::action::AgentError,
    flow::{
        event::GameEvent,
        game::{Game, PendingChoice},
    },
    state::{
        ability::TargetRequirement,
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::ManaCost,
        stack_object::{ActivatedAbilityOnStack, StackObject},
        zone::ZoneType,
    },
};

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
        self.available_mana(player).can_pay(cost)
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
        let mut cost = card_ref.mana_cost.clone()?;
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
            if kicked_cost.is_some_and(|cost| self.can_pay_mana_cost(player, &cost)) {
                self.pending_choice = Some(PendingChoice::KickerChoice { player, card });
                return Ok(());
            }
        }

        self.begin_target_selection(player, card, false)
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
        self.begin_target_selection(player, card, pay)
    }

    /// Start (or continue) target selection at requirement 0.
    fn begin_target_selection(
        &mut self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
    ) -> Result<(), AgentError> {
        self.advance_target_selection(player, card, kicked, 0, Vec::new(), Vec::new())
    }

    /// Move the casting pipeline to `requirement_index`, skipping
    /// requirements with no legal targets when they're optional, and
    /// finishing the cast once all requirements are satisfied.
    fn advance_target_selection(
        &mut self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
        mut requirement_index: usize,
        chosen: Vec<Target>,
        chosen_req_indices: Vec<usize>,
    ) -> Result<(), AgentError> {
        let requirements = self.state.cards[card].target_requirements();
        loop {
            let Some(requirement) = requirements.get(requirement_index) else {
                return self.finish_casting(player, card, kicked, chosen, chosen_req_indices);
            };
            let legal =
                self.remaining_legal_targets(player, card, requirement, &chosen);
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
        chosen: &[Target],
    ) -> Vec<Target> {
        self.legal_targets_for_requirement(player, card, requirement)
            .into_iter()
            .filter(|target| !chosen.contains(target))
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

        let requirements = self.state.cards[card].target_requirements();
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
                requirement_index + 1,
                chosen,
                chosen_req_indices,
            );
        }

        let legal = self.remaining_legal_targets(player, card, requirement, &chosen);
        if legal.is_empty() {
            return self.advance_target_selection(
                player,
                card,
                kicked,
                requirement_index + 1,
                chosen,
                chosen_req_indices,
            );
        }
        self.pending_choice = Some(PendingChoice::ChooseTargets {
            player,
            card,
            kicked,
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
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets,
            });
            return Err(AgentError("wrong player for target choice".to_string()));
        }

        let requirements = self.state.cards[card].target_requirements();
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
            requirement_index + 1,
            chosen,
            chosen_req_indices,
        )
    }

    fn finish_casting(
        &mut self,
        player: PlayerId,
        card: CardId,
        kicked: bool,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
    ) -> Result<(), AgentError> {
        self.pending_choice = None;

        // CR 601.2f-h — Determine and pay costs as the final part of casting.
        if let Some(cost) = self.effective_spell_cost(player, card, kicked) {
            self.produce_mana(player, &cost)?;
            self.spend_mana(player, &cost)?;
        }

        self.cast_spell(player, card, targets, target_req_indices, kicked)?;
        self.state.priority.on_non_pass_action(self.active_player());
        Ok(())
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
        // CR 601.2i — A cast spell is put onto the stack.
        self.push_spell_to_stack(card, player, targets.clone(), target_req_indices, kicked);
        self.emit(GameEvent::SpellCast {
            card,
            targets: targets.clone(),
        });
        // Ward (CR 702.21) — targeted permanents see the spell that
        // targeted them.
        for target in targets {
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
        let Some(ability) = self.state.cards[source_card]
            .activated_abilities
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

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

        self.produce_mana(player, &ability.mana_cost)?;
        self.spend_mana(player, &ability.mana_cost)?;
        self.push_activated_ability(player, permanent_id, ability_index)
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

    /// Can `player` cover `cost` using up to `candidates` waterbend taps
    /// ({1} each, generic only) plus available mana?
    pub(crate) fn can_pay_with_waterbend(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        candidates: usize,
    ) -> bool {
        let tappable = candidates.min(cost.generic() as usize) as u8;
        self.available_mana(player)
            .can_pay(&cost.reduced_generic(tappable))
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

        // Waterbend taps count as tapping for mana — triggered mana
        // abilities (Badgermole Cub) fire and add to the pool immediately.
        self.tap_permanent(tap_permanent, true);
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
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            self.pending_choice = None;
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        let source_card = permanent.card;
        let Some(ability) = self.state.cards[source_card]
            .activated_abilities
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
        self.push_activated_ability(player, permanent_id, ability_index)
    }

    /// Costs are paid: put the activated ability on the stack (handling
    /// sacrifice-as-cost) and reset the priority round.
    fn push_activated_ability(
        &mut self,
        player: PlayerId,
        permanent_id: PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        let source_permanent_object_id = permanent.id;
        let source_card = permanent.card;
        let source_definition_id = self.state.cards[source_card].definition_id;
        let Some(ability) = self.state.cards[source_card]
            .activated_abilities
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

        if ability.sacrifice_source {
            // CR 601.2h, 602.2b — Costs are paid on activation; sacrificing
            // moves the source to its owner's graveyard immediately.
            self.move_card(source_card, ZoneType::Graveyard);
            self.invalidate_mana_cache(player);
        }

        let id = self.state.id_gen.next_id();
        self.push_to_stack(StackObject::ActivatedAbility(ActivatedAbilityOnStack {
            id,
            controller: player,
            source_definition_id,
            source_card,
            source_permanent_object_id,
            ability_index,
            targets: Vec::new(),
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
