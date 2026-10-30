// play.rs
// Spell casting, land plays, and activated abilities.
//
// Casting is a staged pipeline driven by `PendingChoice`:
//   kicker choice (optional) -> spell mode (if any) -> targets per clause -> payment.
// Waterbend activations run their own payment sub-decision before the
// ability reaches the stack. Activations name an `ActivationSource`: a
// permanent, or a card whose ability functions from its owner's hand or
// graveyard (cycling, "return this card from your graveyard").

use crate::{
    agent::action::AgentError,
    flow::{
        event::GameEvent,
        game::{ActivationSource, CastChoices, Game, Payment, PendingChoice},
    },
    state::{
        ability::TargetRequirement,
        card::{ActivatedAbilityDefinition, ActivationZone, AdditionalCost, StaticAbility},
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::{ManaCost, ManaUse},
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
        self.can_pay_automatic_mana(player, cost)
    }

    /// The graveyard casting cost of `card` for its owner, if any (CR 702.34).
    /// Iroh's continuous grant applies only on its controller's turn; printed
    /// flashback applies whenever the card's timing permits. When both apply,
    /// the grant is used: in TLA it never costs more than a printed flashback
    /// (a Lesson's {1}, or a spell's own mana cost).
    pub(crate) fn flashback_cost(&self, player: PlayerId, card: CardId) -> Option<ManaCost> {
        let definition = &self.state.cards[card];
        if !definition.types.is_spell()
            || !self.state.zones.contains(card, ZoneType::Graveyard, player)
        {
            return None;
        }
        let granted = self.active_player() == player
            && self.battlefield_permanents(player).iter().any(|id| {
                let source = self.state.permanents[*id].as_ref().unwrap();
                self.state.cards[source.card]
                    .has_static(&StaticAbility::GraveyardSpellsHaveFlashback)
            });
        if !granted {
            definition.flashback().cloned()
        } else if definition.has_subtype("Lesson") {
            Some(ManaCost::parse("{1}"))
        } else {
            definition.mana_cost.clone()
        }
    }

    /// The cost `player` may pay to cast `card` from outside their hand:
    /// graveyard flashback or an exact-object exile grant. A hand-only
    /// restriction overrides both.
    pub(crate) fn zone_cast_cost(&self, player: PlayerId, card: CardId) -> Option<ManaCost> {
        if self.hand_only_until(player).is_some() {
            return None;
        }
        match self.state.zones.zone_of(card)? {
            ZoneType::Graveyard => self.flashback_cost(player, card),
            ZoneType::Exile => self.grant_cast_cost(self.exile_grant(player, card)?, card),
            _ => None,
        }
    }

    /// The total cost to cast `card` right now with the announced `cast`
    /// choices: its mana cost (none when cast free) with X fixed, plus the
    /// kicker, less affinity-style reductions (CR 601.2f).
    pub(crate) fn effective_spell_cost(
        &self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
    ) -> Option<ManaCost> {
        let card_ref = &self.state.cards[card];
        let mut cost = if cast.free {
            ManaCost::default()
        } else if self.state.zones.zone_of(card) == Some(ZoneType::Hand) {
            card_ref.mana_cost.clone()?
        } else {
            self.zone_cast_cost(player, card)?
        };
        cost = cost.with_x(cast.x);
        cost.purpose = ManaUse::spell(card_ref.subtypes.iter().map(String::as_str));
        if cast.kicked {
            if let Some(kicker) = card_ref.optional_cost() {
                cost = cost.plus(kicker);
            }
        }
        for ability in &card_ref.statics {
            let count = match ability {
                StaticAbility::CostsLessPerPermanent(predicate) => {
                    self.count_battlefield_matching(player, predicate)
                }
                StaticAbility::CostsLessPerGraveyardCard(predicate) => {
                    self.count_graveyard_matching(player, predicate)
                }
                _ => continue,
            };
            cost = cost.reduced_generic(u8::try_from(count).unwrap_or(u8::MAX));
        }
        for permanent_id in self.battlefield_permanents(player) {
            let permanent = self.state.permanents[permanent_id].as_ref()?;
            for ability in &self.state.cards[permanent.card].statics {
                let StaticAbility::SpellsCostLess(reduction) = ability else {
                    continue;
                };
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
        if !self.may_play_from_zone(player, card) {
            return Err(AgentError("land is not in a permitted zone".into()));
        }
        if !self.can_play_land(player) {
            return Err(AgentError("cannot play land now".to_string()));
        }

        // CR 305.2 — Track one normal land play per turn.
        self.journal_turn();
        self.state.turn.lands_played += 1;
        self.move_card(card, ZoneType::Battlefield);

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
                self.has_instant_timing(player, card),
            )
        };

        if !self.may_play_from_zone(player, card) {
            return Err(AgentError("card is not in a permitted casting zone".into()));
        }
        if is_land {
            return Err(AgentError("land cards cannot be cast".to_string()));
        }
        if owner != player && self.exile_grant(player, card).is_none() {
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

        self.begin_cast(player, card, CastChoices::default())
    }

    /// Cast `card` without paying its mana cost while an effect resolves
    /// (CR 608.2g): its timing and zone permissions don't apply.
    pub(crate) fn begin_free_cast(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<(), AgentError> {
        let cast = CastChoices {
            free: true,
            ..CastChoices::default()
        };
        self.begin_cast(player, card, cast)
    }

    /// CR 601.2b — announce X, then any optional additional cost, before
    /// modes and targets. A free cast has X = 0 (CR 107.3b).
    fn begin_cast(
        &mut self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        let has_x = self.state.cards[card]
            .mana_cost
            .as_ref()
            .is_some_and(|cost| cost.x > 0);
        if has_x && !cast.free {
            self.pending_choice = Some(PendingChoice::AnnounceX { player, card, cast });
            return Ok(());
        }
        self.begin_kicker_choice(player, card, cast)
    }

    /// Whether `player` can pay the total cost of `card` with `cast`'s
    /// choices, waterbending any part that allows it.
    pub(crate) fn can_pay_cast(&self, player: PlayerId, card: CardId, cast: CastChoices) -> bool {
        let Some(cost) = self.effective_spell_cost(player, card, cast) else {
            return false;
        };
        let waterbend = self
            .cast_waterbend_generic(player, card, cast)
            .min(cost.generic());
        if waterbend > 0 {
            self.waterbend_payment_possible(player, &cost, waterbend)
        } else {
            self.can_pay_with_mana_abilities(player, &cost, None)
        }
    }

    /// The values of X `player` may announce: zero through the largest
    /// payable value (payability is monotone in X).
    pub(crate) fn legal_x_values(
        &self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
    ) -> Vec<u8> {
        let mut values = vec![0];
        for x in 1..=u8::MAX {
            if !self.can_pay_cast(player, card, CastChoices { x, ..cast }) {
                break;
            }
            values.push(x);
        }
        values
    }

    /// Announce X for the pending cast, then continue the pipeline.
    pub(crate) fn announce_x_action(
        &mut self,
        player: PlayerId,
        x: usize,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::AnnounceX {
            player: chooser,
            card,
            cast,
        }) = self.pending_choice
        else {
            return Err(AgentError("no X announcement is pending".into()));
        };
        let x = u8::try_from(x).map_err(|_| AgentError("X is too large".into()))?;
        if chooser != player || !self.legal_x_values(player, card, cast).contains(&x) {
            return Err(AgentError("illegal value of X".into()));
        }
        self.pending_choice = None;
        self.begin_kicker_choice(player, card, CastChoices { x, ..cast })
    }

    /// CR 601.2b — offer the optional additional cost when the total cost
    /// including it is payable.
    fn begin_kicker_choice(
        &mut self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        let kicked = CastChoices {
            kicked: true,
            ..cast
        };
        if self.state.cards[card].optional_cost().is_some()
            && self.can_pay_cast(player, card, kicked)
        {
            self.pending_choice = Some(PendingChoice::KickerChoice { player, card, cast });
            return Ok(());
        }
        self.begin_mode_selection(player, card, cast)
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
            cast,
        }) = self.pending_choice
        else {
            return Err(AgentError("no kicker choice is pending".to_string()));
        };
        if chooser != player {
            return Err(AgentError("wrong player for kicker choice".to_string()));
        }
        self.pending_choice = None;
        let cast = CastChoices {
            kicked: pay,
            ..cast
        };
        self.begin_mode_selection(player, card, cast)
    }

    /// The generic part of `card`'s total cost that `cast` lets `player`
    /// pay by waterbending: a waterbend additional cost once chosen, or all
    /// of a cost granted as "waterbending {X}" (Hama).
    pub(crate) fn cast_waterbend_generic(
        &self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
    ) -> u8 {
        let card_ref = &self.state.cards[card];
        let waterbend = card_ref
            .additional_costs
            .iter()
            .find_map(|cost| match cost {
                AdditionalCost::Waterbend(waterbend) if cast.kicked => Some(waterbend.generic()),
                _ => None,
            })
            .unwrap_or(0);
        let granted = !cast.free
            && self
                .exile_grant(player, card)
                .is_some_and(|grant| grant.waterbend);
        if granted {
            u8::MAX
        } else {
            waterbend
        }
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
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        if self.state.cards[card].spell_modes().is_some() {
            if self.legal_spell_modes(player, card).is_empty() {
                return Err(AgentError("no legal spell mode".into()));
            }
            self.pending_choice = Some(PendingChoice::SpellMode { player, card, cast });
            return Ok(());
        }
        self.advance_target_selection(player, card, cast, None, TargetSelection::default())
    }

    pub(crate) fn spell_mode_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::SpellMode {
            player: chooser,
            card,
            cast,
        }) = self.pending_choice
        else {
            return Err(AgentError("no spell mode is pending".into()));
        };
        if chooser != player || !self.legal_spell_modes(player, card).contains(&mode) {
            return Err(AgentError("illegal spell mode".into()));
        }
        self.advance_target_selection(player, card, cast, Some(mode), TargetSelection::default())
    }

    /// Move the casting pipeline to `requirement_index`, skipping
    /// requirements with no legal targets when they're optional, and
    /// finishing the cast once all requirements are satisfied.
    fn advance_target_selection(
        &mut self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
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
                return self.finish_casting(player, card, cast, mode, chosen, chosen_req_indices);
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
                cast,
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
            cast,
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
                cast,
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
                cast,
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
                cast,
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
                cast,
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
            cast,
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
            cast,
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
                cast,
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
                cast,
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
            cast,
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
        cast: CastChoices,
        mode: Option<usize>,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
    ) -> Result<(), AgentError> {
        let cost = self.state.cards[card].sacrifice_cost().cloned();
        self.begin_payment(
            player,
            Payment::Spell {
                card,
                cast,
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
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        self.journal_turn();
        self.state.turn.spells_cast[player.0].push(card);
        // CR 601.2i — A cast spell is put onto the stack.
        self.push_spell_to_stack(card, player, targets.clone(), target_req_indices, cast);
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

        if ability.zone != ActivationZone::Battlefield {
            return Err(AgentError(
                "ability does not function on the battlefield".into(),
            ));
        }
        if ability.mana_choices().is_some() {
            return self.activate_mana_ability(player, permanent_id, ability_index);
        }
        self.check_activation_terms(player, &ability)?;
        if permanent.activation_used(ability_index, &ability) {
            return Err(AgentError("ability was already activated".into()));
        }
        if ability.tap_source && !permanent.can_tap(&self.state.cards[source_card]) {
            return Err(AgentError("source cannot tap".into()));
        }
        let source = ActivationSource::Permanent(permanent_id);
        if ability.target_spec().is_some() {
            return self.begin_activated_targets(player, source, ability_index, &ability);
        }

        self.begin_activation_payment(player, source, ability_index, Vec::new())
    }

    /// Activate an ability of a card in `player`'s hand or graveyard. Only
    /// abilities printed for that zone function there (CR 113.6).
    pub(crate) fn activate_card_ability_action(
        &mut self,
        player: PlayerId,
        card: CardId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        let ability = self
            .card_ability_in_zone(player, card, ability_index)
            .ok_or_else(|| AgentError("no ability of this card functions here".into()))?;
        self.check_activation_terms(player, &ability)?;
        let source = ActivationSource::Card(card);
        if ability.target_spec().is_some() {
            return self.begin_activated_targets(player, source, ability_index, &ability);
        }
        self.begin_activation_payment(player, source, ability_index, Vec::new())
    }

    /// The turn and life terms every activation shares, whatever its source.
    pub(crate) fn check_activation_terms(
        &self,
        player: PlayerId,
        ability: &ActivatedAbilityDefinition,
    ) -> Result<(), AgentError> {
        if ability.own_turn_only && self.active_player() != player {
            return Err(AgentError("activation requires your turn".into()));
        }
        if self.state.players[player.0].life < ability.life_cost as i32 {
            return Err(AgentError("cannot pay life cost".into()));
        }
        if ability.sorcery_timing && !self.can_cast_sorceries(player) {
            return Err(AgentError("activation requires sorcery timing".into()));
        }
        Ok(())
    }

    /// The ability at `ability_index` of `card` when the card is in the
    /// zone of `player`'s that the ability functions from, priced as now.
    pub(crate) fn card_ability_in_zone(
        &self,
        player: PlayerId,
        card: CardId,
        ability_index: usize,
    ) -> Option<ActivatedAbilityDefinition> {
        let card_ref = &self.state.cards[card];
        let ability = card_ref.activated_abilities.get(ability_index)?;
        if ability.zone == ActivationZone::Battlefield
            || !self.state.zones.contains(card, ability.zone.zone(), player)
        {
            return None;
        }
        let mut ability = ability.clone();
        let purpose = ManaUse::ability(card_ref.subtypes.iter().map(String::as_str));
        self.price_activation(player, purpose, &mut ability);
        Some(ability)
    }

    /// Record an activation against its exhaust or once-each-turn limit.
    pub(crate) fn record_activation(
        &mut self,
        permanent: PermanentId,
        ability_index: usize,
        ability: &ActivatedAbilityDefinition,
    ) {
        if !ability.exhaust && !ability.once_each_turn {
            return;
        }
        self.journal_permanent(permanent);
        if let Some(permanent) = self.state.permanents[permanent].as_mut() {
            permanent.record_activation(ability_index, ability);
        }
    }

    /// The definition an activation of `source` uses; `None` once the
    /// source no longer exists where the ability functions.
    pub(crate) fn activation_ability(
        &self,
        source: ActivationSource,
        ability_index: usize,
    ) -> Option<ActivatedAbilityDefinition> {
        match source {
            ActivationSource::Permanent(permanent) => self
                .activated_abilities_for(permanent)
                .get(ability_index)
                .cloned(),
            ActivationSource::Card(card) => {
                let owner = self.state.cards[card].owner;
                self.card_ability_in_zone(owner, card, ability_index)
            }
        }
    }

    fn begin_activated_targets(
        &mut self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        ability: &ActivatedAbilityDefinition,
    ) -> Result<(), AgentError> {
        if ability.requires_target()
            && self
                .activated_ability_targets(player, source, ability_index)
                .is_empty()
        {
            return Err(AgentError("no legal ability targets".into()));
        }
        self.pending_choice = Some(PendingChoice::ActivatedTarget {
            player,
            source,
            ability_index,
        });
        Ok(())
    }

    pub(crate) fn activated_ability_targets(
        &self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
    ) -> Vec<Target> {
        let Some(ability) = self.activation_ability(source, ability_index) else {
            return Vec::new();
        };
        ability.target_spec().map_or_else(Vec::new, |spec| {
            self.legal_targets_for_spec(spec, player)
                .into_iter()
                .filter(|target| {
                    !matches!((source, target), (ActivationSource::Permanent(own), Target::Permanent(id)) if own == *id)
                })
                .collect()
        })
    }

    pub(crate) fn decline_activated_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(PendingChoice::ActivatedTarget {
            player: chooser,
            source,
            ability_index,
        }) = self.pending_choice
        else {
            return Err(AgentError("no activated target pending".into()));
        };
        let ability = self
            .activation_ability(source, ability_index)
            .ok_or_else(|| AgentError("missing activation source".into()))?;
        if chooser != player || !ability.target_is_optional() {
            return Err(AgentError("target required".into()));
        }
        self.begin_activation_payment(player, source, ability_index, Vec::new())
    }

    pub(crate) fn choose_activated_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::ActivatedTarget {
            player: chooser,
            source,
            ability_index,
        }) = self.pending_choice
        else {
            return Err(AgentError("no activated target pending".into()));
        };
        if player != chooser
            || !self
                .activated_ability_targets(player, source, ability_index)
                .contains(&target)
        {
            return Err(AgentError("illegal activated target".into()));
        }
        self.begin_activation_payment(player, source, ability_index, vec![target])
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
                    || !matches!(payment, Payment::Activation { source: ActivationSource::Permanent(permanent), .. } if permanent == id)
            })
            .collect()
    }

    fn begin_activation_payment(
        &mut self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        targets: Vec<Target>,
    ) -> Result<(), AgentError> {
        let ability = self
            .activation_ability(source, ability_index)
            .ok_or_else(|| AgentError("missing source".into()))?;
        if ability.discard_card {
            self.pending_choice = Some(PendingChoice::DiscardPayment {
                player,
                source,
                ability_index,
                targets,
            });
            return Ok(());
        }
        let cost = ability.sacrifice;
        self.begin_payment(
            player,
            Payment::Activation {
                source,
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

    /// Discard `card` for the pending "Discard a card" cost, then pay the
    /// rest of the activation cost.
    pub(crate) fn discard_payment_action(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::DiscardPayment {
            player: payer,
            source,
            ability_index,
            targets,
        }) = self.pending_choice.clone()
        else {
            return Err(AgentError("no discard payment pending".into()));
        };
        if payer != player || !self.state.zones.contains(card, ZoneType::Hand, player) {
            return Err(AgentError("discard cost must be a card in hand".into()));
        }
        let sacrifice = self
            .activation_ability(source, ability_index)
            .ok_or_else(|| AgentError("missing source".into()))?
            .sacrifice;
        self.move_card(card, ZoneType::Graveyard);
        let payment = Payment::Activation {
            source,
            ability_index,
            targets,
        };
        self.begin_payment(player, payment, sacrifice)
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
            Payment::Spell { card, cast, .. } => self
                .effective_spell_cost(player, *card, *cast)
                .unwrap_or_default(),
            Payment::Activation {
                source,
                ability_index,
                ..
            } => {
                self.activation_ability(*source, *ability_index)
                    .expect("pending activation source")
                    .mana_cost
            }
        };
        match alternative {
            Some(extra) => base.plus(extra),
            None => base,
        }
    }

    /// The part of `payment`'s generic cost its payer may pay by waterbending
    /// (tapping artifacts and creatures for {1} each): all of a waterbend
    /// ability's generic cost, or a spell's waterbend additional cost.
    pub(crate) fn payment_waterbend_generic(&self, player: PlayerId, payment: &Payment) -> u8 {
        match payment {
            Payment::Spell { card, cast, .. } => {
                let total = self.payment_mana_cost(player, payment, None).generic();
                self.cast_waterbend_generic(player, *card, *cast).min(total)
            }
            Payment::Activation {
                source,
                ability_index,
                ..
            } => self
                .activation_ability(*source, *ability_index)
                .filter(|ability| ability.waterbend)
                .map_or(0, |ability| ability.mana_cost.generic()),
        }
    }

    fn begin_mana_payment(
        &mut self,
        player: PlayerId,
        payment: Payment,
        sacrifice: Option<PermanentId>,
        alternative: Option<ManaCost>,
    ) -> Result<(), AgentError> {
        // Waterbend costs open a payment sub-decision: taps pay part of the
        // generic cost, then mana pays the rest.
        let waterbend = self.payment_waterbend_generic(player, &payment);
        if waterbend > 0 && sacrifice.is_none() && alternative.is_none() {
            self.pending_choice = Some(PendingChoice::Waterbend {
                player,
                payment: Box::new(payment),
                remaining_generic: waterbend,
                tapped: 0,
            });
            return Ok(());
        }
        self.pending_choice = Some(PendingChoice::ManaPayment {
            player,
            payment: Box::new(payment.clone()),
            sacrifice,
            alternative: alternative.clone(),
        });
        if !self.mana_ability_actions(player, sacrifice).is_empty() {
            return Ok(());
        }
        self.commit_payment(player, payment, sacrifice, alternative, 0)
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
        self.commit_payment(player, *payment, sacrifice, alternative, 0)
    }

    /// Pay `payment`'s announced cost, less `waterbent` generic already paid
    /// by waterbend taps, then cast the spell or activate the ability.
    fn commit_payment(
        &mut self,
        player: PlayerId,
        payment: Payment,
        sacrifice: Option<PermanentId>,
        alternative: Option<ManaCost>,
        waterbent: u8,
    ) -> Result<(), AgentError> {
        let cost = self
            .payment_mana_cost(player, &payment, alternative.as_ref())
            .reduced_generic(waterbent);
        self.produce_mana(player, &cost)?;
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
                cast,
                mode,
                targets,
                target_req_indices,
            } => {
                self.cast_spell(player, card, targets, target_req_indices, cast)?;
                if let Some(StackObject::Spell(spell)) = self.state.stack_objects.last_mut() {
                    spell.mode = mode;
                }
                self.state.priority.on_non_pass_action(self.active_player());
                Ok(())
            }
            Payment::Activation {
                source,
                ability_index,
                targets,
            } => {
                let life = self
                    .activation_ability(source, ability_index)
                    .ok_or_else(|| {
                        AgentError("activation source sacrificed as another cost".into())
                    })?
                    .life_cost;
                self.change_life(player, -(life as i32));
                match source {
                    ActivationSource::Permanent(permanent) => {
                        self.push_activated_ability(player, permanent, ability_index, targets)
                    }
                    ActivationSource::Card(card) => {
                        self.push_card_activated_ability(player, card, ability_index, targets)
                    }
                }
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

    /// The card whose spell or ability `payment` pays for.
    pub(crate) fn payment_source_card(&self, payment: &Payment) -> Option<CardId> {
        match payment {
            Payment::Spell { card, .. }
            | Payment::Activation {
                source: ActivationSource::Card(card),
                ..
            } => Some(*card),
            Payment::Activation {
                source: ActivationSource::Permanent(permanent),
                ..
            } => self.state.permanents[*permanent]
                .as_ref()
                .map(|permanent| permanent.card),
        }
    }

    /// The permanent whose ability `payment` activates when that ability
    /// also costs {T}: it can't tap for both.
    pub(crate) fn payment_tapped_source(&self, payment: &Payment) -> Option<PermanentId> {
        let Payment::Activation {
            source: ActivationSource::Permanent(source),
            ability_index,
            ..
        } = payment
        else {
            return None;
        };
        self.activated_abilities_for(*source)
            .get(*ability_index)
            .is_some_and(|ability| ability.tap_source)
            .then_some(*source)
    }

    /// Waterbend candidates for `payment`, less a source it taps anyway.
    pub(crate) fn waterbend_payment_candidates(
        &self,
        player: PlayerId,
        payment: &Payment,
    ) -> Vec<PermanentId> {
        let mut candidates = self.waterbend_candidates(player);
        if let Some(source) = self.payment_tapped_source(payment) {
            candidates.retain(|candidate| *candidate != source);
        }
        candidates
    }

    /// What a pending waterbend payment still owes, and how much of that
    /// generic further taps may pay instead of mana.
    pub(crate) fn waterbend_owed(
        &self,
        player: PlayerId,
        payment: &Payment,
        remaining_generic: u8,
        tapped: u8,
    ) -> (ManaCost, u8) {
        let owed = self
            .payment_mana_cost(player, payment, None)
            .reduced_generic(tapped);
        let generic = remaining_generic.min(owed.generic());
        (owed, generic)
    }

    /// Can `source`'s waterbend ability be paid, its own {T} included?
    pub(crate) fn can_pay_waterbend_activation(
        &self,
        player: PlayerId,
        source: PermanentId,
        ability: &ActivatedAbilityDefinition,
    ) -> bool {
        let cost = &ability.mana_cost;
        if !ability.tap_source {
            return self.waterbend_payment_possible(player, cost, cost.generic());
        }
        let mut probe = self.clone();
        probe.tap_permanent(source, false);
        probe.waterbend_payment_possible(player, cost, cost.generic())
    }

    /// Tap `permanent` to pay {1} of the pending waterbend cost.
    pub(crate) fn waterbend_tap_action(
        &mut self,
        player: PlayerId,
        tap_permanent: PermanentId,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::Waterbend {
            player: payer,
            payment,
            remaining_generic,
            tapped,
        }) = self.pending_choice.clone()
        else {
            return Err(AgentError("no waterbend payment is pending".to_string()));
        };
        if payer != player {
            return Err(AgentError("wrong player for waterbend payment".to_string()));
        }
        if remaining_generic == 0 {
            return Err(AgentError("waterbend cost is already paid".to_string()));
        }
        if !self
            .waterbend_payment_candidates(player, &payment)
            .contains(&tap_permanent)
        {
            return Err(AgentError(
                "permanent cannot be tapped for waterbend".to_string(),
            ));
        }

        // Waterbend pays a cost without activating a mana ability.
        self.tap_permanent(tap_permanent, false);
        let remaining_generic = remaining_generic - 1;
        let tapped = tapped + 1;
        if remaining_generic == 0 {
            self.pending_choice = None;
            return self.commit_payment(player, *payment, None, None, tapped);
        }
        self.pending_choice = Some(PendingChoice::Waterbend {
            player,
            payment,
            remaining_generic,
            tapped,
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
            payment,
            tapped,
            ..
        }) = self.pending_choice.clone()
        else {
            return Err(AgentError("no waterbend payment is pending".to_string()));
        };
        if payer != player {
            return Err(AgentError("wrong player for waterbend payment".to_string()));
        }
        self.pending_choice = None;
        self.commit_payment(player, *payment, None, None, tapped)
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
        self.record_activation(permanent_id, ability_index, &ability);
        if ability.sacrifice_source {
            // CR 601.2h, 602.2b — Costs are paid on activation; sacrificing
            // moves the source to its owner's graveyard immediately.
            self.sacrifice_permanent(permanent_id);
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

    /// Costs are paid: discard the source if the ability says so (CR 602.2b)
    /// and put the ability on the stack. It exists independently of the card
    /// (CR 113.7a) and refers to the card by its identity after costs.
    fn push_card_activated_ability(
        &mut self,
        player: PlayerId,
        card: CardId,
        ability_index: usize,
        targets: Vec<Target>,
    ) -> Result<(), AgentError> {
        let ability = self
            .card_ability_in_zone(player, card, ability_index)
            .ok_or_else(|| AgentError("activation source left its zone".into()))?;
        if ability.discard_source {
            self.move_card(card, ZoneType::Graveyard);
        }
        // The card as it stands once costs are paid: a discarded source is
        // referenced in the graveyard, where "it" still deals its damage.
        let source_ref = self
            .current_object_ref(card)
            .ok_or_else(|| AgentError("missing activation source identity".into()))?;
        let id = self.state.id_gen.next_id();
        self.push_to_stack(StackObject::ActivatedAbility(ActivatedAbilityOnStack {
            id,
            controller: player,
            source_definition_id: self.state.cards[card].definition_id,
            source_card: card,
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
