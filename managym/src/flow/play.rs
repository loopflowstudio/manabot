// play.rs
// Spell casting, land plays, and activated abilities.
//
// Casting is a staged pipeline driven by `PendingChoice`:
//   X (if any) -> kicker choice (optional) -> spell mode (if any) -> targets
//   per clause -> payment.
// Activations announce X (if any), choose targets, pay crew, then pay; a
// waterbend cost runs its own payment sub-decision before the ability
// reaches the stack. Activations name an `ActivationSource`: a
// permanent, or a card whose ability functions from its owner's hand or
// graveyard (cycling, "return this card from your graveyard").

use crate::{
    agent::action::{Action, AgentError},
    flow::{
        decision::{Decision, EffectFrame, FrameCompletion, SpendCompletion},
        event::GameEvent,
        game::{
            pay_pip_with, ActivationSource, CastChoices, CostComponent, Game, Help, Operation,
            Payment, PendingChoice,
        },
        proposed_event::ProposedEvent,
        turn::{PhaseKind, StepKind},
    },
    state::{
        ability::{EffectValue, TargetRequirement},
        card::{
            ActivatedAbilityDefinition, ActivationZone, AdditionalCost, CardType, SacrificeCost,
            StaticAbility, TapCost,
        },
        game_object::{CardId, ObjectId, PermanentId, PlayerId, Target},
        mana::{Color, ManaCost, SpellSnapshot, SpendingContext, SpendingObject},
        payment::{How, PaymentReceipt, Toward},
        predicate::CardPredicate,
        stack_object::ActivatedAbilityOnStack,
        zone::ZoneType,
    },
};

/// X values from `least` through the largest one `can_pay` accepts. Payability
/// is monotone in X, so the scan stops at the first unpayable value.
fn payable_x_values(least: u8, can_pay: impl Fn(u8) -> bool) -> Vec<u8> {
    (least..=u8::MAX).take_while(|&x| can_pay(x)).collect()
}

/// The fields of `PendingChoice::ActivatedTarget`: `player` activates and
/// `requirement` is the clause being chosen.
struct ActivatedTargeting {
    player: PlayerId,
    source: ActivationSource,
    ability_index: usize,
    chosen: Vec<Target>,
    requirement: usize,
}

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
        // CR 305.1, 305.2 — Land plays use sorcery timing and are limited to
        // one per turn, plus each additional land play granted.
        self.can_cast_sorceries(player)
            && self.state.turn.lands_played < 1 + self.additional_land_plays()
    }

    /// Additional land plays every player has on their turns (CR 305.2).
    fn additional_land_plays(&self) -> u32 {
        let permanents = [PlayerId(0), PlayerId(1)]
            .into_iter()
            .flat_map(|player| self.iter_battlefield_permanents(player));
        self.layer_views(permanents)
            .map(|(_, view)| {
                view.static_abilities
                    .iter()
                    .filter(|ability| {
                        matches!(ability, StaticAbility::AdditionalLandPlayEachPlayer)
                    })
                    .count() as u32
            })
            .sum()
    }

    pub fn can_pay_mana_cost(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.can_pay_automatic_mana(player, cost)
    }

    /// The graveyard casting cost of `card` for its owner, if any (CR 702.34).
    /// Iroh's continuous grant applies only on its controller's turn; printed
    /// flashback applies whenever the card's timing permits. When both apply,
    /// the grant is used: in TLA it never costs more than a printed flashback
    /// (a Lesson's {1}, or a spell's own mana cost).
    pub(crate) fn flashback_cost(&self, player: PlayerId, object: ObjectId) -> Option<ManaCost> {
        let card = self.state.card_of(object)?;
        let definition = &self.state[self.state.object_of(card)];
        if !definition.types.is_spell()
            || !self
                .state
                .zones
                .contains(object, ZoneType::Graveyard, player)
        {
            return None;
        }
        let granted = self.active_player() == player
            && self.active_permanent_has(player, |ability| {
                *ability == StaticAbility::GraveyardSpellsHaveFlashback
            });
        if !granted {
            definition.flashback().cloned()
        } else if definition.has_subtype("Lesson") {
            Some(ManaCost::parse("{1}"))
        } else {
            definition.mana_cost.clone()
        }
    }

    /// Whether a layered static ability on the player's battlefield matches.
    fn active_permanent_has(&self, player: PlayerId, has: impl Fn(&StaticAbility) -> bool) -> bool {
        self.layer_views(self.iter_battlefield_permanents(player))
            .any(|(_, view)| view.static_abilities.iter().any(&has))
    }

    /// Whether `player` may cast spells only from their hand: an "until your
    /// next turn" restriction, or an opponent's permanent that says so.
    pub(crate) fn casts_only_from_hand(&self, player: PlayerId) -> bool {
        self.hand_only_until(player).is_some()
            || self.active_permanent_has(PlayerId(1 - player.0), |ability| {
                *ability == StaticAbility::OpponentsCastOnlyFromHand
            })
    }

    /// The top card of `player`'s library, when a permanent they control
    /// lets them cast it as a spell (Hakoda's Ally spells). Lands are never
    /// cast, so they are never offered.
    pub(crate) fn library_top_castable(&self, player: PlayerId) -> Option<CardId> {
        let card = self.library_top_card(player)?;
        let definition = &self.state[self.state.object_of(card)];
        (!definition.types.is_land()
            && self.active_permanent_has(player, |source| {
                matches!(source, StaticAbility::CastFromLibraryTop(predicate) if self.object_matches_predicate(self.state.object_of(card), predicate))
            }))
        .then_some(card)
    }

    /// The cost `player` may pay to cast `card` from outside their hand:
    /// graveyard flashback, an exact-object exile grant, or the top of their
    /// library while a permanent permits it. A hand-only restriction overrides
    /// all three.
    pub(crate) fn zone_cast_cost(&self, player: PlayerId, object: ObjectId) -> Option<ManaCost> {
        if self.casts_only_from_hand(player) {
            return None;
        }
        match self.state.zones.zone_of(object)? {
            ZoneType::Graveyard => self.flashback_cost(player, object),
            ZoneType::Exile => self.grant_cast_cost(self.exile_grant(player, object)?, object),
            ZoneType::Library
                if self
                    .library_top_castable(player)
                    .map(|card| self.state.object_of(card))
                    == Some(object) =>
            {
                self.state[object].mana_cost.clone()
            }
            _ => None,
        }
    }

    /// The selected mana or alternative cost, before announcing X or applying
    /// additional costs and reductions. Announcement and payment share this
    /// choice so a fixed alternative cost cannot grant a free value of X.
    fn casting_mana_cost(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Option<ManaCost> {
        if cast.free || self.uses_exile_alternative(object, cast) {
            Some(ManaCost::default())
        } else if let (true, Some((cost, _))) = (cast.kicked, self.state[object].alternative_mana())
        {
            Some(cost.clone())
        } else if cast.resolving || self.state.zones.zone_of(object) == Some(ZoneType::Hand) {
            self.state[object].mana_cost.clone()
        } else {
            self.zone_cast_cost(player, object)
        }
    }

    /// The total cost to cast `card` right now with the announced `cast`
    /// choices: its mana cost (none when cast free) with X fixed, plus the
    /// kicker, less affinity-style reductions (CR 601.2f). `targets` are the
    /// chosen targets; before they are chosen (`None`), a target-dependent
    /// reduction applies if some legal target would earn it.
    pub(crate) fn effective_spell_cost(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
        targets: Option<&[Target]>,
    ) -> Option<ManaCost> {
        let card_ref = &self.state[object];
        let mut cost = self.casting_mana_cost(player, object, cast)?.with_x(cast.x);
        cost.purpose = SpendingContext::Cast(SpellSnapshot(card_ref.copiable_definition()));
        if let Some(additional) = card_ref.required_waterbend() {
            cost = cost.plus(&additional.with_x(cast.x));
        }
        if let Some(replicate) = card_ref.replicate() {
            for _ in 0..cast.x {
                cost = cost.plus(replicate);
            }
        }
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
                StaticAbility::CostsLessPerPermanentOnBattlefield(predicate) => {
                    [PlayerId(0), PlayerId(1)]
                        .into_iter()
                        .map(|controller| self.count_battlefield_matching(controller, predicate))
                        .sum()
                }
                StaticAbility::CostsLessPerGraveyardCard(predicate) => {
                    self.count_graveyard_matching(player, predicate)
                }
                StaticAbility::CostsLessByGreatestPower(predicate) => {
                    self.controller_value(
                        &EffectValue::GreatestPowerYouControl {
                            predicate: predicate.clone(),
                        },
                        player,
                    )
                    .max(0) as usize
                }
                _ => continue,
            };
            cost = cost.reduced_generic(u8::try_from(count).unwrap_or(u8::MAX));
        }
        if let Some(reduction) = card_ref.cost_reduction_if_targets() {
            let earns = |target: &Target| {
                matches!(target, Target::Permanent(id)
                    if self.permanent_matches_predicate(*id, &reduction.predicate))
            };
            let applies = match targets {
                Some(targets) => targets.iter().any(earns),
                None => card_ref.target_requirements().iter().any(|requirement| {
                    self.legal_targets_for_requirement(player, object, requirement)
                        .iter()
                        .any(earns)
                }),
            };
            if applies {
                cost = cost.reduced_generic(reduction.generic);
            }
        }
        for (_, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            for ability in &view.static_abilities {
                if let StaticAbility::SpellsCostLessMana(reduction) = ability {
                    cost = cost.reduced_by(reduction);
                    continue;
                }
                let StaticAbility::SpellsCostLess(reduction) = ability else {
                    continue;
                };
                if reduction.first_on_own_turn
                    && (self.active_player() != player
                        || self.state.turn.spells_cast[player.0].iter().any(|cast| {
                            self.recorded_object_matches_predicate(*cast, &reduction.predicate)
                        }))
                {
                    continue;
                }
                if self.object_matches_predicate(object, &reduction.predicate)
                    && self.check_static_condition(&reduction.condition, player)
                {
                    cost = cost.reduced_generic(reduction.generic);
                }
            }
        }
        Some(cost)
    }

    pub(crate) fn play_land(&mut self, player: PlayerId, card: CardId) -> Result<(), AgentError> {
        let card_ref = &self.state[self.state.object_of(card)];
        if !card_ref.types.is_land() {
            return Err(AgentError("only land cards can be played".to_string()));
        }
        if card_ref.owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        if !self.may_play_from_zone(player, self.state.object_of(card)) {
            return Err(AgentError("land is not in a permitted zone".into()));
        }
        if !self.can_play_land(player) {
            return Err(AgentError("cannot play land now".to_string()));
        }

        // CR 305.2 — Track one normal land play per turn.
        self.state.turn.lands_played += 1;
        let mut frame = EffectFrame::new(player, Vec::new());
        frame.batch = Some(self.event_batch(vec![self.zone_move_proposal(
            self.state.object_of(card),
            ZoneType::Battlefield,
            Default::default(),
        )]));
        frame.completion = FrameCompletion::Continue;
        self.run_frame(frame);
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
            let card_ref = &self.state[self.state.object_of(card)];
            (
                card_ref.types.is_land(),
                card_ref.owner,
                self.has_instant_timing(player, self.state.object_of(card)),
            )
        };

        if !self.may_play_from_zone(player, self.state.object_of(card)) {
            return Err(AgentError("card is not in a permitted casting zone".into()));
        }
        if is_land {
            return Err(AgentError("land cards cannot be cast".to_string()));
        }
        if owner != player
            && self
                .exile_grant(player, self.state.object_of(card))
                .is_none()
        {
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

        self.begin_cast(player, self.state.object_of(card), CastChoices::default())
    }

    /// Cast `card` while an effect resolves (CR 608.2g), paying its costs
    /// or not: its timing and zone permissions don't apply.
    pub(crate) fn begin_resolution_cast(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        paid: bool,
    ) -> Result<(), AgentError> {
        let cast = CastChoices {
            free: !paid,
            resolving: paid,
            ..CastChoices::default()
        };
        self.begin_cast(player, object, cast)
    }

    /// CR 601.2b — announce X (or a replicate count), then any optional additional cost, before
    /// modes and targets. Replacing a printed X cost with a free cast or a
    /// fixed alternative fixes X to zero (CR 107.3b).
    fn begin_cast(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        if self.cast_needs_x_announcement(player, object, cast) {
            self.pending_choice = Some(PendingChoice::AnnounceX {
                player,
                object,
                cast,
            });
            return Ok(());
        }
        self.begin_kicker_choice(player, object, cast)
    }

    /// Announcement and read-only explanation use the selected casting cost,
    /// including a fixed zone permission that can replace a printed X.
    pub(crate) fn cast_needs_x_announcement(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> bool {
        let card = &self.state[object];
        let printed_x = card.mana_cost.as_ref().is_some_and(|cost| cost.x > 0);
        let selected_x = self
            .casting_mana_cost(player, object, cast)
            .is_some_and(|cost| cost.x > 0);
        // CR 107.3a/b: additional X applies to a free cast when the printed
        // cost has no X; a replaced printed X is zero. Replicate announces
        // its count at the same boundary (CR 702.56a).
        let additional_x = card.required_waterbend().is_some_and(|cost| cost.x > 0);
        selected_x || (!printed_x && additional_x) || card.replicate().is_some()
    }

    /// Whether `player` can pay the total cost of `card` with `cast`'s
    /// choices, waterbending any part that allows it.
    pub(crate) fn can_pay_cast(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> bool {
        self.can_pay_cast_with_targets(player, object, cast, None)
    }

    /// [`Game::can_pay_cast`] once `targets` are chosen (`None` before).
    fn can_pay_cast_with_targets(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
        targets: Option<&[Target]>,
    ) -> bool {
        if self.uses_exile_alternative(object, cast)
            && self.exile_alternative_cards(player, object).is_empty()
        {
            return false;
        }
        let Some(cost) = self.effective_spell_cost(player, object, cast, targets) else {
            return false;
        };
        if !cast.kicked
            && self.state[object]
                .life_alternative()
                .is_some_and(|life| self.payable_life(player) < life as i32)
        {
            return false;
        }
        let card = &self.state[object];
        if card.discard_cost() && self.discard_candidates(player, Some(object)).is_empty() {
            return false;
        }
        if cast.kicked && card.bargain() {
            return self
                .sacrifice_candidates(player, &bargain_cost().predicate)
                .into_iter()
                .any(|id| self.can_pay_with_mana_abilities(player, &cost, Some(id)));
        }
        if let Some(or_mana) = card.forage_or() {
            return self.can_forage(player, object, &cost)
                || self.can_pay_with_mana_abilities(player, &cost.plus(or_mana), None);
        }
        let help = self.spell_help(player, object, cast, &cost);
        if help.keyworded() {
            self.payable_with_help(player, &cost, help, None)
        } else if help.waterbend > 0 {
            self.waterbend_payment_possible(player, &cost, help.waterbend)
        } else {
            self.can_pay_with_mana_abilities(player, &cost, None)
        }
    }

    /// The values of X `player` may announce: zero through the largest
    /// payable value (payability is monotone in X).
    pub(crate) fn legal_x_values(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Vec<u8> {
        let mut values = vec![0];
        values.extend(payable_x_values(1, |x| {
            self.can_pay_cast(player, object, CastChoices { x, ..cast })
        }));
        // "X targets" (CR 601.2c): X can't exceed the legal candidates.
        for requirement in self.state[object].target_requirements() {
            if requirement.exact_x() {
                let legal = self.legal_targets_for_requirement(player, object, &requirement);
                values.retain(|x| usize::from(*x) <= legal.len());
            }
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
            object,
            cast,
        }) = self.pending_choice
        else {
            return Err(AgentError("no X announcement is pending".into()));
        };
        let x = u8::try_from(x).map_err(|_| AgentError("X is too large".into()))?;
        if chooser != player || !self.legal_x_values(player, object, cast).contains(&x) {
            return Err(AgentError("illegal value of X".into()));
        }
        self.pending_choice = None;
        self.begin_kicker_choice(player, object, CastChoices { x, ..cast })
    }

    /// CR 601.2b — offer the optional additional cost when the total cost
    /// including it is payable.
    fn begin_kicker_choice(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        match self.fixed_cast_cost_choices(player, object, cast) {
            Some(fixed) => self.begin_mode_selection(player, object, fixed),
            None => {
                self.pending_choice = Some(PendingChoice::KickerChoice {
                    player,
                    object,
                    cast,
                });
                Ok(())
            }
        }
    }

    /// The announcement and preview share forced cost choices. None means
    /// both the base and optional/alternative payment are still possible.
    pub(crate) fn fixed_cast_cost_choices(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Option<CastChoices> {
        let kicked = CastChoices {
            kicked: true,
            ..cast
        };
        // CR 118.9a: a free cast cannot also take an alternative cost.
        let card = &self.state[object];
        let alternative = !cast.free && self.alternative_cost_available(player, object);
        let optional = card.optional_cost().is_some() || card.bargain();
        if (alternative || optional) && self.can_pay_cast(player, object, kicked) {
            if self.can_pay_cast(player, object, cast) {
                None
            } else {
                Some(kicked)
            }
        } else {
            Some(cast)
        }
    }

    /// Whether `object` has an alternative cost `player` may choose now: an
    /// exile alternative, or alternative mana whose condition holds.
    pub(crate) fn alternative_cost_available(&self, player: PlayerId, object: ObjectId) -> bool {
        let card = &self.state[object];
        card.exile_alternative().is_some()
            || card
                .alternative_mana()
                .is_some_and(|(_, condition)| self.check_static_condition(condition, player))
    }

    /// Resolve the pending kicker choice, then continue the pipeline.
    pub(crate) fn kicker_choice_action(
        &mut self,
        player: PlayerId,
        pay: bool,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::KickerChoice {
            player: chooser,
            object,
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
        self.begin_mode_selection(player, object, cast)
    }

    /// The generic part of `card`'s total cost that `cast` lets `player`
    /// pay by waterbending: a required waterbend additional cost, an
    /// optional one once chosen, or all of a cost granted as
    /// "waterbending {X}" (Hama).
    pub(crate) fn cast_waterbend_generic(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> u8 {
        let card_ref = &self.state[object];
        let waterbend = card_ref
            .additional_costs
            .iter()
            .map(|cost| match cost {
                AdditionalCost::Waterbend(waterbend) if cast.kicked => waterbend.generic(),
                AdditionalCost::RequiredWaterbend(cost) => cost.with_x(cast.x).generic(),
                _ => 0,
            })
            .fold(0u8, u8::saturating_add);
        let granted = !cast.free
            && self
                .exile_grant(player, object)
                .is_some_and(|grant| grant.waterbend);
        if granted {
            u8::MAX
        } else {
            waterbend
        }
    }

    /// How `player` may pay part of `cost`, the mana of the spell `object`,
    /// without mana: its own and granted convoke, improvise and delve, and
    /// the waterbend generic it announced.
    pub(crate) fn spell_help(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
        cost: &ManaCost,
    ) -> Help {
        let mut help = Help {
            waterbend: self
                .cast_waterbend_generic(player, object, cast)
                .min(cost.generic()),
            ..Default::default()
        };
        let mut grant = |cost: &AdditionalCost| match cost {
            AdditionalCost::Convoke => help.convoke = true,
            AdditionalCost::Improvise => help.improvise = true,
            AdditionalCost::Delve => help.delve = Some(object),
            _ => {}
        };
        self.state[object]
            .additional_costs
            .iter()
            .for_each(&mut grant);
        for (_, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            for ability in &view.static_abilities {
                if let StaticAbility::SpellsHave { predicate, cost } = ability {
                    if self.object_matches_predicate(object, predicate) {
                        grant(cost);
                    }
                }
            }
        }
        help
    }

    pub(crate) fn legal_spell_modes(&self, player: PlayerId, object: ObjectId) -> Vec<usize> {
        self.state[object]
            .spell_modes()
            .map_or_else(Vec::new, |modes| {
                (0..modes.len())
                    .filter(|mode| {
                        self.state[object]
                            .target_requirements_for_mode(Some(*mode))
                            .iter()
                            .all(|req| {
                                req.min == 0
                                    || self
                                        .legal_targets_for_requirement(player, object, req)
                                        .len()
                                        >= req.min
                            })
                    })
                    .collect()
            })
    }

    fn begin_mode_selection(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        if self.state[object].spell_modes().is_some() {
            if self.legal_spell_modes(player, object).is_empty() {
                return Err(AgentError("no legal spell mode".into()));
            }
            self.pending_choice = Some(PendingChoice::SpellMode {
                player,
                object,
                cast,
            });
            return Ok(());
        }
        self.advance_target_selection(player, object, cast, None, TargetSelection::default())
    }

    pub(crate) fn spell_mode_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::SpellMode {
            player: chooser,
            object,
            cast,
        }) = self.pending_choice
        else {
            return Err(AgentError("no spell mode is pending".into()));
        };
        if chooser != player || !self.legal_spell_modes(player, object).contains(&mode) {
            return Err(AgentError("illegal spell mode".into()));
        }
        self.advance_target_selection(player, object, cast, Some(mode), TargetSelection::default())
    }

    /// Move the casting pipeline to `requirement_index`, skipping
    /// requirements with no legal targets when they're optional, and
    /// finishing the cast once all requirements are satisfied.
    fn advance_target_selection(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
        mode: Option<usize>,
        selection: TargetSelection,
    ) -> Result<(), AgentError> {
        let TargetSelection {
            mut requirement_index,
            chosen,
            chosen_req_indices,
        } = selection;
        let requirements = self.state[object].target_requirements_for_mode(mode);
        loop {
            let Some(requirement) = requirements.get(requirement_index) else {
                return self.finish_casting(player, object, cast, mode, chosen, chosen_req_indices);
            };
            let legal = self
                .remaining_legal_targets(
                    player,
                    object,
                    requirement,
                    &chosen
                        .iter()
                        .zip(&chosen_req_indices)
                        .filter(|(_, index)| **index == requirement_index)
                        .map(|(target, _)| *target)
                        .collect::<Vec<_>>(),
                    cast.x,
                )
                .into_iter()
                .filter(|target| {
                    self.target_leaves_cost_payable(player, object, cast, &chosen, *target)
                })
                .collect::<Vec<_>>();
            if legal.is_empty() || requirement.max_for(cast.x) == 0 {
                if requirement.min_for(cast.x) == 0 {
                    requirement_index += 1;
                    continue;
                }
                return Err(AgentError("no legal targets".to_string()));
            }
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player,
                object,
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

    /// With a target-dependent reduction, a choice that would leave the
    /// spell's total cost unpayable is not offered (CR 601.2f-h).
    fn target_leaves_cost_payable(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
        chosen: &[Target],
        target: Target,
    ) -> bool {
        if self.state[object].cost_reduction_if_targets().is_none() {
            return true;
        }
        let targets: Vec<Target> = chosen.iter().copied().chain([target]).collect();
        self.can_pay_cast_with_targets(player, object, cast, Some(&targets))
    }

    fn remaining_legal_targets(
        &self,
        player: PlayerId,
        object: ObjectId,
        requirement: &TargetRequirement,
        chosen: &[Target],
        x: u8,
    ) -> Vec<Target> {
        self.legal_targets_for_requirement(player, object, requirement)
            .into_iter()
            .filter(|target| {
                self.within_target_mana_budget(requirement, x, chosen.iter().copied(), *target)
            })
            // Distinctness and the budget are local to this clause.
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
            object,
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
                object,
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
                object,
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

        let requirements = self.state[object].target_requirements_for_mode(mode);
        let requirement = &requirements[requirement_index];
        let chosen_in_req = chosen_req_indices
            .iter()
            .filter(|i| **i == requirement_index)
            .count();

        if chosen_in_req >= requirement.max_for(cast.x) {
            return self.advance_target_selection(
                player,
                object,
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
            object,
            requirement,
            &chosen
                .iter()
                .zip(&chosen_req_indices)
                .filter(|(_, index)| **index == requirement_index)
                .map(|(target, _)| *target)
                .collect::<Vec<_>>(),
            cast.x,
        );
        if legal.is_empty() {
            return self.advance_target_selection(
                player,
                object,
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
            object,
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
            object,
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
                object,
                cast,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                legal_targets,
            });
            return Err(AgentError("wrong player for target choice".to_string()));
        }

        let requirements = self.state[object].target_requirements_for_mode(mode);
        let requirement = &requirements[requirement_index];
        let chosen_in_req = chosen_req_indices
            .iter()
            .filter(|i| **i == requirement_index)
            .count();
        if chosen_in_req < requirement.min_for(cast.x) {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                player: chooser,
                object,
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
            object,
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
        object: ObjectId,
        cast: CastChoices,
        mode: Option<usize>,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
    ) -> Result<(), AgentError> {
        self.begin_payment(
            player,
            Operation::Spell {
                object,
                cast,
                mode,
                targets,
                target_req_indices,
            },
        )
    }

    /// Whether `cast` takes `card`'s exile-from-hand alternative cost: such a
    /// card's kicked flag means the alternative, which costs no mana.
    fn uses_exile_alternative(&self, object: ObjectId, cast: CastChoices) -> bool {
        cast.kicked && self.state[object].exile_alternative().is_some()
    }

    /// The cards `player` may exile for `card`'s exile-from-hand alternative
    /// cost: other matching cards in hand while `card` is cast from hand,
    /// paying its costs, and, if the cost says so, outside `player`'s turn.
    pub(crate) fn exile_alternative_cards(
        &self,
        player: PlayerId,
        object: ObjectId,
    ) -> Vec<CardId> {
        let Some((predicate, not_your_turn)) = self.state[object].exile_alternative() else {
            return Vec::new();
        };
        if (not_your_turn && self.active_player() == player)
            || !self.state.zones.contains(object, ZoneType::Hand, player)
        {
            return Vec::new();
        }
        self.state
            .zone_cards(ZoneType::Hand, player)
            .filter(|card| {
                self.state.object_of(*card) != object
                    && self.object_matches_predicate(self.state.object_of(*card), predicate)
            })
            .collect()
    }

    pub(crate) fn cast_spell(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        targets: Vec<Target>,
        target_req_indices: Vec<usize>,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        // Recorded before the move, so its invalidation precedes the move's
        // board reads; the move then names the stack appearance.
        self.state.turn.spells_cast[player.0].push(object);
        if self.state[object].types.is_creature() {
            self.end_next_creature_cast_flash(player);
        }
        // CR 601.2i — A cast spell is put onto the stack.
        let object =
            self.push_spell_to_stack(object, player, targets.clone(), target_req_indices, cast);
        self.state.name_last_cast(player, object);
        self.emit(GameEvent::SpellCast {
            object_id: object,
            targets: targets.clone(),
        });
        self.emit_permanents_targeted(&targets, Target::StackSpell(object), player);
        Ok(())
    }

    /// Ward (CR 702.21) — each targeted permanent sees, once, the spell or
    /// ability that targeted it as that object is put on the stack.
    pub(crate) fn emit_permanents_targeted(
        &mut self,
        targets: &[Target],
        by: Target,
        by_controller: PlayerId,
    ) {
        let mut notified = std::collections::HashSet::new();
        for target in targets {
            if !notified.insert(*target) {
                continue;
            }
            if let Target::Permanent(permanent) = *target {
                self.emit(GameEvent::PermanentTargeted {
                    permanent,
                    by,
                    by_controller,
                });
            }
        }
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
        if self.state.zones.zone_of(permanent.object) != Some(ZoneType::Battlefield) {
            return Err(AgentError(
                "source permanent must be on battlefield".to_string(),
            ));
        }

        let Some(ability) = self
            .characteristics(permanent_id)
            .activated_abilities
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
        if self.activations_prohibited(permanent_id) {
            return Err(AgentError(
                "its activated abilities can't be activated".into(),
            ));
        }
        if ability.mana_choices().is_some() {
            return self.activate_mana_ability(player, permanent_id, ability_index);
        }
        self.check_activation_terms(player, &ability)?;
        if permanent.activation_used(ability_index, &ability) {
            return Err(AgentError("ability was already activated".into()));
        }
        if ability.tap_source && !self.permanent_can_tap(permanent_id) {
            return Err(AgentError("source cannot tap".into()));
        }
        if ability.mana_cost.x > 0 {
            if self
                .legal_activation_x_values(player, permanent_id, ability_index)
                .is_empty()
            {
                return Err(AgentError("no payable value of X".into()));
            }
            self.pending_choice = Some(PendingChoice::ActivationX {
                player,
                permanent: permanent_id,
                ability_index,
            });
            return Ok(());
        }
        let source = ActivationSource::Permanent(permanent_id);
        if ability.target_spec().is_some() {
            return self.begin_activated_targets(player, source, ability_index, &ability);
        }
        if ability
            .tap_power()
            .is_some_and(|(power, _)| self.crew_power_available(player, permanent_id) < power)
        {
            return Err(AgentError("not enough power to crew or saddle".into()));
        }
        self.begin_payment(
            player,
            Operation::Activation {
                source,
                ability_index,
                targets: Vec::new(),
                x: 0,
            },
        )
    }

    /// Values of X from the ability's least X that its waterbend cost can
    /// pay (CR 602.2b, 601.2f).
    pub(crate) fn legal_activation_x_values(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Vec<u8> {
        let Some(ability) = self
            .characteristics(permanent)
            .activated_abilities
            .get(ability_index)
            .cloned()
        else {
            return Vec::new();
        };
        payable_x_values(ability.x_min, |x| {
            self.can_pay_with_waterbend(player, &ability.mana_cost.with_x(x))
        })
    }

    /// Announce X for the pending activation, then pay its costs.
    pub(crate) fn choose_activation_x_action(
        &mut self,
        player: PlayerId,
        x: usize,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::ActivationX {
            player: chooser,
            permanent,
            ability_index,
        }) = self.pending_choice
        else {
            return Err(AgentError("no value of X is pending".into()));
        };
        let x = u8::try_from(x).map_err(|_| AgentError("illegal value of X".into()))?;
        if chooser != player
            || !self
                .legal_activation_x_values(player, permanent, ability_index)
                .contains(&x)
        {
            return Err(AgentError("illegal value of X".into()));
        }
        self.pending_choice = None;
        self.begin_payment(
            player,
            Operation::Activation {
                source: ActivationSource::Permanent(permanent),
                ability_index,
                targets: Vec::new(),
                x,
            },
        )
    }

    /// Activate an ability of a card in `player`'s hand or graveyard. Only
    /// abilities printed for that zone function there (CR 113.6).
    pub(crate) fn activate_card_ability_action(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        let card = self
            .state
            .card_of(object)
            .ok_or_else(|| AgentError("activation source is not a card".into()))?;
        let ability = self
            .card_ability_in_zone(player, card, ability_index)
            .ok_or_else(|| AgentError("no ability of this card functions here".into()))?;
        self.check_activation_terms(player, &ability)?;
        let source = ActivationSource::Card(card);
        if ability.target_spec().is_some() {
            return self.begin_activated_targets(player, source, ability_index, &ability);
        }
        self.begin_payment(
            player,
            Operation::Activation {
                source,
                ability_index,
                targets: Vec::new(),
                x: 0,
            },
        )
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
        if self.payable_life(player) < ability.life_cost as i32 {
            return Err(AgentError("cannot pay life cost".into()));
        }
        if ability.sorcery_timing && !self.can_cast_sorceries(player) {
            return Err(AgentError("activation requires sorcery timing".into()));
        }
        if ability.before_attackers && !self.before_attackers_declared() {
            return Err(AgentError(
                "activation requires a time before attackers".into(),
            ));
        }
        Ok(())
    }

    /// Whether this turn's attackers are not yet declared: the beginning
    /// phase, the first main phase or the beginning of combat.
    fn before_attackers_declared(&self) -> bool {
        match self.state.turn.current_phase_kind() {
            PhaseKind::Beginning | PhaseKind::PrecombatMain => true,
            PhaseKind::Combat => self.state.turn.current_step_kind() == StepKind::BeginningOfCombat,
            _ => false,
        }
    }

    /// The ability at `ability_index` of `card` when the card is in the
    /// zone of `player`'s that the ability functions from, priced as now.
    pub(crate) fn card_ability_in_zone(
        &self,
        player: PlayerId,
        card: CardId,
        ability_index: usize,
    ) -> Option<ActivatedAbilityDefinition> {
        let definition = &self.state[self.state.object_of(card)];
        let ability = definition.activated_abilities.get(ability_index)?;
        if ability.zone == ActivationZone::Battlefield
            || !self
                .state
                .zones
                .contains(self.state.object_of(card), ability.zone.zone(), player)
        {
            return None;
        }
        let mut ability = ability.clone();
        ability.mana_cost.purpose = SpendingContext::Activate {
            zone: ability.zone.zone(),
            source: SpendingObject::new(definition.subtypes.iter().map(String::as_str)),
        };
        self.price_activation(player, &mut ability);
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
        self.state
            .record_activation(permanent, ability_index, ability);
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
                .characteristics(permanent)
                .activated_abilities
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
        let requirements = ability.target_requirements();
        debug_assert!(
            requirements.len() < 2 || requirements.iter().all(|r| r.min == r.max),
            "several activation clauses each name a fixed number of targets"
        );
        if !self.activation_targets_available(player, source, ability_index) {
            return Err(AgentError("no legal ability targets".into()));
        }
        self.pending_choice = Some(PendingChoice::ActivatedTarget {
            player,
            source,
            ability_index,
            chosen: Vec::new(),
            requirement: 0,
        });
        Ok(())
    }

    /// Whether every required targeting clause has a legal candidate.
    pub(crate) fn activation_targets_available(
        &self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
    ) -> bool {
        let Some(ability) = self.activation_ability(source, ability_index) else {
            return false;
        };
        ability
            .target_requirements()
            .iter()
            .enumerate()
            .all(|(index, requirement)| {
                requirement.min == 0
                    || !self
                        .activated_ability_targets(player, source, ability_index, index, &[])
                        .is_empty()
            })
    }

    /// Targeting clause `requirement` of the activation in progress.
    pub(crate) fn activation_target_requirement(
        &self,
        source: ActivationSource,
        ability_index: usize,
        requirement: usize,
    ) -> Option<TargetRequirement> {
        self.activation_ability(source, ability_index)?
            .target_requirements()
            .get(requirement)
            .cloned()
    }

    /// Who chooses clause `requirement` of `player`'s activation: they do,
    /// or their opponent for "of an opponent's choice".
    pub(crate) fn activation_target_chooser(
        &self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        requirement: usize,
    ) -> PlayerId {
        match self.activation_target_requirement(source, ability_index, requirement) {
            Some(clause) if clause.opponent_chooses => PlayerId(1 - player.0),
            _ => player,
        }
    }

    /// The targets already chosen for clause `requirement`.
    pub(crate) fn activation_clause_chosen(
        &self,
        source: ActivationSource,
        ability_index: usize,
        requirement: usize,
        chosen: &[Target],
    ) -> Vec<Target> {
        let Some(ability) = self.activation_ability(source, ability_index) else {
            return Vec::new();
        };
        let indices = ability.target_clause_indices(chosen.len());
        chosen
            .iter()
            .enumerate()
            .filter(|(slot, _)| indices.get(*slot).copied().unwrap_or(0) == requirement)
            .map(|(_, target)| *target)
            .collect()
    }

    /// Targets clause `requirement` of `player`'s activation can choose
    /// next, given those already `chosen` (all clauses). Legality is from
    /// `player`'s side whoever chooses. Under "that share a creature type",
    /// a target is offered only with the chosen ones and, while the minimum
    /// is unmet, a remaining partner.
    pub(crate) fn activated_ability_targets(
        &self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        requirement: usize,
        chosen: &[Target],
    ) -> Vec<Target> {
        let Some(clause) = self.activation_target_requirement(source, ability_index, requirement)
        else {
            return Vec::new();
        };
        let chosen = self.activation_clause_chosen(source, ability_index, requirement, chosen);
        let candidates: Vec<Target> = self
            .legal_targets_for_spec(&clause.spec, player)
            .into_iter()
            .filter(|target| {
                let is_source = match source {
                    ActivationSource::Permanent(permanent) => {
                        *target == Target::Permanent(permanent)
                    }
                    ActivationSource::Card(_) => false,
                };
                !is_source && !chosen.contains(target)
            })
            .collect();
        if !clause.shares_creature_type {
            return candidates;
        }
        let needs_partner = chosen.len() + 1 < clause.min;
        candidates
            .iter()
            .copied()
            .filter(|target| {
                let with = |extra: &[Target]| {
                    self.targets_share_creature_type(chosen.iter().chain([target]).chain(extra))
                };
                with(&[])
                    && (!needs_partner
                        || candidates.iter().any(|partner| {
                            partner != target && with(std::slice::from_ref(partner))
                        }))
            })
            .collect()
    }

    /// Whether the permanents among `targets` have a creature type in common
    /// ("that share a creature type").
    fn targets_share_creature_type<'a>(&self, targets: impl Iterator<Item = &'a Target>) -> bool {
        let mut common: Option<Vec<String>> = None;
        for target in targets {
            let Target::Permanent(id) = target else {
                return false;
            };
            let types = self.characteristics(*id).creature_types.clone();
            common = Some(match common {
                None => types,
                Some(common) => common.into_iter().filter(|t| types.contains(t)).collect(),
            });
        }
        common.is_some_and(|common| !common.is_empty())
    }

    /// The pending activation targeting.
    fn pending_activated_target(&self) -> Result<ActivatedTargeting, AgentError> {
        let Some(PendingChoice::ActivatedTarget {
            player,
            source,
            ability_index,
            chosen,
            requirement,
        }) = self.pending_choice.clone()
        else {
            return Err(AgentError("no activated target pending".into()));
        };
        Ok(ActivatedTargeting {
            player,
            source,
            ability_index,
            chosen,
            requirement,
        })
    }

    /// Finish choosing the current clause's targets once its minimum is met.
    pub(crate) fn decline_activated_target(&mut self, actor: PlayerId) -> Result<(), AgentError> {
        let ActivatedTargeting {
            player,
            source,
            ability_index,
            chosen,
            requirement,
        } = self.pending_activated_target()?;
        let chooser = self.activation_target_chooser(player, source, ability_index, requirement);
        let clause = self
            .activation_target_requirement(source, ability_index, requirement)
            .ok_or_else(|| AgentError("missing activation source".into()))?;
        let count = self
            .activation_clause_chosen(source, ability_index, requirement, &chosen)
            .len();
        if chooser != actor || count < clause.min {
            return Err(AgentError("target required".into()));
        }
        self.next_activation_clause(player, source, ability_index, requirement + 1, chosen)
    }

    pub(crate) fn choose_activated_target(
        &mut self,
        actor: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let ActivatedTargeting {
            player,
            source,
            ability_index,
            mut chosen,
            requirement,
        } = self.pending_activated_target()?;
        let chooser = self.activation_target_chooser(player, source, ability_index, requirement);
        if actor != chooser
            || !self
                .activated_ability_targets(player, source, ability_index, requirement, &chosen)
                .contains(&target)
        {
            return Err(AgentError("illegal activated target".into()));
        }
        let clause = self
            .activation_target_requirement(source, ability_index, requirement)
            .ok_or_else(|| AgentError("missing activation source".into()))?;
        chosen.push(target);
        let count = self
            .activation_clause_chosen(source, ability_index, requirement, &chosen)
            .len();
        if count >= clause.max
            || self
                .activated_ability_targets(player, source, ability_index, requirement, &chosen)
                .is_empty()
        {
            return self.next_activation_clause(
                player,
                source,
                ability_index,
                requirement + 1,
                chosen,
            );
        }
        self.pending_choice = Some(PendingChoice::ActivatedTarget {
            player,
            source,
            ability_index,
            chosen,
            requirement,
        });
        Ok(())
    }

    /// Move on to clause `requirement`, skipping optional clauses without a
    /// candidate, or to payment once every clause is chosen.
    fn next_activation_clause(
        &mut self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        mut requirement: usize,
        chosen: Vec<Target>,
    ) -> Result<(), AgentError> {
        while let Some(clause) =
            self.activation_target_requirement(source, ability_index, requirement)
        {
            if clause.min > 0
                || !self
                    .activated_ability_targets(player, source, ability_index, requirement, &chosen)
                    .is_empty()
            {
                self.pending_choice = Some(PendingChoice::ActivatedTarget {
                    player,
                    source,
                    ability_index,
                    chosen,
                    requirement,
                });
                return Ok(());
            }
            requirement += 1;
        }
        self.begin_payment(
            player,
            Operation::Activation {
                source,
                ability_index,
                targets: chosen,
                x: 0,
            },
        )
    }

    pub(crate) fn payment_candidates(
        &self,
        player: PlayerId,
        operation: &Operation,
        cost: &SacrificeCost,
    ) -> Vec<PermanentId> {
        self.sacrifice_candidates(player, &cost.predicate)
            .into_iter()
            .filter(|id| {
                !cost.another
                    || !matches!(operation, Operation::Activation { source: ActivationSource::Permanent(permanent), .. } if permanent == id)
            })
            .collect()
    }

    /// What `player` may choose to forage with for `payment`: a Food to
    /// sacrifice, or a graveyard card to exile while three can be. Once a
    /// card is exiled only cards are offered. A Food whose mana the rest of
    /// the cost needs is left out.
    pub(crate) fn forage_objects(&self, player: PlayerId, payment: &Payment) -> Vec<ObjectId> {
        let CostComponent::Forage { exiled, .. } = payment.owed[0] else {
            return Vec::new();
        };
        let mana = payment.mana().0;
        let mut objects = Vec::new();
        if exiled == 0 {
            objects.extend(
                self.sacrifice_candidates(player, &food())
                    .into_iter()
                    .filter(|id| self.can_pay_with_mana_abilities(player, mana, Some(*id)))
                    .filter_map(|id| self.permanent_object(id)),
            );
        }
        let Operation::Spell { object: spell, .. } = payment.operation else {
            unreachable!("forage pays for a spell")
        };
        let mut graveyard = self.graveyard_payers(player, spell);
        graveyard.sort_unstable_by_key(|object| (self.state[*object].definition_id, *object));
        if graveyard.len() >= usize::from(FORAGE_CARDS - exiled)
            && self.can_pay_with_mana_abilities(player, mana, None)
        {
            objects.extend(graveyard);
        }
        objects
    }

    /// The cards in `player`'s graveyard that may be exiled to pay for
    /// `spell` (delve, forage), in graveyard order. A spell cast from the
    /// graveyard waits there until it is paid for, but it is being cast and
    /// is no card its own cost can exile (CR 601.2a).
    pub(crate) fn graveyard_payers(&self, player: PlayerId, spell: ObjectId) -> Vec<ObjectId> {
        self.state
            .zones
            .zone_objects(ZoneType::Graveyard, player)
            .iter()
            .copied()
            .filter(|object| *object != spell)
            .collect()
    }

    /// The cards `player` may discard to pay a "discard a card" cost: the
    /// hand, less the spell being cast from it (`casting`).
    pub(crate) fn discard_candidates(
        &self,
        player: PlayerId,
        casting: Option<ObjectId>,
    ) -> Vec<ObjectId> {
        self.state
            .zone_cards(ZoneType::Hand, player)
            .map(|card| self.state.object_of(card))
            .filter(|object| Some(*object) != casting)
            .collect()
    }

    /// Whether `player` could forage for `spell` beside paying `cost` in mana.
    fn can_forage(&self, player: PlayerId, spell: ObjectId, cost: &ManaCost) -> bool {
        self.graveyard_payers(player, spell).len() >= usize::from(FORAGE_CARDS)
            && self.can_pay_with_mana_abilities(player, cost, None)
            || self
                .sacrifice_candidates(player, &food())
                .into_iter()
                .any(|id| self.can_pay_with_mana_abilities(player, cost, Some(id)))
    }

    /// The parts of `operation`'s total cost its payer chooses how to pay,
    /// in the order they are paid, ending with its mana. The total is fixed
    /// here, once announcement is over (CR 601.2f, 602.2b).
    pub(crate) fn cost_components(
        &self,
        player: PlayerId,
        operation: &Operation,
    ) -> Vec<CostComponent> {
        let mut owed = Vec::new();
        match operation {
            Operation::Spell {
                object,
                cast,
                targets,
                ..
            } => {
                if self.uses_exile_alternative(*object, *cast) {
                    owed.push(CostComponent::ExileFromHand);
                }
                let card = &self.state[*object];
                if let Some(cost) = card.sacrifice_cost() {
                    owed.push(CostComponent::Sacrifice(cost.clone(), Toward::Printed));
                }
                if cast.kicked && card.bargain() {
                    owed.push(CostComponent::Sacrifice(bargain_cost(), Toward::Bargain));
                }
                if let Some(or_mana) = card.forage_or() {
                    owed.push(CostComponent::Forage {
                        or_mana: or_mana.clone(),
                        exiled: 0,
                    });
                }
                if card.discard_cost() {
                    owed.push(CostComponent::Discard);
                }
                let cost = self
                    .effective_spell_cost(player, *object, *cast, Some(targets))
                    .unwrap_or_default();
                let help = self.spell_help(player, *object, *cast, &cost);
                owed.push(CostComponent::Mana { cost, help });
                if let (false, Some(life)) = (cast.kicked, self.state[*object].life_alternative()) {
                    owed.push(CostComponent::Life(life));
                }
            }
            Operation::Activation {
                source,
                ability_index,
                x,
                ..
            } => {
                let ability = self
                    .activation_ability(*source, *ability_index)
                    .expect("announced activation source");
                if let Some((power, toward)) = ability.tap_power().filter(|(power, _)| *power > 0) {
                    owed.push(CostComponent::TapPower { power, toward });
                }
                if let Some((cost, toward)) = ability.chosen_tap() {
                    owed.push(CostComponent::Tap(cost, toward));
                }
                if ability.discard_card {
                    owed.push(CostComponent::Discard);
                }
                if let Some(cost) = ability.sacrifice {
                    owed.push(CostComponent::Sacrifice(cost, Toward::Printed));
                }
                let cost = ability.mana_cost.with_x(*x);
                let help = Help {
                    waterbend: if ability.waterbend { cost.generic() } else { 0 },
                    ..Default::default()
                };
                owed.push(CostComponent::Mana { cost, help });
                if ability.life_cost > 0 {
                    owed.push(CostComponent::Life(ability.life_cost));
                }
                for (owes, how) in [
                    (ability.tap_source, How::Tapped),
                    (ability.sacrifice_source, How::Sacrificed),
                    (ability.exile_source, How::Exiled),
                    (ability.discard_source, How::Discarded),
                ] {
                    if owes {
                        owed.push(CostComponent::Source(how));
                    }
                }
            }
        }
        owed
    }

    /// Announcement is over: fix the operation's cost and start paying it.
    fn begin_payment(&mut self, player: PlayerId, operation: Operation) -> Result<(), AgentError> {
        if let Operation::Activation {
            source,
            ability_index,
            ..
        } = &operation
        {
            if self.activation_ability(*source, *ability_index).is_none() {
                return Err(AgentError("missing source".into()));
            }
        }
        let owed = self.cost_components(player, &operation);
        self.advance_payment(
            player,
            Payment {
                operation,
                owed,
                receipt: PaymentReceipt::default(),
            },
        )
    }

    /// Offer the next owed component. Once only mana is owed and no source
    /// or mana-ability choice remains, pay it and finish.
    fn advance_payment(&mut self, player: PlayerId, payment: Payment) -> Result<(), AgentError> {
        // A waterbend cost always offers its taps beside paying with mana.
        let chooses = !payment.paying_mana() || payment.waterbends();
        self.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment),
        });
        if chooses || self.has_mana_payment_choices(player) || !self.help_actions(player).is_empty()
        {
            return Ok(());
        }
        self.finish_mana_payment(player)
    }

    /// The payment in progress and the component being paid.
    pub(crate) fn paying(&self) -> Option<(PlayerId, &Payment, &CostComponent)> {
        match &self.pending_choice {
            Some(PendingChoice::Pay { player, payment }) => {
                Some((*player, payment, &payment.owed[0]))
            }
            _ => None,
        }
    }

    /// The receipt of the cost being paid: an announced payment's, or that
    /// of a payment suspended during resolution.
    pub(crate) fn pending_receipt(&self) -> Option<&PaymentReceipt> {
        match (&self.state.suspended_decision, &self.pending_choice) {
            (Some(suspended), _) => match &suspended.decision {
                Decision::Payment(payment) => Some(&payment.receipt),
                _ => None,
            },
            (None, Some(PendingChoice::Pay { payment, .. })) => Some(&payment.receipt),
            _ => None,
        }
    }

    fn pending_payment(&self, player: PlayerId) -> Result<Payment, AgentError> {
        match self.paying() {
            Some((payer, payment, _)) if payer == player => Ok(payment.clone()),
            Some(_) => Err(AgentError("wrong payer".into())),
            None => Err(AgentError("no cost payment is pending".into())),
        }
    }

    /// The untapped permanents `player` controls matching a tap cost.
    pub(crate) fn tap_candidates(
        &self,
        player: PlayerId,
        predicate: &CardPredicate,
    ) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| {
                self.state.permanents[*id]
                    .as_ref()
                    .is_some_and(|permanent| !permanent.tapped)
                    && self.permanent_matches_predicate(*id, predicate)
            })
            .collect()
    }

    /// The untapped permanents `player` controls that can pay a `Tap`
    /// component of `operation`. Station taps another creature than its
    /// source, whatever its power.
    pub(crate) fn tap_component_candidates(
        &self,
        player: PlayerId,
        operation: &Operation,
        cost: &TapCost,
        toward: Toward,
    ) -> Vec<PermanentId> {
        let mut candidates = self.tap_candidates(player, &cost.predicate);
        if let (
            Toward::Station,
            Operation::Activation {
                source: ActivationSource::Permanent(source),
                ..
            },
        ) = (toward, operation)
        {
            candidates.retain(|id| id != source);
        }
        candidates
    }

    /// The untapped permanents `player` controls that can pay `ability`'s
    /// tap cost.
    pub(crate) fn tap_cost_candidates(
        &self,
        player: PlayerId,
        ability: &ActivatedAbilityDefinition,
    ) -> Vec<PermanentId> {
        ability
            .tap_cost
            .as_ref()
            .map(|cost| self.tap_candidates(player, &cost.predicate))
            .unwrap_or_default()
    }

    /// Pay the component being paid with `object`: tap it, discard it, exile
    /// it, or commit to sacrificing it. Taps, discards and exiles happen now;
    /// a sacrifice is recorded and happens as the whole cost completes, so no
    /// departure trigger sees a half-paid cost.
    pub(crate) fn pay_with_object(
        &mut self,
        player: PlayerId,
        object: ObjectId,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        let permanent = self.state.permanent_of(object);
        match payment.owed[0].clone() {
            CostComponent::TapPower { power, toward } => {
                let Operation::Activation {
                    source: ActivationSource::Permanent(vehicle),
                    ..
                } = payment.operation
                else {
                    unreachable!("a permanent's ability taps creatures for power")
                };
                let creature = permanent
                    .filter(|id| self.crew_candidates(player, vehicle).contains(id))
                    .ok_or_else(|| AgentError("creature cannot crew".into()))?;
                let power = power - self.characteristics(creature).power;
                self.pending_choice = None;
                self.tap_permanent(creature, false);
                payment.receipt.contribute(object, How::Tapped, toward);
                if power > 0 {
                    payment.owed[0] = CostComponent::TapPower { power, toward };
                } else {
                    payment.owed.remove(0);
                }
            }
            CostComponent::Tap(cost, toward) => {
                let tapped = permanent
                    .filter(|id| {
                        self.tap_component_candidates(player, &payment.operation, &cost, toward)
                            .contains(id)
                    })
                    .ok_or_else(|| {
                        AgentError("tap cost must be an untapped permanent you control".into())
                    })?;
                if !self.tap_activation_payment_possible(player, &payment, Some(tapped)) {
                    return Err(AgentError(
                        "tap cost selection cannot complete the activation".into(),
                    ));
                }
                self.pending_choice = None;
                self.tap_permanent(tapped, false);
                payment.receipt.contribute(object, How::Tapped, toward);
                if cost.count > 1 {
                    payment.owed[0] = CostComponent::Tap(
                        TapCost {
                            count: cost.count - 1,
                            ..cost
                        },
                        toward,
                    );
                } else {
                    payment.owed.remove(0);
                }
            }
            CostComponent::Discard => {
                if !self
                    .discard_candidates(player, payment.operation.spell())
                    .contains(&object)
                {
                    return Err(AgentError("discard cost must be a card in hand".into()));
                }
                self.pending_choice = None;
                let discarded = self.move_object(object, ZoneType::Graveyard);
                payment
                    .receipt
                    .contribute(discarded, How::Discarded, Toward::Printed);
                payment.owed.remove(0);
            }
            CostComponent::ExileFromHand => {
                let Operation::Spell { object: spell, .. } = payment.operation else {
                    unreachable!("an exile alternative pays for a spell")
                };
                if !self
                    .state
                    .card_of(object)
                    .is_some_and(|card| self.exile_alternative_cards(player, spell).contains(&card))
                {
                    return Err(AgentError(
                        "not a card this alternative cost can exile".into(),
                    ));
                }
                self.pending_choice = None;
                let exiled = self.move_object(object, ZoneType::Exile);
                payment
                    .receipt
                    .contribute(exiled, How::Exiled, Toward::Alternative);
                payment.owed.remove(0);
            }
            CostComponent::Sacrifice(cost, toward) => {
                if !permanent.is_some_and(|id| {
                    self.payment_candidates(player, &payment.operation, &cost)
                        .contains(&id)
                }) {
                    return Err(AgentError("illegal sacrifice cost".into()));
                }
                payment.receipt.contribute(object, How::Sacrificed, toward);
                payment.owed.remove(0);
            }
            CostComponent::Forage { or_mana, exiled } => {
                if !self.forage_objects(player, &payment).contains(&object) {
                    return Err(AgentError(
                        "not a Food or graveyard card to forage with".into(),
                    ));
                }
                if permanent.is_some() {
                    payment
                        .receipt
                        .contribute(object, How::Sacrificed, Toward::Forage);
                    payment.owed.remove(0);
                } else {
                    self.pending_choice = None;
                    let card = self.move_object(object, ZoneType::Exile);
                    payment
                        .receipt
                        .contribute(card, How::Exiled, Toward::Forage);
                    if exiled + 1 < FORAGE_CARDS {
                        payment.owed[0] = CostComponent::Forage {
                            or_mana,
                            exiled: exiled + 1,
                        };
                    } else {
                        payment.owed.remove(0);
                    }
                }
            }
            CostComponent::Mana { cost, help } => {
                // Delve: a card exiled from the payer's graveyard pays {1}.
                if !(cost.generic() > 0
                    && help.delve.is_some_and(|spell| {
                        self.graveyard_payers(player, spell).contains(&object)
                    }))
                {
                    return Err(AgentError("no object pays this cost".into()));
                }
                self.pending_choice = None;
                let exiled = self.move_object(object, ZoneType::Exile);
                payment.help_paid(exiled, How::Exiled, Toward::Delve, Color::Generic);
            }
            CostComponent::ConvokeColor { .. } => {
                return Err(AgentError("choose the color this creature pays".into()));
            }
            CostComponent::Life(_) | CostComponent::Source(_) => {
                unreachable!("paid as the payment completes, after its mana")
            }
        }
        self.advance_payment(player, payment)
    }

    /// `PayCost` while a cost is being paid: take a sacrifice cost's mana
    /// branch, or pay the mana still owed.
    pub(crate) fn pay_cost_action(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        let extra = match payment.owed[0].clone() {
            CostComponent::Sacrifice(cost, _) => cost
                .or_mana
                .ok_or_else(|| AgentError("sacrifice required".into()))?,
            CostComponent::Forage { or_mana, exiled: 0 } => or_mana,
            CostComponent::Mana { .. } => return self.finish_mana_payment(player),
            _ => return Err(AgentError("this cost is not paid with mana".into())),
        };
        payment.owed.remove(0);
        payment.owe_mana(&extra);
        self.advance_payment(player, payment)
    }

    /// Pay the mana the pending payment still owes, then cast the spell or
    /// activate the ability.
    pub(crate) fn finish_mana_payment(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let payment = self.pending_payment(player)?;
        if !payment.paying_mana() {
            return Err(AgentError("no mana payment pending".into()));
        }
        let cost = payment.mana().0.clone();
        self.produce_mana(player, &cost)?;
        // Commit the complete announced cost once, before sacrifice triggers
        // can observe a departing permanent or a spell reaches the stack.
        self.pending_choice = None;
        self.begin_mana_spend(
            player,
            cost,
            SpendCompletion::Announced(payment),
            EffectFrame::new(player, Vec::new()),
        )
    }

    /// The whole announced cost is paid: pay what `owed` holds after the
    /// mana (life and the source's own costs), sacrifice what was chosen or
    /// is owed, then cast the spell or put the ability on the stack with the
    /// receipt. The ability is captured before its source taps or leaves.
    pub(crate) fn complete_announced_payment(
        &mut self,
        player: PlayerId,
        mut payment: Payment,
    ) -> Result<(), AgentError> {
        if payment.waterbends() {
            self.record_bending(player, crate::flow::turn::Bending::Water);
        }
        let life = payment.life();
        self.pay_life_cost(player, life, &mut payment.receipt);
        // The source's own discard or exile; its sacrifice is on the receipt.
        let mut leaves = Vec::new();
        let activated = match &payment.operation {
            Operation::Spell { .. } => None,
            Operation::Activation {
                source,
                ability_index,
                targets,
                x,
            } => {
                let ability = match *source {
                    ActivationSource::Permanent(permanent) => self.prepare_activated_ability(
                        player,
                        permanent,
                        *ability_index,
                        targets.clone(),
                        *x,
                        payment.owes(&CostComponent::Source(How::Tapped)),
                    )?,
                    ActivationSource::Card(card) => self.prepare_card_activated_ability(
                        player,
                        card,
                        *ability_index,
                        targets.clone(),
                    )?,
                };
                let source_object = ability.source_object;
                for component in &payment.owed {
                    let CostComponent::Source(how) = *component else {
                        continue;
                    };
                    payment
                        .receipt
                        .contribute(source_object, how, Toward::Printed);
                    let to = match how {
                        How::Tapped | How::Sacrificed => continue,
                        How::Discarded => ZoneType::Graveyard,
                        How::Exiled => ZoneType::Exile,
                    };
                    leaves.push(self.zone_move_proposal(source_object, to, Default::default()));
                }
                Some((ability, *source))
            }
        };
        let events: Vec<_> = payment
            .receipt
            .sacrificed()
            .filter(|object| self.state.permanent_of(*object).is_some())
            .map(|target| ProposedEvent::Sacrifice { target })
            .chain(leaves)
            .collect();
        let completion = match activated {
            None => FrameCompletion::Paid { player, payment },
            Some((mut ability, source)) => {
                ability.paid = payment.receipt;
                FrameCompletion::Activated {
                    ability,
                    source_after_cost: match source {
                        ActivationSource::Card(card) => Some(self.state.object_of(card)),
                        ActivationSource::Permanent(_) => None,
                    },
                }
            }
        };
        let mut frame = EffectFrame::new(player, Vec::new());
        frame.batch = Some(self.event_batch(events));
        frame.completion = completion;
        self.run_frame(frame);
        Ok(())
    }

    /// Other untapped creatures `player` controls that can help crew
    /// `vehicle`. Summoning sickness doesn't matter (crew is not a {T}
    /// cost); creatures without positive power add nothing and are omitted.
    pub(crate) fn crew_candidates(
        &self,
        player: PlayerId,
        vehicle: PermanentId,
    ) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|&id| {
                id != vehicle
                    && self.state.permanents[id]
                        .as_ref()
                        .is_some_and(|permanent| !permanent.tapped)
                    && {
                        let characteristics = self.characteristics(id);
                        characteristics.is_creature() && characteristics.power > 0
                    }
            })
            .collect()
    }

    pub(crate) fn crew_power_available(&self, player: PlayerId, vehicle: PermanentId) -> i32 {
        self.crew_candidates(player, vehicle)
            .into_iter()
            .map(|id| self.characteristics(id).power)
            .sum()
    }

    /// Untapped artifacts/creatures `player` controls, as the layers see
    /// them — the permanents that can each pay {1} of a waterbend cost.
    pub(crate) fn waterbend_candidates(&self, player: PlayerId) -> Vec<PermanentId> {
        self.battlefield_permanents(player)
            .into_iter()
            .filter(|id| {
                self.state.permanents[*id]
                    .as_ref()
                    .is_some_and(|permanent| !permanent.tapped)
                    && {
                        let view = self.characteristics(*id);
                        view.types.is_artifact() || view.is_creature()
                    }
            })
            .collect()
    }

    /// The card whose spell or ability `payment` pays for.
    pub(crate) fn payment_source_object(&self, operation: &Operation) -> Option<ObjectId> {
        match operation {
            Operation::Spell { object, .. } => Some(*object),
            Operation::Activation {
                source: ActivationSource::Card(card),
                ..
            } => Some(self.state.object_of(*card)),
            Operation::Activation {
                source: ActivationSource::Permanent(permanent),
                ..
            } => self.state.permanents[*permanent]
                .as_ref()
                .map(|permanent| permanent.object),
        }
    }

    /// The permanent whose ability `payment` activates when that ability
    /// also costs {T}: it can't tap for both.
    pub(crate) fn payment_tapped_source(&self, operation: &Operation) -> Option<PermanentId> {
        let Operation::Activation {
            source: ActivationSource::Permanent(source),
            ability_index,
            ..
        } = operation
        else {
            return None;
        };
        self.characteristics(*source)
            .activated_abilities
            .get(*ability_index)
            .is_some_and(|ability| ability.tap_source)
            .then_some(*source)
    }

    /// Waterbend candidates for `payment`, less a source it taps anyway.
    pub(crate) fn waterbend_payment_candidates(
        &self,
        player: PlayerId,
        operation: &Operation,
    ) -> Vec<PermanentId> {
        let mut candidates = self.waterbend_candidates(player);
        if let Some(source) = self.payment_tapped_source(operation) {
            candidates.retain(|candidate| *candidate != source);
        }
        candidates
    }

    /// Can `source`'s waterbend ability be paid, its own {T} included?
    pub(crate) fn can_pay_waterbend_activation(
        &self,
        player: PlayerId,
        source: PermanentId,
        ability: &ActivatedAbilityDefinition,
    ) -> bool {
        // An X cost is payable when its least X is.
        let cost = ability.mana_cost.with_x(ability.x_min);
        if !ability.tap_source {
            return self.can_pay_with_waterbend(player, &cost);
        }
        !self
            .waterbend_taps_keeping_payable(player, &cost, cost.generic(), vec![source], None)
            .is_empty()
    }

    /// Can `player` cover `cost` with untapped waterbend candidates
    /// ({1} each, generic only) plus available mana?
    pub(crate) fn can_pay_with_waterbend(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.waterbend_payment_possible(player, cost, cost.generic())
    }

    /// Tap `tapped`, one of `candidates`, to pay {1} of the waterbend part
    /// of `cost` (CR 701.67): the one waterbend tap, for a cast, an
    /// activation and a payment made during resolution. It pays without
    /// activating a mana ability; `receipt` names the permanent tapped.
    pub(crate) fn waterbend_tap(
        &mut self,
        tapped: PermanentId,
        candidates: &[PermanentId],
        cost: &mut ManaCost,
        help: Help,
        receipt: &mut PaymentReceipt,
    ) -> Result<(), AgentError> {
        if help.waterbend_left(cost, receipt) == 0 {
            return Err(AgentError("waterbend cost is already paid".to_string()));
        }
        let object = self
            .permanent_object(tapped)
            .filter(|_| candidates.contains(&tapped))
            .ok_or_else(|| AgentError("permanent cannot be tapped for waterbend".to_string()))?;
        self.tap_permanent(tapped, false);
        pay_pip_with(
            cost,
            receipt,
            object,
            How::Tapped,
            Toward::Waterbend,
            Color::Generic,
        );
        Ok(())
    }

    /// The waterbend taps among `candidates` that leave the rest of `cost`
    /// payable, `waterbend` of its generic mana being payable by taps.
    /// `also_tapped` is the source a cost taps as well.
    pub(crate) fn waterbend_tap_offers(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
        candidates: Vec<PermanentId>,
        also_tapped: Option<PermanentId>,
    ) -> Vec<Action> {
        if waterbend == 0 {
            return Vec::new();
        }
        self.waterbend_taps_keeping_payable(
            player,
            &cost.reduced_generic(1),
            waterbend - 1,
            candidates,
            also_tapped,
        )
        .into_iter()
        .map(|permanent| Action::WaterbendTap { player, permanent })
        .collect()
    }

    /// Pay `life` as a cost (CR 119.4) and record it on `receipt`. The
    /// caller has offered it only to a player who can pay it.
    pub(crate) fn pay_life_cost(
        &mut self,
        player: PlayerId,
        life: u32,
        receipt: &mut PaymentReceipt,
    ) {
        self.change_life(player, -(life as i32));
        receipt.pay_life(life);
    }

    /// Tap `tapped` to pay {1} of the pending waterbend cost.
    pub(crate) fn waterbend_tap_action(
        &mut self,
        player: PlayerId,
        tapped: PermanentId,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        if payment.paying_mana() && payment.help().taps_any_generic() {
            return self.help_tap_action(player, tapped, payment);
        }
        if !payment.paying_mana() || !payment.waterbends() {
            return Err(AgentError("no waterbend payment is pending".to_string()));
        }
        let candidates = self.waterbend_payment_candidates(player, &payment.operation);
        let (cost, help, receipt) = payment.mana_step();
        self.waterbend_tap(tapped, &candidates, cost, help, receipt)?;
        let (owed, waterbend) = payment.mana();
        // A remaining colored cost can still need a mana ability such as
        // sacrificing Treasure. Leave the committed payment open for it.
        let finished = waterbend == 0
            && (owed.mana_value == 0
                || (!self.offers_extra_payment_taps(player)
                    && self.can_pay_mana_cost(player, owed)));
        self.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment),
        });
        if finished {
            return self.finish_mana_payment(player);
        }
        Ok(())
    }

    /// The keyword under which tapping `tapped` pays part of a cost with
    /// `help`: convoke for a creature, improvise for an artifact.
    pub(crate) fn help_tap_toward(&self, help: Help, tapped: PermanentId) -> Option<Toward> {
        let view = self.characteristics(tapped);
        help.tap_toward(view.is_creature(), view.types.is_artifact())
    }

    /// The pips `tapped` may pay of the pending cost by convoke or improvise
    /// such that the rest stays payable: each color it has that the cost
    /// still owes, or {1} (`Color::Generic`) when it has none of them. A
    /// colored pip is never worse for the payer than {1}: whatever mana
    /// would have paid that pip can pay the {1} instead.
    pub(crate) fn help_tap_options(
        &self,
        player: PlayerId,
        payment: &Payment,
        tapped: PermanentId,
    ) -> Vec<Color> {
        let (cost, help) = (payment.mana().0, payment.help());
        let view = self.characteristics(tapped);
        let toward = self.help_tap_toward(help, tapped);
        if toward.is_none()
            || !self
                .waterbend_payment_candidates(player, &payment.operation)
                .contains(&tapped)
        {
            return Vec::new();
        }
        let convokes = toward == Some(Toward::Convoke);
        let mut options: Vec<Color> = view
            .colors
            .iter()
            .copied()
            .filter(|color| convokes && cost.cost[*color as usize] > 0)
            .collect();
        if options.is_empty() && cost.generic() > 0 {
            options.push(Color::Generic);
        }
        options.retain(|color| {
            self.payable_with_help(player, &cost.less_pip(*color), help, Some(tapped))
        });
        options
    }

    /// The taps and exiles that may pay part of the pending mana cost by
    /// convoke, improvise or delve, each leaving the rest payable.
    pub(crate) fn help_actions(&self, player: PlayerId) -> Vec<Action> {
        let Some((payer, payment, CostComponent::Mana { cost, help })) = self.paying() else {
            return Vec::new();
        };
        if payer != player || !help.keyworded() {
            return Vec::new();
        }
        let mut actions = Vec::new();
        if help.taps_any_generic() {
            actions.extend(
                self.waterbend_payment_candidates(player, &payment.operation)
                    .into_iter()
                    .filter(|permanent| {
                        !self
                            .help_tap_options(player, payment, *permanent)
                            .is_empty()
                    })
                    .map(|permanent| Action::WaterbendTap { player, permanent }),
            );
        }
        if let Some(spell) = help.delve.filter(|_| cost.generic() > 0) {
            actions.extend(
                self.graveyard_payers(player, spell)
                    .into_iter()
                    .map(|object| Action::SelectCard { player, object }),
            );
        }
        actions
    }

    /// Tap `tapped` to pay one pip of the pending cost by convoke or
    /// improvise. A creature that could pay more than one owed color is
    /// tapped now and its color chosen next.
    fn help_tap_action(
        &mut self,
        player: PlayerId,
        tapped: PermanentId,
        mut payment: Payment,
    ) -> Result<(), AgentError> {
        let colors = self.help_tap_options(player, &payment, tapped);
        let object = self
            .permanent_object(tapped)
            .filter(|_| !colors.is_empty())
            .ok_or_else(|| AgentError("permanent cannot help pay this cost".to_string()))?;
        let toward = self
            .help_tap_toward(payment.help(), tapped)
            .expect("a tap with options pays by a keyword");
        self.pending_choice = None;
        // Convoke and improvise pay a cost without activating a mana ability.
        self.tap_permanent(tapped, false);
        match colors[..] {
            [color] => payment.help_paid(object, How::Tapped, toward, color),
            _ => payment.owed.insert(
                0,
                CostComponent::ConvokeColor {
                    creature: object,
                    colors,
                },
            ),
        }
        self.advance_payment(player, payment)
    }

    /// Choose which owed color the convoking creature just tapped pays.
    pub(crate) fn convoke_color_action(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let mut payment = self.pending_payment(player)?;
        let CostComponent::ConvokeColor { creature, colors } = payment.owed[0].clone() else {
            return Err(AgentError("no convoke color is being chosen".into()));
        };
        let color = *colors
            .get(mode)
            .ok_or_else(|| AgentError("not a color this creature can pay".into()))?;
        payment.owed.remove(0);
        payment.help_paid(creature, How::Tapped, Toward::Convoke, color);
        self.advance_payment(player, payment)
    }

    /// Capture the ability before its source can leave, and tap the source
    /// when its payment owes that (`tap`). Frame completion puts it on the
    /// stack.
    fn prepare_activated_ability(
        &mut self,
        player: PlayerId,
        permanent_id: PermanentId,
        ability_index: usize,
        targets: Vec<Target>,
        x: u8,
        tap: bool,
    ) -> Result<ActivatedAbilityOnStack, AgentError> {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        let source_object = self
            .permanent_object(permanent_id)
            .ok_or_else(|| AgentError("missing activation source identity".into()))?;
        let attached = permanent.attached_to;
        let source_transforms = permanent.transforms;
        let source_definition_id = self.state[source_object].definition_id;
        let inherited_from = self.inherited_definition_id(permanent_id);
        let Some(ability) = self
            .characteristics(permanent_id)
            .activated_abilities
            .get(ability_index)
            .cloned()
        else {
            return Err(AgentError("invalid ability index".to_string()));
        };

        let program = !self.activation_catalog_program(
            source_object,
            source_definition_id,
            inherited_from,
            ability_index,
            ability.granted_by,
            &ability.effects,
        );
        if tap {
            if !self.permanent_can_tap(permanent_id) {
                return Err(AgentError("source cannot pay tap cost".into()));
            }
            self.tap_permanent(permanent_id, false);
        }
        self.record_activation(permanent_id, ability_index, &ability);

        let id = self.state.id_gen_mut().next_id();
        Ok(ActivatedAbilityOnStack {
            target_req_indices: ability.target_clause_indices(targets.len()),
            effects: ability.effects,
            program,
            source_transforms,
            inherited_from,
            id,
            controller: player,
            source_definition_id,
            source_object,
            ability_index,
            targets,
            attached,
            x,
            granted_by: ability.granted_by,
            paid: PaymentReceipt::default(),
        })
    }

    /// Capture an off-battlefield ability. Frame completion binds the
    /// post-cost source identity (CR 602.2b, 113.7a) before putting the
    /// ability on the stack.
    fn prepare_card_activated_ability(
        &mut self,
        player: PlayerId,
        card: CardId,
        ability_index: usize,
        targets: Vec<Target>,
    ) -> Result<ActivatedAbilityOnStack, AgentError> {
        let ability = self
            .card_ability_in_zone(player, card, ability_index)
            .ok_or_else(|| AgentError("activation source left its zone".into()))?;
        // The completion replaces this identity after a discard cost commits.
        let source_object = self
            .state
            .live(self.state.object_of(card))
            .ok_or_else(|| AgentError("missing activation source identity".into()))?;
        let id = self.state.id_gen_mut().next_id();
        Ok(ActivatedAbilityOnStack {
            target_req_indices: ability.target_clause_indices(targets.len()),
            effects: ability.effects,
            program: false,
            source_transforms: 0,
            inherited_from: None,
            id,
            controller: player,
            source_definition_id: self.state[self.state.object_of(card)].definition_id,
            source_object,
            ability_index,
            targets,
            attached: None,
            x: 0,
            granted_by: None,
            paid: PaymentReceipt::default(),
        })
    }

    /// Legal targets for one targeting requirement of `card`, cast by
    /// `player`. The card being cast is excluded from stack-spell targets
    /// (it is not on the stack yet while targets are chosen).
    pub(crate) fn legal_targets_for_requirement(
        &self,
        player: PlayerId,
        object: ObjectId,
        requirement: &TargetRequirement,
    ) -> Vec<Target> {
        self.legal_targets_for_spec(&requirement.spec, player)
            .into_iter()
            .filter(|target| *target != Target::StackSpell(object))
            .collect()
    }
}

/// The cards forage exiles from a graveyard (CR 701.61a).
const FORAGE_CARDS: u8 = 3;

fn food() -> CardPredicate {
    CardPredicate {
        subtype: Some("Food".into()),
        ..Default::default()
    }
}

/// What bargain sacrifices: an artifact, enchantment or token (CR 702.166a).
fn bargain_cost() -> SacrificeCost {
    SacrificeCost {
        predicate: CardPredicate {
            any_of: vec![
                CardPredicate {
                    card_types_any: vec![CardType::Artifact, CardType::Enchantment],
                    ..Default::default()
                },
                CardPredicate {
                    token: Some(true),
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        another: false,
        or_mana: None,
    }
}
