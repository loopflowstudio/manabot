// play.rs
// Spell casting, land plays, and activated abilities.
//
// Casting is a staged pipeline driven by `PendingChoice`:
//   X (if any) -> kicker choice (optional) -> spell mode (if any) -> targets
//   per clause -> payment.
// Activations announce X (if any) and choose targets, then pay. Both
// pipelines end in `PendingChoice::Pay`; every cost, crew and waterbend
// included, is a component of that payment (`flow/payment.rs`).
// Activations name an `ActivationSource`: a permanent, or a card whose
// ability functions from its owner's hand or graveyard (cycling, "return
// this card from your graveyard").

use crate::state::card::SpellProfile;
use std::borrow::Cow;

use crate::{
    agent::action::AgentError,
    flow::{
        cast_permission::PlayRoute,
        decision::{EffectFrame, FrameCompletion},
        event::GameEvent,
        game::{ActivationSource, CastChoices, Game, PendingChoice},
        payment::{bargain_cost, Help, Operation},
        turn::{PhaseKind, StepKind},
    },
    state::{
        ability::{EffectValue, TargetRequirement},
        card::{ActivatedAbilityDefinition, ActivationZone, AdditionalCost, StaticAbility},
        game_object::{CardId, ObjectId, PermanentId, PlayerId, Target},
        mana::{ManaCost, SpellSnapshot, SpendingContext, SpendingObject},
        payment::PaymentReceipt,
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

impl CastChoices {
    pub(crate) fn spell<'a>(
        self,
        object: &'a crate::state::game_object::Object,
    ) -> Cow<'a, crate::state::game_object::Object> {
        if !self.profile.is_front() {
            let mut spell = object.clone();
            spell.show_spell(self.profile);
            Cow::Owned(spell)
        } else {
            Cow::Borrowed(object)
        }
    }

    fn matches_spell(
        self,
        game: &Game,
        object: ObjectId,
        predicate: &crate::state::predicate::CardPredicate,
    ) -> bool {
        if !self.profile.is_front() {
            let spell = self.spell(&game.state[object]);
            predicate.matches_object_with_power(&spell, spell.power.unwrap_or(0))
        } else {
            game.object_matches_predicate(object, predicate)
        }
    }

    pub(crate) fn from_route(route: PlayRoute, profile: SpellProfile) -> Self {
        Self {
            profile,
            route: (route != PlayRoute::Hand).then_some(route),
            flashback: matches!(
                route,
                PlayRoute::PrintedFlashback | PlayRoute::GrantedFlashback
            ),
            ..Self::default()
        }
    }
    pub(super) fn grant(
        self,
        game: &Game,
        object: ObjectId,
    ) -> Option<&crate::flow::cast_permission::PlayGrant> {
        match self.route {
            Some(PlayRoute::Grant(key)) => game.grant_by_key(object, key),
            _ => None,
        }
    }
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

    /// Iroh grants a Lesson {1}, and other spells their printed mana cost.
    fn granted_flashback_cost(&self, object: ObjectId) -> Option<ManaCost> {
        let card = &self.state[object];
        if card.has_subtype("Lesson") {
            Some(ManaCost::parse("{1}"))
        } else {
            card.mana_cost.clone()
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

    /// Collect the permissions before timing or payment eliminates individual routes.
    pub(super) fn play_routes(
        &self,
        player: PlayerId,
        object: ObjectId,
    ) -> Cow<'static, [PlayRoute]> {
        let land = self.state[object].types.is_land();
        match self.state.zones.zone_of(object) {
            Some(ZoneType::Hand) if self.state[object].owner == player => {
                Cow::Borrowed(&[PlayRoute::Hand])
            }
            _ if !land && self.casts_only_from_hand(player) => Cow::Borrowed(&[]),
            Some(zone @ (ZoneType::Exile | ZoneType::Graveyard)) => {
                let mut routes: Vec<_> = self
                    .live_grants(player, object)
                    .map(|grant| PlayRoute::Grant(grant.key))
                    .collect();
                if zone != ZoneType::Graveyard || self.state[object].owner != player {
                    return routes.into();
                }
                if self.state[object].types.is_spell() {
                    let printed = self.state[object].flashback();
                    if printed.is_some() {
                        routes.push(PlayRoute::PrintedFlashback);
                    }
                    if self.active_player() == player
                        && self.active_permanent_has(player, |ability| {
                            *ability == StaticAbility::GraveyardSpellsHaveFlashback
                        })
                        && self
                            .granted_flashback_cost(object)
                            .as_ref()
                            .is_some_and(|cost| Some(cost) != printed)
                    {
                        routes.push(PlayRoute::GrantedFlashback);
                    }
                }
                if self.active_player() != player || !self.state[object].types.is_permanent() {
                    return routes.into();
                }
                for (id, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
                    if !view
                        .static_abilities
                        .contains(&StaticAbility::PlayGraveyardByPermanentType)
                    {
                        continue;
                    }
                    let grantor = self.state.permanents[id].as_ref().unwrap();
                    let source = grantor.object;
                    for &kind in &self.state[object].types.types {
                        use crate::state::card::CardType::*;
                        if matches!(kind, Instant | Sorcery | Kindred) || (land && kind != Land) {
                            continue;
                        }
                        let route = PlayRoute::GraveyardType {
                            source,
                            kind,
                            permanent_id: grantor.id,
                        };
                        if !self
                            .state
                            .cast_permissions
                            .spent_types
                            .contains(&(player, route))
                        {
                            routes.push(route);
                        }
                    }
                }
                routes.into()
            }
            Some(ZoneType::Library)
                if !land && self.state.zones.top(ZoneType::Library, player) == Some(object) =>
            {
                Cow::Borrowed(&[PlayRoute::LibraryTop])
            }
            _ => Cow::Borrowed(&[]),
        }
    }

    /// A library permission tests the characteristics of the proposed spell.
    pub(super) fn permits_spell(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> bool {
        if self.state[object]
            .spell_characteristics(cast.profile)
            .is_none()
        {
            return false;
        }
        match cast.route.unwrap_or(PlayRoute::Hand) {
            PlayRoute::LibraryTop => self.active_permanent_has(player, |ability| match ability {
                StaticAbility::CastFromLibraryTop(predicate) => {
                    cast.matches_spell(self, object, predicate)
                }
                _ => false,
            }),
            PlayRoute::Hand => true,
            PlayRoute::Grant(key) if !cast.profile.is_front() => self
                .grant_by_key(object, key)
                .is_some_and(|grant| !grant.on_adventure && !grant.foretold),
            _ => cast.profile.is_front(),
        }
    }

    pub(super) fn selected_play_route(
        &self,
        player: PlayerId,
        object: ObjectId,
        selected: Option<PlayRoute>,
    ) -> Result<PlayRoute, AgentError> {
        let routes = self.play_routes(player, object);
        match (selected, routes.as_ref()) {
            (Some(route), _) if routes.contains(&route) => Ok(route),
            (None, [only]) => Ok(*only),
            _ => Err(AgentError("missing or ambiguous play permission".into())),
        }
    }

    pub(super) fn spend_play_route(&mut self, player: PlayerId, route: Option<PlayRoute>) {
        if let Some(route @ PlayRoute::GraveyardType { .. }) = route {
            self.state
                .cast_permissions
                .spent_types
                .push((player, route));
        }
    }

    /// The selected mana or alternative cost, before announcing X or applying
    /// additional costs and reductions. Announcement and payment share this
    /// choice so a fixed alternative cost cannot grant a free value of X.
    pub(crate) fn casting_mana_cost(
        &self,
        object: ObjectId,
        cast: CastChoices,
    ) -> Option<ManaCost> {
        let spell = cast.spell(&self.state[object]);
        if cast.free || self.uses_exile_alternative(object, cast) {
            Some(ManaCost::default())
        } else if let (true, Some((cost, _))) = (cast.kicked, spell.alternative_mana()) {
            Some(cost.clone())
        } else if cast.resolving || self.state.zones.zone_of(object) == Some(ZoneType::Hand) {
            spell.mana_cost.clone()
        } else {
            match cast.route? {
                PlayRoute::PrintedFlashback => spell.flashback().cloned(),
                PlayRoute::GrantedFlashback => self.granted_flashback_cost(object),
                PlayRoute::Grant(_) => self.grant_cost_for_spell(cast.grant(self, object)?, &spell),
                _ => spell.mana_cost.clone(),
            }
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
        let card_ref = cast.spell(&self.state[object]);
        let mut cost = self.casting_mana_cost(object, cast)?.with_x(cast.x);
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
        let mut any_mana = false;
        for (_, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            for ability in &view.static_abilities {
                if let StaticAbility::SpendAnyManaToCast(predicate) = ability {
                    any_mana |= cast.matches_spell(self, object, predicate);
                    continue;
                }
                if let StaticAbility::SpellsFromOutsideHandCostLess(amount) = ability {
                    if self.state.zones.zone_of(object) != Some(ZoneType::Hand) {
                        cost = cost.reduced_generic(*amount);
                    }
                    continue;
                }
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
                if cast.matches_spell(self, object, &reduction.predicate)
                    && self.check_static_condition(&reduction.condition, player)
                {
                    cost = cost.reduced_generic(reduction.generic);
                }
            }
        }
        Some(if any_mana {
            cost.pay_with_any_mana()
        } else {
            cost
        })
    }

    pub(crate) fn play_land(
        &mut self,
        player: PlayerId,
        card: CardId,
        selected: Option<PlayRoute>,
    ) -> Result<(), AgentError> {
        let card_ref = &self.state[self.state.object_of(card)];
        if !card_ref.types.is_land() {
            return Err(AgentError("only land cards can be played".to_string()));
        }
        if card_ref.owner != player {
            return Err(AgentError("card does not belong to player".to_string()));
        }
        let route = self.selected_play_route(player, self.state.object_of(card), selected)?;
        if !self.can_play_land(player) {
            return Err(AgentError("cannot play land now".to_string()));
        }

        // CR 305.2 — Track one normal land play per turn.
        self.spend_play_route(player, Some(route));
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
        object: ObjectId,
        selected: Option<PlayRoute>,
        profile: SpellProfile,
    ) -> Result<(), AgentError> {
        if self.pending_choice.is_some() {
            return Err(AgentError("a choice is already pending".to_string()));
        }

        let spell = self
            .state
            .object(object)
            .map_err(|_| AgentError("spell must name a current object".into()))?;
        if spell.types.is_land() {
            return Err(AgentError("land cards cannot be cast".into()));
        }
        let route = self.selected_play_route(player, object, selected)?;
        let cast = CastChoices::from_route(route, profile);
        if !self.permits_spell(player, object, cast) {
            return Err(AgentError("permission does not allow this spell".into()));
        }
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        if self.has_instant_timing(player, object, cast) {
            if !self.can_cast_instants(player) {
                return Err(AgentError("cannot cast instant now".to_string()));
            }
        } else if !self.can_cast_sorceries(player) {
            return Err(AgentError(
                "cannot cast sorcery-speed spell now".to_string(),
            ));
        }

        self.begin_cast(player, object, cast)
    }

    /// CR 601.2b — announce X (or a replicate count), then any optional additional cost, before
    /// modes and targets. Replacing a printed X cost with a free cast or a
    /// fixed alternative fixes X to zero (CR 107.3b).
    pub(super) fn begin_cast(
        &mut self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Result<(), AgentError> {
        if self.cast_needs_x_announcement(object, cast) {
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
    pub(crate) fn cast_needs_x_announcement(&self, object: ObjectId, cast: CastChoices) -> bool {
        let card = cast.spell(&self.state[object]);
        let printed_x = card.mana_cost.as_ref().is_some_and(|cost| cost.x > 0);
        let selected_x = self
            .casting_mana_cost(object, cast)
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
            && cast
                .spell(&self.state[object])
                .life_alternative()
                .is_some_and(|life| self.payable_life(player) < life as i32)
        {
            return false;
        }
        let card = cast.spell(&self.state[object]);
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
        if help != Help::default() {
            let pool = self.pooled_mana(player);
            self.payable_with_help(player, &pool, &cost, help, help.waterbend, None)
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
        for requirement in cast.spell(&self.state[object]).target_requirements() {
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
        let card = cast.spell(&self.state[object]);
        let alternative = self.alternative_cost_available(player, object, cast);
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
    pub(crate) fn alternative_cost_available(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> bool {
        if cast.free
            || cast.flashback
            || cast
                .grant(self, object)
                .is_some_and(|g| g.cost.is_some() || g.foretold)
        {
            return false;
        }
        let card = cast.spell(&self.state[object]);
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
    pub(crate) fn cast_waterbend_generic(&self, object: ObjectId, cast: CastChoices) -> u8 {
        let card_ref = cast.spell(&self.state[object]);
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
            && cast
                .grant(self, object)
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
                .cast_waterbend_generic(object, cast)
                .min(cost.generic()),
            ..Default::default()
        };
        let mut grant = |cost: &AdditionalCost| match cost {
            AdditionalCost::Convoke => help.convoke = true,
            AdditionalCost::Improvise => help.improvise = true,
            AdditionalCost::Delve => help.delve = Some(object),
            _ => {}
        };
        cast.spell(&self.state[object])
            .additional_costs
            .iter()
            .for_each(&mut grant);
        for (_, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            for ability in &view.static_abilities {
                if let StaticAbility::SpellsHave { predicate, cost } = ability {
                    if cast.matches_spell(self, object, predicate) {
                        grant(cost);
                    }
                }
            }
        }
        help
    }

    pub(crate) fn legal_spell_modes(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Vec<usize> {
        let spell = cast.spell(&self.state[object]);
        spell.spell_modes().map_or_else(Vec::new, |modes| {
            (0..modes.len())
                .filter(|mode| {
                    spell
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
        if cast.spell(&self.state[object]).spell_modes().is_some() {
            if self.legal_spell_modes(player, object, cast).is_empty() {
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
        if chooser != player || !self.legal_spell_modes(player, object, cast).contains(&mode) {
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
        let requirements = cast
            .spell(&self.state[object])
            .target_requirements_for_mode(mode);
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
        if cast
            .spell(&self.state[object])
            .cost_reduction_if_targets()
            .is_none()
        {
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

        let requirements = cast
            .spell(&self.state[object])
            .target_requirements_for_mode(mode);
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

        let requirements = cast
            .spell(&self.state[object])
            .target_requirements_for_mode(mode);
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
    pub(super) fn uses_exile_alternative(&self, object: ObjectId, cast: CastChoices) -> bool {
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
        if cast.spell(&self.state[object]).types.is_creature() {
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
            self.can_pay_with_waterbend(player, &ability.mana_cost.with_x(x), None)
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
        if !ability.exhaust && !ability.power_up && !ability.once_each_turn {
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

    /// Capture the ability before its source can leave, and tap the source
    /// when its payment owes that (`tap`). Frame completion puts it on the
    /// stack.
    pub(super) fn prepare_activated_ability(
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
    pub(super) fn prepare_card_activated_ability(
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
