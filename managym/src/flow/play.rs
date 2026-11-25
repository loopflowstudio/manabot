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

use std::{borrow::Cow, sync::Arc};

use crate::{
    agent::action::{Action, AgentError},
    flow::{
        cast_permission::PlayRoute,
        decision::{Decision, EffectFrame, FrameCompletion, ResolutionCastOffer},
        event::GameEvent,
        game::{
            ActivatedTargeting, ActivationSource, CastChoices, DivisionParent, Game, PendingChoice,
            SpellTargeting,
        },
        payment::{bargain_cost, Help, Operation, Payment},
        proposed_event::{EntryFace, EventBinding, ZoneEntry},
        turn::{PhaseKind, StepKind},
    },
    state::{
        ability::{EffectValue, TargetRequirement},
        card::{
            ActivatedAbilityDefinition, ActivationZone, AdditionalCost, PrintedCharacteristics,
            SpellProfile, StaticAbility,
        },
        effect::Modification,
        game_object::{AbilitySource, CardId, Object, ObjectId, PermanentId, PlayerId, Target},
        mana::{ManaCost, SpellSnapshot, SpendingContext, SpendingObject},
        payment::PaymentReceipt,
        predicate::CardPredicate,
        stack_object::ActivatedAbilityOnStack,
        subtype::CreatureTypes,
        zone::ZoneType,
    },
};

/// X values from `least` through the largest one `can_pay` accepts. Payability
/// is monotone in X, so the scan stops at the first unpayable value.
fn payable_x_values(least: u8, can_pay: impl Fn(u8) -> bool) -> Vec<u8> {
    (least..=u8::MAX).take_while(|&x| can_pay(x)).collect()
}

impl CastChoices {
    pub(crate) fn spell(self, object: &Object) -> &Arc<PrintedCharacteristics> {
        crate::cards::content_pack::default_content_pack()
            .spell_definition(object, self.profile)
            .expect("offered spell profile")
    }

    pub(super) fn matches_spell(
        self,
        game: &Game,
        object: ObjectId,
        predicate: &CardPredicate,
    ) -> bool {
        if self.profile.is_front() {
            return game.object_matches_predicate(object, predicate);
        }
        let card = &game.state[object];
        predicate.matches_printed_spell(self.spell(card), card.kind())
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

/// What permits a cast: a priority play over one route (CR 601.2a), or the
/// resolving instruction that offers it (CR 608.2g).
#[derive(Clone, Copy, Debug)]
pub(crate) enum CastAuthority<'a> {
    Priority(PlayRoute),
    Resolution {
        offer: &'a ResolutionCastOffer,
        paid: bool,
    },
}

impl Decision {
    /// The authority a `CastDuringResolution` decision grants its casts;
    /// None for every other decision.
    pub(crate) fn cast_authority(&self) -> Option<CastAuthority<'_>> {
        match self {
            Decision::CastDuringResolution { offer, paid, .. } => {
                Some(CastAuthority::Resolution { offer, paid: *paid })
            }
            _ => None,
        }
    }
}

impl Game {
    /// The one construction of a cast's choices for every way of casting.
    /// None when `profile` has no definition for `object`.
    pub(crate) fn cast_choices(
        &self,
        player: PlayerId,
        object: ObjectId,
        profile: SpellProfile,
        authority: CastAuthority<'_>,
    ) -> Option<CastChoices> {
        let spell = crate::cards::content_pack::default_content_pack()
            .spell_definition(&self.state[object], profile)?;
        let route = match authority {
            CastAuthority::Resolution { offer, paid } => {
                // The instruction supplies timing and zone permission for
                // every resolution cast, paid or free, so no permission
                // surcharge (flash) is ever bought here.
                return Some(CastChoices {
                    profile,
                    free: !paid,
                    resolving: true,
                    exile_instead_of_graveyard: matches!(
                        offer,
                        ResolutionCastOffer::Graveyard {
                            exile_instants_and_sorceries: true,
                            ..
                        }
                    ) && spell.types.is_instant_or_sorcery(),
                    ..Default::default()
                });
            }
            CastAuthority::Priority(route) => route,
        };
        let grant = match route {
            PlayRoute::Grant(key) => self.grant_by_key(object, key),
            _ => None,
        };
        let mut cast = CastChoices {
            profile,
            free: route == PlayRoute::FreeFromHand,
            route: (route != PlayRoute::Hand).then_some(route),
            flashback: matches!(
                route,
                PlayRoute::PrintedFlashback | PlayRoute::GrantedFlashback
            ) || grant.is_some_and(|grant| grant.flashback),
            from_prepared: grant.is_some_and(|grant| grant.prepared),
            ..Default::default()
        };
        // A surcharge is mandatory only when this is the permission that
        // makes casting timely. Other flash grants still allow normal payment.
        if spell.flash_surcharge().is_some()
            && !self.can_cast_sorceries(player)
            && !self.has_instant_timing(player, object, cast)
        {
            cast.flash_surcharge = true;
        }
        Some(cast)
    }

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
            && self.state.turn.lands_played < 1 + self.additional_land_plays(player)
    }

    fn additional_land_plays(&self, player: PlayerId) -> u32 {
        let granted = self
            .player_effects(player)
            .filter(|effect| {
                effect.modification == crate::state::effect::Modification::AdditionalLandPlay
            })
            .count() as u32;
        self.layer_board().play_permissions[player.0].extra_land_plays + granted
    }

    pub fn can_pay_mana_cost(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.can_pay_automatic_mana(player, cost)
    }

    /// Iroh grants Lessons {1} and other instants and sorceries their mana cost.
    fn granted_flashback_cost(&self, object: ObjectId) -> Option<ManaCost> {
        let card = &self.state[object];
        if card.has_subtype("Lesson") {
            Some(ManaCost::parse("{1}"))
        } else if card.types.is_instant_or_sorcery() {
            card.mana_cost.clone()
        } else {
            None
        }
    }

    /// Whether `player` may cast spells only from their hand: an "until your
    /// next turn" restriction, or an opponent's permanent that says so.
    pub(crate) fn casts_only_from_hand(&self, player: PlayerId) -> bool {
        self.hand_only_until(player).is_some()
            || self.layer_board().play_permissions[1 - player.0].opponents_hand_only
    }

    /// The routes `object`'s zone opens to `player`, before any face is
    /// selected. Which face a route permits, and the cast-only-from-hand
    /// prohibition, belong to `permits_spell` alone; timing and payment
    /// eliminate routes later.
    pub(super) fn play_routes(
        &self,
        player: PlayerId,
        object: ObjectId,
    ) -> Cow<'static, [PlayRoute]> {
        let card = &self.state[object];
        let board = self.layer_board();
        let permissions = &board.play_permissions[player.0];
        let own = card.owner == player;
        let active = self.active_player() == player;
        match self.state.zones.zone_of(object) {
            Some(ZoneType::Hand) if own => {
                if permissions.free_from_hand {
                    Cow::Borrowed(&[PlayRoute::Hand, PlayRoute::FreeFromHand])
                } else {
                    Cow::Borrowed(&[PlayRoute::Hand])
                }
            }
            Some(zone @ (ZoneType::Exile | ZoneType::Graveyard)) => {
                let mut routes: Vec<_> = self
                    .live_grants(player, object)
                    .map(|grant| PlayRoute::Grant(grant.key))
                    .collect();
                if zone == ZoneType::Exile && active && !own {
                    if let Some(counters) = self.state.object_counters.get(&object) {
                        routes.extend(
                            permissions
                                .exile_counter_plays
                                .iter()
                                .filter(|kind| counters.get(**kind) > 0)
                                .map(|kind| PlayRoute::ExileCounter(*kind)),
                        );
                    }
                }
                if zone != ZoneType::Graveyard || !own {
                    return routes.into();
                }
                let printed = card.flashback();
                if printed.is_some() {
                    routes.push(PlayRoute::PrintedFlashback);
                }
                if active
                    && permissions.graveyard_flashback
                    && self
                        .granted_flashback_cost(object)
                        .as_ref()
                        .is_some_and(|cost| Some(cost) != printed)
                {
                    routes.push(PlayRoute::GrantedFlashback);
                }
                if card.graveyard_counter_cast().is_some() {
                    routes.push(PlayRoute::PrintedGraveyard);
                }
                let sources = &permissions.graveyard_type_sources;
                if !active || sources.is_empty() || (!card.types.is_permanent() && !card.modal_back)
                {
                    return routes.into();
                }
                // CR 305.9: a land face (even an artifact land) uses only the
                // Land permission; other permanent types come from spell faces.
                let back = crate::cards::content_pack::default_content_pack()
                    .spell_definition(card, SpellProfile::BackFace);
                let mut kinds = Vec::new();
                for types in std::iter::once(&card.types).chain(back.map(|spell| &spell.types)) {
                    use crate::state::card::CardType::*;
                    let face: Vec<_> = if types.is_land() {
                        vec![Land]
                    } else {
                        types.types.iter().copied().collect()
                    };
                    for kind in face {
                        if !matches!(kind, Instant | Sorcery | Kindred) && !kinds.contains(&kind) {
                            kinds.push(kind);
                        }
                    }
                }
                for &(source, permanent_id) in sources {
                    for &kind in &kinds {
                        let route = PlayRoute::GraveyardType {
                            source,
                            kind,
                            permanent_id,
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
                if self.state.zones.top(ZoneType::Library, player) == Some(object) =>
            {
                Cow::Borrowed(&[PlayRoute::LibraryTop])
            }
            _ => Cow::Borrowed(&[]),
        }
    }

    /// Whether the cast's route permits its selected face: a library or
    /// graveyard permission tests the characteristics of that face.
    pub(super) fn permits_spell(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> bool {
        let Some(face) = crate::cards::content_pack::default_content_pack()
            .spell_definition(&self.state[object], cast.profile)
        else {
            return false;
        };
        let route = cast.route.unwrap_or(PlayRoute::Hand);
        let land = face.types.is_land();
        if !matches!(route, PlayRoute::Hand | PlayRoute::FreeFromHand)
            && self.casts_only_from_hand(player)
            && !land
        {
            return false;
        }
        // The one land-play rule: a land face is played only through a
        // permission to play lands ("you may play"), never one to cast
        // spells (CR 305.9: a land can't be cast).
        match route {
            // Every printed face of the card: its Adventure, Omen or back face too.
            PlayRoute::Hand | PlayRoute::ExileCounter(_) => true,
            // "Cast spells without paying" and "cast from the top of your library".
            PlayRoute::FreeFromHand => !land,
            PlayRoute::LibraryTop => {
                !land
                    && self.layer_board().play_permissions[player.0]
                        .library_spells
                        .iter()
                        .any(|predicate| cast.matches_spell(self, object, predicate))
            }
            PlayRoute::GraveyardType { kind, .. } => {
                (kind == crate::state::card::CardType::Land) == land
                    && face.types.types.contains(&kind)
            }
            PlayRoute::Grant(key) => self.grant_by_key(object, key).is_some_and(|grant| {
                (!land || grant.play_lands)
                    && (cast.profile.is_front()
                        || (!grant.on_adventure && !grant.foretold && !grant.flashback))
            }),
            PlayRoute::PrintedFlashback => {
                !land && cast.profile.is_front() && face.flashback().is_some()
            }
            PlayRoute::GrantedFlashback => {
                !land && cast.profile.is_front() && self.granted_flashback_cost(object).is_some()
            }
            PlayRoute::PrintedGraveyard => {
                !land && cast.profile.is_front() && face.graveyard_counter_cast().is_some()
            }
        }
    }

    /// The routes that permit `profile` of `object`, before timing and
    /// payment. A face with one route leaves its Command unnamed even when
    /// another face of the card overlaps.
    pub(crate) fn permitted_play_routes(
        &self,
        player: PlayerId,
        object: ObjectId,
        profile: SpellProfile,
    ) -> Vec<PlayRoute> {
        self.play_routes(player, object)
            .iter()
            .copied()
            .filter(|&route| {
                self.cast_choices(player, object, profile, CastAuthority::Priority(route))
                    .is_some_and(|cast| self.permits_spell(player, object, cast))
            })
            .collect()
    }

    pub(crate) fn selected_play_route(
        &self,
        player: PlayerId,
        object: ObjectId,
        profile: SpellProfile,
        selected: Option<PlayRoute>,
    ) -> Result<PlayRoute, AgentError> {
        let routes = self.permitted_play_routes(player, object, profile);
        match (selected, routes.as_slice()) {
            (Some(route), _) if routes.contains(&route) => Ok(route),
            (None, [only]) => Ok(*only),
            _ => Err(AgentError("missing or ambiguous play permission".into())),
        }
    }

    /// The route a priority play uses, including the lone permission a
    /// single-route Command leaves unnamed. None for an ordinary hand play
    /// and for every other action.
    pub(crate) fn resolved_play_route(&self, action: &Action) -> Option<PlayRoute> {
        let (Action::CastSpell {
            player,
            object,
            route,
            profile,
        }
        | Action::PlayLand {
            player,
            object,
            route,
            profile,
        }) = action
        else {
            return None;
        };
        // A resolution cast (CR 608.2g) is authorized by the resolving
        // instruction, never by a priority route the object also holds or
        // an action names.
        if self.state.suspended_decision.is_some() {
            return None;
        }
        if route.is_some() {
            return *route;
        }
        // A lone hand route is the ordinary play; overlap always names a route.
        if self.state.zones.zone_of(*object) == Some(ZoneType::Hand)
            && self.state[*object].owner == *player
        {
            return None;
        }
        self.selected_play_route(*player, *object, *profile, None)
            .ok()
            .filter(|route| *route != PlayRoute::Hand)
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
                PlayRoute::Grant(_) => self.grant_base_cost(cast.grant(self, object)?, spell),
                _ => spell.mana_cost.clone(),
            }
        }
    }

    /// The zone `object` is cast from: where it is while its costs are
    /// paid (CR 601.2h precedes the move of CR 601.2i), or the zone a spell
    /// already on the stack left.
    fn casting_zone(&self, object: ObjectId) -> ZoneType {
        match self.state.zones.zone_of(object) {
            Some(ZoneType::Stack) | None => self
                .spell_on_stack(object)
                .map_or(ZoneType::Stack, |spell| spell.cast_from),
            Some(zone) => zone,
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
        cost.purpose = SpendingContext::Cast {
            spell: SpellSnapshot(Arc::clone(card_ref)),
            from: self.casting_zone(object),
        };
        if cast.flash_surcharge {
            cost = cost.plus(card_ref.flash_surcharge()?);
        }
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
                StaticAbility::CostsLess { amount, condition } => {
                    if condition
                        .as_ref()
                        .is_some_and(|condition| !self.check_condition(condition, player, None))
                    {
                        continue;
                    }
                    self.controller_value(amount, player).max(0) as usize
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
        // CR 601.2h: any-type spending applies to the total cost, once
        // additional costs and reductions are settled (CR 601.2f): Tinybones's
        // stash, an exact grant's permission, or a static ability.
        let mut any_mana = matches!(cast.route, Some(PlayRoute::ExileCounter(_)))
            || cast.grant(self, object).is_some_and(|grant| grant.any_mana);
        for (source, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            for ability in &view.static_abilities {
                if let StaticAbility::SpendAnyManaToCast(predicate) = ability {
                    any_mana |= cast.matches_spell(self, object, predicate);
                    continue;
                }
                if let StaticAbility::SpellsFromOutsideHandCostLess(amount) = ability {
                    if self.casting_zone(object) != ZoneType::Hand {
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
                            self.recorded_cast_matches_predicate(*cast, &reduction.predicate)
                        }))
                {
                    continue;
                }
                if cast.matches_spell(self, object, &reduction.predicate)
                    && self.check_static_condition(&reduction.condition, player)
                {
                    let generic = self.static_value(&reduction.generic, source, player);
                    cost = cost.reduced_generic(u8::try_from(generic.max(0)).unwrap_or(u8::MAX));
                }
            }
        }
        Some(if any_mana {
            cost.pay_with_any_mana()
        } else {
            cost
        })
    }

    /// The choices of a land play (`land`) or a cast of `profile` through
    /// `selected`, before timing: the player holds priority with no choice
    /// pending, the face is a land exactly when `land` (CR 305.9) and its
    /// route permits it.
    fn priority_play(
        &self,
        player: PlayerId,
        object: ObjectId,
        selected: Option<PlayRoute>,
        profile: SpellProfile,
        land: bool,
    ) -> Result<CastChoices, AgentError> {
        if self.pending_choice.is_some() {
            return Err(AgentError("a choice is already pending".to_string()));
        }
        if self.state.priority.holder != player {
            return Err(AgentError("player does not have priority".to_string()));
        }
        let route = self.selected_play_route(player, object, profile, selected)?;
        let cast = self
            .cast_choices(player, object, profile, CastAuthority::Priority(route))
            .ok_or_else(|| AgentError("the card has no such face".into()))?;
        if cast.spell(&self.state[object]).types.is_land() != land {
            return Err(AgentError(if land {
                "only land faces can be played".into()
            } else {
                "land faces cannot be cast".into()
            }));
        }
        if !self.permits_spell(player, object, cast) {
            return Err(AgentError("permission does not allow this play".into()));
        }
        Ok(cast)
    }

    pub(crate) fn play_land(
        &mut self,
        player: PlayerId,
        card: CardId,
        selected: Option<PlayRoute>,
        profile: SpellProfile,
    ) -> Result<(), AgentError> {
        let object = self.state.object_of(card);
        let cast = self.priority_play(player, object, selected, profile, true)?;
        if !self.can_play_land(player) {
            return Err(AgentError("cannot play land now".to_string()));
        }

        // CR 305.2 — Track one normal land play per turn.
        let controller = (self.state[object].owner != player).then_some(player);
        self.spend_play_route(player, cast.route);
        self.state.turn.lands_played += 1;
        let mut frame = EffectFrame::new(player, Vec::new());
        frame.batch = Some(self.event_batch(
            EventBinding::None,
            [self.zone_move_proposal(
                object,
                ZoneType::Battlefield,
                ZoneEntry {
                    controller,
                    face: EntryFace::played_as(profile),
                    library_top_departure: self.is_library_top(object),
                    ..Default::default()
                },
            )],
        ));
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
        self.state
            .object(object)
            .map_err(|_| AgentError("spell must name a current object".into()))?;
        let cast = self.priority_play(player, object, selected, profile, false)?;
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
            && self
                .exile_alternative_objects(player, object, cast)
                .is_empty()
        {
            return false;
        }
        let Some(cost) = self.effective_spell_cost(player, object, cast, targets) else {
            return false;
        };
        let card = cast.spell(&self.state[object]);
        if !cast.kicked
            && card
                .life_alternative()
                .is_some_and(|life| self.payable_life(player) < life as i32)
        {
            return false;
        }
        if card.discard_cost() && self.discard_candidates(player, Some(object)).is_empty() {
            return false;
        }
        if cast.route == Some(PlayRoute::PrintedGraveyard)
            && card
                .graveyard_counter_cast()
                .is_some_and(|count| self.removable_counters(player) < u32::from(count))
        {
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
        if cast.spell(&self.state[object]).flash_surcharge().is_some() {
            let paid = CastChoices {
                flash_surcharge: true,
                ..cast
            };
            return (cast.flash_surcharge
                || cast.resolving
                || !self.can_pay_cast(player, object, paid))
            .then_some(cast);
        }
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
        let cast = if cast.spell(&self.state[object]).flash_surcharge().is_some() {
            CastChoices {
                flash_surcharge: pay,
                ..cast
            }
        } else {
            CastChoices {
                kicked: pay,
                ..cast
            }
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
        self.advance_target_selection(SpellTargeting::new(player, object, cast, None))
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
        self.advance_target_selection(SpellTargeting::new(player, object, cast, Some(mode)))
    }

    /// Evaluate a clause's division total at announcement time (CR 601.2d, 602.2b).
    pub(crate) fn announcement_division_total(
        &self,
        value: &EffectValue,
        player: PlayerId,
        source: Option<ObjectId>,
        x: i32,
    ) -> i32 {
        let mut frame = EffectFrame::new(player, Vec::new());
        frame.x = x;
        if let Some(source) = source {
            frame.bind_source(Some(source), Some(source));
        }
        self.effect_value(value, &frame)
    }

    /// Maximum targets selectable for `clause`, capped by any announced division
    /// total (CR 601.2d: each target must receive at least 1).
    pub(crate) fn effective_clause_max(
        &self,
        clause: &TargetRequirement,
        player: PlayerId,
        source: Option<ObjectId>,
        x: i32,
    ) -> usize {
        let base = clause.max_for_x(x);
        match &clause.division {
            Some(value) => {
                let total = self
                    .announcement_division_total(value, player, source, x)
                    .max(0) as usize;
                base.min(total)
            }
            None => base,
        }
    }

    /// Finish target selection for one spell clause, entering sequential division
    /// when `requirement.division` is set and multiple targets need shares.
    fn finish_spell_clause(&mut self, mut targeting: SpellTargeting) -> Result<(), AgentError> {
        let requirements = targeting
            .cast
            .spell(&self.state[targeting.object])
            .target_requirements_for_mode(targeting.mode);
        let division = requirements
            .get(targeting.requirement_index)
            .and_then(|req| req.division.as_ref())
            .cloned();
        if let Some(value) = division {
            let total = self
                .announcement_division_total(
                    &value,
                    targeting.player,
                    Some(targeting.object),
                    i32::from(targeting.cast.x),
                )
                .max(0);
            let slots = targeting.clause_slots(targeting.requirement_index);
            let player = targeting.player;
            return self.begin_division(player, slots, total, DivisionParent::Spell(targeting));
        }
        targeting.requirement_index += 1;
        self.advance_target_selection(targeting)
    }

    /// Begin sequential amount assignment for `slots` summing to `total`,
    /// auto-assigning when no real choice exists (0 targets, 1 target, or
    /// every target forced to 1).
    fn begin_division(
        &mut self,
        player: PlayerId,
        slots: Vec<usize>,
        total: i32,
        mut parent: DivisionParent,
    ) -> Result<(), AgentError> {
        if slots.is_empty() {
            return self.finish_division_parent(player, parent);
        }
        let chosen_len = parent.chosen().len();
        let divisions = parent.divisions_mut();
        if divisions.len() < chosen_len {
            divisions.resize(chosen_len, 0);
        }
        self.advance_division(player, slots, 0, total, parent)
    }

    /// Auto-assign remaining slots when no choice remains, otherwise suspend on
    /// `PendingChoice::DivideAmount`.
    fn advance_division(
        &mut self,
        player: PlayerId,
        slots: Vec<usize>,
        next: usize,
        remaining: i32,
        mut parent: DivisionParent,
    ) -> Result<(), AgentError> {
        let rem_targets = (slots.len() - next) as i32;
        if rem_targets == 1 {
            parent.divisions_mut()[slots[next]] = remaining;
            return self.finish_division_parent(player, parent);
        }
        if remaining <= rem_targets {
            let divisions = parent.divisions_mut();
            for &slot in &slots[next..] {
                divisions[slot] = 1;
            }
            return self.finish_division_parent(player, parent);
        }
        self.pending_choice = Some(PendingChoice::DivideAmount {
            player,
            slots,
            next,
            remaining,
            parent: Box::new(parent),
        });
        Ok(())
    }

    /// Assign `mode` as the divided amount for the current target slot.
    pub(crate) fn choose_division_amount(
        &mut self,
        actor: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let Some(PendingChoice::DivideAmount {
            player,
            slots,
            next,
            remaining,
            mut parent,
        }) = self.pending_choice.take()
        else {
            return Err(AgentError("no division choice is pending".into()));
        };
        let rem_targets = (slots.len() - next) as i32;
        let max_amt = remaining - (rem_targets - 1);
        let amount = i32::try_from(mode).unwrap_or(0);
        if actor != player || amount < 1 || amount > max_amt {
            self.pending_choice = Some(PendingChoice::DivideAmount {
                player,
                slots,
                next,
                remaining,
                parent,
            });
            return Err(AgentError("illegal division amount".into()));
        }
        parent.divisions_mut()[slots[next]] = amount;
        self.advance_division(player, slots, next + 1, remaining - amount, *parent)
    }

    fn finish_division_parent(
        &mut self,
        _player: PlayerId,
        parent: DivisionParent,
    ) -> Result<(), AgentError> {
        match parent {
            DivisionParent::Spell(mut targeting) => {
                targeting.requirement_index += 1;
                self.advance_target_selection(targeting)
            }
            DivisionParent::Activation(targeting) => self.next_activation_clause(targeting),
        }
    }

    /// Move the casting pipeline to `targeting.requirement_index`, skipping
    /// requirements with no legal targets when they're optional, and
    /// finishing the cast once all requirements are satisfied.
    fn advance_target_selection(
        &mut self,
        mut targeting: SpellTargeting,
    ) -> Result<(), AgentError> {
        let requirements = targeting
            .cast
            .spell(&self.state[targeting.object])
            .target_requirements_for_mode(targeting.mode);
        loop {
            let Some(requirement) = requirements.get(targeting.requirement_index) else {
                return self.finish_casting(targeting);
            };
            let max = self.effective_clause_max(
                requirement,
                targeting.player,
                Some(targeting.object),
                i32::from(targeting.cast.x),
            );
            let clause_chosen = targeting.clause_targets(targeting.requirement_index);
            let legal = self
                .remaining_legal_targets(
                    targeting.player,
                    targeting.object,
                    requirement,
                    &clause_chosen,
                    targeting.cast.x,
                )
                .into_iter()
                .filter(|target| {
                    self.target_leaves_cost_payable(
                        targeting.player,
                        targeting.object,
                        targeting.cast,
                        &targeting.chosen,
                        *target,
                    )
                })
                .collect::<Vec<_>>();
            if legal.is_empty() || max == 0 {
                if requirement.min_for(targeting.cast.x) == 0 {
                    targeting.requirement_index += 1;
                    continue;
                }
                return Err(AgentError("no legal targets".to_string()));
            }
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                targeting,
                legal_targets: legal,
            });
            return Ok(());
        }
    }

    /// With a target-dependent reduction, a choice that would leave the
    /// spell's total cost unpayable is not offered (CR 601.2f-h).
    pub(crate) fn target_leaves_cost_payable(
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
                self.can_add_clause_target(
                    requirement,
                    i32::from(x),
                    chosen.iter().copied(),
                    *target,
                )
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
            mut targeting,
            legal_targets,
        }) = self.pending_choice.take()
        else {
            return Err(AgentError("no target choice is pending".to_string()));
        };

        if targeting.player != player {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                targeting,
                legal_targets,
            });
            return Err(AgentError("wrong player for target choice".to_string()));
        }
        if !legal_targets.contains(&target) {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                targeting,
                legal_targets,
            });
            return Err(AgentError("target is not legal".to_string()));
        }

        targeting.choose(target);

        let requirements = targeting
            .cast
            .spell(&self.state[targeting.object])
            .target_requirements_for_mode(targeting.mode);
        let requirement = &requirements[targeting.requirement_index];
        let clause_chosen = targeting.clause_targets(targeting.requirement_index);
        let effective_max = self.effective_clause_max(
            requirement,
            targeting.player,
            Some(targeting.object),
            i32::from(targeting.cast.x),
        );

        if clause_chosen.len() >= effective_max {
            return self.finish_spell_clause(targeting);
        }

        let legal = self.remaining_legal_targets(
            targeting.player,
            targeting.object,
            requirement,
            &clause_chosen,
            targeting.cast.x,
        );
        if legal.is_empty() {
            return self.finish_spell_clause(targeting);
        }
        self.pending_choice = Some(PendingChoice::ChooseTargets {
            targeting,
            legal_targets: legal,
        });
        Ok(())
    }

    /// Stop choosing targets for an "up to N" requirement (its minimum has
    /// been met) and advance to the next requirement.
    pub(crate) fn finish_targets_action(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(PendingChoice::ChooseTargets {
            targeting,
            legal_targets,
        }) = self.pending_choice.take()
        else {
            return Err(AgentError("no target choice is pending".to_string()));
        };
        if targeting.player != player {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                targeting,
                legal_targets,
            });
            return Err(AgentError("wrong player for target choice".to_string()));
        }

        let requirements = targeting
            .cast
            .spell(&self.state[targeting.object])
            .target_requirements_for_mode(targeting.mode);
        let requirement = &requirements[targeting.requirement_index];
        let chosen_in_req = targeting.clause_slots(targeting.requirement_index).len();
        if chosen_in_req < requirement.min_for(targeting.cast.x) {
            self.pending_choice = Some(PendingChoice::ChooseTargets {
                targeting,
                legal_targets,
            });
            return Err(AgentError(
                "minimum number of targets not chosen".to_string(),
            ));
        }
        self.finish_spell_clause(targeting)
    }

    fn finish_casting(&mut self, targeting: SpellTargeting) -> Result<(), AgentError> {
        self.begin_payment(
            targeting.player,
            Operation::Spell {
                object: targeting.object,
                cast: targeting.cast,
                mode: targeting.mode,
                targets: targeting.chosen,
                target_req_indices: targeting.chosen_req_indices,
                target_divisions: targeting.target_divisions,
            },
        )
    }

    /// Whether `cast` takes `card`'s exile-from-hand alternative cost: such a
    /// card's kicked flag means the alternative, which costs no mana.
    pub(super) fn uses_exile_alternative(&self, object: ObjectId, cast: CastChoices) -> bool {
        cast.kicked
            && cast
                .spell(&self.state[object])
                .exile_alternative()
                .is_some()
    }

    /// The cards `player` may exile for `card`'s exile-from-hand alternative
    /// cost: other matching cards in hand while `card` is cast from hand,
    /// paying its costs, and, if the cost says so, outside `player`'s turn.
    pub(crate) fn exile_alternative_objects(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Vec<ObjectId> {
        let Some((predicate, not_your_turn)) = cast.spell(&self.state[object]).exile_alternative()
        else {
            return Vec::new();
        };
        if (not_your_turn && self.active_player() == player)
            || !self.state.zones.contains(object, ZoneType::Hand, player)
        {
            return Vec::new();
        }
        self.state
            .zone_cards(ZoneType::Hand, player)
            .map(|card| self.state.object_of(card))
            .filter(|&other| other != object && self.object_matches_predicate(other, predicate))
            .collect()
    }

    pub(super) fn commit_spell_payment(&mut self, player: PlayerId, payment: Payment) {
        let Operation::Spell {
            object,
            cast,
            mode,
            targets,
            target_req_indices,
            target_divisions,
        } = payment.operation
        else {
            unreachable!("activation prepared before costs")
        };
        // Recorded before the move, so its invalidation precedes the move's
        // board reads; the move then names the stack appearance.
        self.state.turn.spells_cast[player.0].push(crate::flow::turn::RecordedCast {
            object,
            profile: cast.profile,
        });
        if cast.spell(&self.state[object]).types.is_creature() {
            self.end_next_creature_cast_flash(player);
        }
        // CR 601.2i — A cast spell is put onto the stack.
        let object =
            self.push_spell_to_stack(object, player, targets.clone(), target_req_indices, cast);
        self.state.name_last_cast(player, object);
        if self.state.effects.awaits_spell(player) {
            self.state.effects.bind_next_spell(player, object);
        }
        let Some(crate::state::stack_object::StackObject::Spell(spell)) =
            self.state.stack_objects_mut().last_mut()
        else {
            unreachable!("just pushed spell")
        };
        let paid = u32::from(payment.receipt.total());
        spell.mode = mode;
        spell.spent = payment.receipt;
        spell.target_divisions = target_divisions;
        let total = &mut self.state.turn.history[player.0].mana_spent_casting;
        let spend_before = *total;
        *total = total.saturating_add(paid);
        let spend_after = *total;
        self.emit(GameEvent::SpellCast {
            caster: player,
            spend_before,
            spend_after,
            object_id: object,
            targets: targets.clone(),
        });
        self.emit_permanents_targeted(&targets, Target::StackSpell(object), player);
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
        self.check_permanent_activation_terms(player, permanent_id, &ability)?;
        if !permanent.can_pay_counters(&ability) {
            return Err(AgentError("cannot pay counter cost".into()));
        }
        if permanent.activation_used(ability_index, &ability) {
            return Err(AgentError("ability was already activated".into()));
        }
        if ability.tap_source && !self.permanent_can_tap(permanent_id) {
            return Err(AgentError("source cannot tap".into()));
        }
        if !self.can_tap_grantor(&ability) {
            return Err(AgentError("grantor cannot tap".into()));
        }
        if ability.announces_x() {
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
        if ability
            .tap_power()
            .is_some_and(|(power, _)| self.crew_power_available(player, permanent_id) < power)
        {
            return Err(AgentError("not enough power to crew or saddle".into()));
        }
        self.begin_activation_targets(
            player,
            ActivationSource::Permanent(permanent_id),
            ability_index,
            &ability,
            0,
        )
    }

    /// Legal X values for `permanent`'s `ability_index`: payable with the
    /// remaining loyalty and mana resources, and with legal targets available
    /// for every required clause at that X (CR 602.2b, 601.2b-c).
    pub(crate) fn legal_activation_x_values(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Vec<i32> {
        let view = self.characteristics(permanent);
        let ability = &view.activated_abilities[ability_index];
        let least = i32::from(ability.x_min);
        let mut most = if ability.loyalty == Some(crate::state::card::LoyaltyCost::MinusX) {
            self.state.permanents[permanent]
                .as_ref()
                .expect("activation source")
                .available_loyalty(ability)
        } else {
            i32::from(u8::MAX)
        };
        if ability.mana_cost.x > 0 {
            let limit = most.min(i32::from(u8::MAX));
            most = (least..=limit)
                .take_while(|&x| {
                    self.activation_x_payable(player, permanent, ability_index, x as u8)
                })
                .last()
                .unwrap_or(least - 1);
        } else if (ability.mana_cost.mana_value > 0 || ability.waterbend)
            && !self.activation_x_payable(player, permanent, ability_index, 0)
        {
            return Vec::new();
        }
        if most < least {
            return Vec::new();
        }
        let source = ActivationSource::Permanent(permanent);
        if ability.target_requirements().is_empty() {
            return (least..=most).collect();
        }
        (least..=most)
            .filter(|&x| {
                self.activation_targets_available(player, source, ability_index, ability, x)
            })
            .collect()
    }

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
        let x = i32::try_from(x).map_err(|_| AgentError("illegal value of X".into()))?;
        if chooser != player
            || !self
                .legal_activation_x_values(player, permanent, ability_index)
                .contains(&x)
        {
            return Err(AgentError("illegal value of X".into()));
        }
        self.pending_choice = None;
        self.begin_activation_targets(
            player,
            ActivationSource::Permanent(permanent),
            ability_index,
            &self.characteristics(permanent).activated_abilities[ability_index],
            x,
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
        self.begin_activation_targets(
            player,
            ActivationSource::Card(card),
            ability_index,
            &ability,
            0,
        )
    }

    pub(crate) fn check_permanent_activation_terms(
        &self,
        player: PlayerId,
        source: PermanentId,
        ability: &ActivatedAbilityDefinition,
    ) -> Result<(), AgentError> {
        if ability.loyalty.is_some() && !self.can_cast_sorceries(player) {
            let instant_timing =
                self.player_effects(player)
                    .any(|record| match &record.modification {
                        Modification::LoyaltyAtInstantSpeed(predicate) => {
                            self.permanent_matches_predicate(source, predicate)
                        }
                        _ => false,
                    });
            if !instant_timing {
                return Err(AgentError(
                    "loyalty activation requires sorcery timing".into(),
                ));
            }
        }
        self.check_activation_terms(player, ability)
    }

    /// The turn, life and condition terms every activation shares, whatever
    /// its source, mana abilities included: the one authority for "activate
    /// only ..." restrictions (CR 602.5).
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
        if let Some(condition) = &ability.condition {
            if !self.check_static_condition(condition, player) {
                return Err(AgentError("activation condition does not hold".into()));
            }
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
            source: SpendingObject::new(definition.subtypes.iter().map(String::as_str), false),
        };
        self.price_activation(player, &mut ability);
        Some(ability)
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

    fn begin_activation_targets(
        &mut self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        ability: &ActivatedAbilityDefinition,
        x: i32,
    ) -> Result<(), AgentError> {
        let targeting = ActivatedTargeting {
            player,
            source,
            ability_index,
            x,
            chosen: Vec::new(),
            target_req_indices: Vec::new(),
            target_divisions: Vec::new(),
            requirement: 0,
        };
        if !ability.target_requirements().is_empty() {
            if !self.activation_targets_available(player, source, ability_index, ability, x) {
                return Err(AgentError("no legal ability targets".into()));
            }
            let first = &ability.target_requirements()[0];
            let source_object = self.activation_ability_source(source).map(|s| s.object());
            let first_max = self.effective_clause_max(first, player, source_object, x);
            if first.min_for_x(x) == 0
                && (first_max == 0 || self.activated_ability_targets(&targeting).is_empty())
            {
                return self.next_activation_clause(targeting);
            }
            self.pending_choice = Some(PendingChoice::ActivatedTarget(targeting));
            Ok(())
        } else {
            self.pay_activation(targeting)
        }
    }

    /// Whether every required targeting clause has a legal candidate.
    pub(crate) fn activation_targets_available(
        &self,
        player: PlayerId,
        source: ActivationSource,
        ability_index: usize,
        ability: &ActivatedAbilityDefinition,
        x: i32,
    ) -> bool {
        let source_object = self.activation_ability_source(source).map(|s| s.object());
        ability
            .target_requirements()
            .iter()
            .enumerate()
            .all(|(index, requirement)| {
                let min = requirement.min_for_x(x);
                min == 0
                    || (self.effective_clause_max(requirement, player, source_object, x) >= min
                        && self
                            .activated_ability_targets(&ActivatedTargeting {
                                player,
                                source,
                                ability_index,
                                requirement: index,
                                x,
                                chosen: Vec::new(),
                                target_req_indices: Vec::new(),
                                target_divisions: Vec::new(),
                            })
                            .len()
                            >= min)
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
    /// The exact object an activation of `source` comes from, as its stacked
    /// ability will record it; `None` once a permanent source has left.
    fn activation_ability_source(&self, source: ActivationSource) -> Option<AbilitySource> {
        let object = match source {
            ActivationSource::Permanent(permanent) => self.permanent_object(permanent)?,
            ActivationSource::Card(card) => self.state.object_of(card),
        };
        Some(AbilitySource::new(object, Some(object)))
    }

    /// Targets clause `requirement` of `player`'s activation can choose
    /// next, given those already `chosen` (all clauses). Legality is from
    /// `player`'s side whoever chooses. Under "that share a creature type",
    /// a target is offered only with the chosen ones and, while the minimum
    /// is unmet, a remaining partner.
    pub(crate) fn activated_ability_targets(&self, targeting: &ActivatedTargeting) -> Vec<Target> {
        let ActivatedTargeting {
            player,
            source,
            ability_index,
            requirement,
            x,
            ..
        } = *targeting;
        let Some(clause) = self.activation_target_requirement(source, ability_index, requirement)
        else {
            return Vec::new();
        };
        let Some(ability_source) = self.activation_ability_source(source) else {
            return Vec::new();
        };
        let chosen = targeting.clause_targets(requirement);
        let candidates: Vec<Target> = self
            .legal_clause_targets(&clause, player, ability_source)
            .filter(|target| {
                self.can_add_clause_target(&clause, x, chosen.iter().copied(), *target)
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
                !needs_partner
                    || candidates.iter().any(|partner| {
                        self.can_add_clause_target(
                            &clause,
                            x,
                            chosen.iter().chain([target]).copied(),
                            *partner,
                        )
                    })
            })
            .collect()
    }

    /// Whether the permanents among `targets` have a creature type in common
    /// ("that share a creature type").
    pub(super) fn targets_share_creature_type(
        &self,
        targets: impl Iterator<Item = Target>,
    ) -> bool {
        let mut common: Option<CreatureTypes> = None;
        for target in targets {
            let Target::Permanent(id) = target else {
                return false;
            };
            let types = self.characteristics(id).creature_types.clone();
            common = Some(match common {
                None => types,
                Some(common) => common.intersection(&types),
            });
        }
        common.is_some_and(|common| !common.is_empty())
    }

    /// The pending activation targeting.
    fn pending_activated_target(&self) -> Result<ActivatedTargeting, AgentError> {
        match &self.pending_choice {
            Some(PendingChoice::ActivatedTarget(targeting)) => Ok(targeting.clone()),
            _ => Err(AgentError("no activated target pending".into())),
        }
    }

    /// Finish choosing the current clause's targets once its minimum is met.
    pub(crate) fn decline_activated_target(&mut self, actor: PlayerId) -> Result<(), AgentError> {
        let targeting = self.pending_activated_target()?;
        let ActivatedTargeting {
            player,
            source,
            ability_index,
            requirement,
            ..
        } = targeting;
        let chooser = self.activation_target_chooser(player, source, ability_index, requirement);
        let clause = self
            .activation_target_requirement(source, ability_index, requirement)
            .ok_or_else(|| AgentError("missing activation source".into()))?;
        let count = targeting.clause_targets(requirement).len();
        if chooser != actor || count < clause.min_for_x(targeting.x) {
            return Err(AgentError("target required".into()));
        }
        self.finish_activation_clause(targeting)
    }

    pub(crate) fn choose_activated_target(
        &mut self,
        actor: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let mut targeting = self.pending_activated_target()?;
        let ActivatedTargeting {
            player,
            source,
            ability_index,
            requirement,
            x,
            ..
        } = targeting;
        let chooser = self.activation_target_chooser(player, source, ability_index, requirement);
        if actor != chooser || !self.activated_ability_targets(&targeting).contains(&target) {
            return Err(AgentError("illegal activated target".into()));
        }
        let clause = self
            .activation_target_requirement(source, ability_index, requirement)
            .ok_or_else(|| AgentError("missing activation source".into()))?;
        targeting.choose(target, requirement);
        let count = targeting.clause_targets(requirement).len();
        let source_object = self.activation_ability_source(source).map(|s| s.object());
        let effective_max = self.effective_clause_max(&clause, player, source_object, x);
        if count >= effective_max || self.activated_ability_targets(&targeting).is_empty() {
            return self.finish_activation_clause(targeting);
        }
        self.pending_choice = Some(PendingChoice::ActivatedTarget(targeting));
        Ok(())
    }

    /// Finish selecting targets for the current activation clause, entering
    /// sequential division if the clause divides an amount across its targets.
    fn finish_activation_clause(
        &mut self,
        targeting: ActivatedTargeting,
    ) -> Result<(), AgentError> {
        let clause = self.activation_target_requirement(
            targeting.source,
            targeting.ability_index,
            targeting.requirement,
        );
        if let Some(value) = clause.and_then(|c| c.division) {
            let source_object = self
                .activation_ability_source(targeting.source)
                .map(|s| s.object());
            let total = self
                .announcement_division_total(&value, targeting.player, source_object, targeting.x)
                .max(0);
            let slots = targeting.clause_slots(targeting.requirement);
            let player = targeting.player;
            return self.begin_division(
                player,
                slots,
                total,
                DivisionParent::Activation(targeting),
            );
        }
        self.next_activation_clause(targeting)
    }

    /// Skip later optional clauses with no candidate, then pay the announced costs.
    fn next_activation_clause(
        &mut self,
        mut targeting: ActivatedTargeting,
    ) -> Result<(), AgentError> {
        let source_object = self
            .activation_ability_source(targeting.source)
            .map(|s| s.object());
        targeting.requirement += 1;
        while let Some(clause) = self.activation_target_requirement(
            targeting.source,
            targeting.ability_index,
            targeting.requirement,
        ) {
            let max =
                self.effective_clause_max(&clause, targeting.player, source_object, targeting.x);
            if clause.min_for_x(targeting.x) > 0
                || (max > 0 && !self.activated_ability_targets(&targeting).is_empty())
            {
                self.pending_choice = Some(PendingChoice::ActivatedTarget(targeting));
                return Ok(());
            }
            targeting.requirement += 1;
        }
        self.pay_activation(targeting)
    }

    fn pay_activation(&mut self, targeting: ActivatedTargeting) -> Result<(), AgentError> {
        self.begin_payment(
            targeting.player,
            Operation::Activation {
                source: targeting.source,
                ability_index: targeting.ability_index,
                targets: targeting.chosen,
                target_req_indices: targeting.target_req_indices,
                target_divisions: targeting.target_divisions,
                x: targeting.x,
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
        x: i32,
        tap: bool,
    ) -> Result<ActivatedAbilityOnStack, AgentError> {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return Err(AgentError("source permanent does not exist".to_string()));
        };
        let source_object = self
            .permanent_object(permanent_id)
            .ok_or_else(|| AgentError("missing activation source identity".into()))?;
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
        self.state
            .record_activation(permanent_id, ability_index, &ability);

        let id = self.state.id_gen_mut().next_id();
        Ok(ActivatedAbilityOnStack {
            target_req_indices: Vec::new(),
            target_divisions: Vec::new(),
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
            attached: self.state.permanents[permanent_id]
                .as_ref()
                .and_then(|p| p.attached_to),
            x,
            loyalty_change: ability.loyalty.map(|cost| cost.counter_change(x)),
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
            target_req_indices: Vec::new(),
            target_divisions: Vec::new(),
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
            loyalty_change: None,
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
        self.legal_clause_targets(requirement, player, AbilitySource::new(object, None))
            .filter(|target| *target != Target::StackSpell(object))
            .collect()
    }
}
