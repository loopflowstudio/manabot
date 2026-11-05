//! Derived characteristics and rules questions. `layers::Context` owns CR 613
//! evaluation; this module caches its immutable board and exposes the one public
//! view. Activation prices and dynamic mana outputs are finalized afterwards.
use crate::{
    flow::{
        decision::EffectFrame,
        game::{Game, Revisions},
        layers::{matches_view, Context, SourceSeed},
        triggers::TriggerIndex,
    },
    state::{
        ability::{Condition, PaymentCost},
        card::{
            ActivatedAbilityDefinition, CardTypes, Keyword, Keywords, PowerCda, Restriction,
            StaticAbility, StaticScope,
        },
        game_object::{CardId, ObjectRef, PermanentId, PlayerId, Target},
        mana::ManaUse,
        permanent::CounterKind,
        predicate::CardPredicate,
        zone::ZoneType,
    },
};
use std::{collections::BTreeMap, sync::Arc};
/// A battlefield permanent's characteristics with every static ability and
/// continuous-effect record applied (CR 613). A permanent that has left the
/// battlefield reads as the default: nameless, typeless, 0/0, no abilities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Characteristics {
    pub mana_abilities: Vec<crate::state::card::ManaAbility>,
    pub triggered_mana_abilities: Vec<crate::state::card::TriggeredManaAbility>,
    pub colors: crate::state::mana::Colors,
    pub static_abilities: Vec<StaticAbility>,
    pub triggered_abilities: Vec<crate::state::ability::EffectiveTrigger>,
    /// Set by a characteristic-setting Aura, or printed (CR 201.2).
    pub name: String,
    /// Whether a characteristic-setting Aura replaced its name, creature
    /// types and base power/toughness.
    pub characteristics_set: bool,
    /// Every subtype it has: printed (creature types replaced by a
    /// characteristic-setting Aura's), then added by records.
    pub subtypes: Vec<String>,
    /// Creature subtypes added by records after any characteristic-setting
    /// Aura (layer 4), in record order.
    pub added_subtypes: Vec<String>,
    /// Distinct creature types (CR 205.3m), sorted.
    pub creature_types: Vec<String>,
    /// Layer 4 card types: printed, added by static type grants, and
    /// Creature when animated.
    pub types: CardTypes,
    /// Layer 4 supertype: its copiable values' legendary supertype, unless a
    /// `Modification::Nonlegendary` record removes it. The legend rule and
    /// the card row read this.
    pub is_legendary: bool,
    /// A creature by something other than its printed type: earthbend, its
    /// own counter threshold, or an effect.
    pub is_animated: bool,
    /// Layer 6 keywords.
    pub keywords: Keywords,
    /// Firebending N granted by records after any ability loss, summed.
    pub firebending: u8,
    /// Whether it has lost all abilities (a record, or a characteristic-
    /// setting Aura); grants with later timestamps still apply.
    pub abilities_lost: bool,
    /// Its printed ward cost (layer 6), unless it has lost its abilities.
    pub ward: Option<PaymentCost>,
    pub power: i32,
    pub toughness: i32,
    /// Effective activations in layer/timestamp order, with current costs.
    /// Its index binds offers and payments; stacking captures the chosen effects.
    pub activated_abilities: Vec<ActivatedAbilityDefinition>,
}

impl Characteristics {
    /// Dynamic outputs and reductions can change without changing source plans.
    fn has_dynamic_activations(&self) -> bool {
        self.activated_abilities.iter().any(|ability| {
            ability.cost_reduction_per.is_some()
                || matches!(
                    ability.effects.as_slice(),
                    [crate::state::ability::Effect::If { .. }
                        | crate::state::ability::Effect::AddOneColorManaValue { .. }]
                )
        })
    }

    pub(crate) fn final_chapter(&self) -> Option<u8> {
        if !self.has_subtype("Saga") {
            return None;
        }
        self.triggered_abilities
            .iter()
            .filter_map(|ability| match &ability.ability {
                crate::state::ability::Ability::Triggered {
                    condition: crate::state::ability::TriggerCondition::Chapter { chapter },
                    ..
                } => Some(*chapter),
                _ => None,
            })
            .max()
    }

    pub fn pt(&self) -> (i32, i32) {
        (self.power, self.toughness)
    }

    pub fn is_creature(&self) -> bool {
        self.types.is_creature()
    }

    /// Printed or granted land.
    pub fn is_land(&self) -> bool {
        self.types.is_land()
    }

    pub fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
    }
}

/// Revision-scoped derived values. The raw board feeds explicit rules reads;
/// final entries additionally price activations and materialize mana outputs.
#[derive(Clone, Default)]
pub(crate) struct DerivedCache {
    revisions: Revisions,
    board: Option<Arc<LayerBoard>>,
    /// Copiable payloads survive unrelated board revisions and are shared by forks.
    sources: Vec<Option<Arc<SourceSeed>>>,
    /// Indexed by PermanentId, as are the authoritative permanent slots.
    characteristics: Vec<Option<Arc<Characteristics>>>,
    pub(super) trigger_index: Option<Arc<TriggerIndex>>,
}

#[derive(Debug, PartialEq)]
struct LayerBoard {
    objects: Vec<Option<Arc<Characteristics>>>,
    reads_mana: bool,
    reads_tapped: bool,
    /// Finalization can inspect state beyond the raw layer view. Ordinary
    /// fixed activations retain their finalized entry across pool/tap writes.
    dynamic_activations: bool,
}
impl std::fmt::Debug for DerivedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DerivedCache")
    }
}
impl DerivedCache {
    #[inline]
    pub(super) fn at(&mut self, revisions: Revisions) -> &mut Self {
        if self.revisions != revisions {
            self.invalidate(revisions);
        }
        self
    }

    fn invalidate(&mut self, revisions: Revisions) {
        let mana_changed = self.revisions.mana_pools != revisions.mana_pools;
        let tapped_changed = self.revisions.tapped != revisions.tapped;
        #[cfg(feature = "benchmark")]
        if mana_changed {
            crate::benchmark_layers::record(|m| m.mana_invalidations += 1);
        }
        let characteristics_changed = self.revisions.characteristics != revisions.characteristics;
        let board_changed = characteristics_changed
            || (mana_changed && self.board.as_ref().is_some_and(|board| board.reads_mana))
            || (tapped_changed && self.board.as_ref().is_some_and(|board| board.reads_tapped));
        if board_changed
            || ((mana_changed || tapped_changed)
                && self
                    .board
                    .as_ref()
                    .is_some_and(|board| board.dynamic_activations))
        {
            self.characteristics.clear();
        }
        // Event kinds come from raw layered triggers. Activation prices and
        // mana outputs cannot change that index; matching still reads live LKI.
        if board_changed || self.revisions.triggers != revisions.triggers {
            self.trigger_index = None;
        }
        if board_changed {
            self.board = None;
        }
        self.revisions = revisions;
    }
}
impl Game {
    fn compute_board(&self, sources: Option<&mut Vec<Option<Arc<SourceSeed>>>>) -> LayerBoard {
        let context = Context::evaluate(self, sources);
        LayerBoard {
            reads_mana: context.reads_mana,
            reads_tapped: context.reads_tapped,
            dynamic_activations: context
                .objects
                .values()
                .any(|object| object.view.has_dynamic_activations()),
            objects: {
                let mut objects = vec![None; self.state.permanents.len()];
                for (id, object) in context.objects {
                    objects[id.0] = Some(object.view.clone());
                }
                objects
            },
        }
    }
    fn layer_board(&self) -> Arc<LayerBoard> {
        let revisions = self.state.revisions();
        let cached = self.derived.borrow_mut().at(revisions).board.clone();
        if let Some(board) = cached {
            debug_assert_eq!(*board, self.compute_board(None), "stale layer context");
            return board;
        }
        let mut sources = std::mem::take(&mut self.derived.borrow_mut().sources);
        let board = Arc::new(self.compute_board(Some(&mut sources)));
        self.derived.borrow_mut().sources = sources;
        debug_assert_eq!(*board, self.compute_board(None), "stale layer source plan");
        self.derived.borrow_mut().at(revisions).board = Some(board.clone());
        board
    }
    pub(super) fn layer_view(&self, id: PermanentId) -> Arc<Characteristics> {
        self.layer_board()
            .objects
            .get(id.0)
            .and_then(Clone::clone)
            .unwrap_or_default()
    }
    /// One immutable board read for a whole-board consumer. In debug this
    /// validates the entire board once, rather than once per permanent.
    pub(super) fn layer_views(
        &self,
        ids: impl Iterator<Item = PermanentId>,
    ) -> impl Iterator<Item = (PermanentId, Arc<Characteristics>)> {
        let board = self.layer_board();
        ids.filter_map(move |id| {
            board
                .objects
                .get(id.0)
                .and_then(Clone::clone)
                .map(|view| (id, view))
        })
    }
    pub fn characteristics(&self, id: PermanentId) -> Arc<Characteristics> {
        self.characteristics_from(id, || self.layer_view(id))
    }
    /// Read several final views at one immutable game state, preserving the
    /// caller's order. The shared raw board is validated once, and each cached
    /// finalized view still has its independent debug recomputation.
    pub(crate) fn characteristics_many<'a>(
        &'a self,
        ids: impl Iterator<Item = PermanentId> + 'a,
    ) -> impl Iterator<Item = (PermanentId, Arc<Characteristics>)> + 'a {
        self.layer_views(ids)
            .map(move |(id, raw)| (id, self.characteristics_from(id, || raw)))
    }
    fn characteristics_from(
        &self,
        id: PermanentId,
        raw: impl FnOnce() -> Arc<Characteristics>,
    ) -> Arc<Characteristics> {
        let revisions = self.state.revisions();
        let cached = self
            .derived
            .borrow_mut()
            .at(revisions)
            .characteristics
            .get(id.0)
            .and_then(Clone::clone);
        if let Some(view) = cached {
            debug_assert_eq!(
                view,
                self.finalize_characteristics(id, raw()),
                "stale characteristics"
            );
            return view;
        }
        let view = self.finalize_characteristics(id, raw());
        let mut derived = self.derived.borrow_mut();
        let cache = &mut derived.at(revisions).characteristics;
        cache.resize(self.state.permanents.len(), None);
        if let Some(slot) = cache.get_mut(id.0) {
            *slot = Some(view.clone());
        }
        view
    }
    fn finalize_characteristics(
        &self,
        id: PermanentId,
        mut view: Arc<Characteristics>,
    ) -> Arc<Characteristics> {
        // Only activations need finalization. Every other view is already
        // complete and can share the board's immutable allocation across reads.
        if view.activated_abilities.is_empty() {
            return view;
        }
        let Some(permanent) = self.state.permanents.get(id.0).and_then(Option::as_ref) else {
            return view;
        };
        let purpose = ManaUse::ability(view.subtypes.iter().map(String::as_str));
        // Most fixed activations already have an ordinary payment purpose.
        // Preserve the shared allocation when finalization would write nothing.
        if !view.has_dynamic_activations()
            && view
                .activated_abilities
                .iter()
                .all(|ability| ability.mana_cost.purpose == purpose)
        {
            return view;
        }

        for ability in &mut Arc::make_mut(&mut view).activated_abilities {
            if let [effect] = ability.effects.as_slice() {
                if let Some(effect) = self.current_mana_output(effect, permanent.controller) {
                    ability.effects = vec![effect];
                }
            }
            self.price_activation(permanent.controller, purpose, ability);
        }
        view
    }
    pub(crate) fn effective_pt(&self, id: PermanentId) -> (i32, i32) {
        self.layer_view(id).pt()
    }
    pub(super) fn layer_state_condition(
        &self,
        condition: &Condition,
        controller: PlayerId,
        source: PermanentId,
    ) -> bool {
        debug_assert!(!matches!(
            condition,
            Condition::ControlsAtLeast { .. } | Condition::AnyOf(_)
        ));
        self.condition_holds(
            condition,
            controller,
            None,
            self.permanent_object_ref(source),
            None,
        )
    }
    /// Power in the card's current zone (CR 604.3). On the stack "you"
    /// means the spell's controller; elsewhere off the battlefield it means
    /// the owner (CR 109.5), including owned outside-game copies.
    pub fn card_power(&self, card_id: CardId) -> i32 {
        if let Some(permanent) = self.state.card_to_permanent[card_id] {
            return self.characteristics(permanent).power;
        }
        let card = &self.state.cards[card_id];
        let Some(cda) = card.power_cda() else {
            return card.power.unwrap_or(0);
        };
        let controller = self.stack_spell_controller(card_id).unwrap_or(card.owner);
        self.cda_power(cda, controller)
    }

    /// Contextual predicates must read CDA power rather than a printed star's
    /// placeholder. Layer computation itself uses its explicit board context.
    pub(crate) fn card_matches_predicate(&self, card: CardId, predicate: &CardPredicate) -> bool {
        if let Some(permanent) = self.state.card_to_permanent[card] {
            return self.permanent_matches_predicate(permanent, predicate);
        }
        let power = if predicate.reads_power() {
            self.card_power(card)
        } else {
            0
        };
        predicate.matches_card_with_power(&self.state.cards[card], power)
    }

    fn cda_power(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .layer_views(self.iter_battlefield_permanents(controller))
                .filter(|(_, view)| view.is_land())
                .filter_map(|(id, _)| self.state.permanents[id].as_ref())
                .map(|p| p.counter_count(CounterKind::PlusOnePlusOne))
                .sum(),
            PowerCda::CreaturesYouControl => self
                .layer_views(self.iter_battlefield_permanents(controller))
                .filter(|(_, view)| view.is_creature())
                .count() as i32,
            PowerCda::GraveyardMatching(predicate) => {
                self.count_graveyard_matching(controller, predicate) as i32
            }
        }
    }

    pub(crate) fn static_sources(
        &self,
    ) -> impl Iterator<Item = (PermanentId, PlayerId, Arc<Characteristics>)> + '_ {
        self.layer_views((0..self.state.permanents.len()).map(PermanentId))
            .filter(|(_, view)| !view.static_abilities.is_empty())
            .map(|(id, view)| {
                (
                    id,
                    self.state.permanents[id].as_ref().unwrap().controller,
                    view,
                )
            })
    }
    pub(crate) fn spell_keywords(&self, card: CardId, controller: PlayerId) -> Keywords {
        let mut keywords = self.state.cards[card].keywords;
        for (_, owner, view) in self.static_sources() {
            for ability in &view.static_abilities {
                if let StaticAbility::KeywordGrant(grant) = ability {
                    if let StaticScope::SpellsYouControl(predicate) = &grant.scope {
                        if owner == controller
                            && self.card_matches_predicate(card, predicate)
                            && grant
                                .condition
                                .as_ref()
                                .is_none_or(|c| self.check_static_condition(c, controller))
                        {
                            keywords = keywords.union(grant.keywords);
                        }
                    }
                }
            }
        }
        keywords
    }
    pub(crate) fn static_scope_applies(
        &self,
        scope: &StaticScope,
        source: PermanentId,
        controller: PlayerId,
        target: PermanentId,
    ) -> bool {
        let Some(permanent) = self.state.permanents[target].as_ref() else {
            return false;
        };
        let predicate = match scope {
            StaticScope::This => return source == target,
            StaticScope::Attached => return self.attached_permanent(source) == Some(target),
            StaticScope::SpellsYouControl(_) => return false,
            StaticScope::YouControl(p) if permanent.controller == controller => p,
            StaticScope::OtherYouControl(p)
                if permanent.controller == controller && source != target =>
            {
                p
            }
            StaticScope::All(p) => p,
            StaticScope::Other(p) if source != target => p,
            _ => return false,
        };
        self.permanent_matches_with_source(target, predicate, Some(source))
    }

    /// Whether the permanent is attacking or blocking (CR 506.4).
    pub(crate) fn permanent_in_combat(&self, permanent_id: PermanentId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| permanent.attacking)
            || self.state.combat.as_ref().is_some_and(|combat| {
                combat
                    .attacker_to_blockers
                    .values()
                    .any(|blockers| blockers.contains(&permanent_id))
            })
    }

    /// The permanent an Aura or Equipment is attached to, if that exact
    /// object is still on the battlefield.
    pub(crate) fn attached_permanent(&self, source: PermanentId) -> Option<PermanentId> {
        let attached = self.state.permanents[source].as_ref()?.attached_to?;
        self.lookup_current_permanent(attached).ok()
    }

    /// The attached permanent, if it is still a creature.
    pub(crate) fn attached_creature(&self, source: PermanentId) -> Option<PermanentId> {
        self.attached_permanent(source)
            .filter(|&creature| self.layer_view(creature).is_creature())
    }

    pub(crate) fn has_restriction(&self, target: PermanentId, kind: Restriction) -> bool {
        self.static_sources().any(|(source, controller, view)| {
            view.static_abilities.iter().any(|ability| {
                let StaticAbility::Restriction(restriction) = ability else {
                    return false;
                };
                restriction.kind == kind
                    && restriction
                        .unless
                        .as_ref()
                        .is_none_or(|condition| !self.check_static_condition(condition, controller))
                    && self.static_scope_applies(&restriction.scope, source, controller, target)
            })
        })
    }
    pub(crate) fn greatest_shared_creature_type(&self, controller: PlayerId) -> usize {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (_, view) in self.layer_views(self.iter_battlefield_permanents(controller)) {
            for creature_type in &view.creature_types {
                *counts.entry(creature_type.clone()).or_default() += 1;
            }
        }
        counts.into_values().max().unwrap_or(0)
    }
    /// Instant timing for `player` casting `card`: printed instant or flash,
    /// or a conditional "as though it had flash" that currently holds.
    pub(crate) fn has_instant_timing(&self, player: PlayerId, card: CardId) -> bool {
        let granted_flash = self
            .exile_grant(player, card)
            .is_some_and(|grant| grant.flash);
        let card = &self.state.cards[card];
        granted_flash
            || card.is_instant_speed()
            || card
                .flash_if()
                .is_some_and(|condition| self.check_static_condition(condition, player))
    }

    /// Answer a game-state condition for `controller` ("you"): statics and
    /// `ActiveIf`. Resolution-only and source questions have no frame or
    /// source here and answer false; definition checks keep them out of
    /// these contexts.
    pub(crate) fn check_static_condition(
        &self,
        condition: &Condition,
        controller: PlayerId,
    ) -> bool {
        debug_assert!(
            !condition.is_resolution_only() && !condition.reads_source(),
            "resolution-only or source condition outside Effect::If: {condition:?}"
        );
        self.condition_holds(condition, controller, None, None, None)
    }

    /// Answer an intervening "if" (CR 603.4) of the ability of the exact
    /// battlefield object `source`, as its event occurs and as it resolves.
    /// `triggering` is the permanent whose event fired it.
    pub(crate) fn check_intervening_if(
        &self,
        condition: &Condition,
        controller: PlayerId,
        source: Option<ObjectRef>,
        triggering: Option<Target>,
    ) -> bool {
        debug_assert!(
            !condition.is_resolution_only(),
            "resolution-only condition outside Effect::If: {condition:?}"
        );
        self.condition_holds(condition, controller, None, source, triggering)
    }

    /// Answer `condition` for `controller`, reading `frame` for the
    /// resolution-only questions (CR 608.2c).
    pub(crate) fn check_condition(
        &self,
        condition: &Condition,
        controller: PlayerId,
        frame: Option<&EffectFrame>,
    ) -> bool {
        self.condition_holds(
            condition,
            controller,
            frame,
            frame.and_then(|frame| frame.source_ref),
            frame.and_then(|frame| frame.context_target),
        )
    }

    /// The source questions read `source`, the ability's exact object, which
    /// answers nothing once it has left the battlefield; the triggering
    /// questions read `triggering`, as it is or last existed.
    fn condition_holds(
        &self,
        condition: &Condition,
        controller: PlayerId,
        frame: Option<&EffectFrame>,
        source: Option<ObjectRef>,
        triggering: Option<Target>,
    ) -> bool {
        let source_permanent = || {
            source
                .and_then(|source| self.lookup_current_permanent(source).ok())
                .and_then(|id| self.state.permanents[id].as_ref())
        };
        match condition {
            Condition::Always => true,
            Condition::YourTurn => self.active_player() == controller,
            Condition::CardsDrawnAtLeast { count } => {
                self.state.turn.history[controller.0].cards_drawn >= *count
            }
            Condition::GraveyardAtLeast { count, predicate } => {
                self.count_graveyard_matching(controller, predicate) >= *count
            }
            Condition::ControlsAtLeast { count, predicate } => {
                let source = source.and_then(|source| self.lookup_current_permanent(source).ok());
                self.iter_battlefield_permanents(controller)
                    .filter(|&id| self.permanent_matches_with_source(id, predicate, source))
                    .take(*count)
                    .count()
                    == *count
            }
            Condition::UnspentManaAtLeast { count } => {
                u32::from(self.pooled_mana(controller).total()) >= *count
            }
            Condition::AttackedThisTurn => self.state.turn.history[controller.0].attacked,
            Condition::LandEnteredThisTurn => self.state.turn.history[controller.0].land_entered,
            Condition::BentAllFourThisTurn => self.state.turn.history[controller.0].bent == 0b1111,
            Condition::SacrificedThisTurnAtLeast { count } => {
                self.state.turn.history[controller.0].permanents_sacrificed >= *count
            }
            Condition::AnyOf(conditions) => conditions.iter().any(|condition| {
                self.condition_holds(condition, controller, frame, source, triggering)
            }),
            Condition::Kicked => frame.is_some_and(|frame| frame.kicked),
            Condition::TargetMatches(predicate) => frame
                .and_then(|frame| self.present_primary_target(frame))
                .is_some_and(|target| {
                    matches!(target, Target::Permanent(id)
                        if self.permanent_matches_predicate(id, predicate))
                }),
            Condition::SourceCountersAtLeast { kind, count } => {
                source_permanent().is_some_and(|permanent| permanent.counter_count(*kind) >= *count)
            }
            Condition::SourceMatches(predicate) => source
                .and_then(|r| self.lookup_current_permanent(r).ok())
                .is_some_and(|id| self.permanent_matches_predicate(id, predicate)),
            Condition::SourceTapped => source_permanent().is_some_and(|permanent| permanent.tapped),
            Condition::TriggeringHadCounters => match triggering {
                Some(Target::Permanent(id)) => match self.state.permanents[id].as_ref() {
                    Some(permanent) => permanent.has_counters(),
                    None => self
                        .departed_permanent_lki(id)
                        .is_some_and(|lki| !lki.counters.is_empty()),
                },
                _ => false,
            },
            Condition::Moved => frame.is_some_and(|frame| !frame.moved.is_empty()),
            Condition::TargetControlledByYou => {
                let Some(Target::Permanent(permanent_id)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                let last_controller = match self.state.permanents[permanent_id].as_ref() {
                    Some(permanent) => Some(permanent.controller),
                    None => self
                        .departed_permanent_lki(permanent_id)
                        .map(|lki| lki.controller),
                };
                last_controller == Some(controller)
            }
            Condition::DiscardedThisWay(predicate) => frame.is_some_and(|frame| {
                frame
                    .this_way
                    .iter()
                    .any(|card| self.card_matches_predicate(*card, predicate))
            }),
            Condition::TriggeringSpellMatches(predicate) => matches!(
                frame.and_then(|frame| frame.context_target),
                Some(Target::StackSpell(card)) if self.card_matches_predicate(card, predicate)
            ),
            Condition::TargetCardExiled(predicate) => {
                let Some(Target::GraveyardCard(object)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                let card = CardId::from(object.entity);
                self.current_object_ref(card) != Some(object)
                    && self.state.zones.zone_of(card) == Some(ZoneType::Exile)
                    && self.card_matches_predicate(card, predicate)
            }
            Condition::TargetDealtDamageThisTurn => {
                let Some(Target::Permanent(permanent_id)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                // A departed target answers from its last known information.
                match self.state.permanents[permanent_id].as_ref() {
                    Some(permanent) => permanent.dealt_damage_this_turn,
                    None => self
                        .departed_permanent_lki(permanent_id)
                        .is_some_and(|lki| lki.dealt_damage_this_turn),
                }
            }
        }
    }

    pub(crate) fn permanent_is_battlefield_creature(&self, id: PermanentId) -> bool {
        self.layer_view(id).is_creature()
    }
    pub(crate) fn permanent_matches_predicate(
        &self,
        id: PermanentId,
        predicate: &CardPredicate,
    ) -> bool {
        self.permanent_matches_with_source(id, predicate, None)
    }
    fn permanent_matches_with_source(
        &self,
        id: PermanentId,
        predicate: &CardPredicate,
        source: Option<PermanentId>,
    ) -> bool {
        let view = self.layer_view(id);
        let source = source.map(|id| self.layer_view(id));
        matches_view(
            self,
            id,
            predicate,
            &view,
            source.as_ref().map(|s| s.name.as_str()),
        )
    }
    /// Whether the permanent can pay a {T} cost or attack (CR 302.6): it is
    /// untapped, and a creature under its controller's control since the
    /// turn began or with haste, as its characteristics report it. A
    /// temporary grant must not erase the restriction that applies again when
    /// the grant expires (CR 702.10).
    pub(crate) fn permanent_can_tap(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if permanent.tapped {
            return false;
        }
        if !permanent.summoning_sick {
            return true;
        }
        let characteristics = self.characteristics(permanent_id);
        !characteristics.is_creature() || characteristics.keywords.has(Keyword::Haste)
    }

    /// Lethal-damage check (CR 704.5g) against the permanent's
    /// `characteristics`.
    pub(crate) fn has_lethal_damage(
        &self,
        permanent_id: PermanentId,
        characteristics: &Characteristics,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        if !characteristics.is_creature() {
            return false;
        }
        if permanent.deathtouch_damage && permanent.damage > 0 {
            return true;
        }
        permanent.damage >= characteristics.toughness
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::state::{
        ability::Effect,
        card::{LayerValue, StaticContinuous},
        effect::{EffectDuration, Modification},
        mana::Mana,
        player::PlayerConfig,
    };

    #[test]
    fn mana_revision_reuses_independent_board_but_reprices_dynamic_outputs() {
        let deck = [("Mountain".into(), 24), ("Gray Ogre".into(), 16)].into();
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck),
                PlayerConfig::new("b", [("Mountain".into(), 40)].into()),
            ],
            244,
            true,
        );
        game.play();
        let id = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .unwrap();
        let raw = game.layer_view(id);
        assert!(Arc::ptr_eq(&raw, &game.characteristics(id)));
        *game.state.mana_pool_mut(PlayerId(0)) = Mana::parse("R");
        assert!(Arc::ptr_eq(&raw, &game.characteristics(id)));
        let affected = game.affected_permanent(id).unwrap();
        game.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                effects: vec![Effect::If {
                    condition: Condition::UnspentManaAtLeast { count: 2 },
                    then: vec![Effect::add_mana(Mana::parse("RR"))],
                    otherwise: vec![Effect::add_mana(Mana::parse("R"))],
                }],
                ..Default::default()
            })),
            EffectDuration::Indefinite,
        );
        let board = game.layer_board();
        let before = game.characteristics(id);
        let mut branch = game.clone();
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("RR");
        let after = branch.characteristics(id);
        assert!(Arc::ptr_eq(&board, &branch.layer_board()));
        assert_ne!(before.activated_abilities, after.activated_abilities);
        assert_eq!(
            game.characteristics(id),
            before,
            "clone owns its mana and finalized view"
        );

        // A granted static whose nested condition reads mana must invalidate
        // layers, including when the condition was previously false.
        let affected = branch.affected_permanent(id).unwrap();
        branch.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::GrantStatic(Box::new(StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::This,
                condition: Some(Condition::AnyOf(vec![Condition::UnspentManaAtLeast {
                    count: 3,
                }])),
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(2),
                    toughness: LayerValue::Fixed(2),
                }],
            }))),
            EffectDuration::Indefinite,
        );
        assert_eq!(branch.characteristics(id).pt(), (2, 2));
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("RRR");
        assert_eq!(branch.characteristics(id).pt(), (4, 4));
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::default();
        assert_eq!(branch.characteristics(id).pt(), (2, 2));
    }

    #[test]
    fn tap_revision_tracks_nested_granted_predicates_and_stun_counters() {
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", [("Gray Ogre".into(), 40)].into()),
                PlayerConfig::new("b", [("Mountain".into(), 40)].into()),
            ],
            244,
            true,
        );
        game.play();
        let id = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .unwrap();
        let board = game.layer_board();
        let mut branch = game.clone();
        branch.state.tap(id);
        assert!(Arc::ptr_eq(&board, &branch.layer_board()));
        assert!(!game.state.permanents[id].as_ref().unwrap().tapped);

        let affected = branch.affected_permanent(id).unwrap();
        branch.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                effects: vec![Effect::add_mana(Mana::parse("R"))],
                ..Default::default()
            })),
            EffectDuration::Indefinite,
        );
        let fixed = branch.characteristics(id);
        branch.state.untap(id);
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("R");
        assert!(Arc::ptr_eq(&fixed, &branch.characteristics(id)));

        let affected = branch.affected_permanent(id).unwrap();
        branch.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                effects: vec![Effect::If {
                    condition: Condition::ControlsAtLeast {
                        count: 1,
                        predicate: CardPredicate {
                            tapped: true,
                            ..Default::default()
                        },
                    },
                    then: vec![Effect::add_mana(Mana::parse("RR"))],
                    otherwise: vec![Effect::add_mana(Mana::parse("R"))],
                }],
                ..Default::default()
            })),
            EffectDuration::Indefinite,
        );
        let before = branch.characteristics(id);
        let board = branch.layer_board();
        branch.state.tap(id);
        let after = branch.characteristics(id);
        assert!(Arc::ptr_eq(&board, &branch.layer_board()));
        assert_eq!(
            before.activated_abilities[1].effects,
            vec![Effect::add_mana(Mana::parse("R"))]
        );
        assert_eq!(
            after.activated_abilities[1].effects,
            vec![Effect::add_mana(Mana::parse("RR"))]
        );

        // Both a nested target predicate and a nested source condition must
        // invalidate, even when the granting node itself reads neither.
        for in_scope in [true, false] {
            let mut branch = game.clone();
            let tapped = CardPredicate {
                any_of: vec![CardPredicate {
                    tapped: true,
                    ..Default::default()
                }],
                ..Default::default()
            };
            let affected = branch.affected_permanent(id).unwrap();
            branch.apply_continuous(
                None,
                PlayerId(0),
                [affected],
                Modification::GrantStatic(Box::new(StaticAbility::Continuous(StaticContinuous {
                    scope: if in_scope {
                        StaticScope::YouControl(tapped.clone())
                    } else {
                        StaticScope::This
                    },
                    condition: (!in_scope)
                        .then(|| Condition::AnyOf(vec![Condition::SourceMatches(tapped)])),
                    characteristic_defining: false,
                    modifications: vec![Modification::ModifyPt {
                        power: LayerValue::Fixed(2),
                        toughness: LayerValue::Fixed(2),
                    }],
                }))),
                EffectDuration::Indefinite,
            );
            assert_eq!(branch.characteristics(id).pt(), (2, 2));
            branch.state.tap(id);
            assert_eq!(branch.characteristics(id).pt(), (4, 4));
            branch.state.untap(id);
            assert_eq!(branch.characteristics(id).pt(), (2, 2));
        }

        // Untapping through a stun counter keeps the tap state, but changes
        // counter-dependent characteristics and must still evict the board.
        let permanent = game.state.permanents[id].as_mut().unwrap();
        permanent.tapped = true;
        permanent.change_counters(CounterKind::Stun, 1);
        let board = game.layer_board();
        game.state.untap(id);
        assert!(game.state.permanents[id].as_ref().unwrap().tapped);
        assert_eq!(
            game.state.permanents[id]
                .as_ref()
                .unwrap()
                .counter_count(CounterKind::Stun),
            0
        );
        assert!(!Arc::ptr_eq(&board, &game.layer_board()));
    }
}
