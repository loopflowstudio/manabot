//! Derived characteristics and rules questions. `layers::Context` owns CR 613
//! evaluation; this module caches its immutable board and exposes the one public
//! view. Activation prices and dynamic mana outputs are finalized afterwards.
use crate::{
    flow::{
        decision::EffectFrame,
        game::{Game, Revisions},
        layers::{matches_view, Context, PredicateSource, SourceSeed},
        proposed_event::{EntryFace, ZoneEntry},
        triggers::TriggerIndex,
        turn::RecordedCast,
    },
    state::{
        ability::{Condition, EffectValue, PaymentCost},
        card::{
            ActivatedAbilityDefinition, CardTypes, Keyword, Keywords, PowerCda, Restriction,
            SpellProfile, StaticAbility, StaticScope,
        },
        game_object::{ObjectId, ObjectKind, PermanentId, PlayerId, Target},
        permanent::CounterKind,
        predicate::CardPredicate,
        subtype::{is_creature_type, CreatureTypes},
        zone::ZoneType,
    },
};
use std::{collections::BTreeMap, sync::Arc};
/// Stop as soon as the threshold is met. Ordinary counts never read a name.
pub(super) fn controlled_threshold<T, N: AsRef<str> + Ord>(
    matching: impl Iterator<Item = T>,
    count: usize,
    distinct_names: bool,
    name: impl FnMut(T) -> N,
) -> bool {
    if distinct_names {
        let mut names = std::collections::BTreeSet::new();
        matching
            .map(name)
            .filter(|n| !n.as_ref().is_empty())
            .filter_map(|n| names.insert(n).then_some(()))
            .take(count)
            .count()
            == count
    } else {
        matching.take(count).count() == count
    }
}

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
    /// Whether a layer 3 effect set its name.
    pub name_set: bool,
    /// Whether a layer 4 effect replaced its creature types (Kellan's
    /// upgrades, a Citizen, a Skeleton) or removed them with Creature. A
    /// later grant of all creature types clears it, so it never accompanies
    /// `CreatureTypes::All`.
    pub creature_types_set: bool,
    /// Every subtype it has: printed or set, then added by records.
    pub subtypes: Vec<String>,
    /// Subtypes added on top of its printed or set ones (layer 4), in record
    /// order; creature types a record replaced are in `creature_types`.
    pub added_subtypes: Vec<String>,
    /// Its creature types (CR 205.3m); none unless it is a creature.
    pub creature_types: CreatureTypes,
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
            ability.power_up
                || ability.reduce_by_source_power
                || ability.cost_reduction_per.is_some()
                || ability.has_dynamic_mana_output()
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

    /// Whether it has every creature type. Mid-pass, `CreatureTypes::All`
    /// only marks the grant, which counts for a creature alone (CR 205.3d).
    pub fn has_all_creature_types(&self) -> bool {
        self.is_creature() && self.creature_types.is_all()
    }

    /// Whether a summoning-sick permanent with these characteristics can't
    /// attack or pay {T} (CR 302.6): it is a creature without haste.
    pub fn restricted_by_summoning_sickness(&self) -> bool {
        self.is_creature() && !self.keywords.has(Keyword::Haste)
    }

    /// Printed or granted land.
    pub fn is_land(&self) -> bool {
        self.types.is_land()
    }

    /// Whether it has this subtype, counting all creature types.
    pub fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
            || (self.has_all_creature_types() && is_creature_type(subtype))
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
    /// Live `BoardScan`s of this game; a fork starts with none.
    #[cfg(debug_assertions)]
    scans: ScanCount,
}

/// Clones to zero: a `BoardScan` borrows one game and says nothing of its forks.
#[cfg(debug_assertions)]
#[derive(Default)]
struct ScanCount(u32);
#[cfg(debug_assertions)]
impl Clone for ScanCount {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// One immutable scan of the game. Debug recomputes the cached board on read to
/// catch a write that advanced no revision. The guard's shared borrow excludes
/// every such write while it lives, so the board is validated when the scan
/// begins and its reads skip the repeat. Scans that read the board per source or
/// per ability would otherwise rebuild it hundreds of times per decision.
pub(crate) struct BoardScan<'a> {
    #[cfg(debug_assertions)]
    game: &'a Game,
    #[cfg(not(debug_assertions))]
    game: std::marker::PhantomData<&'a Game>,
}
#[cfg(debug_assertions)]
impl Drop for BoardScan<'_> {
    fn drop(&mut self) {
        self.game.derived.borrow_mut().scans.0 -= 1;
    }
}

/// Permissions granted by this player's active permanents. Priority checks these
/// for each candidate card and face during the same immutable board scan.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct StaticPlayPermissions {
    pub look_at_top: bool,
    pub free_from_hand: bool,
    pub opponents_hand_only: bool,
    pub graveyard_flashback: bool,
    pub library_spells: Vec<CardPredicate>,
    pub flash_spells: Vec<CardPredicate>,
    /// Each permanent letting this player play a permanent spell or land of
    /// each permanent type from their graveyard once per turn (Muldrotha):
    /// its object and public id.
    pub graveyard_type_sources: Vec<(ObjectId, crate::state::game_object::PublicId)>,
    /// Counter kinds marking exiled cards this player doesn't own that they
    /// may play during their turn, spending mana of any type (Tinybones).
    pub exile_counter_plays: Vec<CounterKind>,
    /// Land plays beyond the first this turn (CR 305.2) granted by
    /// permanents: the seat's own grants plus each "each player" grant.
    pub extra_land_plays: u32,
}

#[derive(Debug, PartialEq)]
pub(super) struct LayerBoard {
    objects: Vec<Option<Arc<Characteristics>>>,
    /// Layered permissions for both seats, computed during the board's scan.
    pub play_permissions: [StaticPlayPermissions; 2],
    restrictions: Vec<(PermanentId, PlayerId, crate::state::card::StaticRestriction)>,
    pub(super) land_type_overwrites: Vec<CardPredicate>,
    reads_mana: bool,
    reads_tapped: bool,
    reads_life: bool,
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
        let life_changed = self.revisions.life != revisions.life;
        #[cfg(feature = "benchmark")]
        if mana_changed {
            crate::benchmark_layers::record(|m| m.mana_invalidations += 1);
        }
        let characteristics_changed = self.revisions.characteristics != revisions.characteristics;
        let board_changed = characteristics_changed
            || (mana_changed && self.board.as_ref().is_some_and(|board| board.reads_mana))
            || (tapped_changed && self.board.as_ref().is_some_and(|board| board.reads_tapped))
            || (life_changed && self.board.as_ref().is_some_and(|board| board.reads_life));
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
        let mut objects = vec![None; self.state.permanents.len()];
        let mut restrictions = Vec::new();
        let mut land_plays_each_player = 0;
        let mut play_permissions: [StaticPlayPermissions; 2] = Default::default();
        let mut dynamic_activations = false;
        for (id, object) in context.objects {
            dynamic_activations = dynamic_activations || object.view.has_dynamic_activations();
            if !object.view.static_abilities.is_empty() && !self.is_phased_out(id) {
                let permanent = self.state.permanents[id].as_ref().unwrap();
                let controller = permanent.controller;
                let permissions = &mut play_permissions[controller.0];
                for ability in &object.view.static_abilities {
                    match ability {
                        StaticAbility::Restriction(rule) => {
                            restrictions.push((id, controller, rule.clone()))
                        }
                        StaticAbility::AdditionalLandPlayEachPlayer => land_plays_each_player += 1,
                        StaticAbility::AdditionalLandPlay => permissions.extra_land_plays += 1,
                        StaticAbility::PlayUnownedExiledWithCounter(kind) => {
                            if !permissions.exile_counter_plays.contains(kind) {
                                permissions.exile_counter_plays.push(*kind)
                            }
                        }
                        StaticAbility::PlayGraveyardByPermanentType => permissions
                            .graveyard_type_sources
                            .push((permanent.object, permanent.id)),
                        StaticAbility::LookAtLibraryTop => permissions.look_at_top = true,
                        StaticAbility::CastFromHandWithoutManaCost => {
                            permissions.free_from_hand = true
                        }
                        StaticAbility::OpponentsCastOnlyFromHand => {
                            permissions.opponents_hand_only = true
                        }
                        StaticAbility::GraveyardSpellsHaveFlashback => {
                            permissions.graveyard_flashback = true
                        }
                        StaticAbility::CastFromLibraryTop(predicate) => {
                            permissions.library_spells.push(predicate.clone())
                        }
                        StaticAbility::CastSpellsAsThoughFlash(predicate) => {
                            permissions.flash_spells.push(predicate.clone())
                        }
                        _ => {}
                    }
                }
            }
            objects[id.0] = Some(object.view.clone());
        }
        for permissions in &mut play_permissions {
            permissions.extra_land_plays += land_plays_each_player;
        }
        LayerBoard {
            reads_mana: context.reads_mana,
            reads_tapped: context.reads_tapped,
            reads_life: context.reads_life,
            dynamic_activations,
            restrictions,
            land_type_overwrites: context.land_type_overwrites,
            objects,
            play_permissions,
        }
    }

    /// Both seats share the same layered board and its invalidation rules.
    pub(crate) fn visible_library_tops(&self) -> [bool; 2] {
        self.layer_board()
            .play_permissions
            .each_ref()
            .map(|p| p.look_at_top)
    }

    /// Whether `player` may look at the top card of their library (CR 401.2).
    pub(crate) fn looks_at_library_top(&self, player: PlayerId) -> bool {
        self.visible_library_tops()[player.0]
    }
    /// Begin an immutable scan; see `BoardScan`. Release builds do nothing.
    pub(crate) fn board_scan(&self) -> BoardScan<'_> {
        #[cfg(debug_assertions)]
        {
            self.layer_board();
            self.derived.borrow_mut().scans.0 += 1;
            BoardScan { game: self }
        }
        #[cfg(not(debug_assertions))]
        BoardScan {
            game: std::marker::PhantomData,
        }
    }
    pub(super) fn layer_board(&self) -> Arc<LayerBoard> {
        let revisions = self.state.revisions();
        let cached = self.derived.borrow_mut().at(revisions).board.clone();
        if let Some(board) = cached {
            #[cfg(debug_assertions)]
            if self.derived.borrow().scans.0 == 0 {
                assert_eq!(*board, self.compute_board(None), "stale layer context");
            }
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
        let reduction = self.activated_cost_reduction(id, permanent.controller, &view);
        // Preserve the shared allocation when finalization would write nothing.
        if reduction == 0 && !view.has_dynamic_activations() {
            return view;
        }

        let power = view.power.clamp(0, u8::MAX.into()) as u8;
        for ability in &mut Arc::make_mut(&mut view).activated_abilities {
            if ability.has_dynamic_mana_output() {
                if let Some(output) = self.current_mana_output(
                    &ability.effects[0],
                    permanent.controller,
                    permanent.chosen.as_deref(),
                ) {
                    ability.effects[0] = output;
                }
            }
            if ability.power_up && permanent.entered_this_turn(self.state.turn.turn_number) {
                if let Some(cost) = &self.state[permanent.object].mana_cost {
                    ability.mana_cost = ability.mana_cost.reduced_by(cost);
                }
            }
            if ability.reduce_by_source_power {
                ability.mana_cost = ability.mana_cost.reduced_generic(power);
            }
            self.price_activation(permanent.controller, ability);
            let floor = ability.mana_cost.mana_value.saturating_sub(1);
            ability.mana_cost = ability.mana_cost.reduced_generic(reduction.min(floor));
        }
        view
    }

    /// Generic mana the controller's "activated abilities of [predicate] you
    /// control cost less" statics take off each activation of `id`.
    fn activated_cost_reduction(
        &self,
        id: PermanentId,
        controller: PlayerId,
        view: &Characteristics,
    ) -> u8 {
        let mut total = 0u8;
        for (_, source) in self.layer_views(self.iter_battlefield_permanents(controller)) {
            for ability in &source.static_abilities {
                if let StaticAbility::ActivatedAbilitiesCostLess { predicate, generic } = ability {
                    if matches_view(self, id, predicate, view, None) {
                        total = total.saturating_add(*generic);
                    }
                }
            }
        }
        total
    }
    pub(super) fn layer_state_condition(
        &self,
        condition: &Condition,
        controller: PlayerId,
        source: PermanentId,
    ) -> bool {
        debug_assert!(!matches!(
            condition,
            Condition::Compare { .. }
                | Condition::ControlsAtLeast { .. }
                | Condition::BattlefieldAtLeast { .. }
                | Condition::AllOf(_)
                | Condition::AnyOf(_)
                | Condition::Not(_)
        ));
        self.condition_holds(
            condition,
            controller,
            None,
            self.permanent_object(source),
            None,
        )
    }
    /// Power and toughness in the object's current zone (CR 604.3): a
    /// characteristic-defining ability applies everywhere. On the stack
    /// "you" means the spell's controller; elsewhere off the battlefield it
    /// means the owner (CR 109.5), including owned outside-game copies.
    pub fn object_pt(&self, object_id: ObjectId) -> (i32, i32) {
        if let Some(permanent) = self.state.permanent_of(object_id) {
            return self.layer_view(permanent).pt();
        }
        // An outside-game card is allocated but in no zone.
        let object = self
            .state
            .object(object_id)
            .unwrap_or_else(|_| self.state.entry_object(object_id));
        let printed = (object.power.unwrap_or(0), object.toughness.unwrap_or(0));
        let Some((cda, sets_toughness)) = object.pt_cda() else {
            return printed;
        };
        let controller = self
            .spell_on_stack(object_id)
            .map(|spell| spell.controller)
            .unwrap_or(object.owner);
        let value = self.cda_value(cda, controller);
        (value, if sets_toughness { value } else { printed.1 })
    }

    /// Contextual predicates must read CDA power rather than a printed star's
    /// placeholder. Layer computation itself uses its explicit board context.
    pub(crate) fn object_matches_predicate(
        &self,
        object_id: ObjectId,
        predicate: &CardPredicate,
    ) -> bool {
        self.object_matches_with_source(object_id, predicate, None)
    }

    pub(super) fn object_matches_with_source(
        &self,
        object_id: ObjectId,
        predicate: &CardPredicate,
        source: Option<PermanentId>,
    ) -> bool {
        if let Some(permanent) = self.state.permanent_of(object_id) {
            return self.permanent_matches_with_source(permanent, predicate, source);
        }
        self.off_battlefield_matches_choice(
            object_id,
            predicate,
            || self.object_pt(object_id).0,
            source.and_then(|s| self.live_permanent(s).chosen.as_deref()),
        )
    }

    /// A layer calculation supplies the card's CDA power from its partial board.
    pub(super) fn off_battlefield_matches(
        &self,
        object_id: ObjectId,
        predicate: &CardPredicate,
        power: impl FnOnce() -> i32,
    ) -> bool {
        self.off_battlefield_matches_choice(object_id, predicate, power, None)
    }

    /// Check graveyard-arrival flags and return `predicate` with them cleared
    /// for printing-level matching, or `None` if the object fails them.
    fn strip_graveyard_arrival<'a>(
        &self,
        object: ObjectId,
        predicate: &'a CardPredicate,
    ) -> Option<std::borrow::Cow<'a, CardPredicate>> {
        if !predicate.from_battlefield_this_turn && !predicate.put_into_graveyard_this_turn {
            return Some(std::borrow::Cow::Borrowed(predicate));
        }
        if self.state.zones.zone_of(object) != Some(ZoneType::Graveyard)
            || (predicate.from_battlefield_this_turn
                && !self.state.turn.in_graveyard_from_battlefield(object))
            || (predicate.put_into_graveyard_this_turn
                && !self.state.turn.in_graveyard_this_turn(object))
        {
            return None;
        }
        Some(std::borrow::Cow::Owned(CardPredicate {
            from_battlefield_this_turn: false,
            put_into_graveyard_this_turn: false,
            ..predicate.clone()
        }))
    }

    fn off_battlefield_matches_choice(
        &self,
        object_id: ObjectId,
        predicate: &CardPredicate,
        power: impl FnOnce() -> i32,
        chosen: Option<&crate::state::permanent::EntryChoice>,
    ) -> bool {
        let Some(predicate) = self.strip_graveyard_arrival(object_id, predicate) else {
            return false;
        };
        let object = &self.state[object_id];
        predicate.matches_printing_with_choice(
            object,
            object.kind(),
            self.object_mana_value(object_id),
            if predicate.reads_power() { power() } else { 0 },
            &[],
            chosen,
        )
    }

    pub(crate) fn lki_matches_predicate(
        &self,
        lki: &crate::state::game_object::ObjectLki,
        predicate: &CardPredicate,
    ) -> bool {
        super::layers::matches_departure(lki, predicate)
    }

    /// A recorded event can still ask what its departed card was. Read the
    /// retained base and LKI, never the later appearance occupying its slot.
    pub(crate) fn recorded_object_matches_predicate(
        &self,
        object: ObjectId,
        predicate: &CardPredicate,
    ) -> bool {
        if self.state.live(object).is_some() {
            return self.object_matches_predicate(object, predicate);
        }
        if let Some(lki) = self.object_lki(object) {
            return super::layers::matches_departure(lki, predicate);
        }
        self.printing_matches_predicate(object, predicate)
    }

    /// Match the printing an object's card retains, not its current
    /// appearance: a face-down manifest's hidden card, or a departed object
    /// without last known information. Sampling reads a hidden slot this way.
    pub(crate) fn printing_matches_predicate(
        &self,
        object: ObjectId,
        predicate: &CardPredicate,
    ) -> bool {
        let (_, printing) = self
            .state
            .objects
            .retained_base(object)
            .expect("retained printing");
        let (_, owner, kind) = self
            .state
            .objects
            .provenance(object)
            .expect("recorded provenance");
        let power = if predicate.reads_power() {
            self.printed_power(printing, owner)
        } else {
            0
        };
        predicate.matches_printing(
            printing,
            kind,
            printing
                .mana_cost
                .as_ref()
                .map_or(0, |cost| cost.mana_value),
            power,
            &[],
        )
    }

    /// A cast this turn asks what was cast: while the spell is on the stack
    /// its appearance answers; once it departs, the profile it was cast as
    /// selects the printed spell from the card's retained base, never the
    /// front face the card shows again (an Adventure that resolved).
    pub(crate) fn recorded_cast_matches_predicate(
        &self,
        cast: RecordedCast,
        predicate: &CardPredicate,
    ) -> bool {
        self.recorded_spell_matches_predicate(cast.object, cast.profile, predicate)
    }

    /// Live spells already show their chosen printing. After departure, read
    /// the profile of that exact stack incarnation from its last known information.
    pub(super) fn spell_matches_predicate(
        &self,
        object: ObjectId,
        predicate: &CardPredicate,
    ) -> bool {
        let profile = self
            .state
            .spell_lki
            .get(&object)
            .map_or(SpellProfile::Front, |spell| spell.profile);
        self.recorded_spell_matches_predicate(object, profile, predicate)
    }

    /// The printed spell `object` was cast as (front face, inset Adventure or
    /// Omen, or back face) from its retained base.
    pub(crate) fn recorded_spell_printing(
        &self,
        object: ObjectId,
        profile: SpellProfile,
    ) -> &crate::state::card::PrintedCharacteristics {
        let (definition, printing) = self.state.source_base(object);
        crate::cards::content_pack::default_content_pack()
            .recorded_spell(definition, printing, profile)
    }

    fn recorded_spell_matches_predicate(
        &self,
        object: ObjectId,
        profile: SpellProfile,
        predicate: &CardPredicate,
    ) -> bool {
        if profile.is_front() || self.state.live(object).is_some() {
            return self.recorded_object_matches_predicate(object, predicate);
        }
        let (_, _, kind) = self
            .state
            .objects
            .provenance(object)
            .expect("recorded provenance");
        predicate.matches_printed_spell(self.recorded_spell_printing(object, profile), kind)
    }

    /// Match the face entering under its prospective controller (CR 614.12).
    /// A resolving permanent spell copy enters as a token (CR 608.3f).
    pub(super) fn entry_matches_predicate(
        &self,
        object: ObjectId,
        entry: &ZoneEntry,
        predicate: &CardPredicate,
    ) -> bool {
        let Some(predicate) = self.strip_graveyard_arrival(object, predicate) else {
            return false;
        };
        let Some(face) = self.battlefield_entry_face(object, entry) else {
            return false;
        };
        let row = self.state.entry_object(object);
        let power = if predicate.reads_power() {
            self.printed_power(face, entry.controller.unwrap_or(row.owner))
        } else {
            0
        };
        // A manifested permanent is a face-down 2/2 with mana value 0
        // (CR 708.2), whatever card it hides.
        let mana_value = match entry.face {
            EntryFace::Manifested => 0,
            EntryFace::Melded { partner, .. } => self.meld_mana_value(
                self.state.card_of(object).expect("melded primary card"),
                partner,
            ),
            EntryFace::Transformed if !row.retained_base().1.modal_back => row
                .retained_base()
                .1
                .mana_cost
                .as_ref()
                .map_or(0, |cost| cost.mana_value),
            EntryFace::Front | EntryFace::ModalBack | EntryFace::Transformed => {
                face.mana_cost.as_ref().map_or(0, |cost| cost.mana_value)
            }
        };
        let kind = if row.kind() == ObjectKind::SpellCopy {
            ObjectKind::Token
        } else {
            row.kind()
        };
        predicate.matches_printing(face, kind, mana_value, power, &[])
    }

    fn creature_count(&self, controller: PlayerId) -> usize {
        self.layer_views(self.iter_battlefield_permanents(controller))
            .filter(|(_, view)| view.is_creature())
            .count()
    }

    /// Power as printed, with a characteristic-defining ability evaluated
    /// for `you` (CR 604.3) instead of the printed star's placeholder.
    fn printed_power(
        &self,
        printing: &crate::state::card::PrintedCharacteristics,
        you: PlayerId,
    ) -> i32 {
        printing.pt_cda().map_or_else(
            || printing.power.unwrap_or(0),
            |(cda, _)| self.cda_value(cda, you),
        )
    }

    fn cda_value(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .layer_views(self.iter_battlefield_permanents(controller))
                .filter(|(_, view)| view.is_land())
                .filter_map(|(id, _)| self.state.permanents[id].as_ref())
                .map(|p| p.counter_count(CounterKind::PlusOnePlusOne))
                .sum(),
            PowerCda::CreaturesYouControl => self.creature_count(controller) as i32,
            PowerCda::GraveyardMatching(predicate) => {
                self.count_graveyard_matching(controller, predicate) as i32
            }
            PowerCda::GraveyardMatchingBy { who, predicate } => (0..self.state.players.len())
                .map(PlayerId)
                .filter(|player| who.includes(controller, *player))
                .map(|player| self.count_graveyard_matching(player, predicate) as i32)
                .sum(),
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
    pub(crate) fn spell_keywords(&self, object: ObjectId, controller: PlayerId) -> Keywords {
        let mut keywords = self.state[object].keywords;
        // Mana spent on the spell can give it a keyword (CR 106.6).
        if let Some(spell) = self.spell_on_stack(object) {
            keywords = keywords.union(spell.spent.keywords_gained());
        }
        for (_, owner, view) in self.static_sources() {
            for ability in &view.static_abilities {
                if let StaticAbility::KeywordGrant(grant) = ability {
                    if let StaticScope::SpellsYouControl(predicate) = &grant.scope {
                        if owner == controller
                            && self.object_matches_predicate(object, predicate)
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
        super::layers::matches_static_scope(
            scope,
            source,
            controller,
            self.attached_permanent(source),
            target,
            permanent.controller,
            |predicate| self.permanent_matches_with_source(target, predicate, Some(source)),
        )
    }

    /// Read a permanent while it remains on the battlefield, or from its last
    /// known information after departure (CR 608.2h).
    pub(crate) fn with_permanent_or_lki<T>(
        &self,
        id: PermanentId,
        live: impl FnOnce(&crate::state::permanent::Permanent) -> T,
        departed: impl FnOnce(&crate::state::game_object::ObjectLki) -> T,
    ) -> Option<T> {
        match self.state.permanents[id].as_ref() {
            Some(permanent) => Some(live(permanent)),
            None => self.departed_permanent_lki(id).map(departed),
        }
    }

    /// The permanent's current controller, or its last known controller after departure.
    pub(crate) fn last_known_controller(&self, id: PermanentId) -> Option<PlayerId> {
        self.with_permanent_or_lki(id, |p| p.controller, |lki| lki.controller)
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

    /// The power of the creature `source` is attached to, at least 0; 0 when
    /// it is attached to no creature (`EffectValue::AttachedCreaturePower`).
    pub(crate) fn attached_creature_power(&self, source: PermanentId) -> i32 {
        self.attached_creature(source)
            .map_or(0, |creature| self.layer_view(creature).power.max(0))
    }

    /// `value` as a static ability of `source` reads it: values relative to
    /// that permanent, else as its controller's quantity.
    pub(crate) fn static_value(
        &self,
        value: &EffectValue,
        source: PermanentId,
        controller: PlayerId,
    ) -> i32 {
        match value {
            EffectValue::AttachedCreaturePower => self.attached_creature_power(source),
            value => self.controller_value(value, controller),
        }
    }

    pub(crate) fn has_restriction(&self, target: PermanentId, kind: Restriction) -> bool {
        self.layer_board()
            .restrictions
            .iter()
            .any(|(source, controller, rule)| {
                rule.kind == kind
                    && rule.unless.as_ref().is_none_or(|condition| {
                        !self.check_static_condition(condition, *controller)
                    })
                    && self.static_scope_applies(&rule.scope, *source, *controller, target)
            })
    }

    /// Static prohibitions follow the source's current name in its zone.
    pub(super) fn nonmana_activation_prohibited(&self, object: ObjectId) -> bool {
        if let Some(id) = self.state.permanent_of(object) {
            return self.has_restriction(id, Restriction::CantActivateNonmanaAbilities);
        }
        self.layer_board().restrictions.iter().any(|(source, controller, rule)| {
            rule.kind == Restriction::CantActivateNonmanaAbilities &&
            rule.unless.as_ref().is_none_or(|c| !self.check_static_condition(c, *controller)) &&
            matches!(&rule.scope, StaticScope::All(p) if self.object_matches_with_source(object,p,Some(*source)))
        })
    }

    /// "attacks each combat if able" (CR 508.1d): the declaration offers it
    /// no choice to stay home.
    pub(crate) fn must_attack(&self, attacker: PermanentId) -> bool {
        self.has_restriction(attacker, Restriction::AttacksEachCombat)
    }

    pub(crate) fn greatest_shared_creature_type(&self, controller: PlayerId) -> usize {
        // A creature with all creature types shares each listed one.
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut all = 0;
        for (_, view) in self.layer_views(self.iter_battlefield_permanents(controller)) {
            match &view.creature_types {
                CreatureTypes::All => all += 1,
                CreatureTypes::Listed(listed) => {
                    for creature_type in listed {
                        *counts.entry(creature_type.clone()).or_default() += 1;
                    }
                }
            }
        }
        counts.into_values().max().unwrap_or(0) + all
    }
    /// Instant timing for `player` casting `card`: printed instant or flash,
    /// or a conditional "as though it had flash" that currently holds.
    pub(crate) fn has_instant_timing(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: crate::flow::game::CastChoices,
    ) -> bool {
        let granted_flash = cast.grant(self, object).is_some_and(|grant| grant.flash);
        let spell = cast.spell(&self.state[object]);
        cast.flash_surcharge
            || granted_flash
            || spell.is_instant_speed()
            || self.layer_board().play_permissions[player.0]
                .flash_spells
                .iter()
                .any(|predicate| cast.matches_spell(self, object, predicate))
            || (spell.types.is_creature() && self.has_next_creature_cast_flash(player))
            || spell
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
        source: Option<ObjectId>,
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
            frame.and_then(|frame| frame.source_ref()),
            frame.and_then(|frame| frame.context_target),
        )
    }

    /// The source questions read `source`, the ability's exact object, which
    /// answers nothing once it has left the battlefield; the triggering
    /// questions read `triggering`, as it is or last existed.
    pub(crate) fn condition_holds(
        &self,
        condition: &Condition,
        controller: PlayerId,
        frame: Option<&EffectFrame>,
        source: Option<ObjectId>,
        triggering: Option<Target>,
    ) -> bool {
        let source_id = || source.and_then(|source| self.lookup_current_permanent(source).ok());
        let source_permanent = || source_id().and_then(|id| self.state.permanents[id].as_ref());
        match condition {
            Condition::OwnsMeldPair { partner } => source.is_some_and(|object| {
                self.meld_partners(controller, object, partner)
                    .next()
                    .is_some()
            }),
            Condition::AtMostHalfStartingLife => {
                let player = &self.state.players[controller.0];
                player.life <= player.starting_life / 2
            }
            Condition::Compare { left, op, right } => {
                let context;
                let frame = match frame {
                    Some(frame) => frame,
                    None => {
                        context = EffectFrame {
                            source_identity: crate::state::game_object::AbilitySource::optional(
                                source, source,
                            ),
                            context_target: triggering,
                            ..EffectFrame::new(controller, Vec::new())
                        };
                        &context
                    }
                };
                op.test(
                    self.effect_value(left, frame),
                    self.effect_value(right, frame),
                )
            }
            Condition::Always => true,
            Condition::YourTurn => self.active_player() == controller,
            Condition::CardsDrawnAtLeast { count } => {
                self.state.turn.history[controller.0].cards_drawn >= *count
            }
            Condition::GraveyardAtLeast { count, predicate } => {
                self.count_graveyard_matching(controller, predicate) >= *count
            }
            Condition::AnyGraveyardAtLeast { count } => [PlayerId(0), PlayerId(1)]
                .into_iter()
                .any(|player| self.state.zones.size(ZoneType::Graveyard, player) >= *count),
            Condition::ControlsAtLeast {
                count,
                predicate,
                distinct_names,
            } => {
                let source_view = self.predicate_source_view(source_id(), predicate);
                let predicate_source = self.predicate_source(source_view.as_ref());
                let matching = self
                    .layer_views(self.iter_battlefield_permanents(controller))
                    .filter(|(id, view)| {
                        matches_view(self, *id, predicate, view, predicate_source)
                    });
                controlled_threshold(matching, *count, *distinct_names, |(_, view)| {
                    view.name.clone()
                })
            }
            Condition::OpponentControlsMoreCreatures => {
                self.creature_count(PlayerId(1 - controller.0)) > self.creature_count(controller)
            }
            Condition::BattlefieldAtLeast { count, predicate } => {
                let source_view = self.predicate_source_view(source_id(), predicate);
                let predicate_source = self.predicate_source(source_view.as_ref());
                self.layer_views(
                    [PlayerId(0), PlayerId(1)]
                        .into_iter()
                        .flat_map(|player| self.iter_battlefield_permanents(player)),
                )
                .filter(|(id, view)| matches_view(self, *id, predicate, view, predicate_source))
                .take(*count)
                .count()
                    == *count
            }
            Condition::ControlsUnownedAtLeast { count } => {
                self.iter_battlefield_permanents(controller)
                    .filter(|&id| {
                        self.state.permanents[id].as_ref().is_some_and(|permanent| {
                            self.state[permanent.object].owner != controller
                        })
                    })
                    .take(*count)
                    .count()
                    == *count
            }
            Condition::OpponentHasMoreCardsInHand => {
                self.state
                    .zones
                    .size(ZoneType::Hand, PlayerId(1 - controller.0))
                    > self.state.zones.size(ZoneType::Hand, controller)
            }
            Condition::CardsInHandAtLeast { count } => {
                self.state.zones.size(ZoneType::Hand, controller) >= *count
            }
            Condition::LifeAtLeast { amount } => self.state.players[controller.0].life >= *amount,
            Condition::UnspentManaAtLeast { count } => {
                u32::from(self.pooled_mana(controller).total()) >= *count
            }
            Condition::AttackedThisTurn => self.state.turn.history[controller.0].attacked,
            Condition::CastThisTurn(predicate) => self.state.turn.spells_cast[controller.0]
                .iter()
                .any(|&cast| self.recorded_cast_matches_predicate(cast, predicate)),
            Condition::LandEnteredThisTurn => self.state.turn.history[controller.0].land_entered,
            Condition::PermanentLeftThisTurn { who, nonland } => {
                self.state.turn.any_history(*who, controller, |history| {
                    if *nonland {
                        history.nonland_permanents_left > 0
                    } else {
                        history.permanents_left > 0
                    }
                })
            }
            Condition::WarpedThisTurn { who } => {
                self.state
                    .turn
                    .any_history(*who, controller, |history| history.warped_this_turn)
            }
            Condition::BentAllFourThisTurn => self.state.turn.history[controller.0].bent == 0b1111,
            Condition::SacrificedThisTurnAtLeast { count } => {
                self.state.turn.history[controller.0].permanents_sacrificed >= *count
            }
            // Two players: the one opponent, as `PlayerSelector::EachOpponent`.
            Condition::OpponentLostLifeThisTurnAtLeast { amount } => {
                self.state.turn.history[1 - controller.0].life_lost >= *amount
            }
            Condition::CreatureDiedThisTurn => self.state.turn.creature_died_this_turn(),
            Condition::AllOf(conditions) => conditions.iter().all(|condition| {
                self.condition_holds(condition, controller, frame, source, triggering)
            }),
            Condition::AnyOf(conditions) => conditions.iter().any(|condition| {
                self.condition_holds(condition, controller, frame, source, triggering)
            }),
            Condition::Not(condition) => {
                !self.condition_holds(condition, controller, frame, source, triggering)
            }
            Condition::CastFromGraveyard => source
                .and_then(|object| self.spell_on_stack(object))
                .is_some_and(|spell| self.was_cast_from_graveyard(spell)),
            Condition::Kicked => frame.map_or_else(
                || {
                    source
                        .and_then(|object| self.state.spell_entry(object))
                        .is_some_and(|entry| entry.kicked)
                },
                |frame| frame.kicked,
            ),
            Condition::TargetMatches(predicate) => frame
                .and_then(|frame| self.present_primary_target(frame))
                .is_some_and(|target| {
                    matches!(target, Target::Permanent(id)
                        if self.permanent_matches_predicate(id, predicate))
                }),
            // A departed source answers from its last known information
            // (CR 603.10a; Nine-Lives Familiar's intervening-if).
            Condition::SourceCountersAtLeast { kind, count } => source_permanent()
                .map(|permanent| permanent.counter_count(*kind))
                .or_else(|| {
                    source
                        .and_then(|object| self.object_lki(object))
                        .map(|lki| lki.counters.get(*kind))
                })
                .is_some_and(|n| n >= *count),
            Condition::SourceMatches(predicate) => {
                source.is_some_and(|object| match self.lookup_current_permanent(object) {
                    Ok(id) => self.permanent_matches_predicate(id, predicate),
                    Err(_) => self
                        .object_lki(object)
                        .is_some_and(|lki| super::layers::matches_departure(lki, predicate)),
                })
            }
            Condition::SourceUnprepared => {
                source_permanent().is_some_and(|p| self.prepared_copy(p.object).is_none())
            }
            Condition::SourceTapped => source_permanent().is_some_and(|permanent| permanent.tapped),
            Condition::SourceSaddled => source_id().is_some_and(|id| {
                self.permanent_has_modification(id, &crate::state::effect::Modification::Saddled)
            }),
            Condition::TriggeringHadCounters => matches!(triggering, Some(Target::Permanent(id))
                if self.with_permanent_or_lki(
                    id,
                    |permanent| permanent.has_counters(),
                    |lki| !lki.counters.is_empty(),
                ).unwrap_or(false)),
            Condition::TriggeringMatches(predicate) => {
                matches!(triggering, Some(Target::Permanent(id))
                if self.with_permanent_or_lki(
                    id,
                    |_| self.permanent_matches_predicate(id, predicate),
                    |lki| self.lki_matches_predicate(lki, predicate),
                ).unwrap_or(false))
            }
            Condition::Moved => frame.is_some_and(|frame| !frame.moved.is_empty()),
            Condition::AttachedCountersAtLeast { kind, count } => frame
                .and_then(|frame| self.frame_attached(frame))
                .and_then(|id| self.state.permanents[id].as_ref())
                .is_some_and(|permanent| permanent.counter_count(*kind) >= *count),
            Condition::TargetControlledByYou => {
                let Some(Target::Permanent(permanent_id)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                self.last_known_controller(permanent_id) == Some(controller)
            }
            Condition::CardsThisWayMatch(predicate) => frame.is_some_and(|frame| {
                frame
                    .this_way
                    .iter()
                    .any(|object| self.recorded_object_matches_predicate(*object, predicate))
            }),
            Condition::TriggeringSpellMatches(predicate) => matches!(
                triggering,
                Some(Target::StackSpell(object_id)) if self.spell_matches_predicate(object_id, predicate)
            ),
            Condition::TargetCardExiled(predicate) => {
                let Some(Target::GraveyardCard(object)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                // CR 400.7j: the exiled object the targeted card became.
                self.state.zones.successor(object).is_some_and(|exiled| {
                    self.state.zones.zone_of(exiled) == Some(ZoneType::Exile)
                        && self.object_matches_predicate(exiled, predicate)
                })
            }
            Condition::TargetDealtDamageThisTurn => {
                let Some(Target::Permanent(permanent_id)) =
                    frame.and_then(EffectFrame::primary_target)
                else {
                    return false;
                };
                // A departed target answers from its last known information.
                self.with_permanent_or_lki(
                    permanent_id,
                    |permanent| permanent.dealt_damage_this_turn,
                    |lki| lki.dealt_damage_this_turn,
                )
                .unwrap_or(false)
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
    fn predicate_source_view(
        &self,
        source: Option<PermanentId>,
        predicate: &CardPredicate,
    ) -> Option<(PermanentId, Arc<Characteristics>)> {
        source
            .filter(|_| predicate.reads_source())
            .map(|id| (id, self.layer_view(id)))
    }

    fn predicate_source<'a>(
        &'a self,
        source: Option<&'a (PermanentId, Arc<Characteristics>)>,
    ) -> Option<PredicateSource<'a>> {
        source.map(|(id, view)| PredicateSource {
            name: view.name.as_str(),
            chosen: self.live_permanent(*id).chosen.as_deref(),
        })
    }

    pub(super) fn permanent_matches_with_source(
        &self,
        id: PermanentId,
        predicate: &CardPredicate,
        source: Option<PermanentId>,
    ) -> bool {
        let view = self.layer_view(id);
        let source_view = self.predicate_source_view(source, predicate);
        matches_view(
            self,
            id,
            predicate,
            &view,
            self.predicate_source(source_view.as_ref()),
        )
    }

    /// Whether an ability's "tap [its grantor]" cost can be paid: the
    /// granting permanent is untapped. Summoning sickness doesn't matter
    /// (CR 302.6). True for an ability without that cost.
    pub(crate) fn can_tap_grantor(&self, ability: &ActivatedAbilityDefinition) -> bool {
        !ability.tap_grantor
            || ability
                .granted_by
                .and_then(|grant| self.state.permanent_of(grant.grantor))
                .and_then(|id| self.state.permanents[id].as_ref())
                .is_some_and(|grantor| !grantor.tapped)
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
        !permanent.summoning_sick
            || !self
                .characteristics(permanent_id)
                .restricted_by_summoning_sickness()
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

    /// A game whose cached board disagrees with its state at the same
    /// revisions, as a write that advanced no revision would leave it.
    #[cfg(debug_assertions)]
    fn with_stale_board(game: &Game) {
        let board = game.layer_board();
        game.derived.borrow_mut().board = Some(Arc::new(LayerBoard {
            play_permissions: board.play_permissions.clone(),
            restrictions: board.restrictions.clone(),
            land_type_overwrites: board.land_type_overwrites.clone(),
            objects: board.objects.clone(),
            reads_mana: !board.reads_mana,
            reads_tapped: board.reads_tapped,
            reads_life: board.reads_life,
            dynamic_activations: board.dynamic_activations,
        }));
    }

    #[cfg(debug_assertions)]
    fn scan_fixture() -> Game {
        let deck: std::collections::BTreeMap<String, usize> = [("Mountain".into(), 40)].into();
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck.clone()),
                PlayerConfig::new("b", deck),
            ],
            244,
            true,
        );
        game.play();
        game
    }

    #[cfg(debug_assertions)]
    fn rejects_stale_board(read: impl FnOnce()) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(read)).is_err_and(|panic| {
            panic
                .downcast_ref::<String>()
                .is_some_and(|m| m.contains("stale"))
        })
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_stale_board_is_rejected_by_a_plain_read_and_by_a_new_scan() {
        let game = scan_fixture();
        with_stale_board(&game);
        assert!(rejects_stale_board(|| {
            game.layer_board();
        }));
        assert!(rejects_stale_board(|| {
            let _scan = game.board_scan();
        }));
        assert_eq!(game.derived.borrow().scans.0, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn validation_resumes_only_when_every_nested_scan_ends() {
        let game = scan_fixture();
        let outer = game.board_scan();
        let inner = game.board_scan();
        with_stale_board(&game);
        game.layer_board();
        drop(inner);
        game.layer_board();
        drop(outer);
        assert!(rejects_stale_board(|| {
            game.layer_board();
        }));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn a_fork_taken_during_a_scan_validates_its_own_reads() {
        let game = scan_fixture();
        let _scan = game.board_scan();
        let fork = game.clone();
        with_stale_board(&fork);
        assert!(rejects_stale_board(|| {
            fork.layer_board();
        }));
    }

    #[test]
    fn recording_creature_damage_reuses_the_layer_board() {
        use crate::flow::proposed_event::ProposedDamageTarget;

        let player = PlayerId(0);
        let deck = [("Gray Ogre".into(), 40)].into();
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck),
                PlayerConfig::new("b", [("Mountain".into(), 40)].into()),
            ],
            418,
            true,
        );
        game.play();
        let source = game
            .scenario_force_battlefield(player, "Gray Ogre", true)
            .unwrap();
        let target = game
            .scenario_force_battlefield(player, "Gray Ogre", true)
            .unwrap();
        let source = game.permanent_object(source).unwrap();
        let target = game.permanent_object(target).unwrap();
        let board = game.layer_board();

        assert!(
            game.commit_damage(Some(source), ProposedDamageTarget::Object(target), 1, true)
                .committed
        );
        assert!(game.state.turn.damage_pairs.contains(source, target));
        assert!(Arc::ptr_eq(&board, &game.layer_board()));
    }

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
        *game.state.mana_pool_mut(PlayerId(0)) = Mana::parse("R").into();
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
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("RR").into();
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
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("RRR").into();
        assert_eq!(branch.characteristics(id).pt(), (4, 4));
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::default().into();
        assert_eq!(branch.characteristics(id).pt(), (2, 2));
    }

    #[test]
    fn power_comparison_uses_layer_creatures_and_invalidates_on_tap() {
        use crate::state::ability::{Aggregate, CompareOp, ObjectProperty, ObjectSelector};
        let deck = [("Gray Ogre".into(), 40)].into();
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
        let other = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .unwrap();
        let affected = game.affected_permanent(id).unwrap();
        game.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::GrantStatic(Box::new(StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::This,
                condition: Some(Condition::compare(
                    ObjectSelector::each_you_control(CardPredicate {
                        tapped: true,
                        ..CardPredicate::creature()
                    })
                    .property(ObjectProperty::Power, Aggregate::Sum),
                    CompareOp::Ge,
                    2,
                )),
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(0),
                }],
            }))),
            EffectDuration::Indefinite,
        );
        assert_eq!(game.characteristics(id).power, 2);
        game.state.tap(other);
        assert_eq!(game.characteristics(id).power, 3);
        game.state.untap(other);
        assert_eq!(game.characteristics(id).power, 2);
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
        *branch.state.mana_pool_mut(PlayerId(0)) = Mana::parse("R").into();
        assert!(Arc::ptr_eq(&fixed, &branch.characteristics(id)));

        let affected = branch.affected_permanent(id).unwrap();
        branch.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::GrantActivated(Box::new(ActivatedAbilityDefinition {
                effects: vec![Effect::If {
                    condition: Condition::ControlsAtLeast {
                        distinct_names: false,
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
