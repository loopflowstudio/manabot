// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel answers a question two ways with one result:
// directly, by arithmetic over the pool and the fixed outputs, when every
// source in play for the question taps itself for one fixed output and the
// obligations are one mana cost with an optional split that taps may pay
// (`PaymentModel::direct`); otherwise by searching identity-preserving
// resource steps. The witness search prunes with optimistic whole-payment and
// color-set bounds and memoizes feasibility across interchangeable sources
// (PaymentSearch), always returning the first concrete path in source order.
// The search defines a legal payment; debug builds run it beside every direct
// answer and assert equal affordability and an equal witness.
// Game owns their events and explicit Decision continuations, and shares one
// player's sources across the questions of a `PaymentScan`.
// Source lists are immutable shared values; reservations detach on writes.
// Pool and spend-purpose restrictions belong to state/mana.rs. Game also retains
// bounded exact queries (PaymentCache); debug builds recompute every hit.

mod autotapper;
mod help;
mod preview;
pub use preview::PaymentPreview;

use crate::state::player::{ManaFloats, ManaPayment};
use std::{cell::RefCell, collections::VecDeque, num::NonZeroUsize, sync::Arc};

use crate::{
    agent::action::{Action, ActionSpace, AgentError},
    flow::{
        decision::{
            Decision, EffectFrame, ManaProduction, PendingManaBonus, SpendCompletion,
            SuspendedResolution, TappedForMana,
        },
        game::{ActivationSource, Game, PendingChoice},
        payment::{CostComponent, Help, Operation, Payment},
        statics::Characteristics,
    },
    state::{
        ability::Effect,
        card::{
            ActivatedAbilityDefinition, ActivationZone, ReplacementEffect, TriggeredManaAbility,
            TriggeredManaOutput,
        },
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::{
            Color, Mana, ManaCost, ManaPool, ManaSpendChoice, SpellSnapshot, SpendOrder,
            SpendingContext,
        },
        payment::{How, ManaOrigin, PaymentReceipt, SpentMana},
        zone::ZoneType,
    },
};

/// One finite resource transition model. A source is reserved before its
/// upfront obligation, so filters cannot finance themselves or form seedless
/// cycles. Witnesses retain identities and concrete typed pool remainders.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Obligation {
    Pay {
        cost: ManaCost,
        taps: Taps,
    },
    Output {
        // Alternatives and candidate sets are immutable query payloads. Branches
        // and memo keys share them while keeping their own progress and pool.
        choices: Arc<Vec<Mana>>,
        life_cost: u32,
    },
    Tap {
        candidates: Arc<Vec<PermanentId>>,
        remaining: usize,
    },
}

impl Obligation {
    /// `cost`, paid only with mana.
    fn mana(cost: ManaCost) -> Self {
        Obligation::Pay {
            cost,
            taps: Taps::None,
        }
    }
}

/// Which untapped artifacts and creatures may pay a `Pay` obligation's
/// generic mana by tapping, {1} each, instead of with mana.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Taps {
    /// None: the cost is paid with mana.
    None,
    /// Those the cost's convoke or improvise names (`ManaSource::keyword`).
    Keyword,
    /// Any of them: a waterbend cost (`ManaSource::waterbend`).
    Any,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManaSource {
    permanent: PermanentId,
    object: ObjectId,
    untapped: bool,
    fixed: Option<Mana>,
    abilities: Vec<ManaActivation>,
    /// An untapped artifact or creature, whose tap can pay {1} of a
    /// waterbend cost.
    waterbend: bool,
    /// Whether the cost being paid lets this tap pay any of its generic
    /// mana (convoke, improvise; `Game::mark_help_taps`).
    keyword: bool,
}

impl ManaSource {
    /// Whether tapping this source may pay {1} of an obligation that `taps`
    /// may pay.
    fn pays_by_tap(&self, taps: Taps) -> bool {
        match taps {
            Taps::None => false,
            Taps::Keyword => self.keyword,
            Taps::Any => self.waterbend,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManaActivation {
    tap_source: bool,
    tap_count: usize,
    tap_candidates: Arc<Vec<PermanentId>>,
    limited: bool,
    sacrifice_source: bool,
    ability_index: usize,
    cost: ManaCost,
    outputs: Arc<Vec<Mana>>,
    life_cost: u32,
}

/// The union of output modes, projected onto one spending purpose. The bound
/// only asks color feasibility, so it needs no owned restriction or rider
/// metadata. Maxima are taken within each class *before* combining classes:
/// two modes with different classes can both contribute to the old bound.
fn optimistic_pool(outputs: &[Mana], cost: &ManaCost) -> Mana {
    let mut upper = Mana::default();
    for output in outputs {
        for (amount, more) in upper.mana.iter_mut().zip(output.mana) {
            *amount = (*amount).max(more);
        }
    }
    for (index, output) in outputs.iter().enumerate() {
        for (class, _) in output.classes() {
            if !class.allows(&cost.purpose)
                || outputs[..index]
                    .iter()
                    .any(|previous| previous.classes().iter().any(|(seen, _)| seen == class))
            {
                continue;
            }
            let mut amounts = [0; 6];
            for mode in &outputs[index..] {
                for (amount, more) in amounts.iter_mut().zip(mode.class_of(class)) {
                    *amount = (*amount).max(more);
                }
            }
            for (amount, more) in upper.mana.iter_mut().zip(amounts) {
                *amount = amount.saturating_add(more);
            }
        }
    }
    upper
}

/// Reserve incompatible uses, not the whole object. A tapped mana creature
/// can still be sacrificed later, but an announced tap cannot be spent twice.
#[derive(Clone, Copy)]
struct SourceReservation {
    permanent: PermanentId,
    tap: bool,
    sacrifice: bool,
}

impl SourceReservation {
    /// `permanent` is tapped for something other than mana.
    fn tap(permanent: PermanentId) -> Self {
        Self {
            permanent,
            tap: true,
            sacrifice: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "benchmark", derive(serde::Serialize))]
enum PaymentStep {
    Fixed(PermanentId),
    Activate(PermanentId, usize),
    Spend,
    Output(usize),
    Waterbend(PermanentId),
    Tap(ObjectId),
}

impl PaymentStep {
    fn action(&self, player: PlayerId) -> Option<Action> {
        match self {
            Self::Activate(permanent, ability_index) => Some(Action::ActivateAbility {
                player,
                permanent: *permanent,
                ability_index: *ability_index,
            }),
            Self::Tap(object) => Some(Action::SelectCard {
                player,
                object: *object,
            }),
            Self::Output(mode) => Some(Action::ChooseMode {
                player,
                mode: *mode,
            }),
            Self::Waterbend(permanent) => Some(Action::WaterbendTap {
                player,
                permanent: *permanent,
            }),
            Self::Fixed(_) | Self::Spend => None,
        }
    }
}

/// A source's presence, untapped resource and per-ability limited uses are
/// independent. A tapped source can still pay a sacrifice cost; a repeatable
/// ability can consume other objects without consuming its own source.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PaymentResources {
    present: Vec<bool>,
    untapped: Vec<bool>,
    /// Sorted (source index, ability index) flags. Only limited abilities
    /// insert a use; ordinary mana transitions allocate no per-source rows.
    used: Vec<(usize, usize)>,
}

impl PaymentResources {
    /// Omit trailing false flags: this keeps the dense row's lexicographic
    /// order, because every nonempty suffix ends with a true flag.
    fn used_on(&self, index: usize) -> Vec<bool> {
        let mut flags = Vec::new();
        for &(_, ability) in self.used.iter().filter(|(source, _)| *source == index) {
            flags.resize(ability + 1, false);
            flags[ability] = true;
        }
        flags
    }

    fn use_ability(&mut self, source: usize, ability: usize) {
        let entry = (source, ability);
        if let Err(index) = self.used.binary_search(&entry) {
            self.used.insert(index, entry);
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PaymentSearchState {
    resources: PaymentResources,
    pool: Mana,
    life: u32,
    obligations: Vec<Obligation>,
    /// Fixed-source budget until the first Spend; later obligations are unrestricted.
    prefix: Option<usize>,
}

/// Which transitions may be automatic at the caller's current boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourcePolicy {
    All,
    FixedOnly,
    AutomaticPrefix,
}

/// Memoize feasibility modulo interchangeable resources, but never rewrite the
/// live state or a returned witness. The first concrete DFS path still wins.
///
/// Ordinary plans expand a handful of states, so a search starts with exact
/// identity keys and only the root's color-set check. Once it has expanded
/// `GROWTH` states it classifies the interchangeable sources once, re-keys
/// what it visited and from then on canonicalizes every key and applies the
/// color-set bound to every state. Both protections only reject states with
/// no witness, so when they start does not change the path returned.
#[derive(Default)]
struct PaymentSearch {
    /// Counted-tap sets of the root obligations; with the abilities' own sets
    /// they decide which sources are interchangeable.
    root_taps: Vec<Arc<Vec<PermanentId>>>,
    /// Classes of interchangeable sources, built when the search grows.
    groups: Option<Vec<Vec<usize>>>,
    visited: std::collections::HashSet<Arc<PaymentSearchState>>,
    #[cfg(test)]
    max_states: Option<usize>,
    /// Reference search for tests: skip the color-set bound.
    #[cfg(test)]
    total_bound_only: bool,
    #[cfg(test)]
    gross_bound_only: bool,
}

impl PaymentSearch {
    /// Visited states after which a search pays for source classes and the
    /// per-state color-set bound.
    const GROWTH: usize = 16;

    fn new(obligations: &[Obligation]) -> Self {
        Self {
            root_taps: obligations
                .iter()
                .filter_map(|obligation| match obligation {
                    Obligation::Tap { candidates, .. } => Some(candidates.clone()),
                    _ => None,
                })
                .collect(),
            ..Default::default()
        }
    }

    /// Whether the color-set bound applies to the state about to be checked:
    /// at the root, and everywhere once the search has grown.
    fn bounds_colors(&self) -> bool {
        self.visited.is_empty() || self.groups.is_some()
    }

    fn groups(&self, model: &PaymentModel) -> Vec<Vec<usize>> {
        // Membership in every possible counted-tap set is part of identity:
        // equal mana producers need not be equal candidates for a saved cost.
        let candidates = model
            .sources
            .iter()
            .flat_map(|source| &source.abilities)
            .map(|ability| &ability.tap_candidates)
            .chain(&self.root_taps)
            .collect::<Vec<_>>();
        // Most boards have few duplicates. Do not allocate a member vector
        // for every singleton just to discard it after classification.
        let mut representatives: Vec<(usize, Option<usize>)> = Vec::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (index, source) in model.sources.iter().enumerate() {
            let equivalent = representatives.iter_mut().find(|(representative, _)| {
                let other = &model.sources[*representative];
                source.fixed == other.fixed
                    && source.abilities == other.abilities
                    && source.waterbend == other.waterbend
                    && source.keyword == other.keyword
                    && candidates.iter().all(|set| {
                        set.contains(&source.permanent) == set.contains(&other.permanent)
                    })
            });
            if let Some((representative, group)) = equivalent {
                let group = group.get_or_insert_with(|| {
                    groups.push(vec![*representative]);
                    groups.len() - 1
                });
                groups[*group].push(index);
            } else {
                representatives.push((index, None));
            }
        }
        groups
    }

    /// The state with each class's resources in sorted order.
    fn canonical(
        groups: &[Vec<usize>],
        state: &Arc<PaymentSearchState>,
    ) -> Arc<PaymentSearchState> {
        let mut key = state.clone();
        let mut changed = false;
        let resources = &state.resources;
        let resource_key = |index: usize| {
            (
                resources.present[index],
                resources.untapped[index],
                resources.used_on(index),
            )
        };
        for group in groups {
            // Source-order success usually consumes an already sorted prefix.
            // Allocate a permutation only for branches that need one.
            if group
                .windows(2)
                .all(|pair| resource_key(pair[0]) <= resource_key(pair[1]))
            {
                continue;
            }
            let mut order = group.clone();
            order.sort_unstable_by_key(|&index| resource_key(index));
            let key = Arc::make_mut(&mut key);
            changed = true;
            key.resources
                .used
                .retain(|(source, _)| !group.contains(source));
            for (&to, from) in group.iter().zip(order) {
                key.resources.present[to] = resources.present[from];
                key.resources.untapped[to] = resources.untapped[from];
                key.resources.used.extend(
                    resources
                        .used
                        .iter()
                        .filter(|(source, _)| *source == from)
                        .map(|(_, ability)| (to, *ability)),
                );
            }
        }
        if changed {
            Arc::make_mut(&mut key).resources.used.sort_unstable();
        }
        key
    }

    fn insert(&mut self, model: &PaymentModel, state: &Arc<PaymentSearchState>) -> bool {
        if self.groups.is_none() && self.visited.len() >= Self::GROWTH {
            // A visited identity state failed or is being expanded, and so is
            // every state interchangeable with it: its canonical key is one
            // the grouped search would hold too.
            let groups = self.groups(model);
            self.visited = std::mem::take(&mut self.visited)
                .iter()
                .map(|key| Self::canonical(&groups, key))
                .collect();
            self.groups = Some(groups);
        }
        let key = match &self.groups {
            Some(groups) => Self::canonical(groups, state),
            None => state.clone(),
        };
        let inserted = self.visited.insert(key);
        #[cfg(test)]
        if let Some(limit) = self.max_states {
            assert!(
                self.visited.len() <= limit,
                "payment exceeded {limit} states"
            );
        }
        inserted
    }
}

impl PaymentSearchState {
    /// Mana by kind that the pending payment could spend, for the optimistic
    /// bounds: restricted mana counts if any pending cost can spend it.
    /// Future activation costs are not owed yet; the gross outputs the bounds
    /// add already overestimate what converting restricted mana could buy.
    fn usable(&self, mana: &Mana) -> [u64; 6] {
        let mut usable = mana.mana.map(u64::from);
        for (class, amounts) in mana.classes() {
            if self.obligations.iter().any(|obligation| {
                matches!(obligation, Obligation::Pay { cost, .. }
                    if class.allows(&cost.purpose))
            }) {
                for (amount, more) in usable.iter_mut().zip(amounts) {
                    *amount += u64::from(*more);
                }
            }
        }
        usable
    }
}

/// Exact derived queries, copied by value with a fork and excluded from rules
/// state. Each fork owns its FIFO; immutable entries can share their payload.
/// FIFO bounds retention; equality includes ordered sources and callers,
/// not just board revisions. A selected tap is part of the query identity.
#[derive(Clone, Debug, Default)]
pub(crate) struct PaymentCache {
    entries: VecDeque<Arc<CachedPayment>>,
}

#[cfg(feature = "benchmark")]
impl PaymentCache {
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

#[derive(Debug)]
struct CachedPayment {
    sources: Arc<Vec<ManaSource>>,
    life: u32,
    pool: Mana,
    obligations: Vec<Obligation>,
    policy: SourcePolicy,
    tap: Option<PermanentId>,
    path: Option<Vec<PaymentStep>>,
}

struct PaymentModel<'a> {
    cache: Option<&'a RefCell<PaymentCache>>,
    sources: Arc<Vec<ManaSource>>,
    life: u32,
}

/// A question the direct answer covers: `cost` paid with mana, then `split`,
/// generic that mana or the taps `taps` names may pay.
struct DirectPayment<'a> {
    cost: &'a ManaCost,
    split: Option<&'a ManaCost>,
    taps: Taps,
}

impl DirectPayment<'_> {
    fn split_generic(&self) -> u32 {
        self.split.map_or(0, |split| u32::from(split.generic()))
    }
}

/// What tapping one source offers a direct payment.
#[derive(Clone, Copy)]
struct DirectTap<'a> {
    permanent: PermanentId,
    /// The fixed output of an untapped source, tapped in source order even
    /// when the cost can spend none of it.
    output: Option<&'a Mana>,
    /// How much of `output` the cost's purpose may spend.
    mana: u32,
    /// Whether the tap may instead pay {1} of the split.
    waterbend: bool,
}

impl DirectTap<'_> {
    /// A waterbend tap that gives up no mana. A source worth mana is never
    /// better tapped for waterbend, which pays exactly {1}.
    fn spare(self) -> bool {
        self.waterbend && self.mana == 0
    }

    /// The most this tap adds towards the split.
    fn worth(self) -> u32 {
        self.mana.max(self.waterbend.into())
    }
}

/// One player's unreserved sources, kept while a `PaymentScan` is live.
#[derive(Debug, Default)]
pub(crate) struct SharedSources {
    scans: u32,
    sources: [Option<Arc<Vec<ManaSource>>>; 2],
}

/// One legal-action computation. The sources a player could tap are a
/// function of the game alone, and the guard's shared borrow excludes every
/// write to it, so the payment questions asked while it lives build those
/// sources once per player. Reservations of the pending payment, a sacrifice
/// under consideration and payable life are still applied per question.
/// Debug builds rebuild the sources on every reuse and assert they match.
pub(crate) struct PaymentScan<'a> {
    game: &'a Game,
}

impl Drop for PaymentScan<'_> {
    fn drop(&mut self) {
        let mut shared = self.game.payment_sources.borrow_mut();
        shared.scans -= 1;
        if shared.scans == 0 {
            shared.sources = Default::default();
        }
    }
}

impl PaymentModel<'_> {
    fn reserve(&mut self, reservation: SourceReservation) {
        let Some(index) = self.sources.iter().position(|source| {
            source.permanent == reservation.permanent
                && (reservation.tap
                    && (source.untapped || source.fixed.is_some() || source.waterbend)
                    || source.abilities.iter().any(|ability| {
                        reservation.tap && ability.tap_source
                            || reservation.sacrifice && ability.sacrifice_source
                    }))
        }) else {
            return;
        };
        let source = &mut Arc::make_mut(&mut self.sources)[index];
        if reservation.tap {
            source.fixed = None;
            source.waterbend = false;
            source.keyword = false;
            source.untapped = false;
        }
        source.abilities.retain(|ability| {
            !(reservation.tap && ability.tap_source
                || reservation.sacrifice && ability.sacrifice_source)
        });
    }

    /// Gross usable output and net new units. Each tap can choose one mode.
    fn source_bounds(
        &self,
        resources: &PaymentResources,
        life: u32,
        usable_total: impl Fn(&Mana) -> u64,
        pays_by_tap: impl Fn(&ManaSource) -> bool,
    ) -> (u64, u64) {
        let mut net_total = 0;
        let mut total = 0;
        for (index, source) in self.sources.iter().enumerate() {
            if !resources.present[index] {
                continue;
            }
            // Fixed mana, tapping abilities and waterbending compete for the
            // same tap. Non-tapping activations can still use this object later.
            let mut tap_total = 0;
            let mut net_tap_total = 0;
            if resources.untapped[index] {
                tap_total = source.fixed.as_ref().map_or(0, &usable_total);
                net_tap_total = source
                    .fixed
                    .as_ref()
                    .map_or(0, |output| u64::from(output.total()));
                if pays_by_tap(source) {
                    tap_total = tap_total.max(1);
                    net_tap_total = net_tap_total.max(1);
                }
            }
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let uses = self.uses(resources, index, ability_index) as u64;
                if uses == 0 || ability.life_cost > life {
                    continue;
                }
                let output = uses * ability.outputs.iter().map(&usable_total).max().unwrap_or(0);
                let net_output = uses
                    * ability
                        .outputs
                        .iter()
                        .map(|output| {
                            u64::from(output.total())
                                .saturating_sub(u64::from(ability.cost.mana_value))
                        })
                        .max()
                        .unwrap_or(0);
                if ability.tap_source {
                    net_tap_total = net_tap_total.max(net_output);
                    tap_total = tap_total.max(output);
                } else {
                    total += output;
                    net_total += net_output;
                }
            }
            total += tap_total;
            net_total += net_tap_total;
        }
        (total, net_total)
    }

    fn resources(&self) -> PaymentResources {
        PaymentResources {
            present: vec![true; self.sources.len()],
            untapped: self.sources.iter().map(|source| source.untapped).collect(),
            used: Vec::new(),
        }
    }

    /// An optimistic count, ignoring competition between different abilities.
    /// Every activation consumes presence, a tap, a limited use, or counted taps.
    /// Summing these upper bounds also covers multiple uses of one source.
    fn uses(&self, resources: &PaymentResources, index: usize, ability_index: usize) -> usize {
        if !resources.present[index] {
            return 0;
        }
        let ability = &self.sources[index].abilities[ability_index];
        if resources.used.contains(&(index, ability_index))
            || (ability.tap_source && !resources.untapped[index])
        {
            return 0;
        }
        // A tapped, sacrificed or limited source activates at most once.
        let cap = if ability.tap_source || ability.sacrifice_source || ability.limited {
            1
        } else {
            usize::MAX
        };
        // Only a counted tap cost reads the board, and only as far as `cap`
        // activations need: a repeatable cost counts every candidate for its
        // optimistic production bound.
        let uses = if let Some(tap_count) = NonZeroUsize::new(ability.tap_count) {
            self.tap_candidates(resources, &ability.tap_candidates)
                .take(cap.saturating_mul(tap_count.get()))
                .count()
                / tap_count.get()
        } else {
            cap
        };
        assert_ne!(
            uses,
            usize::MAX,
            "mana activation must consume a finite resource"
        );
        uses
    }

    fn tap_candidates<'a>(
        &'a self,
        resources: &'a PaymentResources,
        candidates: &'a [PermanentId],
    ) -> impl Iterator<Item = usize> + 'a {
        self.sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| {
                (resources.present[index]
                    && resources.untapped[index]
                    && candidates.contains(&source.permanent))
                .then_some(index)
            })
    }

    fn witness(
        &self,
        state: PaymentSearchState,
        policy: SourcePolicy,
        visited: &mut PaymentSearch,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::call();
        if state.obligations.is_empty() {
            return Some(Vec::new());
        }
        // No saved caller reads the final remainder. Exact affordability is
        // enough here; only a payment with later obligations needs to search
        // unit assignments. Execution still offers the native spending choices.
        if let [Obligation::Pay { cost, .. }] = state.obligations.as_slice() {
            if state.pool.can_pay(cost) {
                return Some(vec![PaymentStep::Spend]);
            }
        }
        // These transitions have no branch to memoize or bound. Advance the
        // same ordered obligations before scanning every source; the next
        // branching state still proves affordability and retains every step.
        match &state.obligations[0] {
            Obligation::Pay { cost, .. } if cost.mana_value == 0 => {
                let mut next = state;
                next.obligations.remove(0);
                next.prefix = None;
                return self.witness(next, policy, visited).map(|mut path| {
                    path.insert(0, PaymentStep::Spend);
                    path
                });
            }
            Obligation::Tap { remaining: 0, .. } => {
                let mut next = state;
                next.obligations.remove(0);
                return self.witness(next, policy, visited);
            }
            Obligation::Output { choices, life_cost } if choices.len() == 1 => {
                let life = state.life.checked_sub(*life_cost)?;
                let mut next = state;
                let Obligation::Output { choices, .. } = next.obligations.remove(0) else {
                    unreachable!()
                };
                next.life = life;
                next.pool.add(&choices[0]);
                return self.witness(next, policy, visited).map(|mut path| {
                    path.insert(0, PaymentStep::Output(0));
                    path
                });
            }
            _ => {}
        }
        // Presence and untapped resources only decrease during payment. Once
        // any saved counted tap lacks candidates, no production or spending
        // order can restore it. Reject before exploring the earlier costs;
        // checking only when that tap becomes first repeats doomed payments.
        if state.obligations.iter().any(|obligation| {
            matches!(obligation, Obligation::Tap { candidates, remaining }
                if self.tap_candidates(&state.resources, candidates).take(*remaining).count() < *remaining)
        }) {
            return None;
        }
        // Bound the whole pending payment, not just its first obligation:
        // otherwise an impossible later cost retries every way to fund an
        // earlier one.
        let usable_total = |mana: &Mana| -> u64 { state.usable(mana).iter().sum() };
        // Whether some pending cost lets tapping `source` pay {1} of it.
        let pays_by_tap = |source: &ManaSource| {
            state.obligations.iter().any(|obligation| {
                matches!(obligation, Obligation::Pay { taps, .. } if source.pays_by_tap(*taps))
            })
        };
        // A second conservation bound counts *all* pool classes, even those
        // unusable by pending costs, and only the net gain of future
        // activations. Restricted seeds can fund filters without being lost
        // from this accounting. Already saved outputs remain gross supply;
        // their upfront fees are explicit pending Pay obligations.
        let (gross, net) =
            self.source_bounds(&state.resources, u32::MAX, usable_total, pays_by_tap);
        let mut total = usable_total(&state.pool) + gross;
        let mut net_total = u64::from(state.pool.total()) + net;
        for obligation in &state.obligations {
            if let Obligation::Output { choices, .. } = obligation {
                total += choices.iter().map(&usable_total).max().unwrap_or(0);
                net_total += choices
                    .iter()
                    .map(|output| u64::from(output.total()))
                    .max()
                    .unwrap_or(0);
            }
        }
        let owed = state
            .obligations
            .iter()
            .filter_map(|obligation| match obligation {
                Obligation::Pay { cost, .. } => Some(u64::from(cost.mana_value)),
                _ => None,
            })
            .sum::<u64>();
        #[cfg(test)]
        if visited.gross_bound_only {
            net_total = u64::MAX;
        }
        // Only states that need exploration need an owned visited key.
        if total < owed
            || net_total < owed
            || (visited.bounds_colors() && !self.colors_coverable(&state, visited))
        {
            return None;
        }
        // Retain this immutable branch state as its own memo key. Only a
        // canonical permutation needs a separate copy; concrete children own
        // their mutations and never rewrite the retained key.
        let state = Arc::new(state);
        if !visited.insert(self, &state) {
            return None;
        }
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::state();
        match &state.obligations[0] {
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                for index in self.tap_candidates(&state.resources, candidates) {
                    let mut next = state.as_ref().clone();
                    next.resources.untapped[index] = false;
                    next.obligations[0] = Obligation::Tap {
                        candidates: candidates.clone(),
                        remaining: *remaining - 1,
                    };
                    if let Some(mut path) = self.witness(next, policy, visited) {
                        path.insert(0, PaymentStep::Tap(self.sources[index].object));
                        return Some(path);
                    }
                }
            }
            Obligation::Output {
                choices, life_cost, ..
            } => {
                let life = state.life.checked_sub(*life_cost)?;
                for (mode, output) in choices.iter().enumerate() {
                    let mut pool = state.pool.clone();
                    pool.add(output);
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            pool,
                            life,
                            resources: state.resources.clone(),
                            obligations: state.obligations[1..].to_vec(),
                            prefix: state.prefix,
                        },
                        policy,
                        visited,
                    ) {
                        path.insert(0, PaymentStep::Output(mode));
                        return Some(path);
                    }
                }
            }
            Obligation::Pay { cost, taps } => {
                let pooled = state.pool.can_pay(cost);
                if !pooled && !self.cost_coverable(&state, cost, *taps) {
                    return None;
                }
                // Pool the whole current cost before spending starts;
                // production cannot interleave with its unit choices. If the
                // pool cannot cover it, no first unit can leave a payable rest.
                if pooled {
                    for choice in state.pool.spending_choices(cost) {
                        let mut pool = state.pool.clone();
                        pool.spend(&choice.unit);
                        if !pool.can_pay(&choice.remaining) {
                            continue;
                        }
                        let mut next = PaymentSearchState {
                            pool,
                            resources: state.resources.clone(),
                            obligations: state.obligations.clone(),
                            life: state.life,
                            prefix: state.prefix,
                        };
                        next.obligations[0] = Obligation::Pay {
                            cost: choice.remaining,
                            taps: *taps,
                        };
                        // The zero-cost arm owns completion and the Spend boundary,
                        // including when this was the last unit of a nonzero cost.
                        if let Some(path) = self.witness(next, policy, visited) {
                            return Some(path);
                        }
                    }
                }
                if state.prefix == Some(0) {
                    return None;
                }
                for (index, source) in self.sources.iter().enumerate() {
                    if !state.resources.present[index] {
                        continue;
                    }
                    if state.resources.untapped[index] {
                        if let Some(output) = &source.fixed {
                            let mut next = state.as_ref().clone();
                            next.resources.untapped[index] = false;
                            next.pool.add(output);
                            next.prefix = next.prefix.map(|n| n - 1);
                            if let Some(mut path) = self.witness(next, policy, visited) {
                                path.insert(0, PaymentStep::Fixed(source.permanent));
                                return Some(path);
                            }
                        }
                    }
                    // Chosen activations cannot nest before the automatic
                    // prefix's first Spend, so its budget needs no depth.
                    if policy == SourcePolicy::FixedOnly
                        || (policy == SourcePolicy::AutomaticPrefix && state.prefix.is_some())
                    {
                        continue;
                    }
                    for (ability_index, ability) in source.abilities.iter().enumerate() {
                        if self.uses(&state.resources, index, ability_index) == 0 {
                            continue;
                        }
                        let mut obligations = Self::activation_obligations(ability);
                        obligations.extend_from_slice(&state.obligations);
                        let mut next = PaymentSearchState {
                            resources: state.resources.clone(),
                            pool: state.pool.clone(),
                            life: state.life,
                            obligations,
                            prefix: None,
                        };
                        if ability.tap_source {
                            next.resources.untapped[index] = false;
                        }
                        if ability.sacrifice_source {
                            next.resources.present[index] = false;
                        }
                        if ability.limited {
                            next.resources.use_ability(index, ability_index);
                        }
                        if let Some(mut path) = self.witness(next, policy, visited) {
                            path.insert(
                                0,
                                PaymentStep::Activate(source.permanent, ability.ability_index),
                            );
                            return Some(path);
                        }
                    }
                    if source.pays_by_tap(*taps)
                        && state.resources.untapped[index]
                        && cost.generic() > 0
                    {
                        let mut next = state.as_ref().clone();
                        next.resources.untapped[index] = false;
                        next.obligations[0] = Obligation::Pay {
                            cost: cost.reduced_generic(1),
                            taps: *taps,
                        };
                        next.prefix = next.prefix.map(|n| n - 1);
                        if let Some(mut path) = self.witness(next, policy, visited) {
                            path.insert(0, PaymentStep::Waterbend(source.permanent));
                            return Some(path);
                        }
                    }
                }
            }
        }
        None
    }

    /// Optimistic color/restriction supply for the current cost. Shared
    /// resources are relaxed and every output mode contributes its colors;
    /// the whole-payment and color-set bounds handle competing obligations.
    fn cost_coverable(&self, state: &PaymentSearchState, cost: &ManaCost, taps: Taps) -> bool {
        let mut optimistic = optimistic_pool(std::slice::from_ref(&state.pool), cost);
        let mut usable = u64::from(state.pool.usable_total(cost));
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            if state.resources.untapped[index] {
                if let Some(output) = &source.fixed {
                    optimistic.add(&optimistic_pool(std::slice::from_ref(output), cost));
                    usable += u64::from(output.usable_total(cost));
                }
                if source.pays_by_tap(taps) {
                    optimistic.add(&Mana {
                        mana: [0, 0, 0, 0, 0, 1],
                        ..Default::default()
                    });
                    usable += 1;
                }
            }
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let uses = self.uses(&state.resources, index, ability_index);
                if uses == 0 {
                    continue;
                }
                let upper = optimistic_pool(&ability.outputs, cost);
                for _ in 0..uses {
                    optimistic.add(&upper);
                }
                usable += uses as u64
                    * ability
                        .outputs
                        .iter()
                        .map(|m| u64::from(m.usable_total(cost)))
                        .max()
                        .unwrap_or(0);
            }
        }
        usable >= u64::from(cost.mana_value) && optimistic.can_pay(cost)
    }

    /// Hall's condition over the colors owed: for every set S of the mana
    /// kinds named by pending `Pay` pips (W/U/B/R/G and {C}), the pips payable
    /// only from S need that many units of mana in S. A pip is confined to S
    /// when it is a colored or {C} pip of S or a hybrid pip with both halves
    /// in S; generic pips and waterbend taps are left to the total bound.
    ///
    /// Supply is optimistic in the same way as the total bound: mana counts
    /// when the payment could spend it (`PaymentSearchState::usable`), one tap yields the best
    /// of the fixed output and the tapping abilities, and non-tapping
    /// abilities and pending outputs add their best choice. Unlike the
    /// per-cost check, each use contributes one output choice, not the union
    /// of its modes, so one "U or R" source cannot cover {U}{R}. Mana is never
    /// spent as another color during payment (`Mana::spending_choices`), so
    /// a failed set has no witness and pruning it keeps the first witness.
    fn colors_coverable(&self, state: &PaymentSearchState, search: &PaymentSearch) -> bool {
        #[cfg(test)]
        if search.total_bound_only {
            return true;
        }
        #[cfg(not(test))]
        let _ = search;
        const KINDS: usize = 6;
        let mut pips = [0_u64; KINDS];
        let mut hybrid: Vec<u8> = Vec::new();
        let mut owed_kinds = 0_u8;
        let mut pays = 0;
        for obligation in &state.obligations {
            let Obligation::Pay { cost, .. } = obligation else {
                continue;
            };
            pays += 1;
            for (kind, owed) in pips.iter_mut().enumerate() {
                *owed += u64::from(cost.cost[kind]);
                if cost.cost[kind] > 0 {
                    owed_kinds |= 1 << kind;
                }
            }
            for [left, right] in &cost.hybrid {
                let pip = 1 << (*left as usize) | 1 << (*right as usize);
                owed_kinds |= pip;
                hybrid.push(pip);
            }
        }
        // One color owed by one cost is already bounded by its own check.
        if owed_kinds == 0 || (owed_kinds.count_ones() < 2 && pays < 2) {
            return true;
        }
        let amounts = |mana: &Mana| state.usable(mana);
        // Each group is one exclusive choice: the supply of S is the pool
        // plus, per group, its best member counted inside S.
        let mut choices: Vec<[u64; KINDS]> = Vec::new();
        let mut groups: Vec<usize> = Vec::new();
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            // Fixed mana and tapping abilities compete for the same tap.
            let tap_group = choices.len();
            if state.resources.untapped[index] {
                if let Some(output) = &source.fixed {
                    choices.push(amounts(output));
                }
            }
            let mut separate: Vec<(u64, &[Mana])> = Vec::new();
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let uses = self.uses(&state.resources, index, ability_index) as u64;
                if uses == 0 {
                    continue;
                }
                if ability.tap_source {
                    choices.extend(ability.outputs.iter().map(&amounts));
                } else {
                    separate.push((uses, &ability.outputs));
                }
            }
            if choices.len() > tap_group {
                groups.push(choices.len());
            }
            for (uses, outputs) in separate {
                choices.extend(
                    outputs
                        .iter()
                        .map(|output| amounts(output).map(|amount| amount.saturating_mul(uses))),
                );
                groups.push(choices.len());
            }
        }
        for obligation in &state.obligations {
            if let Obligation::Output { choices: modes, .. } = obligation {
                choices.extend(modes.iter().map(&amounts));
                groups.push(choices.len());
            }
        }
        let pool = amounts(&state.pool);
        let within = |amounts: &[u64; KINDS], set: u8| -> u64 {
            (0..KINDS)
                .filter(|kind| set & (1 << kind) != 0)
                .fold(0_u64, |sum, kind| sum.saturating_add(amounts[kind]))
        };
        // Every non-empty subset of the owed kinds, largest first.
        let mut set = owed_kinds;
        while set != 0 {
            let owed =
                within(&pips, set) + hybrid.iter().filter(|pip| **pip & !set == 0).count() as u64;
            let mut supply = within(&pool, set);
            let mut start = 0;
            for &end in &groups {
                if supply >= owed {
                    break;
                }
                supply = supply.saturating_add(
                    choices[start..end]
                        .iter()
                        .map(|choice| within(choice, set))
                        .max()
                        .unwrap_or(0),
                );
                start = end;
            }
            if supply < owed {
                return false;
            }
            set = (set - 1) & owed_kinds;
        }
        true
    }

    fn activation_obligations(ability: &ManaActivation) -> Vec<Obligation> {
        let mut obligations = Vec::new();
        if ability.tap_count > 0 {
            obligations.push(Obligation::Tap {
                candidates: ability.tap_candidates.clone(),
                remaining: ability.tap_count,
            });
        }
        obligations.push(Obligation::mana(ability.cost.clone()));
        obligations.push(Obligation::Output {
            choices: ability.outputs.clone(),
            life_cost: ability.life_cost,
        });
        obligations
    }

    /// Consume one selected tap before searching the remaining obligations.
    /// Ordinary and nested activations use the same reserved resources and
    /// unpaid count, so neither can offer an object promised to another cost.
    fn plan_after_tap(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        permanent: PermanentId,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::begin_plan(self.sources.len());
        let path = self.cached_plan(pool, obligations, SourcePolicy::All, Some(permanent));
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::end_plan(&path);
        path
    }

    fn search_after_tap(
        &self,
        pool: Mana,
        mut obligations: Vec<Obligation>,
        permanent: PermanentId,
    ) -> Option<Vec<PaymentStep>> {
        let Obligation::Tap {
            candidates,
            remaining,
        } = obligations.first_mut()?
        else {
            return None;
        };
        let mut resources = self.resources();
        let index = self
            .tap_candidates(&resources, candidates)
            .find(|index| self.sources[*index].permanent == permanent)?;
        *remaining = remaining.checked_sub(1)?;
        resources.untapped[index] = false;
        let mut search = PaymentSearch::new(&obligations);
        self.witness(
            PaymentSearchState {
                resources,
                pool,
                life: self.life,
                obligations,
                prefix: None,
            },
            SourcePolicy::All,
            &mut search,
        )
    }

    /// The first witness for `obligations`, or `None` when they cannot be paid.
    fn plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::begin_plan(self.sources.len());
        let path = self.cached_plan(pool, obligations, policy, None);
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::end_plan(&path);
        path
    }

    /// The over-tap criterion, asked of the board after the payment's taps:
    /// without one source's complete `output` (triggered bonuses included),
    /// `pool` still holds `cost` and the saved callers in `tail` stay payable.
    /// Callers first check that the pool exceeds the cost at all.
    fn is_surplus(&self, pool: &Mana, output: &Mana, cost: &ManaCost, tail: &[Obligation]) -> bool {
        let mut remaining = pool.clone();
        remaining.spend(output);
        if !remaining.can_pay(cost) {
            return false;
        }
        if tail.is_empty() {
            return true;
        }
        let mut obligations = vec![Obligation::mana(cost.clone())];
        obligations.extend_from_slice(tail);
        self.payable(remaining, obligations, SourcePolicy::All)
    }

    fn cached_plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
        tap: Option<PermanentId>,
    ) -> Option<Vec<PaymentStep>> {
        // Arithmetic plans are cheaper than retaining and comparing a full
        // board key. Keep the cache for identity-preserving searches.
        if tap.is_none() {
            if let Some(direct) = self.direct(&obligations, policy) {
                let path = self.direct_witness(pool.clone(), &direct);
                debug_assert_eq!(path, self.search(pool, obligations.clone(), policy));
                return path;
            }
        }
        let uncached = || match tap {
            Some(permanent) => self.search_after_tap(pool.clone(), obligations.clone(), permanent),
            None => {
                #[cfg(test)]
                tests::SEARCHED.with(|searched| searched.set(searched.get() + 1));
                self.search(pool.clone(), obligations.clone(), policy)
            }
        };
        let Some(cache) = self.cache else {
            return uncached();
        };
        let cached = cache
            .borrow()
            .entries
            .iter()
            .rev()
            .find(|entry| {
                entry.life == self.life
                    && entry.policy == policy
                    && entry.tap == tap
                    && entry.pool == pool
                    && entry.obligations == obligations
                    && entry.sources == self.sources
            })
            .map(|entry| entry.path.clone());
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::cache(cached.is_some());
        if let Some(path) = cached {
            debug_assert_eq!(
                path,
                uncached(),
                "cached payment witness differs from exact query"
            );
            return path;
        }
        let path = uncached();
        let mut cache = cache.borrow_mut();
        if cache.entries.len() == 64 {
            cache.entries.pop_front();
        }
        cache.entries.push_back(Arc::new(CachedPayment {
            sources: self.sources.clone(),
            life: self.life,
            pool,
            obligations,
            policy,
            tap,
            path: path.clone(),
        }));
        path
    }

    /// The pool already includes the tap's output. Reserve its untapped
    /// resource before asking whether all remaining costs can be paid.
    fn payable_after_tap(
        &self,
        permanent: PermanentId,
        pool: Mana,
        obligations: Vec<Obligation>,
    ) -> bool {
        self.after_tap(permanent)
            .payable(pool, obligations, SourcePolicy::All)
    }

    /// This model once `permanent` is tapped for something else.
    fn after_tap(&self, permanent: PermanentId) -> Self {
        let mut after = PaymentModel {
            cache: None,
            sources: self.sources.clone(),
            life: self.life,
        };
        after.reserve(SourceReservation::tap(permanent));
        after
    }

    /// Whether `obligations` can be paid: `plan(..).is_some()` without
    /// building the witness when the direct answer applies.
    fn payable(&self, pool: Mana, obligations: Vec<Obligation>, policy: SourcePolicy) -> bool {
        let Some(direct) = self.direct(&obligations, policy) else {
            return self.plan(pool, obligations, policy).is_some();
        };
        let payable = self.direct_payable(&pool, &direct);
        debug_assert_eq!(
            payable,
            self.search(pool, obligations.clone(), policy).is_some(),
            "direct payment answer differs from the search"
        );
        payable
    }

    /// The question as the direct answer reads it, or `None` when only the
    /// search is exact.
    ///
    /// Direct class: the obligations are one mana cost, optionally followed
    /// by one generic-only split of the same purpose that taps may pay
    /// (`help_obligations`), and no source offers a searched ability.
    /// Tapping a fixed source then only adds mana, and mana of one purpose
    /// only adds ways to pay, so affordability is one sum and the search's
    /// first witness is a single pass in source order. `FixedOnly` never
    /// searches abilities, so a lone cost is direct whatever the sources
    /// offer; it never taps for waterbend, so a split is not. The automatic
    /// prefix minimizes taps instead of taking them in order and stays with
    /// the search, as do counted taps, pending outputs and life.
    fn direct<'a>(
        &self,
        obligations: &'a [Obligation],
        policy: SourcePolicy,
    ) -> Option<DirectPayment<'a>> {
        let direct = match obligations {
            [Obligation::Pay {
                cost,
                taps: Taps::None,
            }] => DirectPayment {
                cost,
                split: None,
                taps: Taps::None,
            },
            [Obligation::Pay {
                cost,
                taps: Taps::None,
            }, Obligation::Pay { cost: split, taps }]
                if *taps != Taps::None
                    && policy == SourcePolicy::All
                    && split.purpose == cost.purpose
                    && split.mana_value == split.generic() =>
            {
                DirectPayment {
                    cost,
                    split: Some(split),
                    taps: *taps,
                }
            }
            _ => return None,
        };
        let searched = match policy {
            SourcePolicy::FixedOnly => false,
            SourcePolicy::All => self
                .sources
                .iter()
                .any(|source| !source.abilities.is_empty()),
            SourcePolicy::AutomaticPrefix => true,
        };
        (!searched).then_some(direct)
    }

    fn direct_taps<'a>(
        &'a self,
        payment: &'a DirectPayment,
    ) -> impl Iterator<Item = DirectTap<'a>> + Clone {
        self.sources.iter().map(|source| {
            let output = source.fixed.as_ref().filter(|_| source.untapped);
            DirectTap {
                permanent: source.permanent,
                output,
                mana: output.map_or(0, |output| output.usable_total(payment.cost).into()),
                waterbend: source.untapped && source.pays_by_tap(payment.taps),
            }
        })
    }

    /// Affordability by arithmetic, as `(slack, spare taps)`: every fixed
    /// source tapped for mana pays the cost with `slack` usable mana left
    /// over, and that slack plus the spare taps covers the split. `None` when
    /// the payment cannot be made.
    fn direct_margin<'a>(
        pool: &Mana,
        taps: impl Iterator<Item = DirectTap<'a>> + Clone,
        payment: &DirectPayment,
    ) -> Option<(u32, u32)> {
        let mut total = pool.clone();
        for output in taps.clone().filter_map(|tap| tap.output) {
            total.add(output);
        }
        if !total.can_pay(payment.cost) {
            return None;
        }
        let slack = u32::from(total.usable_total(payment.cost))
            .checked_sub(payment.cost.mana_value.into())?;
        let spare_taps = taps.filter(|tap| tap.spare()).count() as u32;
        (slack + spare_taps >= payment.split_generic()).then_some((slack, spare_taps))
    }

    fn direct_payable(&self, pool: &Mana, payment: &DirectPayment) -> bool {
        Self::direct_margin(pool, self.direct_taps(payment), payment).is_some()
    }

    /// The witness the search returns first, built without searching.
    ///
    /// The search spends as soon as the pool pays the cost and the split can
    /// still be covered afterwards, and until then taps the first fixed source
    /// whose tap leaves the whole payment possible. That skips only a
    /// worthless mana tap of a waterbend source the split needs. After the
    /// spend it covers the split the same way, preferring each source's mana
    /// tap to its waterbend tap. A feasible state stays feasible along this
    /// path, so the search never backtracks out of it and one pass in source
    /// order visits every decision it makes.
    fn direct_witness(&self, mut pool: Mana, payment: &DirectPayment) -> Option<Vec<PaymentStep>> {
        let cost = payment.cost;
        let owed = u32::from(cost.mana_value);
        let mut need = payment.split_generic();
        let taps = self.direct_taps(payment).collect::<Vec<_>>();
        // Tapping a source for mana moves its output into the pool and
        // leaves the slack alone.
        let (slack, mut spare_taps) = Self::direct_margin(&pool, taps.iter().copied(), payment)?;
        // What the untapped sources can still add towards the split.
        let mut ahead = taps.iter().map(|tap| tap.worth()).sum::<u32>();
        let mut tapped = vec![false; taps.len()];
        let mut path = Vec::new();
        let mut index = 0;
        while !(pool.can_pay(cost) && u32::from(pool.usable_total(cost)) - owed + ahead >= need) {
            let tap = *taps
                .get(index)
                .expect("a payable cost is paid before its fixed sources run out");
            let needed_for_split = tap.spare() && slack + spare_taps <= need;
            if let Some(output) = tap.output.filter(|_| !needed_for_split) {
                pool.add(output);
                ahead -= tap.worth();
                spare_taps -= u32::from(tap.spare());
                tapped[index] = true;
                path.push(PaymentStep::Fixed(tap.permanent));
            }
            index += 1;
        }
        path.push(PaymentStep::Spend);
        let Some(split) = payment.split else {
            return Some(path);
        };
        // Both costs have the same purpose and the split is generic, so every
        // legal spending choice leaves the same usable total. The witness
        // never chooses units or commits a hypothetical pool remainder.
        let mut have = u32::from(pool.usable_total(split)) - owed;
        for (tap, tapped) in taps.iter().zip(tapped) {
            if have >= need {
                break;
            }
            if tapped {
                continue;
            }
            let rest = ahead - tap.worth();
            match tap.output {
                Some(_) if have + tap.mana + rest >= need => {
                    have += tap.mana;
                    path.push(PaymentStep::Fixed(tap.permanent));
                }
                _ if tap.waterbend => {
                    need -= 1;
                    path.push(PaymentStep::Waterbend(tap.permanent));
                }
                _ => continue,
            }
            ahead = rest;
        }
        path.push(PaymentStep::Spend);
        Some(path)
    }

    /// Before AutomaticPrefix's first Spend, only fixed production and
    /// waterbend can add payment capacity. Even ignoring colors and counted
    /// tap competition, the largest outputs must cover the pool's shortfall.
    /// Budgets below this lower bound have no witness. Other leading
    /// obligations retain the ordinary search's zero start.
    fn prefix_lower_bound(&self, pool: &Mana, obligations: &[Obligation]) -> usize {
        let Some(Obligation::Pay { cost, taps }) = obligations.first() else {
            return 0;
        };
        let mut need =
            u64::from(cost.mana_value).saturating_sub(u64::from(pool.usable_total(cost)));
        if need == 0 {
            return 0;
        }
        let mut outputs: Vec<u64> = self
            .sources
            .iter()
            .filter(|source| source.untapped)
            .map(|source| {
                let fixed = source
                    .fixed
                    .as_ref()
                    .map_or(0, |output| u64::from(output.usable_total(cost)));
                fixed.max(u64::from(source.pays_by_tap(*taps) && cost.generic() > 0))
            })
            .collect();
        outputs.sort_unstable_by(|left, right| right.cmp(left));
        for (count, output) in outputs.into_iter().enumerate() {
            need = need.saturating_sub(output);
            if need == 0 {
                return count + 1;
            }
        }
        // No automatic prefix can fill the shortfall.
        self.sources.len() + 1
    }

    fn search(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
        let mut search = PaymentSearch::new(&obligations);
        // Offers need one executable witness, not a proof of the fewest taps.
        // Only a nested automatic prefix minimizes early commitments.
        if policy != SourcePolicy::AutomaticPrefix {
            return self.witness(
                PaymentSearchState {
                    resources: self.resources(),
                    pool,
                    life: self.life,
                    obligations,
                    prefix: None,
                },
                policy,
                &mut search,
            );
        }
        // Minimize taps before the first Spend without starving a saved caller.
        for budget in self.prefix_lower_bound(&pool, &obligations)..=self.sources.len() {
            search.visited.clear();
            if let Some(path) = self.witness(
                PaymentSearchState {
                    resources: self.resources(),
                    pool: pool.clone(),
                    life: self.life,
                    obligations: obligations.clone(),
                    prefix: Some(budget),
                },
                policy,
                &mut search,
            ) {
                return Some(path);
            }
        }
        None
    }
}

/// A pending payment's hold on its own source, from what it still owes.
fn owed_reservation(payment: &Payment) -> Option<SourceReservation> {
    let Operation::Activation {
        source: ActivationSource::Permanent(permanent),
        ..
    } = payment.operation
    else {
        return None;
    };
    let owes = |how| payment.owes(&CostComponent::Source(how));
    let tap = owes(How::Tapped);
    Some(SourceReservation {
        permanent,
        tap,
        sacrifice: tap || owes(How::Sacrificed) || owes(How::Exiled) || owes(How::Returned),
    })
}

impl Game {
    /// Mana abilities resolve at once (CR 605.3a), so an output that depends
    /// on the game is fixed when the ability is offered: a branch between two
    /// immediate outputs, or X mana of one chosen color. Activation costs do
    /// not change these readings for current content.
    pub(crate) fn current_mana_output(
        &self,
        effect: &Effect,
        controller: PlayerId,
    ) -> Option<Effect> {
        match effect {
            Effect::If {
                condition,
                then,
                otherwise,
            } if [then, otherwise].iter().all(|branch| {
                matches!(
                    branch.as_slice(),
                    [Effect::AddMana {
                        until_end_of_combat: false,
                        ..
                    }]
                )
            }) =>
            {
                let branch = if self.check_static_condition(condition, controller) {
                    then
                } else {
                    otherwise
                };
                Some(branch[0].clone())
            }
            Effect::AddOneColorManaValue { amount } => {
                let amount = self.controller_value(amount, controller).max(0) as usize;
                if amount == 0 {
                    return Some(Effect::add_mana(Mana::default()));
                }
                Some(Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(|color| Mana::parse(&color.repeat(amount))),
                ))
            }
            _ => None,
        }
    }

    /// The activation's mana cost as paid now: per-permanent reductions apply
    /// to its generic part (CR 601.2f, 602.2b). Its purpose already names the
    /// source (CR 106.6).
    pub(crate) fn price_activation(
        &self,
        controller: PlayerId,
        ability: &mut ActivatedAbilityDefinition,
    ) {
        if let Some(predicate) = &ability.cost_reduction_per {
            let count = self.count_battlefield_matching(controller, predicate);
            ability.mana_cost = ability
                .mana_cost
                .reduced_generic(u8::try_from(count).unwrap_or(u8::MAX));
        }
    }

    pub(crate) fn capture_priority_mana(&mut self, player: PlayerId) {
        if self
            .card_capture
            .as_ref()
            .and_then(|cards| cards.exposure.as_ref())
            .is_none()
            || player != self.active_player()
        {
            return;
        }
        let model = self.payment_model(player, None);
        let (_, net) = model.source_bounds(&model.resources(), model.life, |_| 0, |_| false);
        let supply = net + u64::from(self.pooled_mana(player).total());
        if let Some(turn) = self
            .exposure_mut()
            .and_then(|capture| capture.mana_turns[player.0].last_mut())
        {
            turn.available = turn.available.max(turn.paid + supply);
        }
    }

    fn payment_model(&self, player: PlayerId, sacrifice: Option<PermanentId>) -> PaymentModel<'_> {
        let _scan = self.board_scan();
        let mut model = PaymentModel {
            cache: Some(&self.payment_cache),
            sources: self.shared_payment_sources(player),
            life: self.payable_life(player).max(0) as u32,
        };
        if let Some(source) = sacrifice {
            model.reserve(SourceReservation {
                permanent: source,
                tap: false,
                sacrifice: true,
            });
        }
        // An announcement survives nested mana decisions in pending_choice;
        // its resources are owed in every phase, not only mana spending.
        if let Some(PendingChoice::Pay { payment, .. }) = &self.pending_choice {
            self.reserve_announced_payment(payment, &mut model);
            self.mark_help_taps(&mut model, payment.help());
        }
        fn reserve(decision: &Decision, game: &Game, model: &mut PaymentModel) {
            match decision {
                Decision::ManaSpend { completion, .. } => match completion.as_ref() {
                    SpendCompletion::Upfront(next) => reserve(next, game, model),
                    SpendCompletion::Announced(payment) => {
                        game.reserve_announced_payment(payment, model);
                    }
                    _ => {}
                },
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    resume,
                    ..
                } => {
                    let ability =
                        &game.characteristics(*permanent).activated_abilities[*ability_index];
                    model.reserve(SourceReservation {
                        permanent: *permanent,
                        tap: ability.tap_source,
                        sacrifice: ability.tap_source || ability.source_leaves(),
                    });
                    if let Some(next) = resume {
                        reserve(next, game, model);
                    }
                }
                Decision::ManaChoice {
                    resume: Some(next), ..
                } => reserve(next, game, model),
                _ => {}
            }
        }
        if let Some(suspended) = &self.state.suspended_decision {
            reserve(&suspended.decision, self, &mut model);
        }
        model
    }

    /// Begin one legal-action computation; see `PaymentScan`.
    pub(crate) fn payment_scan(&self) -> PaymentScan<'_> {
        self.payment_sources.borrow_mut().scans += 1;
        PaymentScan { game: self }
    }

    /// `player`'s unreserved sources, reused inside a `PaymentScan`.
    fn shared_payment_sources(&self, player: PlayerId) -> Arc<Vec<ManaSource>> {
        if let Some(sources) = &self.payment_sources.borrow().sources[player.0] {
            debug_assert_eq!(
                **sources,
                self.payment_sources(player),
                "stale shared payment sources"
            );
            return sources.clone();
        }
        let sources = Arc::new(self.payment_sources(player));
        let mut shared = self.payment_sources.borrow_mut();
        if shared.scans > 0 {
            shared.sources[player.0] = Some(sources.clone());
        }
        sources
    }

    /// Every permanent `player` controls as a mana source, before any
    /// reservation: a function of the game alone.
    fn payment_sources(&self, player: PlayerId) -> Vec<ManaSource> {
        let _scan = self.board_scan();
        let triggers = self.mana_triggers();
        self.characteristics_many(self.iter_battlefield_permanents(player))
            .map(|(permanent, characteristics)| {
                let source = self.state.permanents[permanent].as_ref().unwrap();
                // A source that lost its abilities offers no zero-mana tap.
                let can_tap = self.permanent_can_tap(permanent);
                let prohibited = self.activations_prohibited(permanent);
                let fixed = (can_tap && !prohibited && !characteristics.mana_abilities.is_empty())
                    .then(|| self.fixed_source_mana(player, permanent, &characteristics, &triggers))
                    .flatten();
                let abilities = characteristics
                    .activated_abilities
                    .iter()
                    .enumerate()
                    .filter(|(index, ability)| {
                        !prohibited
                            && (!ability.tap_source || can_tap)
                            && (!ability.own_turn_only || self.active_player() == player)
                            && !source.activation_used(*index, ability)
                    })
                    .filter_map(|(index, ability)| {
                        let mut outputs = ability.mana_choices()?;
                        if ability.tap_source {
                            let bonuses = self.tap_bonuses(&triggers, player, permanent);
                            outputs = production_alternatives(outputs, &bonuses);
                        }
                        Some(ManaActivation {
                            tap_source: ability.tap_source,
                            tap_count: ability.tap_cost.as_ref().map_or(0, |cost| cost.count),
                            tap_candidates: self.tap_cost_candidates(player, ability).into(),
                            limited: ability.is_use_limited(),
                            sacrifice_source: ability.sacrifice_source,
                            ability_index: index,
                            cost: ability.mana_cost.clone(),
                            outputs: outputs.into(),
                            life_cost: ability.life_cost,
                        })
                    })
                    .collect();
                ManaSource {
                    permanent,
                    object: source.object,
                    untapped: !source.tapped,
                    fixed,
                    abilities,
                    keyword: false,
                    waterbend: !source.tapped
                        && (characteristics.types.is_artifact() || characteristics.is_creature()),
                }
            })
            .collect()
    }

    /// Reserve costs still owed by an announced payment: its source's own
    /// tap or sacrifice, its life, and the permanents already chosen to be
    /// sacrificed as it completes. ManaUpfront's life fee instead lives on
    /// its Output obligation and must not be debited here.
    fn reserve_announced_payment(&self, payment: &Payment, model: &mut PaymentModel) {
        if let Some(reservation) = owed_reservation(payment) {
            model.reserve(reservation);
        }
        model.life = model.life.saturating_sub(payment.life());
        let grantors = payment.owed.iter().filter_map(|component| match component {
            CostComponent::ExileGrantor(object) => Some(*object),
            _ => None,
        });
        for object in payment.receipt.sacrificed().chain(grantors) {
            if let Some(permanent) = self.state.permanent_of(object) {
                model.reserve(SourceReservation {
                    permanent,
                    tap: false,
                    sacrifice: true,
                });
            }
        }
    }

    /// The model for paying `payment`, or `None` when its life cost alone
    /// cannot be paid. The pending payment is reserved by every model; one
    /// only being considered at priority is reserved here.
    fn announced_model(&self, player: PlayerId, operation: &Operation) -> Option<PaymentModel<'_>> {
        let mut model = self.payment_model(player, None);
        if self.paying().is_some() {
            return Some(model);
        }
        if let Some(reservation) = self.payment_reservation(operation) {
            model.reserve(reservation);
        }
        if let Operation::Activation {
            source,
            ability_index,
            ..
        } = operation
        {
            let ability = self.activation_ability(*source, *ability_index)?;
            if ability.exile_grantor {
                model.reserve(SourceReservation {
                    permanent: self.state.permanent_of(ability.granted_by?.grantor)?,
                    tap: false,
                    sacrifice: true,
                });
            }
            model.life = model.life.checked_sub(ability.life_cost)?;
        }
        Some(model)
    }

    /// The hold an activation only being considered at priority would place
    /// on its own source; once announced, `owed_reservation` reads its cost.
    fn payment_reservation(&self, operation: &Operation) -> Option<SourceReservation> {
        match operation {
            Operation::Activation {
                source: ActivationSource::Permanent(permanent),
                ability_index,
                ..
            } => {
                let ability = &self.characteristics(*permanent).activated_abilities[*ability_index];
                Some(SourceReservation {
                    permanent: *permanent,
                    tap: ability.tap_source,
                    sacrifice: ability.tap_source || ability.source_leaves(),
                })
            }
            _ => None,
        }
    }

    fn payment_obligations(&self, player: PlayerId) -> Vec<Obligation> {
        self.payment_obligations_owing(player, None)
    }

    /// The obligations with the payment in progress owing `owed` mana
    /// instead of what it still records.
    fn payment_obligations_owing(
        &self,
        player: PlayerId,
        owed: Option<&ManaCost>,
    ) -> Vec<Obligation> {
        fn append(decision: &Decision, game: &Game, result: &mut Vec<Obligation>) {
            match decision {
                Decision::ManaSpend {
                    cost, completion, ..
                } => {
                    result.push(Obligation::mana(cost.clone()));
                    if let SpendCompletion::Upfront(next) = completion.as_ref() {
                        let mut saved = Vec::new();
                        append(next, game, &mut saved);
                        result.extend(saved.into_iter().skip(1));
                    }
                }
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    tap_remaining,
                    resume,
                    ..
                } => {
                    let ability = game.characteristics(*permanent).activated_abilities
                        [*ability_index]
                        .clone();
                    let mut outputs = ability.mana_choices().unwrap();
                    if ability.tap_source {
                        let bonuses =
                            game.tap_bonuses(&game.mana_triggers(), decision.player(), *permanent);
                        outputs = production_alternatives(outputs, &bonuses);
                    }
                    if *tap_remaining > 0 {
                        result.push(Obligation::Tap {
                            candidates: game
                                .tap_cost_candidates(decision.player(), &ability)
                                .into(),
                            remaining: *tap_remaining,
                        });
                    }
                    result.push(Obligation::mana(ability.mana_cost));
                    result.push(Obligation::Output {
                        choices: outputs.into(),
                        life_cost: ability.life_cost,
                    });
                    if let Some(next) = resume {
                        append(next, game, result);
                    }
                }
                Decision::ManaChoice {
                    resume: Some(next), ..
                } => append(next, game, result),
                // A life cost is paid or declined in one action: it never
                // commits, so it never reaches the solver as an obligation.
                Decision::Payment(progress) if !progress.can_decline() => {
                    if let CostComponent::Mana { cost, .. } = &progress.owed {
                        result.push(Obligation::Pay {
                            cost: cost.clone(),
                            taps: if progress.waterbends() {
                                Taps::Any
                            } else {
                                Taps::None
                            },
                        });
                    }
                }
                _ => {}
            }
        }
        let mut result = Vec::new();
        if let Some(suspended) = &self.state.suspended_decision {
            append(&suspended.decision, self, &mut result);
        }
        match self.paying() {
            Some((payer, payment, CostComponent::Mana { .. })) if payer == player => {
                let (recorded, waterbend) = payment.mana();
                result.extend(self.help_obligations(
                    player,
                    owed.unwrap_or(recorded),
                    payment.help(),
                    waterbend,
                ));
            }
            _ => {}
        }
        result
    }

    fn activation_witness(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Option<Vec<PaymentStep>> {
        let mut model = self.payment_model(player, None);
        let source = model
            .sources
            .iter()
            .find(|source| source.permanent == permanent)?;
        let ability = source
            .abilities
            .iter()
            .find(|ability| ability.ability_index == ability_index)
            .cloned()?;
        model.reserve(SourceReservation {
            permanent,
            tap: ability.tap_source,
            sacrifice: ability.tap_source || ability.sacrifice_source,
        });
        let obligations = PaymentModel::activation_obligations(&ability);
        if ability.limited {
            for source in Arc::make_mut(&mut model.sources) {
                if source.permanent == permanent {
                    source
                        .abilities
                        .retain(|a| a.ability_index != ability_index);
                }
            }
        }
        let pool = self.pooled_mana(player);
        let mut path = self.while_paying(player, &mut model, &mut |model, owed| {
            let mut obligations = obligations.clone();
            obligations.extend(owed);
            model.plan(pool.clone(), obligations, SourcePolicy::All)
        })?;
        path.insert(0, PaymentStep::Activate(permanent, ability_index));
        Some(path)
    }

    pub(crate) fn can_activate_mana_ability(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> bool {
        self.activation_witness(player, permanent, ability_index)
            .is_some()
    }

    pub(crate) fn mana_ability_actions(&self, player: PlayerId) -> Vec<Action> {
        self.payment_model(player, None)
            .sources
            .iter()
            .flat_map(|source| {
                source.abilities.iter().filter_map(|ability| {
                    let ability_index = ability.ability_index;
                    self.activation_witness(player, source.permanent, ability_index)
                        .map(|_| Action::ActivateAbility {
                            player,
                            permanent: source.permanent,
                            ability_index,
                        })
                })
            })
            .collect()
    }

    /// Append fixed sources after the ordinary offers. A committed payment
    /// must still have a witness after the tap, including any saved callers.
    pub(crate) fn append_mana_floats(&self, player: PlayerId, actions: &mut Vec<Action>) {
        if self.mana_float_mode(player).is_off()
            || (self.automatic_mana_payment(player)
                && (self.pending_choice.is_some() || self.state.suspended_decision.is_some()))
        {
            return;
        }
        let obligations = self.payment_obligations(player);
        let model = self.payment_model(player, None);
        let ordinary_len = actions.len();
        for source in model.sources.iter().filter(|source| source.untapped) {
            let Some(output) = &source.fixed else {
                continue;
            };
            let tap = Action::TapForMana {
                player,
                permanent: source.permanent,
            };
            if actions[..ordinary_len].contains(&tap) {
                continue;
            }
            if !obligations.is_empty() {
                let mut pool = self.pooled_mana(player);
                pool.add(output);
                let mut after = model.after_tap(source.permanent);
                let payable = self.while_paying(player, &mut after, &mut |model, owed| {
                    model
                        .payable(pool.clone(), owed, SourcePolicy::All)
                        .then_some(())
                });
                if payable.is_none() {
                    continue;
                }
            }
            actions.push(tap);
        }
    }

    /// Whether the seat deciding for `player` was set up with mana tap offers.
    pub(crate) fn offers_extra_payment_taps(&self, player: PlayerId) -> bool {
        !self.mana_float_mode(player).is_off() && !self.automatic_mana_payment(player)
    }

    pub(crate) fn automatic_mana_payment(&self, player: PlayerId) -> bool {
        self.mana_payment != [ManaPayment::Manual; 2]
            && self.mana_payment[self.decision_controller(player).0] == ManaPayment::Auto
    }

    /// Follow one step of the live payment witness. The automatic payer pays
    /// mana with mana and never picks an object: units of one mana class are
    /// interchangeable, creatures and cards are not. Every other component is
    /// the player's, and so is a convoke or improvise tap or a delve exile
    /// when mana alone cannot pay; the payer resumes once it can. Waterbend
    /// taps are the one exception and are still made for the player. Optional
    /// costs still offer Pay/Decline, and a deliberate priority float still
    /// chooses its output.
    pub(crate) fn automatic_payment_action(&self, space: &ActionSpace) -> Option<Action> {
        let player = space.player?;
        if !self.automatic_mana_payment(player) {
            return None;
        }
        // Whether convoke, improvise or delve may pay part of this mana step.
        let helped = match self.state.suspended_decision.as_ref().map(|s| &s.decision) {
            Some(Decision::Payment(progress)) if !progress.can_decline() => false,
            Some(Decision::ManaUpfront { .. }) => false,
            // The most specific unit that can pay goes first, so mana only
            // this kind of cost can use is not left floating beside
            // ordinary mana; a rider this cost would waste goes last. The
            // offers keep their order: this is the automatic payer's choice.
            Some(Decision::ManaSpend { cost, .. }) => {
                let order = |choice: &ManaSpendChoice| {
                    choice
                        .unit
                        .classes()
                        .first()
                        .map_or(SpendOrder::Ordinary, |(class, _)| {
                            class.spend_order(&cost.purpose)
                        })
                };
                let choices = self.mana_spend_choices(player, cost);
                let mode = (0..choices.len()).min_by_key(|mode| order(&choices[*mode]))?;
                return space.actions.get(mode).cloned();
            }
            Some(Decision::ManaChoice {
                choices, resume, ..
            }) if resume.is_some() || self.pending_choice.is_some() => {
                return space
                    .actions
                    .iter()
                    .min_by_key(|action| {
                        let Action::ChooseMode { mode, .. } = action else {
                            unreachable!()
                        };
                        let (pool, obligations) = self.after_mana_choice(player, &choices[*mode]);
                        self.payment_model(player, None)
                            .plan(pool, obligations, SourcePolicy::All)
                            .map_or(usize::MAX, |remaining| remaining.len())
                    })
                    .cloned();
            }
            None => match self.paying() {
                Some((_, _, CostComponent::Mana { help, .. })) => help.keyworded(),
                _ => return None,
            },
            _ => return None,
        };
        if let Some(pay) = space
            .actions
            .iter()
            .find(|a| matches!(a, Action::PayCost { .. }))
        {
            return Some(pay.clone());
        }
        // Mana alone cannot pay: which creature or artifact to tap, or card
        // to exile, is the player's pick.
        if helped {
            return None;
        }
        // The witness models the ordinary and waterbend portions as separate
        // spends, but the game charges their combined remainder only once.
        // Every offered waterbend tap preserves a witness for that remainder.
        if let Some(tap) = space
            .actions
            .iter()
            .find(|a| matches!(a, Action::WaterbendTap { .. }))
        {
            return Some(tap.clone());
        }
        let path = self
            .payment_model(player, None)
            .plan(
                self.pooled_mana(player),
                self.payment_obligations(player),
                SourcePolicy::All,
            )
            .expect("committed cost has a payment witness");
        let next = if self
            .paying()
            .is_some_and(|(_, payment, _)| payment.waterbends())
            && self.state.suspended_decision.is_none()
        {
            path.iter().find(|step| !matches!(step, PaymentStep::Spend))
        } else {
            path.first()
        };
        match next.expect("committed payment has a next step") {
            PaymentStep::Spend => Some(Action::PayCost { player }),
            PaymentStep::Fixed(permanent) => Some(Action::TapForMana {
                player,
                permanent: *permanent,
            }),
            step => step.action(player),
        }
    }

    pub(crate) fn mana_float_mode(&self, player: PlayerId) -> ManaFloats {
        if self.mana_floats == [ManaFloats::Off; 2] {
            return ManaFloats::Off;
        }
        self.mana_floats[self.decision_controller(player).0]
    }

    fn has_mana_floats(&self, player: PlayerId) -> bool {
        let mut taps = Vec::new();
        self.append_mana_floats(player, &mut taps);
        !taps.is_empty()
    }

    /// Ordinary source choices make progress toward the announced cost.
    pub(crate) fn useful_payment_taps(&self, player: PlayerId) -> Vec<Action> {
        let Some((payer, payment, CostComponent::Mana { cost, .. })) = self.paying() else {
            return Vec::new();
        };
        if payer != player || payment.waterbends() {
            return Vec::new();
        }
        let cost = cost.clone();
        let pool = self.pooled_mana(player);
        let spendable = pool.spendable_units(&cost);
        let missing = u16::from(cost.mana_value).saturating_sub(spendable);
        if missing == 0 {
            return Vec::new();
        }
        let model = self.payment_model(player, None);
        let tail: Vec<_> = self
            .payment_obligations(player)
            .into_iter()
            .skip(1)
            .collect();
        // With one payment already covered by fixed sources, a tap just moves
        // that source's output into the pool. The aggregate resources do not
        // change, so every useful tap preserves affordability. Nested callers
        // and payments needing activated sources still use the exact model.
        let obligations = vec![Obligation::mana(cost.clone())];
        let helped = payment.help().keyworded();
        let fixed_only = !helped
            && tail.is_empty()
            && model.payable(pool.clone(), obligations.clone(), SourcePolicy::FixedOnly);
        let mut offered_units = 0_u16;
        let actions: Vec<Action> = model
            .sources
            .iter()
            .filter(|source| source.untapped)
            .filter_map(|source| {
                let output = source.fixed.as_ref()?;
                let mut tapped = pool.clone();
                tapped.add(output);
                if tapped.spendable_units(&cost) <= spendable {
                    return None;
                }
                let payable = || {
                    if helped {
                        // Taps and exiled cards pay part of this cost: ask
                        // with them, not for the whole cost in mana.
                        let mut after = model.after_tap(source.permanent);
                        return self
                            .while_paying(player, &mut after, &mut |model, owed| {
                                model
                                    .payable(tapped.clone(), owed, SourcePolicy::All)
                                    .then_some(())
                            })
                            .is_some();
                    }
                    let mut obligations = obligations.clone();
                    obligations.extend_from_slice(&tail);
                    model.payable_after_tap(source.permanent, tapped.clone(), obligations)
                };
                if fixed_only {
                    debug_assert!(payable(), "a fixed tap changed aggregate affordability");
                } else if !payable() {
                    return None;
                }
                offered_units = offered_units.saturating_add(output.total());
                Some(Action::TapForMana {
                    player,
                    permanent: source.permanent,
                })
            })
            .collect();
        // With no more units on offer than the cost needs, paying is the
        // only choice. A helped cost chooses between a tap for mana and a
        // tap or card that pays directly, so its taps are always offered.
        if offered_units <= missing && (!helped || self.help_actions(player).is_empty()) {
            return Vec::new();
        }
        actions
    }

    /// Tap one offered fixed source; an announced payment then advances
    /// (`advance_payment`), finishing once no mana is owed.
    pub(crate) fn tap_for_mana_action(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
    ) -> Result<(), AgentError> {
        let before = (self.pending_choice.is_none()
            && self.state.suspended_decision.is_none()
            && self.mana_capture(player).is_some())
        .then(|| self.state.players[player.0].mana_pool.total());
        let start = self.state.players[player.0].mana_pool.unit_count();
        self.tap_fixed_mana_source(player, permanent);
        if let Some(before) = before {
            let produced = self.state.players[player.0].mana_pool.total() - before;
            self.mana_capture(player).unwrap().float(produced);
        }
        if self.state.suspended_decision.is_some() {
            return self.advance_mana_upfront();
        }
        self.commit_produced_mana(player, start);
        match self.paying() {
            Some((_, _, CostComponent::Mana { .. })) => {
                let payment = self.pending_payment(player)?.clone();
                return self.advance_payment(player, payment);
            }
            None => self.state.priority_mut().on_non_pass_action(player),
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn has_mana_payment_choices(&self, player: PlayerId) -> bool {
        !self.mana_ability_actions(player).is_empty()
            || !self.useful_payment_taps(player).is_empty()
            || self.has_mana_floats(player)
    }

    pub(crate) fn can_pay_with_mana_abilities(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        sacrifice: Option<PermanentId>,
    ) -> bool {
        self.payment_model(player, sacrifice).payable(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::All,
        )
    }

    /// Feasibility of `payment`'s chosen-object tap cost and its mana
    /// together. The owed taps include the hypothetical `selected` one; the
    /// shared transition checks and consumes it before searching the rest.
    pub(crate) fn tap_activation_payment_possible(
        &self,
        player: PlayerId,
        payment: &Payment,
        selected: Option<PermanentId>,
    ) -> bool {
        let Some((tap, toward)) = payment.owed.iter().find_map(|component| match component {
            CostComponent::Tap(tap, toward) => Some((tap, *toward)),
            _ => None,
        }) else {
            return false;
        };
        let Some(model) = self.announced_model(player, &payment.operation) else {
            return false;
        };
        let (owed, waterbend) = payment.mana();
        let mut obligations = vec![Obligation::Tap {
            candidates: self
                .tap_component_candidates(player, &payment.operation, tap, toward)
                .into(),
            remaining: tap.count,
        }];
        obligations.extend(self.help_obligations(player, owed, payment.help(), waterbend));
        let pool = self.pooled_mana(player);
        match selected {
            Some(permanent) => model.plan_after_tap(pool, obligations, permanent).is_some(),
            None => model.payable(pool, obligations, SourcePolicy::All),
        }
    }

    /// Whether `operation`, considered at priority, can pay `cost` beside
    /// its source's own costs and a `sacrifice` chosen for it.
    pub(crate) fn can_pay_announced_activation(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        operation: &Operation,
        sacrifice: Option<PermanentId>,
    ) -> bool {
        let Some(mut model) = self.announced_model(player, operation) else {
            return false;
        };
        if let Some(permanent) = sacrifice {
            model.reserve(SourceReservation {
                permanent,
                tap: false,
                sacrifice: true,
            });
        }
        model.payable(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::All,
        )
    }

    /// Whether choosing `mana` at the pending `ManaChoice` leaves a complete
    /// witness, counting the triggered mana abilities that still resolve
    /// after it from the same production.
    pub(crate) fn mana_choice_completes_payment(&self, player: PlayerId, mana: &Mana) -> bool {
        let (pool, obligations) = self.after_mana_choice(player, mana);
        self.payment_model(player, None)
            .payable(pool, obligations, SourcePolicy::All)
    }

    fn after_mana_choice(&self, player: PlayerId, mana: &Mana) -> (Mana, Vec<Obligation>) {
        let mut pool = self.pooled_mana(player);
        pool.add(mana);
        let mut obligations = self.payment_obligations(player);
        if let Some(SuspendedResolution {
            decision: Decision::ManaChoice { production, .. },
            ..
        }) = &self.state.suspended_decision
        {
            let (produced, bonuses) = match production {
                ManaProduction::Output {
                    tapped: Some(tapped),
                    ..
                } => (mana, self.production_bonuses(player, tapped)),
                ManaProduction::Output { .. } => (mana, Vec::new()),
                ManaProduction::Bonus {
                    produced, pending, ..
                } => (produced, pending.clone()),
            };
            let bonuses: Vec<_> = bonuses.into_iter().map(|bonus| bonus.output).collect();
            let after = bonus_alternatives(produced, &bonuses);
            if after != [Mana::default()] {
                obligations.insert(
                    0,
                    Obligation::Output {
                        choices: after.into(),
                        life_cost: 0,
                    },
                );
            }
        }
        (pool, obligations)
    }

    pub(crate) fn activate_mana_ability(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Result<(), AgentError> {
        if !self.can_activate_mana_ability(player, permanent, ability_index) {
            return Err(AgentError("illegal mana activation".into()));
        }
        let ability = self.characteristics(permanent).activated_abilities[ability_index].clone();
        self.record_activation(permanent, ability_index, &ability);
        let (frame, resume) = match self.state.suspended_decision_mut().take() {
            Some(caller) => (caller.frame, Some(Box::new(caller.decision))),
            None => (EffectFrame::new(player, Vec::new()), None),
        };
        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
            frame,
            decision: Decision::ManaUpfront {
                player,
                permanent,
                ability_index,
                tap_remaining: ability.tap_cost.as_ref().map_or(0, |cost| cost.count),
                resume,
            },
        });
        self.advance_mana_upfront()
    }

    /// Keep only selections that leave a complete witness for every caller.
    pub(crate) fn mana_tap_cost_actions(&self, player: PlayerId) -> Vec<Action> {
        let obligations = self.payment_obligations(player);
        let Some(Obligation::Tap { candidates, .. }) = obligations.first() else {
            return Vec::new();
        };
        let model = self.payment_model(player, None);
        let pool = self.pooled_mana(player);
        let mut actions = Vec::new();
        for permanent in candidates.iter() {
            if model
                .plan_after_tap(pool.clone(), obligations.clone(), *permanent)
                .is_some()
            {
                actions.push(Action::SelectCard {
                    player,
                    object: self.state.permanents[*permanent].as_ref().unwrap().object,
                });
            }
        }
        actions
    }

    pub(crate) fn pay_mana_tap_cost(
        &mut self,
        player: PlayerId,
        object: ObjectId,
    ) -> Result<(), AgentError> {
        if !self
            .mana_tap_cost_actions(player)
            .contains(&Action::SelectCard { player, object })
        {
            return Err(AgentError("illegal mana tap cost selection".into()));
        }
        let permanent = self.state.permanent_of(object).unwrap();
        self.tap_permanent(permanent, false);
        if let Some(SuspendedResolution {
            decision: Decision::ManaUpfront { tap_remaining, .. },
            ..
        }) = self.state.suspended_decision_mut()
        {
            *tap_remaining -= 1;
        }
        self.advance_mana_upfront()
    }

    /// Commit the automatic fixed-source prefix, then enter native spending.
    /// Neither spending nor chosen output is selected by a search witness.
    pub(crate) fn advance_mana_upfront(&mut self) -> Result<(), AgentError> {
        let Some(suspended) = &self.state.suspended_decision else {
            return Ok(());
        };
        let Decision::ManaUpfront {
            player,
            permanent,
            ability_index,
            tap_remaining,
            ..
        } = suspended.decision
        else {
            return Ok(());
        };
        if tap_remaining > 0 {
            return Ok(());
        }
        // With taps on, a mana-costed activation waits for the payer's taps
        // or PayCost instead of committing the automatic prefix.
        if self.characteristics(permanent).activated_abilities[ability_index]
            .mana_cost
            .mana_value
            > 0
            && self.has_mana_floats(player)
        {
            return Ok(());
        }
        self.pay_mana_upfront(player)
    }

    fn automatic_upfront_path(&self, player: PlayerId) -> Option<Vec<PaymentStep>> {
        let path = self.payment_model(player, None).plan(
            self.pooled_mana(player),
            self.payment_obligations(player),
            SourcePolicy::AutomaticPrefix,
        )?;
        path.iter()
            .take_while(|step| !matches!(step, PaymentStep::Spend))
            .all(|step| step.action(player).is_none())
            .then_some(path)
    }

    pub(crate) fn can_pay_mana_upfront(&self, player: PlayerId) -> bool {
        self.automatic_upfront_path(player).is_some()
    }

    pub(crate) fn pay_mana_upfront(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(path) = self.automatic_upfront_path(player) else {
            return Ok(());
        };
        let Decision::ManaUpfront {
            permanent,
            ability_index,
            ..
        } = self.state.suspended_decision.as_ref().unwrap().decision
        else {
            unreachable!()
        };
        for step in path {
            match step {
                PaymentStep::Fixed(source) => self.tap_fixed_mana_source(player, source),
                PaymentStep::Spend => break,
                _ => unreachable!("validated automatic prefix"),
            }
        }
        let cost = self.characteristics(permanent).activated_abilities[ability_index]
            .mana_cost
            .clone();
        let caller = self.state.suspended_decision_mut().take().unwrap();
        self.begin_mana_spend(
            player,
            cost,
            SpendCompletion::Upfront(Box::new(caller.decision)),
            caller.frame,
        )
    }

    /// Finish the paid activation using its owned caller. Costs commit once;
    /// only the output decision is restored after the source's effects run.
    fn complete_mana_upfront(&mut self, mut caller: SuspendedResolution) {
        let Decision::ManaUpfront {
            player,
            permanent,
            ability_index,
            resume,
            ..
        } = caller.decision
        else {
            unreachable!()
        };
        let ability = self.characteristics(permanent).activated_abilities[ability_index].clone();
        // Tapping the source for mana (CR 106.12) records what the source
        // was, so a sacrifice cost leaves its last known information; the
        // triggered mana abilities it triggers are read from the board once
        // the output is produced (CR 106.12a).
        let mut tapped = None;
        if ability.tap_source {
            self.tap_permanent(permanent, true);
            tapped = Some(self.tapped_for_mana(permanent));
        }
        if ability.life_cost > 0 {
            self.change_life(player, -(ability.life_cost as i32));
        }
        // Retain the output and original payment frame before any cost
        // effect can suspend. Resumption must not spend mana, tap or pay life
        // again, nor expose this output before sacrifice/zone work finishes.
        let source_ref = self.permanent_object(permanent);
        caller.decision = Decision::ManaChoice {
            player,
            choices: ability.mana_choices().unwrap(),
            production: ManaProduction::Output {
                source: source_ref,
                tapped,
            },
            resume,
        };
        // The sacrifice cost, then the ability's non-mana instructions, run
        // once as the source's own child frame (CR 605.3b): damage it deals
        // names the source and can be prevented. Resuming the choice never
        // reaches this frame again.
        let mut effects = Vec::new();
        if ability.sacrifice_source {
            effects.push(Effect::sacrifice_this());
        }
        effects.extend_from_slice(ability.mana_suffix());
        if !effects.is_empty() {
            let mut child = EffectFrame::new(player, effects);
            child.bind_source(
                self.state.permanents[permanent].as_ref().map(|p| p.object),
                source_ref,
            );
            self.run_before_decision(child, caller);
        } else {
            *self.state.suspended_decision_mut() = Some(caller);
        }
    }

    /// Tap a source of fixed mana for mana and resolve its triggered mana
    /// abilities. The planner offers only sources whose bonuses need no
    /// choice (`fixed_source_mana`), so this never suspends.
    fn tap_fixed_mana_source(&mut self, player: PlayerId, permanent: PermanentId) {
        self.perform_automatic_tap(player, AutomaticTap::fixed(permanent));
    }

    fn perform_automatic_tap(&mut self, player: PlayerId, tap: AutomaticTap) {
        let permanent = tap.permanent;
        let produced = tap
            .output
            .unwrap_or_else(|| fixed_output(&self.characteristics(permanent)));
        let source = self
            .permanent_object(permanent)
            .expect("a tapped fixed source is on the battlefield");
        self.tap_permanent(permanent, true);
        let origin = self.mana_origin(source);
        self.state
            .mana_pool_mut(player)
            .add(&produced, Some(origin));
        let tapped = self.tapped_for_mana(permanent);
        let bonuses = self.production_bonuses(player, &tapped);
        let resolved = self.resolve_mana_bonuses(player, source, &produced, bonuses, None);
        assert!(
            resolved.is_none(),
            "a fixed source's bonuses need no choice"
        );
    }

    /// Fixed tap output plus its triggered mana abilities' mana, or `None`
    /// when a bonus would need a choice among several produced types (no
    /// current fixed source produces more than one type). Callers check that
    /// the source has mana abilities, can tap and has not lost its abilities.
    fn fixed_source_mana(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        view: &Characteristics,
        triggers: &[ManaTrigger],
    ) -> Option<Mana> {
        let bonuses = self.tap_bonuses(triggers, player, permanent);
        let output = fixed_output(view);
        if bonuses.is_empty() {
            return Some(output);
        }
        let mut outputs = production_alternatives(vec![output], &bonuses);
        (outputs.len() == 1).then(|| outputs.pop().unwrap())
    }

    /// Every triggered mana ability on the battlefield, with its source and
    /// controller, read from one board.
    fn mana_triggers(&self) -> Vec<ManaTrigger> {
        let mut triggers = Vec::new();
        for controller in [PlayerId(0), PlayerId(1)] {
            for (source, view) in self.layer_views(self.iter_battlefield_permanents(controller)) {
                for ability in &view.triggered_mana_abilities {
                    triggers.push(ManaTrigger {
                        source,
                        controller,
                        ability: ability.clone(),
                    });
                }
            }
        }
        triggers
    }

    /// The planner's estimate of the outputs of the triggered mana abilities
    /// that `producer` tapping `tapped` for mana would trigger, from the
    /// current board. Play reads them at production (`production_bonuses`).
    fn tap_bonuses(
        &self,
        triggers: &[ManaTrigger],
        producer: PlayerId,
        tapped: PermanentId,
    ) -> Vec<TriggeredManaOutput> {
        triggers
            .iter()
            .filter(|trigger| {
                trigger.ability.who.includes(trigger.controller, producer)
                    && self.permanent_matches_predicate(tapped, &trigger.ability.predicate)
            })
            .map(|trigger| trigger.ability.output.clone())
            .collect()
    }

    /// Record `permanent`, just tapped for mana, against every triggered
    /// mana ability predicate on the board.
    fn tapped_for_mana(&self, permanent: PermanentId) -> TappedForMana {
        let mut matched = Vec::new();
        for trigger in self.mana_triggers() {
            let predicate = trigger.ability.predicate;
            if !matched.contains(&predicate)
                && self.permanent_matches_predicate(permanent, &predicate)
            {
                matched.push(predicate);
            }
        }
        TappedForMana { matched }
    }

    /// The triggered mana abilities that `producer` tapping a permanent
    /// for mana triggers, read from the board as the mana is produced
    /// (CR 106.12a): a trigger source removed while the ability's costs were
    /// paid triggers nothing, while the tapped permanent answers from
    /// `tapped` even after it left. In board order.
    pub(crate) fn production_bonuses(
        &self,
        producer: PlayerId,
        tapped: &TappedForMana,
    ) -> Vec<PendingManaBonus> {
        self.mana_triggers()
            .into_iter()
            .filter(|trigger| {
                trigger.ability.who.includes(trigger.controller, producer)
                    && tapped.matched.contains(&trigger.ability.predicate)
            })
            .filter_map(|trigger| {
                Some(PendingManaBonus {
                    source: self.permanent_object(trigger.source)?,
                    output: trigger.ability.output,
                })
            })
            .collect()
    }

    /// Resolve `bonuses` after `tapped`'s production of `produced` into
    /// `player`'s pool (CR 605.4a): a fixed or single-type bonus adds at
    /// once; a choice among several produced types suspends as a
    /// `ManaChoice` naming its source and holding the rest. Returns that
    /// choice, or the caller's `resume` when every bonus resolved.
    pub(crate) fn resolve_mana_bonuses(
        &mut self,
        player: PlayerId,
        tapped: ObjectId,
        produced: &Mana,
        bonuses: Vec<PendingManaBonus>,
        resume: Option<Box<Decision>>,
    ) -> Option<Box<Decision>> {
        let mut pending = bonuses.into_iter();
        while let Some(bonus) = pending.next() {
            let mut choices = bonus.output.alternatives(produced);
            if choices.len() > 1 {
                return Some(Box::new(Decision::ManaChoice {
                    player,
                    choices,
                    production: ManaProduction::Bonus {
                        source: bonus.source,
                        tapped,
                        produced: produced.clone(),
                        pending: pending.collect(),
                    },
                    resume,
                }));
            }
            if let Some(mana) = choices.pop() {
                let origin = self.mana_origin(bonus.source);
                self.state.mana_pool_mut(player).add(&mana, Some(origin));
            }
        }
        resume
    }

    /// The same witness used by automatic production determines PayCost offers.
    pub(crate) fn can_pay_automatic_mana(&self, player: PlayerId, cost: &ManaCost) -> bool {
        let model = self.payment_model(player, None);
        model.payable(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::FixedOnly,
        ) || self.automatic_choice_taps(player, cost, model).is_some()
    }

    /// Both pools combined (regular + until-end-of-combat).
    pub(crate) fn pooled_mana(&self, player: PlayerId) -> Mana {
        let mut total = self.state.players[player.0].mana_pool.mana().clone();
        total.add_combat(&self.state.players[player.0].combat_mana_pool);
        total
    }

    /// `object` as the producer of mana it is adding to a pool.
    pub(crate) fn mana_origin(&self, object: ObjectId) -> ManaOrigin {
        ManaOrigin {
            object,
            definition: self.state.source_base(object).0,
        }
    }

    pub(crate) fn produce_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        for tap in self.automatic_taps(player, cost)? {
            self.perform_automatic_tap(player, tap);
        }
        Ok(())
    }

    /// The exact production prefix shared by execution and inspection.
    fn automatic_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<Vec<AutomaticTap>, AgentError> {
        let model = self.payment_model(player, None);
        let Some(path) = model.plan(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::FixedOnly,
        ) else {
            return self
                .automatic_choice_taps(player, cost, model)
                .ok_or_else(|| AgentError("not enough automatic mana".into()));
        };
        // The witness proved the cost payable; the choice selects
        // which fixed sources pay it. Only the payer's own hand and
        // permanents are read.
        let choice = model.choose_taps(
            &self.pooled_mana(player),
            cost,
            &self.saved_obligations(player),
            &path,
            || self.source_reluctance(&model),
            || self.payment_followups(player, cost, &model),
        );
        let sources = model.sources;
        Ok(choice
            .taps
            .into_iter()
            .map(|index| AutomaticTap::fixed(sources[index].permanent))
            .collect())
    }

    /// If fixed sources cannot fund a payment, let the existing witness
    /// select colors from simple tap abilities. Costs such as sacrificing a
    /// Treasure and non-mana instructions still require an explicit activation.
    fn automatic_choice_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        mut model: PaymentModel<'_>,
    ) -> Option<Vec<AutomaticTap>> {
        if !model.sources.iter().any(|s| !s.abilities.is_empty()) {
            return None;
        }
        let mut has_choice = false;
        for source in Arc::make_mut(&mut model.sources) {
            if source.abilities.is_empty() {
                continue;
            }
            let view = self.characteristics(source.permanent);
            source.abilities.retain(|activation| {
                let ability = &view.activated_abilities[activation.ability_index];
                ability.tap_source && !ability.mana_activation_has_side_effects()
                    && ability.mana_cost.mana_value == 0 && ability.mana_cost.x == 0
                    && !ability.is_use_limited()
                    && self.check_activation_terms(player, ability).is_ok()
                    // A bonus with its own output choice needs the ordinary
                    // activation flow. Here each witness index is a printed mode.
                    && ability.mana_choices().as_deref() == Some(activation.outputs.as_slice())
            });
            has_choice |= !source.abilities.is_empty();
        }
        if !has_choice {
            return None;
        }
        let path = model.plan(
            self.pooled_mana(player),
            vec![Obligation::mana(cost.clone())],
            SourcePolicy::All,
        )?;
        let mut taps = Vec::new();
        let mut steps = path.into_iter();
        while let Some(step) = steps.next() {
            match step {
                PaymentStep::Fixed(source) => taps.push(AutomaticTap::fixed(source)),
                PaymentStep::Activate(permanent, index) => {
                    // The chosen ability pays zero, then chooses its output.
                    let (Some(PaymentStep::Spend), Some(PaymentStep::Output(mode))) =
                        (steps.next(), steps.next())
                    else {
                        unreachable!("simple tap witness")
                    };
                    let output = self.characteristics(permanent).activated_abilities[index]
                        .mana_choices()
                        .unwrap()[mode]
                        .clone();
                    taps.push(AutomaticTap {
                        permanent,
                        output: Some(output),
                    });
                }
                PaymentStep::Spend => break,
                _ => unreachable!("automatic witness contains only tap abilities"),
            }
        }
        Some(taps)
    }

    /// Costs of the callers suspended behind the payment now being made.
    fn saved_obligations(&self, player: PlayerId) -> Vec<Obligation> {
        self.payment_obligations(player)
            .into_iter()
            .skip(1)
            .collect()
    }

    /// How reluctantly the smart payer taps each of `model`'s sources, read
    /// from the layered view: a creature (a land that is currently one
    /// included), then any other nonland permanent, then a land with an
    /// activated ability on the battlefield, mana abilities with a choice or
    /// a cost among them, then a land without the basic supertype, then a
    /// basic land. Static and triggered abilities do not count: tapping the
    /// land for mana gives none of them up.
    fn source_reluctance(&self, model: &PaymentModel) -> Vec<autotapper::Reluctance> {
        use autotapper::Reluctance;
        let _scan = self.board_scan();
        self.characteristics_many(model.sources.iter().map(|source| source.permanent))
            .zip(model.sources.iter())
            .map(|((id, view), source)| {
                debug_assert_eq!(id, source.permanent);
                let basic = || {
                    self.state[source.object]
                        .supertypes
                        .iter()
                        .any(|supertype| supertype.eq_ignore_ascii_case("basic"))
                };
                if view.is_creature() {
                    Reluctance::Creature
                } else if !view.is_land() {
                    Reluctance::Permanent
                } else if view
                    .activated_abilities
                    .iter()
                    .any(|ability| ability.zone == ActivationZone::Battlefield)
                {
                    Reluctance::UtilityLand
                } else if basic() {
                    Reluctance::BasicLand
                } else {
                    Reluctance::Land
                }
            })
            .collect()
    }

    /// Bounded mana-cost options for automatic source preference. Hand costs
    /// are deduplicated; this is not a count of playable cards. Each other
    /// castable card in the payer's hand at its cost with default cast
    /// choices, less one copy of the spell being paid for, which is still in
    /// hand at this boundary. Then each non-mana battlefield ability of a
    /// permanent the payer controls with a positive mana cost as charged now.
    ///
    /// An ability is listed when its source's activations are not prohibited,
    /// its use limit is not spent, its turn restriction and life cost hold
    /// and, for a {T} cost, its source can tap; `tapping` then names that
    /// source, which neither pays for the ability nor survives a payment
    /// that taps it. Timing within the turn, targets and the other costs
    /// (sacrifice, counted taps, crew, discard, waterbend taps) are not
    /// checked, as for hand cards. Single-use abilities matching the paid
    /// cost are excluded by cost equality, not activation identity; this
    /// can also exclude another source with the same cost.
    fn payment_followups(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        model: &PaymentModel,
    ) -> autotapper::PaymentProbe {
        let mut paying = match &cost.purpose {
            SpendingContext::Cast(spell) => Some(spell.clone()),
            _ => None,
        };
        let mut probe = autotapper::PaymentProbe::default();
        let mut instant = Vec::new();
        for card in self.state.zone_cards(ZoneType::Hand, player) {
            let object_id = self.state.object_of(card);
            let object = &self.state[object_id];
            if !object.types.is_castable() {
                continue;
            }
            let Some(followup) =
                self.effective_spell_cost(player, object_id, Default::default(), None)
            else {
                continue;
            };
            if paying.as_ref() == Some(&SpellSnapshot(object.copiable_definition())) {
                paying = None;
                continue;
            }
            if followup.mana_value == 0 {
                continue;
            }
            let timing = self.has_instant_timing(player, object_id);
            match probe.followups.iter().position(|known| *known == followup) {
                Some(index) => instant[index] |= timing,
                None => {
                    probe.followups.push(followup);
                    instant.push(timing);
                }
            }
        }
        probe.followups.truncate(autotapper::FOLLOWUPS);
        probe.instant = instant
            .iter()
            .take(autotapper::FOLLOWUPS)
            .enumerate()
            .fold(0, |set, (i, timing)| set | u64::from(*timing) << i);
        probe.hand = match probe.followups.len() {
            autotapper::FOLLOWUPS => u64::MAX,
            cards => (1 << cards) - 1,
        };
        probe.tapping = vec![None; probe.followups.len()];
        // One validated board for every view; a permanent with no activated
        // ability (a basic land's mana ability is not one) costs nothing more.
        let _scan = self.board_scan();
        let life = self.payable_life(player);
        let views = self.characteristics_many(model.sources.iter().map(|source| source.permanent));
        for (index, (id, view)) in views.enumerate() {
            if view.activated_abilities.is_empty() {
                continue;
            }
            let source = &model.sources[index];
            debug_assert_eq!(id, source.permanent);
            let Some(permanent) = self.state.permanents[source.permanent].as_ref() else {
                continue;
            };
            if self.activations_prohibited(source.permanent) {
                continue;
            }
            for (ability_index, ability) in view.activated_abilities.iter().enumerate() {
                if probe.followups.len() == autotapper::FOLLOWUPS {
                    return probe;
                }
                let single_use =
                    ability.tap_source || ability.sacrifice_source || ability.is_use_limited();
                if ability.zone != ActivationZone::Battlefield
                    || ability.mana_cost.mana_value == 0
                    || ability.mana_choices().is_some()
                    || permanent.activation_used(ability_index, ability)
                    || (ability.own_turn_only && self.active_player() != player)
                    || life < ability.life_cost as i32
                    || (ability.tap_source && !self.permanent_can_tap(source.permanent))
                    || (single_use && ability.mana_cost == *cost)
                {
                    continue;
                }
                let tapping = ability.tap_source.then_some(index);
                let known = probe
                    .followups
                    .iter()
                    .zip(&probe.tapping)
                    .skip(probe.hand.count_ones() as usize)
                    .any(|(known, tap)| *known == ability.mana_cost && *tap == tapping);
                if !known {
                    probe.followups.push(ability.mana_cost.clone());
                    probe.tapping.push(tapping);
                }
            }
        }
        probe
    }

    /// Commit precisely the selected unit, retaining its expiry identity.
    fn spend_mana_unit(
        &mut self,
        player: PlayerId,
        unit: &Mana,
        cost: &ManaCost,
    ) -> Vec<SpentMana> {
        let (regular, combat) = self.state.mana_pools_mut(player);
        if let Some((class, amounts)) = unit.classes().first().filter(|(class, _)| class.combat) {
            let mut class = class.clone();
            class.combat = false;
            let unit = if class.restriction.is_none() && class.effect.is_none() {
                Mana {
                    mana: *amounts,
                    ..Mana::default()
                }
            } else {
                Mana::of_class(class, *amounts)
            };
            combat.spend(&unit, &cost.purpose)
        } else {
            let spent = regular.spend(unit, &cost.purpose);
            let pooled = regular.total();
            if let Some(exposure) = self.card_capture.as_mut().and_then(|c| c.exposure.as_mut()) {
                exposure.mana[player.0].remove(unit.total(), pooled, false);
            }
            spent
        }
    }

    /// Feasible unit selections preserve every saved caller, including the
    /// output of an in-progress filter before the outer cast's remaining cost.
    pub(crate) fn mana_spend_choices(
        &self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Vec<ManaSpendChoice> {
        let pool = self.pooled_mana(player);
        let mut tail = self.payment_obligations(player);
        if !tail.is_empty() {
            tail.remove(0);
        }
        let model = (!tail.is_empty()).then(|| self.payment_model(player, None));
        // When every usable unit is needed, only the first feasible ordering
        // matters: all orders leave the same receipt and remaining pool.
        let limit = if pool.usable_total(cost) == u16::from(cost.mana_value) {
            1
        } else {
            usize::MAX
        };
        pool.spending_choices(cost)
            .into_iter()
            .filter(|choice| {
                let mut remaining = pool.clone();
                remaining.spend(&choice.unit);
                if !remaining.can_pay(&choice.remaining) {
                    return false;
                }
                let Some(model) = &model else {
                    return true;
                };
                let mut obligations = vec![Obligation::mana(choice.remaining.clone())];
                obligations.extend(tail.clone());
                model.payable(remaining, obligations, SourcePolicy::All)
            })
            .take(limit)
            .collect()
    }

    pub(crate) fn begin_mana_spend(
        &mut self,
        player: PlayerId,
        cost: ManaCost,
        completion: SpendCompletion,
        frame: EffectFrame,
    ) -> Result<(), AgentError> {
        *self.state.suspended_decision_mut() = Some(SuspendedResolution {
            frame,
            decision: Decision::ManaSpend {
                player,
                cost,
                spent: PaymentReceipt::default(),
                completion: Box::new(completion),
            },
        });
        self.advance_mana_spend(None)
    }

    pub(crate) fn advance_mana_spend(
        &mut self,
        mut selected: Option<usize>,
    ) -> Result<(), AgentError> {
        loop {
            let Some(suspended) = self.state.suspended_decision.as_ref() else {
                return Ok(());
            };
            let Decision::ManaSpend {
                player,
                cost,
                completion,
                ..
            } = &suspended.decision
            else {
                return Ok(());
            };
            let player = *player;
            let cost = cost.clone();
            if cost.mana_value > 0 {
                let choices = self.mana_spend_choices(player, &cost);
                let index = match selected.take() {
                    Some(index) => index,
                    None if choices.len() == 1 => 0,
                    None if !choices.is_empty() => return Ok(()),
                    None => return Err(AgentError("no feasible mana spending choice".into())),
                };
                let choice = choices
                    .get(index)
                    .ok_or_else(|| AgentError("illegal mana spending choice".into()))?;
                let count_paid = !matches!(**completion, SpendCompletion::Upfront(_));
                let receipt = self.spend_mana_unit(player, &choice.unit, &cost);
                if count_paid {
                    if let Some(exposure) = self.exposure_mut() {
                        exposure.paid(player, u64::from(choice.unit.total()));
                    }
                }
                let suspended = self.state.suspended_decision_mut().as_mut().unwrap();
                let Decision::ManaSpend { cost, spent, .. } = &mut suspended.decision else {
                    unreachable!()
                };
                *cost = choice.remaining.clone();
                spent.record_mana(receipt);
                continue;
            }
            let suspended = self.state.suspended_decision_mut().take().unwrap();
            let Decision::ManaSpend {
                spent, completion, ..
            } = suspended.decision
            else {
                unreachable!()
            };
            match *completion {
                SpendCompletion::Announced(mut payment) => {
                    payment.receipt.record_mana(spent.mana().iter().cloned());
                    self.complete_announced_payment(player, payment)?;
                }
                SpendCompletion::Upfront(decision) => {
                    self.complete_mana_upfront(SuspendedResolution {
                        frame: suspended.frame,
                        decision: *decision,
                    });
                }
                SpendCompletion::Resolution(mut payment) => {
                    payment.receipt.record_mana(spent.mana().iter().cloned());
                    self.finish_resolution_payment(payment, suspended.frame);
                }
                SpendCompletion::Foretell(card) => self.complete_foretell(player, card),
            }
            return Ok(());
        }
    }

    /// The mana `player` could produce now by tapping each untapped
    /// permanent's mana abilities once, read from the layered board.
    pub fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        let triggers = self.mana_triggers();
        for (permanent_id, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            if view.mana_abilities.is_empty()
                || !self.permanent_can_tap(permanent_id)
                || self.activations_prohibited(permanent_id)
            {
                continue;
            }
            if let Some(mana) = self.fixed_source_mana(player, permanent_id, &view, &triggers) {
                total.add(&mana);
            }
        }
        total
    }

    /// CR 106.4 — regular pools empty as each step/phase ends. The combat
    /// pool (until-end-of-combat mana) survives; see
    /// `clear_combat_mana_pools`.
    pub(crate) fn clear_mana_pools(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            if self.state.players[player.0].mana_pool.total() == 0 {
                continue;
            }
            let pool = std::mem::take(self.state.mana_pool_mut(player));
            let before = pool.total();
            let kept = self.unspent_mana_retained(player, pool);
            if let Some(capture) = self.mana_capture(player) {
                capture.remove(before - kept.total(), kept.total(), true);
            }
            *self.state.mana_pool_mut(player) = kept;
        }
    }

    /// Until-end-of-combat mana (firebending) empties as combat ends. What
    /// `unspent_mana_retained` keeps joins the ordinary pool.
    pub(crate) fn clear_combat_mana_pools(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            if self.state.players[player.0].combat_mana_pool.total() == 0 {
                continue;
            }
            let combat = std::mem::take(self.state.combat_mana_pool_mut(player));
            let before = combat.total();
            let kept = self.unspent_mana_retained(player, combat);
            if let Some(capture) = self.mana_capture(player) {
                capture.combat_expired += u64::from(before - kept.total());
            }
            self.state.mana_pool_mut(player).absorb(kept);
        }
    }

    /// What survives of `player`'s unspent `pool` as it would empty: all of
    /// it as red under Ozai (CR 614.1a), its red mana while they keep red
    /// mana this turn, otherwise nothing.
    fn unspent_mana_retained(&self, player: PlayerId, mut pool: ManaPool) -> ManaPool {
        if self.unspent_mana_becomes_red(player) {
            pool.become_red();
            pool
        } else if self.keeps_unspent_mana(player, Color::Red) {
            pool.red_only()
        } else {
            ManaPool::default()
        }
    }

    /// Whether `player` controls a permanent replacing mana loss (CR 614.1a).
    fn unspent_mana_becomes_red(&self, player: PlayerId) -> bool {
        self.iter_battlefield_permanents(player).any(|id| {
            self.state.permanents[id].as_ref().is_some_and(|permanent| {
                self.state[permanent.object]
                    .replacement_effects
                    .contains(&ReplacementEffect::UnspentManaBecomesRed)
            }) && !self.characteristics(id).abilities_lost
        })
    }
}

/// A triggered mana ability on the battlefield and its controller, the
/// "you" its `who` is relative to.
struct ManaTrigger {
    source: PermanentId,
    controller: PlayerId,
    ability: TriggeredManaAbility,
}

/// Each distinct total the triggered mana abilities `bonuses` can add after a
/// production of `produced`: one per combination of their alternatives.
fn bonus_alternatives(produced: &Mana, bonuses: &[TriggeredManaOutput]) -> Vec<Mana> {
    let mut totals = vec![Mana::default()];
    for bonus in bonuses {
        let alternatives = bonus.alternatives(produced);
        if alternatives.is_empty() {
            continue;
        }
        let mut next = Vec::new();
        for total in &totals {
            for alternative in &alternatives {
                let mut sum = total.clone();
                sum.add(alternative);
                if !next.contains(&sum) {
                    next.push(sum);
                }
            }
        }
        totals = next;
    }
    totals
}

/// The mana a fixed source adds when tapped: every fixed production at once.
pub(crate) fn fixed_output(view: &Characteristics) -> Mana {
    let mut output = Mana::default();
    for ability in &view.mana_abilities {
        output.add(&ability.mana);
    }
    output
}

/// Fixed sources derive their output from their characteristics. A simple
/// choice-output activation carries the specific printed mode the payer chose.
struct AutomaticTap {
    permanent: PermanentId,
    output: Option<Mana>,
}

impl AutomaticTap {
    fn fixed(permanent: PermanentId) -> Self {
        Self {
            permanent,
            output: None,
        }
    }
}

/// Each distinct pool addition of a mana ability with `outputs` followed by
/// the triggered mana abilities it triggers. The planner reads these as one
/// output choice; play resolves the bonuses' own choices afterward.
fn production_alternatives(mut outputs: Vec<Mana>, bonuses: &[TriggeredManaOutput]) -> Vec<Mana> {
    if bonuses.is_empty() {
        // Keep the owned modes and their class metadata. Deduplicate in place,
        // preserving the first occurrence and therefore the first witness.
        let mut index = 0;
        while index < outputs.len() {
            if outputs[..index].contains(&outputs[index]) {
                outputs.remove(index);
            } else {
                index += 1;
            }
        }
        return outputs;
    }
    let mut totals = Vec::new();
    for mut output in outputs {
        let mut alternatives = bonus_alternatives(&output, bonuses);
        // A deterministic bonus needs no copy of the original production.
        if alternatives.len() == 1 {
            output.add(&alternatives.pop().unwrap());
            if !totals.contains(&output) {
                totals.push(output);
            }
        } else {
            for bonus in alternatives {
                let mut sum = output.clone();
                sum.add(&bonus);
                if !totals.contains(&sum) {
                    totals.push(sum);
                }
            }
        }
    }
    totals
}

/// `cost` as obligations: the rest paid with mana, then `keyword` of its
/// generic that the taps convoke or improvise names may pay, then
/// `waterbend` that any artifact's or creature's tap may pay. Every
/// keyword tapper may also waterbend, so a tap pays the waterbend part only
/// when nothing else could.
fn tap_obligations(cost: &ManaCost, keyword: u8, waterbend: u8) -> Vec<Obligation> {
    let generic = |amount: u8| {
        ManaCost {
            purpose: cost.purpose.clone(),
            ..ManaCost::default()
        }
        .with_generic(amount)
    };
    let mut obligations = vec![Obligation::mana(cost.reduced_generic(keyword + waterbend))];
    for (amount, taps) in [(keyword, Taps::Keyword), (waterbend, Taps::Any)] {
        if amount > 0 {
            obligations.push(Obligation::Pay {
                cost: generic(amount),
                taps,
            });
        }
    }
    obligations
}

#[cfg(test)]
mod tests;
