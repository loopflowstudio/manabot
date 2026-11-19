// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel answers a question two ways with one result:
// directly, by arithmetic over the pool and the fixed outputs, when every
// source in play for the question taps itself for one fixed output and the
// obligations are one mana cost with an optional waterbend split
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
        game::{ActivationSource, CostComponent, Game, Help, Operation, Payment, PendingChoice},
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
        payment::{ManaOrigin, PaymentReceipt, SpentMana},
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
        waterbend: bool,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManaSource {
    permanent: PermanentId,
    object: ObjectId,
    untapped: bool,
    fixed: Option<Mana>,
    abilities: Vec<ManaActivation>,
    waterbend: bool,
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
/// generic that mana or waterbend taps may pay.
struct DirectPayment<'a> {
    cost: &'a ManaCost,
    split: Option<&'a ManaCost>,
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
            source.untapped = false;
        }
        source.abilities.retain(|ability| {
            !(reservation.tap && ability.tap_source
                || reservation.sacrifice && ability.sacrifice_source)
        });
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
        let waterbend = state.obligations.iter().any(|obligation| {
            matches!(
                obligation,
                Obligation::Pay {
                    waterbend: true,
                    ..
                }
            )
        });
        // A second conservation bound counts *all* pool classes, even those
        // unusable by pending costs, and only the net gain of future
        // activations. Restricted seeds can fund filters without being lost
        // from this accounting. Already saved outputs remain gross supply;
        // their upfront fees are explicit pending Pay obligations.
        let mut net_total = u64::from(state.pool.total());
        let mut total = usable_total(&state.pool);
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            // Fixed mana, tapping abilities and waterbending compete for the
            // same tap. Non-tapping activations can still use this object later.
            let mut tap_total = 0;
            let mut net_tap_total = 0;
            if state.resources.untapped[index] {
                tap_total = source.fixed.as_ref().map_or(0, &usable_total);
                net_tap_total = source
                    .fixed
                    .as_ref()
                    .map_or(0, |output| u64::from(output.total()));
                if waterbend && source.waterbend {
                    tap_total = tap_total.max(1);
                    net_tap_total = net_tap_total.max(1);
                }
            }
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let uses = self.uses(&state.resources, index, ability_index) as u64;
                if uses == 0 {
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
            Obligation::Pay { cost, waterbend } => {
                let pooled = state.pool.can_pay(cost);
                if !pooled && !self.cost_coverable(&state, cost, *waterbend) {
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
                            waterbend: *waterbend,
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
                    if *waterbend
                        && source.waterbend
                        && state.resources.untapped[index]
                        && cost.generic() > 0
                    {
                        let mut next = state.as_ref().clone();
                        next.resources.untapped[index] = false;
                        next.obligations[0] = Obligation::Pay {
                            cost: cost.reduced_generic(1),
                            waterbend: true,
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
    fn cost_coverable(&self, state: &PaymentSearchState, cost: &ManaCost, waterbend: bool) -> bool {
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
                if waterbend && source.waterbend {
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
        obligations.push(Obligation::Pay {
            cost: ability.cost.clone(),
            waterbend: false,
        });
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
        let mut obligations = vec![Obligation::Pay {
            cost: cost.clone(),
            waterbend: false,
        }];
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
        after.reserve(SourceReservation {
            permanent,
            tap: true,
            sacrifice: false,
        });
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
    /// by the generic-only waterbend split of the same purpose
    /// (`waterbend_obligations`), and no source offers a searched ability.
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
                waterbend: false,
            }] => DirectPayment { cost, split: None },
            [Obligation::Pay {
                cost,
                waterbend: false,
            }, Obligation::Pay {
                cost: split,
                waterbend: true,
            }] if policy == SourcePolicy::All
                && split.purpose == cost.purpose
                && split.mana_value == split.generic() =>
            {
                DirectPayment {
                    cost,
                    split: Some(split),
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
                waterbend: source.untapped && source.waterbend && payment.split.is_some(),
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
        let Some(Obligation::Pay { cost, waterbend }) = obligations.first() else {
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
                fixed.max(u64::from(
                    *waterbend && source.waterbend && cost.generic() > 0,
                ))
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
    let tap = payment.owes(&CostComponent::TapSource);
    Some(SourceReservation {
        permanent,
        tap,
        sacrifice: tap || payment.owes(&CostComponent::SacrificeSource),
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
            self.restrict_help_taps(&mut model, payment.help());
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
                        sacrifice: ability.tap_source || ability.sacrifice_source,
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
                            limited: ability.once_each_turn || ability.exhaust,
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
        for object in payment.receipt.sacrificed() {
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
                    sacrifice: ability.tap_source || ability.sacrifice_source,
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
                    result.push(Obligation::Pay {
                        cost: cost.clone(),
                        waterbend: false,
                    });
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
                    result.push(Obligation::Pay {
                        cost: ability.mana_cost,
                        waterbend: false,
                    });
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
                Decision::Payment(progress) if !progress.can_decline() => {
                    result.push(Obligation::Pay {
                        cost: progress.remaining.mana_cost(),
                        waterbend: matches!(
                            progress.remaining,
                            crate::state::ability::PaymentCost::Waterbend { .. }
                        ),
                    });
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
                let (owed, split) =
                    self.helped_cost(player, owed.unwrap_or(recorded), payment.help(), waterbend);
                result.extend(waterbend_obligations(&owed, split));
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
        // The engine also commits purely non-mana activation costs through
        // the mana step. That bookkeeping step opens no mana window.
        if matches!(self.paying(), Some((_, payment, CostComponent::Mana { .. })) if !payment.waterbends() && matches!(payment.operation, Operation::Activation { .. }))
            && self.state.suspended_decision.is_none()
            && matches!(obligations.as_slice(), [Obligation::Pay { cost, .. }] if cost.mana_value == 0)
        {
            return;
        }
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
        let obligations = vec![Obligation::Pay {
            cost: cost.clone(),
            waterbend: false,
        }];
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

    /// Tap one offered fixed source, then finish automatically once no
    /// source or mana-ability choice remains.
    pub(crate) fn tap_for_mana_action(
        &mut self,
        player: PlayerId,
        permanent: PermanentId,
    ) -> Result<(), AgentError> {
        let before = (self.pending_choice.is_none()
            && self.state.suspended_decision.is_none()
            && self.mana_capture(player).is_some())
        .then(|| self.state.players[player.0].mana_pool.total());
        self.tap_fixed_mana_source(player, permanent);
        if let Some(before) = before {
            let produced = self.state.players[player.0].mana_pool.total() - before;
            self.mana_capture(player).unwrap().float(produced);
        }
        if self.state.suspended_decision.is_some() {
            return self.advance_mana_upfront();
        }
        match self.paying() {
            Some((_, payment, CostComponent::Mana { .. }))
                if !payment.waterbends()
                    && !self.has_mana_payment_choices(player)
                    && self.help_actions(player).is_empty() =>
            {
                return self.finish_mana_payment(player);
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
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
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
        let obligations = vec![
            Obligation::Tap {
                candidates: self
                    .tap_component_candidates(player, &payment.operation, tap, toward)
                    .into(),
                remaining: tap.count,
            },
            Obligation::Pay {
                cost: payment.mana().0.clone(),
                waterbend: payment.waterbends(),
            },
        ];
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
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
            SourcePolicy::All,
        )
    }

    /// Convoke and improvise name which taps may pay generic mana: creatures
    /// and artifacts respectively, where waterbend alone takes either. A
    /// cost with both waterbend and one of them keeps only the keyword's
    /// taps, which may hide a legal waterbend tap and never offers an
    /// illegal one.
    fn restrict_help_taps(&self, model: &mut PaymentModel, help: Help) {
        if !help.taps_any_generic() || !model.sources.iter().any(|source| source.waterbend) {
            return;
        }
        for source in Arc::make_mut(&mut model.sources)
            .iter_mut()
            .filter(|source| source.waterbend)
        {
            source.waterbend = self.help_tap_toward(help, source.permanent).is_some();
        }
    }

    /// `cost` as the payment model sees it under `help`: delve pays as much
    /// generic as the graveyard holds, and the second value is the generic
    /// that a tap may pay instead of mana.
    fn helped_cost(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
    ) -> (ManaCost, u8) {
        let cost = match help.delve {
            Some(spell) => {
                let cards = self.graveyard_payers(player, spell).len();
                cost.reduced_generic(cards.min(usize::from(u8::MAX)) as u8)
            }
            None => cost.clone(),
        };
        let split = if help.taps_any_generic() {
            cost.generic()
        } else {
            waterbend.min(cost.generic())
        };
        (cost, split)
    }

    /// The creatures that may convoke under `model`, each with its colors.
    fn convokers(&self, model: &PaymentModel) -> Vec<(PermanentId, Vec<Color>)> {
        model
            .sources
            .iter()
            .filter(|source| source.waterbend)
            .map(|source| {
                let colors = &self.characteristics(source.permanent).colors;
                (source.permanent, colors.iter().copied().collect())
            })
            .collect()
    }

    /// Can `player` pay `cost` with mana and `help`, once `tapped` is tapped
    /// for something else? Delve and taps for generic mana are part of the
    /// payment model's question; a convoking creature paying a colored pip
    /// is tried by `convoke_search`.
    pub(crate) fn payable_with_help(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        tapped: Option<PermanentId>,
    ) -> bool {
        let mut model = self.payment_model(player, None);
        if let Some(permanent) = tapped {
            model.reserve(SourceReservation {
                permanent,
                tap: true,
                sacrifice: false,
            });
        }
        self.restrict_help_taps(&mut model, help);
        let convokers = if help.convoke {
            self.convokers(&model)
        } else {
            Vec::new()
        };
        let pool = self.pooled_mana(player);
        self.convoke_search(&mut model, cost, &convokers, &mut |model, cost| {
            let (relaxed, split) = self.helped_cost(player, cost, help, help.waterbend);
            model
                .payable(
                    pool.clone(),
                    waterbend_obligations(&relaxed, split),
                    SourcePolicy::All,
                )
                .then_some(())
        })
        .is_some()
    }

    /// `attempt` against the obligations `player` has pending. A convoke
    /// payment in progress is asked as it stands and then with each set of
    /// creatures paying colored pips of it, since the payment model itself
    /// only taps for generic mana.
    fn while_paying<T>(
        &self,
        player: PlayerId,
        model: &mut PaymentModel,
        attempt: &mut dyn FnMut(&PaymentModel, Vec<Obligation>) -> Option<T>,
    ) -> Option<T> {
        match self.paying() {
            Some((payer, payment, CostComponent::Mana { .. }))
                if payer == player && payment.help().convoke =>
            {
                let convokers = self.convokers(model);
                let owed = payment.mana().0.clone();
                self.convoke_search(model, &owed, &convokers, &mut |model, owed| {
                    attempt(model, self.payment_obligations_owing(player, Some(owed)))
                })
            }
            _ => attempt(model, self.payment_obligations(player)),
        }
    }

    /// `attempt` to pay `cost`, then `cost` less each set of colored pips
    /// that `convokers` can pay, those creatures tapped. The search is
    /// exhaustive over those assignments, so it grows with the number of
    /// colored creatures when no attempt succeeds. Creatures are tried in
    /// one order, so each set of taps is visited once.
    fn convoke_search<T>(
        &self,
        model: &mut PaymentModel,
        cost: &ManaCost,
        convokers: &[(PermanentId, Vec<Color>)],
        attempt: &mut dyn FnMut(&PaymentModel, &ManaCost) -> Option<T>,
    ) -> Option<T> {
        if let Some(found) = attempt(model, cost) {
            return Some(found);
        }
        for (position, (permanent, colors)) in convokers.iter().enumerate() {
            let Some(index) = model
                .sources
                .iter()
                .position(|source| source.permanent == *permanent)
            else {
                continue;
            };
            for color in colors
                .iter()
                .filter(|color| cost.cost[**color as usize] > 0)
            {
                let untouched = model.sources[index].clone();
                model.reserve(SourceReservation {
                    permanent: *permanent,
                    tap: true,
                    sacrifice: false,
                });
                let found = self.convoke_search(
                    model,
                    &cost.less_pip(*color),
                    &convokers[position + 1..],
                    attempt,
                );
                Arc::make_mut(&mut model.sources)[index] = untouched;
                if found.is_some() {
                    return found;
                }
            }
        }
        None
    }

    /// Can `player` pay `cost`, `waterbend` of its generic with mana or
    /// waterbend taps and the rest with mana?
    pub(crate) fn waterbend_payment_possible(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
    ) -> bool {
        self.payment_model(player, None).payable(
            self.pooled_mana(player),
            waterbend_obligations(cost, waterbend),
            SourcePolicy::All,
        )
    }

    /// The `candidates` whose waterbend tap, beside `also_tapped`, leaves
    /// `cost` payable, without copying the game. A tap reserves its source
    /// and produces no mana. One decision's candidates are judged against one
    /// model, each tap reserved for its own question and then restored. The
    /// debug oracle retains the full fork calculation if future rules add a
    /// tap dependency.
    pub(crate) fn waterbend_taps_keeping_payable(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
        candidates: Vec<PermanentId>,
        also_tapped: Option<PermanentId>,
    ) -> Vec<PermanentId> {
        let tap = |permanent| SourceReservation {
            permanent,
            tap: true,
            sacrifice: false,
        };
        let mut model = self.payment_model(player, None);
        if let Some(permanent) = also_tapped {
            model.reserve(tap(permanent));
        }
        let pool = self.pooled_mana(player);
        let obligations = waterbend_obligations(cost, waterbend);
        candidates
            .into_iter()
            .filter(|&candidate| {
                let index = model
                    .sources
                    .iter()
                    .position(|source| source.permanent == candidate);
                let untouched = index.map(|index| model.sources[index].clone());
                model.reserve(tap(candidate));
                let possible = model.payable(pool.clone(), obligations.clone(), SourcePolicy::All);
                if let (Some(index), Some(source)) = (index, untouched) {
                    Arc::make_mut(&mut model.sources)[index] = source;
                }
                debug_assert_eq!(possible, {
                    let mut probe = self.clone();
                    for permanent in std::iter::once(candidate).chain(also_tapped) {
                        probe.tap_permanent(permanent, false);
                    }
                    probe.waterbend_payment_possible(player, cost, waterbend)
                });
                possible
            })
            .collect()
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

    pub fn cached_producible_mana(&mut self, player: PlayerId) -> Mana {
        if let Some(cached) = &self.state.mana_cache[player.0] {
            return cached.clone();
        }
        let mana = self.producible_mana(player);
        self.state.mana_cache_mut()[player.0] = Some(mana.clone());
        mana
    }

    pub fn invalidate_mana_cache(&mut self, player: PlayerId) {
        self.state.mana_cache_mut()[player.0] = None;
    }

    /// The same witness used by automatic production determines PayCost offers.
    pub(crate) fn can_pay_automatic_mana(&self, player: PlayerId, cost: &ManaCost) -> bool {
        let model = self.payment_model(player, None);
        model.payable(
            self.pooled_mana(player),
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
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
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
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
                    && !ability.exhaust && !ability.once_each_turn
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
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
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
                let single_use = ability.tap_source
                    || ability.sacrifice_source
                    || ability.once_each_turn
                    || ability.exhaust;
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
                let mut obligations = vec![Obligation::Pay {
                    cost: choice.remaining.clone(),
                    waterbend: false,
                }];
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
            let Decision::ManaSpend { player, cost, .. } = &suspended.decision else {
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
                let receipt = self.spend_mana_unit(player, &choice.unit, &cost);
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
                SpendCompletion::Resolution(payment) => {
                    if let Some(key) = payment.spends {
                        self.spend_once_each_turn(key);
                    }
                    if matches!(
                        payment.remaining,
                        crate::state::ability::PaymentCost::Waterbend { .. }
                    ) {
                        self.record_bending(player, crate::flow::turn::Bending::Water);
                    }
                    let mut frame = suspended.frame;
                    frame.prepend_effects(payment.if_paid.into_iter());
                    self.run_frame(frame);
                }
                SpendCompletion::Foretell(card) => self.complete_foretell(player, card),
            }
            return Ok(());
        }
    }

    pub(crate) fn producible_mana(&self, player: PlayerId) -> Mana {
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
fn fixed_output(view: &Characteristics) -> Mana {
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

/// `cost` as obligations: `waterbend` of its generic that taps may also
/// pay, and the rest paid only with mana.
fn waterbend_obligations(cost: &ManaCost, waterbend: u8) -> Vec<Obligation> {
    let mut obligations = vec![Obligation::Pay {
        cost: cost.reduced_generic(waterbend),
        waterbend: false,
    }];
    if waterbend > 0 {
        obligations.push(Obligation::Pay {
            cost: ManaCost {
                purpose: cost.purpose.clone(),
                ..ManaCost::default()
            }
            .with_generic(waterbend),
            waterbend: true,
        });
    }
    obligations
}

#[cfg(test)]
mod tests {
    use super::{
        waterbend_obligations as waterbend, ManaActivation, ManaSource, Obligation, PaymentModel,
        PaymentSearch, PaymentSearchState, PaymentStep, SourcePolicy,
    };
    use crate::cards::content_pack::ContentPack;
    use crate::state::{
        game_object::{ObjectId, PermanentId},
        mana::{Mana, ManaCost, ManaRestriction, SpendEffect, SpendingContext},
    };
    use std::sync::Arc;

    #[test]
    fn production_totals_preserve_first_modes_classes_and_trigger_choices() {
        use crate::state::card::TriggeredManaOutput;

        let restricted =
            Mana::restricted_to(ManaRestriction::only(["Ally"], []), [0, 1, 0, 0, 0, 0]);
        let outputs = vec![
            Mana::parse("UR"),
            restricted.clone(),
            Mana::default(),
            Mana::parse("UR"),
            Mana::parse("G"),
            restricted,
        ];
        for bonuses in [
            vec![],
            vec![TriggeredManaOutput::Fixed(Mana::parse("G"))],
            vec![TriggeredManaOutput::OneProducedType],
            vec![
                TriggeredManaOutput::OneProducedType,
                TriggeredManaOutput::Fixed(Mana::parse("U")),
                TriggeredManaOutput::OneProducedType,
            ],
        ] {
            let mut expected = Vec::new();
            for output in &outputs {
                for bonus in super::bonus_alternatives(output, &bonuses) {
                    let mut sum = output.clone();
                    sum.add(&bonus);
                    if !expected.contains(&sum) {
                        expected.push(sum);
                    }
                }
            }
            assert_eq!(
                super::production_alternatives(outputs.clone(), &bonuses),
                expected
            );
        }
        assert_eq!(
            super::production_alternatives(Vec::new(), &[]),
            Vec::<Mana>::new()
        );
    }

    /// Synthetic announced life fee: the same obligation moves from
    /// Waterbend to spending, while a mana caller has its own Output fee.
    #[test]
    fn announced_payment_reserves_life_across_nested_callers() {
        use crate::flow::{
            decision::{Decision, EffectFrame, SpendCompletion, SuspendedResolution},
            game::{ActivationSource, CostComponent, Game, Operation, Payment, PendingChoice},
        };
        use crate::state::{
            card::ActivatedAbilityDefinition, game_object::PlayerId, player::PlayerConfig,
        };
        use std::collections::BTreeMap;

        let player = PlayerId(0);
        let deck = BTreeMap::from([("Gray Ogre".into(), 8), ("Sunbaked Canyon".into(), 8)]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("p0", deck.clone()),
                PlayerConfig::new("p1", deck),
            ],
            350,
            true,
        );
        let source = game
            .scenario_force_battlefield(player, "Gray Ogre", true)
            .unwrap();
        let canyon = game
            .scenario_force_battlefield(player, "Sunbaked Canyon", true)
            .unwrap();
        let object = game.state.permanents[source].as_ref().unwrap().object;
        game.state[object].activated_abilities = vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("1"),
            waterbend: true,
            tap_source: true,
            life_cost: 2,
            ..Default::default()
        }];
        let operation = Operation::Activation {
            source: ActivationSource::Permanent(source),
            ability_index: 0,
            targets: vec![],
            x: 0,
        };
        let payment = Payment {
            owed: game.cost_components(player, &operation),
            operation,
            receipt: Default::default(),
        };
        assert_eq!(payment.help().waterbend, 1);
        assert_eq!(
            payment.owed[1..],
            [CostComponent::Life(2), CostComponent::TapSource]
        );
        for life in [2, 4] {
            game.scenario_set_life(player, life);
            game.pending_choice = Some(PendingChoice::Pay {
                player,
                payment: Box::new(payment.clone()),
            });
            assert_eq!(game.can_activate_mana_ability(player, canyon, 0), life == 4);
            // The life fee remains reserved after the announcement moves
            // into spending, including when a nested upfront owns that caller.
            game.pending_choice = None;
            let announced = Decision::ManaSpend {
                player,
                cost: ManaCost::parse("1"),
                spent: Default::default(),
                completion: Box::new(SpendCompletion::Announced(payment.clone())),
            };
            for decision in [
                announced.clone(),
                Decision::ManaUpfront {
                    player,
                    permanent: canyon,
                    ability_index: 0,
                    tap_remaining: 0,
                    resume: Some(Box::new(announced)),
                },
            ] {
                *game.state.suspended_decision_mut() = Some(SuspendedResolution {
                    frame: EffectFrame::new(player, vec![]),
                    decision,
                });
                let model = game.payment_model(player, None);
                assert_eq!(model.life, (life - 2) as u32);
                let source = model
                    .sources
                    .iter()
                    .find(|s| s.permanent == source)
                    .unwrap();
                assert!(!source.waterbend);
                let payable = model.payable(
                    Mana::default(),
                    game.payment_obligations(player),
                    SourcePolicy::All,
                );
                assert_eq!(payable, life == 4);
            }
            *game.state.suspended_decision_mut() = None;
        }
        // Execute the nearby payable case: Canyon's fee commits before its
        // color choice; the announced fee is still owed, exactly once.
        game.pending_choice = Some(PendingChoice::Pay {
            player,
            payment: Box::new(payment),
        });
        game.activate_mana_ability(player, canyon, 0).unwrap();
        assert_eq!(game.state.players[0].life, 3);
        assert!(game.state.permanents[canyon].as_ref().unwrap().tapped);
        assert!(game.mana_choice_completes_payment(player, &Mana::parse("R")));
        let mut fork = game.clone();
        for candidate in [&mut game, &mut fork] {
            candidate
                .execute_action(&crate::agent::action::Action::ChooseMode { player, mode: 0 })
                .unwrap();
            candidate.finish_action_step();
            assert!(candidate.pending_choice.is_none());
            assert!(candidate.state.suspended_decision.is_none());
            assert_eq!(candidate.state.players[0].life, 1);
            assert_eq!(candidate.state.players[0].mana_pool, Mana::default());
            assert!(candidate.state.permanents[source].as_ref().unwrap().tapped);
            assert_eq!(candidate.state.stack_objects.len(), 0);
        }
        assert_eq!(
            crate::search_state::witness(&game),
            crate::search_state::witness(&fork)
        );
    }

    #[test]
    fn smart_tap_prunes_surplus_and_preserves_mana_options() {
        use crate::flow::game::Game;
        use crate::state::{game_object::PlayerId, player::PlayerConfig};
        use std::collections::BTreeMap;

        let player = PlayerId(0);
        let board = |lands: &[&str], creature: Option<&str>, hand: Option<&str>| {
            let deck = BTreeMap::from([
                ("Mountain".into(), 8),
                ("Forest".into(), 8),
                ("Lightning Bolt".into(), 4),
                ("Llanowar Elves".into(), 4),
                ("Shivan Dragon".into(), 4),
            ]);
            let mut game = Game::new(
                vec![
                    PlayerConfig::new("p0", deck.clone()),
                    PlayerConfig::new("p1", deck),
                ],
                347,
                true,
            );
            game.scenario_clear_hand(player);
            let lands: Vec<PermanentId> = lands
                .iter()
                .map(|land| game.scenario_force_battlefield(player, land, true).unwrap())
                .collect();
            if let Some(creature) = creature {
                game.scenario_force_battlefield(player, creature, true)
                    .unwrap();
            }
            if let Some(card) = hand {
                game.scenario_force_card_in_hand(player, card).unwrap();
            }
            (game, lands)
        };
        // The lands tapped by paying `cost`, by position.
        let tapped = |game: &Game, lands: &[PermanentId], cost: &str| {
            let mut game = game.clone();
            game.produce_mana(player, &ManaCost::parse(cost)).unwrap();
            let tapped: Vec<usize> = (0..lands.len())
                .filter(|index| {
                    game.state.permanents[lands[*index]]
                        .as_ref()
                        .unwrap()
                        .tapped
                })
                .collect();
            tapped
        };

        // The first witness remains in source order.
        let (game, lands) = board(
            &["Forest", "Mountain", "Forest"],
            None,
            Some("Lightning Bolt"),
        );
        let model = game.payment_model(player, None);
        let witness = model
            .plan(Mana::default(), vec![pay("1G")], SourcePolicy::FixedOnly)
            .unwrap();
        assert_eq!(
            witness,
            [
                PaymentStep::Fixed(lands[0]),
                PaymentStep::Fixed(lands[1]),
                PaymentStep::Spend
            ]
        );
        // The chosen payment keeps the Mountain for Lightning Bolt.
        assert_eq!(tapped(&game, &lands, "1G"), vec![0, 2]);
        let probe = game.payment_followups(player, &ManaCost::parse("1G"), &model);
        assert_eq!(
            (probe.followups.len(), probe.hand, probe.instant),
            (1, 1, 1)
        );

        // Mountain, Mountain, Forest for {1}{G}: two taps instead of three.
        let (game, lands) = board(&["Mountain", "Mountain", "Forest"], None, None);
        assert_eq!(tapped(&game, &lands, "1G"), vec![0, 2]);

        // Shivan Dragon's {R} ability is a follow-up like a card in hand.
        let (game, lands) = board(
            &["Forest", "Mountain", "Forest"],
            Some("Shivan Dragon"),
            None,
        );
        let model = game.payment_model(player, None);
        let probe = game.payment_followups(player, &ManaCost::parse("1G"), &model);
        assert_eq!((probe.followups.len(), probe.hand), (1, 0));
        assert_eq!(probe.tapping, [None]);
        assert_eq!(tapped(&game, &lands, "1G"), [0, 2]);

        // One of each kept either way: the card in hand beats the ability.
        let (game, lands) = board(
            &["Forest", "Mountain"],
            Some("Shivan Dragon"),
            Some("Llanowar Elves"),
        );
        assert_eq!(tapped(&game, &lands, "1"), [1]);
    }

    /// The kind of a source breaks ties among payments that keep the same
    /// follow-ups: a basic land pays before Treetop Village or Llanowar
    /// Elves, which pay when that keeps a card in hand castable.
    #[test]
    fn smart_tap_prefers_basic_lands_until_a_follow_up_needs_them() {
        use super::autotapper::Reluctance;
        use crate::flow::game::Game;
        use crate::state::{game_object::PlayerId, player::PlayerConfig};
        use std::collections::BTreeMap;

        let player = PlayerId(0);
        let board = |sources: &[&str], hand: Option<&str>| {
            let deck = BTreeMap::from([
                ("Mountain".into(), 8),
                ("Forest".into(), 8),
                ("Lightning Bolt".into(), 4),
                ("Llanowar Elves".into(), 4),
                ("Treetop Village".into(), 4),
                ("Sol Ring".into(), 4),
                ("Secret Tunnel".into(), 4),
            ]);
            let mut game = Game::new(
                vec![
                    PlayerConfig::new("p0", deck.clone()),
                    PlayerConfig::new("p1", deck),
                ],
                347,
                true,
            );
            game.scenario_clear_hand(player);
            let sources: Vec<PermanentId> = sources
                .iter()
                .map(|name| game.scenario_force_battlefield(player, name, true).unwrap())
                .collect();
            if let Some(card) = hand {
                game.scenario_force_card_in_hand(player, card).unwrap();
            }
            (game, sources)
        };
        // The sources tapped by paying `cost`, by position.
        let tapped = |game: &Game, sources: &[PermanentId], cost: &str| {
            let mut game = game.clone();
            game.produce_mana(player, &ManaCost::parse(cost)).unwrap();
            (0..sources.len())
                .filter(|index| {
                    game.state.permanents[sources[*index]]
                        .as_ref()
                        .unwrap()
                        .tapped
                })
                .collect::<Vec<usize>>()
        };

        let (game, sources) = board(
            &[
                "Llanowar Elves",
                "Sol Ring",
                "Treetop Village",
                "Secret Tunnel",
                "Forest",
            ],
            None,
        );
        let model = game.payment_model(player, None);
        assert_eq!(
            game.source_reluctance(&model),
            [
                Reluctance::Creature,
                Reluctance::Permanent,
                Reluctance::UtilityLand,
                Reluctance::UtilityLand,
                Reluctance::BasicLand,
            ]
        );
        // Tap the Forest first, then the
        // colorless Secret Tunnel ahead of the green Treetop Village.
        assert_eq!(tapped(&game, &sources, "1"), [4]);
        assert_eq!(tapped(&game, &sources, "2"), [3, 4]);
        assert_eq!(tapped(&game, &sources, "3"), [2, 3, 4]);

        // A generic pip with nothing in hand: the Mountain. With Lightning
        // Bolt in hand the Mountain stays for it and the other source pays.
        for other in ["Treetop Village", "Llanowar Elves"] {
            for order in [[other, "Mountain"], ["Mountain", other]] {
                let mountain = usize::from(order[1] == "Mountain");
                let (game, sources) = board(&order, None);
                assert_eq!(tapped(&game, &sources, "1"), [mountain]);
                let (game, sources) = board(&order, Some("Lightning Bolt"));
                assert_eq!(tapped(&game, &sources, "1"), [1 - mountain]);
            }
        }

        // Mana floating in the pool is spent before a source is tapped.
        let (mut game, sources) = board(&["Forest", "Forest", "Forest"], None);
        game.state
            .mana_pool_mut(player)
            .add(&Mana::parse("G"), None);
        assert_eq!(tapped(&game, &sources, "1G"), [0]);
        assert_eq!(tapped(&game, &sources, "G"), Vec::<usize>::new());
        game.state
            .mana_pool_mut(player)
            .add(&Mana::parse("G"), None);
        assert_eq!(tapped(&game, &sources, "1G"), Vec::<usize>::new());
    }

    thread_local! {
        /// Questions this thread answered by searching rather than directly;
        /// the debug parity searches beside direct answers are not counted.
        pub(in crate::flow::mana) static SEARCHED: std::cell::Cell<usize> =
            const { std::cell::Cell::new(0) };
    }

    pub(in crate::flow::mana) fn tapping_ability(outputs: Vec<Mana>) -> ManaActivation {
        ManaActivation {
            tap_source: true,
            tap_count: 0,
            tap_candidates: Arc::new(Vec::new()),
            limited: false,
            sacrifice_source: false,
            ability_index: 0,
            cost: ManaCost::default(),
            outputs: outputs.into(),
            life_cost: 0,
        }
    }

    // Payment models take distinct appearances from the ordinary allocator.
    fn payment_object(index: usize) -> ObjectId {
        static OBJECTS: std::sync::LazyLock<Vec<ObjectId>> = std::sync::LazyLock::new(|| {
            let mut zones = crate::state::zone::ZoneManager::default();
            (0..128).map(|_| zones.allocate()).collect()
        });
        OBJECTS[index]
    }

    pub(in crate::flow::mana) fn source(
        index: usize,
        fixed: Option<&str>,
        abilities: Vec<ManaActivation>,
    ) -> ManaSource {
        ManaSource {
            permanent: PermanentId(index),
            object: payment_object(index),
            untapped: true,
            fixed: fixed.map(Mana::parse),
            abilities,
            waterbend: false,
        }
    }

    fn forests(count: usize) -> PaymentModel<'static> {
        PaymentModel {
            cache: None,
            sources: Arc::new(
                (0..count)
                    .map(|index| source(index, Some("G"), Vec::new()))
                    .collect::<Vec<_>>(),
            ),
            life: 20,
        }
    }

    fn pay(text: &str) -> Obligation {
        Obligation::Pay {
            cost: ManaCost::parse(text),
            waterbend: false,
        }
    }

    fn search_state(model: &PaymentModel, obligations: Vec<Obligation>) -> PaymentSearchState {
        PaymentSearchState {
            resources: model.resources(),
            pool: Mana::default(),
            life: model.life,
            obligations,
            prefix: None,
        }
    }

    /// Compare full concrete witnesses with the identity-keyed DFS on small
    /// boards; on large boards fail during exploration, before a stall can grow.
    /// A search holds up to `GROWTH` identity states before it classifies its
    /// sources, so a smaller limit binds from there; zero asserts that the
    /// answer needs no state expansion.
    fn bounded_search(
        model: &PaymentModel,
        state: PaymentSearchState,
        policy: SourcePolicy,
        limit: usize,
        compare: bool,
    ) -> Option<Vec<PaymentStep>> {
        let mut search = PaymentSearch::new(&state.obligations);
        search.max_states = Some(if limit == 0 {
            0
        } else {
            limit.max(PaymentSearch::GROWTH)
        });
        let path = model.witness(state.clone(), policy, &mut search);
        if compare {
            // No classes: identity keys throughout.
            let mut identities = PaymentSearch {
                groups: Some(Vec::new()),
                gross_bound_only: true,
                max_states: Some(20_000),
                ..Default::default()
            };
            let original = model.witness(state, policy, &mut identities);
            assert_eq!(path, original);
        }
        path
    }

    #[test]
    fn payment_shared_outputs_keep_cached_witness_after_source_change() {
        let cache = std::cell::RefCell::new(super::PaymentCache::default());
        let mut model = PaymentModel {
            cache: Some(&cache),
            sources: Arc::new(vec![modal_source(0, &["U", "R"], "")]),
            life: 20,
        };
        let original = model.sources.clone();
        let blue = model.plan(Mana::default(), vec![pay("U")], SourcePolicy::All);
        assert!(blue.is_some());
        let ability = &mut Arc::make_mut(&mut model.sources)[0].abilities[0];
        Arc::make_mut(&mut ability.outputs)[0] = Mana::parse("G");
        assert!(model
            .plan(Mana::default(), vec![pay("U")], SourcePolicy::All)
            .is_none());
        assert!(model
            .plan(Mana::default(), vec![pay("G")], SourcePolicy::All)
            .is_some());
        model.sources = original;
        assert_eq!(
            model.plan(Mana::default(), vec![pay("U")], SourcePolicy::All),
            blue
        );
    }

    #[test]
    fn payment_cache_keeps_selected_taps_and_exact_mutations() {
        let cache = std::cell::RefCell::new(super::PaymentCache::default());
        let mut model = PaymentModel {
            cache: Some(&cache),
            sources: Arc::new(vec![source(0, Some("U"), vec![]), source(1, None, vec![])]),
            life: 3,
        };
        let obligations = vec![
            Obligation::Tap {
                candidates: vec![PermanentId(0), PermanentId(1)].into(),
                remaining: 1,
            },
            pay("U"),
        ];
        // These queries have identical unmodified obligations and models.
        // Only the selected tap differs; one spends the sole blue producer.
        for _ in 0..2 {
            assert!(model
                .plan_after_tap(Mana::default(), obligations.clone(), PermanentId(0))
                .is_none());
            assert_eq!(
                model.plan_after_tap(Mana::default(), obligations.clone(), PermanentId(1)),
                Some(vec![PaymentStep::Fixed(PermanentId(0)), PaymentStep::Spend])
            );
        }
        assert_eq!(cache.borrow().entries.len(), 2);
        for mutation in 0..6 {
            match mutation {
                0 => Arc::make_mut(&mut model.sources)[0].untapped = false,
                1 => {
                    Arc::make_mut(&mut model.sources)[0].fixed = Some(Mana::restricted_to(
                        ManaRestriction::only([], ["Ally"]),
                        [0, 1, 0, 0, 0, 0],
                    ))
                }
                2 => Arc::make_mut(&mut model.sources)[0].untapped = true,
                3 => model.life = 0,
                4 => Arc::make_mut(&mut model.sources).reverse(),
                _ => model.life = 3,
            }
            for pool in [Mana::default(), Mana::parse("U")] {
                for tail in [
                    vec![],
                    vec![pay("1")],
                    vec![Obligation::Output {
                        choices: vec![Mana::parse("U")].into(),
                        life_cost: 1,
                    }],
                ] {
                    let mut owed = vec![pay("U")];
                    owed.extend(tail);
                    for policy in [
                        SourcePolicy::All,
                        SourcePolicy::FixedOnly,
                        SourcePolicy::AutomaticPrefix,
                    ] {
                        let expected = model.search(pool.clone(), owed.clone(), policy);
                        for _ in 0..2 {
                            assert_eq!(model.plan(pool.clone(), owed.clone(), policy), expected);
                        }
                    }
                }
            }
        }
        assert_eq!(cache.borrow().entries.len(), 64);
        let fork = cache.borrow().clone();
        cache.borrow_mut().entries.clear();
        assert_eq!(fork.entries.len(), 64);
        // FIFO retention remains bounded across many distinct life queries.
        for life in 0..100 {
            model.life = life;
            model.plan(Mana::default(), vec![pay("U"), pay("G")], SourcePolicy::All);
        }
        let cache = cache.borrow();
        assert_eq!(cache.entries.len(), 64);
        assert_eq!(cache.entries.front().unwrap().life, 36);
        assert_eq!(cache.entries.back().unwrap().life, 99);
    }

    #[test]
    fn direct_payment_matches_the_search_on_fixed_sources() {
        use rand::{seq::SliceRandom, Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(339);
        let lesson = || ManaRestriction::spell(["Lesson"]);
        // Empty and unusable outputs are worth nothing to a non-Lesson cost,
        // which is where a waterbend tap beats a mana tap.
        let outputs = [
            None,
            Some(Mana::default()),
            Some(Mana::parse("G")),
            Some(Mana::parse("U")),
            Some(Mana::parse("UG")),
            Some(Mana::parse("C")),
            Some(Mana::restricted_to(lesson(), [0, 1, 0, 0, 0, 0])),
            Some(Mana::restricted_to(lesson(), [0, 0, 0, 0, 1, 1])),
        ];
        let pools = [
            Mana::default(),
            Mana::parse("U"),
            Mana::parse("GC"),
            Mana::restricted_to(lesson(), [0, 1, 0, 0, 1, 0]),
        ];
        let costs = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "U",
            "G",
            "UU",
            "1U",
            "2U",
            "2G",
            "3G",
            "UG",
            "{U/G}",
            "1{U/G}{U/G}",
            "C",
            "3UG",
        ];
        let purposes = [
            SpendingContext::default(),
            SpendingContext::cast(["Lesson"], []),
        ];
        let (mut direct, mut payable, mut waterbent) = (0, 0, 0);
        for _ in 0..20_000 {
            let mut model = PaymentModel {
                cache: None,
                sources: Arc::new(
                    (0..rng.gen_range(0..7))
                        .map(|index| {
                            let untapped = rng.gen_bool(0.85);
                            ManaSource {
                                permanent: PermanentId(index),
                                object: payment_object(index),
                                untapped,
                                fixed: outputs.choose(&mut rng).unwrap().clone(),
                                abilities: Vec::new(),
                                waterbend: untapped && rng.gen_bool(0.4),
                            }
                        })
                        .collect::<Vec<_>>(),
                ),
                life: 20,
            };
            let cost = ManaCost {
                purpose: purposes.choose(&mut rng).unwrap().clone(),
                ..ManaCost::parse(costs.choose(&mut rng).unwrap())
            };
            let split = if rng.gen_bool(0.5) {
                cost.generic().min(3)
            } else {
                rng.gen_range(0..=cost.generic().min(3))
            };
            let mut policy = SourcePolicy::All;
            if split == 0 && rng.gen_bool(0.3) {
                // Automatic production ignores abilities, so they stay direct.
                policy = SourcePolicy::FixedOnly;
                if let Some(source) = Arc::make_mut(&mut model.sources).first_mut() {
                    source.abilities = vec![tapping_ability(vec![Mana::parse("U")])];
                }
            }
            let obligations = waterbend(&cost, split);
            let pool = pools.choose(&mut rng).unwrap().clone();
            let question = model
                .direct(&obligations, policy)
                .expect("fixed sources and mana costs are direct");
            let searched = model.search(pool.clone(), obligations.clone(), policy);
            assert_eq!(model.direct_witness(pool.clone(), &question), searched);
            assert_eq!(model.direct_payable(&pool, &question), searched.is_some());
            direct += 1;
            if let Some(path) = searched {
                payable += 1;
                waterbent += usize::from(
                    path.iter()
                        .any(|step| matches!(step, PaymentStep::Waterbend(_))),
                );
            }
        }
        // The comparison is vacuous unless both answers and waterbend taps occur.
        assert!(payable > direct / 4 && payable < 3 * direct / 4);
        assert!(
            waterbent > direct / 50,
            "{waterbent} of {direct} ({payable} payable)"
        );
    }

    #[test]
    fn direct_payment_leaves_searched_sources_and_obligations_to_the_search() {
        let cost = ManaCost::parse("1U");
        let mut model = forests(3);
        assert!(model
            .direct(&[pay("1U")], SourcePolicy::AutomaticPrefix)
            .is_none());
        assert!(model
            .direct(&[pay("1"), pay("U")], SourcePolicy::All)
            .is_none());
        assert!(model
            .direct(&waterbend(&cost, 1), SourcePolicy::FixedOnly)
            .is_none());
        let counted = Obligation::Tap {
            candidates: vec![PermanentId(0)].into(),
            remaining: 1,
        };
        assert!(model
            .direct(&[counted, pay("U")], SourcePolicy::All)
            .is_none());
        let output = Obligation::Output {
            choices: vec![Mana::parse("U")].into(),
            life_cost: 0,
        };
        assert!(model
            .direct(&[pay("1"), output], SourcePolicy::All)
            .is_none());
        // A waterbend split spent for another purpose is not one sum.
        let mut other = waterbend(&cost, 1);
        if let Obligation::Pay { cost, .. } = &mut other[1] {
            cost.purpose = SpendingContext::cast(["Lesson"], []);
        }
        assert!(model.direct(&other, SourcePolicy::All).is_none());
        assert!(model
            .direct(&waterbend(&cost, 1), SourcePolicy::All)
            .is_some());

        // A modal land chooses its color and a filter has an upfront cost.
        let modal = tapping_ability(vec![Mana::parse("U"), Mana::parse("R")]);
        let mut filter = tapping_ability(vec![Mana::parse("U")]);
        filter.cost = ManaCost::parse("1");
        for ability in [modal, filter] {
            Arc::make_mut(&mut model.sources).push(source(3, None, vec![ability]));
            assert!(model.direct(&[pay("1U")], SourcePolicy::All).is_none());
            assert!(model
                .direct(&[pay("1U")], SourcePolicy::FixedOnly)
                .is_some());
            // The search still answers, through `plan` and `payable` alike.
            assert!(model.payable(Mana::default(), vec![pay("1U")], SourcePolicy::All));
            let path = model
                .plan(Mana::default(), vec![pay("1U")], SourcePolicy::All)
                .unwrap();
            assert!(path
                .iter()
                .any(|step| matches!(step, PaymentStep::Activate(PermanentId(3), 0))));
            Arc::make_mut(&mut model.sources).pop();
        }
    }

    #[test]
    fn a_payment_scan_keeps_sources_until_its_last_guard_ends() {
        use crate::{
            flow::game::Game,
            state::{game_object::PlayerId, player::PlayerConfig},
        };
        let deck: std::collections::BTreeMap<String, usize> = [("Mountain".into(), 40)].into();
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck.clone()),
                PlayerConfig::new("b", deck),
            ],
            339,
            true,
        );
        game.play();
        let player = PlayerId(0);
        let kept = |game: &Game| game.payment_sources.borrow().sources[player.0].clone();

        // Outside a scan every question builds its own sources.
        let unshared = game.payment_model(player, None).sources;
        assert_eq!(kept(&game), None);
        {
            let _outer = game.payment_scan();
            {
                let _inner = game.payment_scan();
                assert_eq!(game.payment_model(player, None).sources, unshared);
            }
            // The first question's sources outlive the nested computation,
            // and a later question reads them.
            assert_eq!(kept(&game), Some(unshared.clone()));
            assert_eq!(game.payment_model(player, None).sources, unshared);
            // A clone can be written to, so it starts with nothing shared.
            assert_eq!(kept(&game.clone()), None);
        }
        assert_eq!(kept(&game), None);
        game.payment_cache.borrow_mut().entries.clear();
        let binding = crate::search_state::witness(&game);
        game.payment_model(player, None).plan(
            Mana::default(),
            vec![pay("U"), pay("")],
            SourcePolicy::All,
        );
        assert_eq!(game.payment_cache.borrow().entries.len(), 1);
        let fork = game.clone();
        assert_eq!(crate::search_state::witness(&fork), binding);
        assert_eq!(fork.payment_cache.borrow().entries.len(), 1);
        assert_eq!(
            fork.payment_model(player, None).plan(
                Mana::parse("U"),
                vec![pay("U"), pay("")],
                SourcePolicy::All
            ),
            Some(vec![PaymentStep::Spend, PaymentStep::Spend])
        );
        assert_eq!(fork.payment_cache.borrow().entries.len(), 2);
        assert_eq!(game.payment_cache.borrow().entries.len(), 1);
        assert_eq!(crate::search_state::witness(&game), binding);
    }

    #[test]
    fn direct_payment_taps_in_source_order_and_keeps_waterbend_taps_it_needs() {
        // Source order, not usefulness: the Forest ahead of the Islands is tapped.
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(vec![
                source(0, Some("G"), Vec::new()),
                source(1, Some("U"), Vec::new()),
                source(2, Some("U"), Vec::new()),
            ]),
            life: 20,
        };
        let path = model
            .plan(Mana::default(), vec![pay("U")], SourcePolicy::All)
            .unwrap();
        assert_eq!(
            path,
            vec![
                PaymentStep::Fixed(PermanentId(0)),
                PaymentStep::Fixed(PermanentId(1)),
                PaymentStep::Spend,
            ]
        );
        assert!(!model.payable(Mana::default(), vec![pay("UUU")], SourcePolicy::All));

        // {1}{U} with one generic waterbendable: the creature covers the
        // split, so the Island pays {U} and nothing is tapped for {1}.
        let mut creature = source(0, None, Vec::new());
        creature.waterbend = true;
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(vec![creature, source(1, Some("U"), Vec::new())]),
            life: 20,
        };
        let cost = ManaCost::parse("1U");
        let path = model
            .plan(Mana::default(), waterbend(&cost, 1), SourcePolicy::All)
            .unwrap();
        assert_eq!(
            path,
            vec![
                PaymentStep::Fixed(PermanentId(1)),
                PaymentStep::Spend,
                PaymentStep::Waterbend(PermanentId(0)),
                PaymentStep::Spend,
            ]
        );
        assert!(!model.payable(Mana::default(), waterbend(&cost, 0), SourcePolicy::All));
    }

    #[test]
    fn payment_search_bounds_identical_land_subsets_in_automatic_prefixes() {
        for count in [8, 24] {
            let model = forests(count);
            let mut state = search_state(&model, vec![pay(&count.to_string())]);
            state.prefix = Some(count - 1);
            assert!(bounded_search(
                &model,
                state,
                SourcePolicy::AutomaticPrefix,
                count,
                count == 8
            )
            .is_none());
            let path = model
                .plan(
                    Mana::default(),
                    vec![pay(&count.to_string())],
                    SourcePolicy::AutomaticPrefix,
                )
                .unwrap();
            assert_eq!(
                path[..count],
                (0..count)
                    .map(|index| PaymentStep::Fixed(PermanentId(index)))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn payment_search_bounds_modal_and_restricted_bottlenecks() {
        for count in [8, 24] {
            let mut model = forests(count);
            // One Herbalist-shaped source: any color, or two restricted mana.
            Arc::make_mut(&mut model.sources).push(ManaSource {
                permanent: PermanentId(count),
                object: payment_object(count),
                untapped: true,
                fixed: None,
                waterbend: true,
                abilities: vec![tapping_ability(vec![
                    Mana::parse("U"),
                    Mana::parse("R"),
                    Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 2, 0, 0, 0, 0]),
                ])],
            });
            let state = search_state(&model, vec![pay("U"), pay("U")]);
            assert!(bounded_search(
                &model,
                state,
                SourcePolicy::All,
                20 * (count + 1),
                count == 8
            )
            .is_none());
            // Fixed-only probes must not enumerate the lands trying to reach
            // a color available solely from a visible modal activation.
            let state = search_state(&model, vec![pay("U")]);
            assert!(bounded_search(
                &model,
                state,
                SourcePolicy::FixedOnly,
                count + 1,
                count == 8
            )
            .is_none());
        }
    }

    #[test]
    fn payment_search_bounds_counted_taps_competing_with_mana() {
        for count in [8, 24] {
            let model = forests(count);
            let state = search_state(
                &model,
                vec![
                    Obligation::Tap {
                        candidates: model
                            .sources
                            .iter()
                            .map(|source| source.permanent)
                            .collect::<Vec<_>>()
                            .into(),
                        remaining: count / 2,
                    },
                    pay(&(count / 2 + 1).to_string()),
                ],
            );
            assert!(bounded_search(&model, state, SourcePolicy::All, count, count == 8).is_none());
        }
    }

    #[test]
    fn payment_search_bounds_seedless_nested_filters() {
        for count in [6, 24] {
            let mut model = forests(count);
            let mut filter = tapping_ability(vec![Mana::parse("U")]);
            filter.cost = ManaCost::parse("U");
            for index in count..count + 2 {
                Arc::make_mut(&mut model.sources).push(source(index, None, vec![filter.clone()]));
            }
            let state = search_state(&model, vec![pay("U")]);
            assert!(bounded_search(
                &model,
                state,
                SourcePolicy::All,
                10 * (count + 1),
                count == 6
            )
            .is_none());
        }
    }

    #[test]
    fn payment_search_preserves_witnesses_and_counted_tap_membership() {
        let mut model = forests(4);
        let mut sacrifice = tapping_ability(vec![Mana::parse("U"), Mana::parse("R")]);
        sacrifice.tap_source = false;
        sacrifice.sacrifice_source = true;
        for source in Arc::make_mut(&mut model.sources) {
            source.abilities.push(sacrifice.clone());
            source.waterbend = true;
        }
        // Only the first land can pay a later counted tap. A failed branch
        // spending it must not hide the successful branch spending land #1.
        let obligations = vec![
            pay("1"),
            Obligation::Tap {
                candidates: vec![PermanentId(0)].into(),
                remaining: 1,
            },
            pay("U"),
        ];
        let state = search_state(&model, obligations);
        let path = bounded_search(&model, state, SourcePolicy::All, 1000, true).unwrap();
        assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(1))));
        // Vary initial resources, colors and prefix policy; compare the
        // complete first witness, not only feasibility or a final mana total.
        for mask in 0..16 {
            for text in ["2", "GU", "UU", "UR", "5"] {
                for policy in [
                    SourcePolicy::All,
                    SourcePolicy::FixedOnly,
                    SourcePolicy::AutomaticPrefix,
                ] {
                    let mut state = search_state(&model, vec![pay(text)]);
                    for index in 0..4 {
                        state.resources.untapped[index] = mask & (1 << index) == 0;
                    }
                    if policy == SourcePolicy::AutomaticPrefix {
                        state.prefix = Some(2);
                    }
                    bounded_search(&model, state, policy, 2000, true);
                }
            }
        }
    }

    #[test]
    fn payment_search_keeps_future_tap_membership_and_limited_uses() {
        let mut model = forests(4);
        let mut filter = tapping_ability(vec![Mana::parse("U")]);
        filter.cost = ManaCost::parse("1");
        filter.tap_count = 1;
        filter.tap_candidates = vec![PermanentId(0)].into();
        Arc::make_mut(&mut model.sources).push(source(4, None, vec![filter]));
        let state = search_state(&model, vec![pay("U")]);
        let path = bounded_search(&model, state, SourcePolicy::All, 200, true).unwrap();
        assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(1))));

        let mut model = forests(3);
        let mut limited = tapping_ability(vec![Mana::parse("U")]);
        limited.tap_source = false;
        limited.limited = true;
        limited.life_cost = 1;
        for source in Arc::make_mut(&mut model.sources) {
            source.abilities.push(limited.clone());
        }
        for mask in 0..8 {
            let mut state = search_state(&model, vec![pay("UU")]);
            state.life = 1; // Output life costs, too, survive canonicalization.
            for index in 0..3 {
                if mask & (1 << index) != 0 {
                    state.resources.use_ability(index, 0);
                }
                state.resources.present[index] = index != 2;
            }
            assert!(bounded_search(&model, state.clone(), SourcePolicy::All, 200, true).is_none());
            state.life = 2;
            bounded_search(&model, state, SourcePolicy::All, 200, true);
        }
    }

    #[test]
    fn payment_search_bounds_repeat_activations_sharing_counted_taps() {
        for count in [6, 24] {
            let mut model = forests(count);
            // Two Cabbage Merchant-shaped abilities share the same Food taps.
            for source in Arc::make_mut(&mut model.sources) {
                source.fixed = None;
            }
            let mut counted = tapping_ability(vec![Mana::parse("U")]);
            counted.tap_source = false;
            counted.tap_count = 2;
            counted.tap_candidates = model
                .sources
                .iter()
                .map(|source| source.permanent)
                .collect::<Vec<_>>()
                .into();
            for index in count..count + 2 {
                Arc::make_mut(&mut model.sources).push(source(index, None, vec![counted.clone()]));
            }
            let state = search_state(&model, vec![pay(&"U".repeat(count / 2 + 1))]);
            assert!(bounded_search(&model, state, SourcePolicy::All, 1000, count == 6).is_none());
            let state = search_state(&model, vec![pay(&"U".repeat(count / 2))]);
            assert!(bounded_search(&model, state, SourcePolicy::All, 1000, count == 6).is_some());
        }
    }

    /// A single tap cannot both produce mana and help waterbend. Modal and
    /// restricted outputs still contribute only what this cost can spend.
    #[test]
    fn payment_bound_does_not_search_subsets_of_alternative_taps() {
        let sources = (0..4)
            .map(|index| ManaSource {
                permanent: PermanentId(index),
                object: payment_object(index),
                untapped: true,
                fixed: Some(Mana::parse("G")),
                abilities: vec![
                    tapping_ability(vec![Mana::parse("U"), Mana::parse("R")]),
                    tapping_ability(vec![Mana::restricted_to(
                        ManaRestriction::spell(["Lesson"]),
                        [0, 0, 0, 0, 2, 0],
                    )]),
                ],
                waterbend: true,
            })
            .collect();
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(sources),
            life: 20,
        };
        let mut visited = PaymentSearch::default();
        let state = PaymentSearchState {
            resources: model.resources(),
            pool: Mana::default(),
            life: 20,
            obligations: vec![
                Obligation::Pay {
                    cost: ManaCost::default(),
                    waterbend: false,
                },
                Obligation::Pay {
                    cost: ManaCost::parse("5"),
                    waterbend: true,
                },
            ],
            prefix: None,
        };
        assert!(model
            .witness(state, SourcePolicy::All, &mut visited)
            .is_none());
        // Count explored states instead of asserting a host-dependent duration.
        assert!(visited.visited.is_empty());

        let path = model
            .plan(
                Mana::default(),
                vec![Obligation::Pay {
                    cost: ManaCost::parse("4"),
                    waterbend: true,
                }],
                SourcePolicy::All,
            )
            .unwrap();
        assert_eq!(path.len(), 5);
        for (index, step) in path[..4].iter().enumerate() {
            assert!(matches!(step, PaymentStep::Fixed(id) if *id == PermanentId(index)));
        }
        assert!(matches!(path[4], PaymentStep::Spend));
    }

    /// Tapping and then sacrificing one object are independent resources.
    #[test]
    fn payment_bound_keeps_non_tapping_production_after_a_tap() {
        let mut sacrifice = tapping_ability(vec![Mana::parse("U")]);
        sacrifice.tap_source = false;
        sacrifice.sacrifice_source = true;
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(vec![ManaSource {
                permanent: PermanentId(0),
                object: payment_object(0),
                untapped: true,
                fixed: Some(Mana::parse("G")),
                abilities: vec![sacrifice],
                waterbend: true,
            }]),
            life: 20,
        };
        let path = model
            .plan(
                Mana::default(),
                vec![Obligation::Pay {
                    cost: ManaCost::parse("GU"),
                    waterbend: false,
                }],
                SourcePolicy::All,
            )
            .unwrap();
        assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(0))));
        assert!(matches!(path[1], PaymentStep::Activate(PermanentId(0), 0)));
    }

    #[test]
    fn payment_bound_counts_restricted_mana_for_each_pending_purpose() {
        let mut pool = Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 1, 0, 0, 0, 0]);
        pool.add(&Mana::restricted_to(
            ManaRestriction::spell(["Shrine"]),
            [0, 1, 0, 0, 0, 0],
        ));
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(Vec::new()),
            life: 20,
        };
        let obligations = ["Lesson", "Shrine"]
            .map(|subtype| Obligation::Pay {
                cost: ManaCost {
                    purpose: SpendingContext::cast([subtype], []),
                    ..ManaCost::parse("U")
                },
                waterbend: false,
            })
            .to_vec();
        assert!(model.plan(pool, obligations, SourcePolicy::All).is_some());
    }

    #[test]
    fn payment_bound_counts_spend_effect_and_combat_classes() {
        let mut pool = Mana::of_class(
            SpendEffect::CreatureEntersWithPlusOneCounter {
                without_subtype: "Human".into(),
            }
            .into(),
            [0, 1, 0, 0, 0, 0],
        );
        pool.add_combat(&Mana::parse("R"));
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(Vec::new()),
            life: 20,
        };
        assert!(model
            .plan(pool, vec![pay("U"), pay("R")], SourcePolicy::All)
            .is_some());
    }

    #[test]
    fn funded_final_payment_needs_no_unit_search() {
        let model = forests(1);
        let mut pool = Mana::parse("U");
        pool.add_combat(&Mana::parse("R"));
        pool.add(&Mana::of_class(
            SpendEffect::CreatureEntersWithPlusOneCounter {
                without_subtype: "Human".into(),
            }
            .into(),
            [2, 0, 0, 0, 0, 0],
        ));
        // Hybrid assignment must backtrack (R for U/R leaves U for U/B),
        // while the generic pip can use the rider. No source needs tapping.
        for policy in [
            SourcePolicy::All,
            SourcePolicy::FixedOnly,
            SourcePolicy::AutomaticPrefix,
        ] {
            let mut state = search_state(&model, vec![pay("{1}{U/R}{U/B}")]);
            state.pool = pool.clone();
            state.prefix = Some(0);
            assert_eq!(
                bounded_search(&model, state, policy, 0, false),
                Some(vec![PaymentStep::Spend])
            );
        }
    }

    /// Restricted mana can finance a future activation even when it cannot
    /// pay the current debt. That activation's output covers the debt instead.
    #[test]
    fn payment_bound_allows_restricted_mana_to_fund_a_filter() {
        let mut filter = tapping_ability(vec![Mana::parse("U")]);
        filter.cost = ManaCost {
            purpose: SpendingContext::ability(["Ally"]),
            ..ManaCost::parse("1")
        };
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(vec![source(0, None, vec![filter])]),
            life: 20,
        };
        let pool = Mana::restricted_to(ManaRestriction::only([], ["Ally"]), [0, 0, 0, 0, 1, 0]);
        let path = model
            .plan(
                pool,
                vec![Obligation::Pay {
                    cost: ManaCost::parse("U"),
                    waterbend: false,
                }],
                SourcePolicy::All,
            )
            .unwrap();
        assert!(matches!(path[0], PaymentStep::Activate(PermanentId(0), 0)));
        assert!(matches!(path[1], PaymentStep::Spend));
        assert!(matches!(path[2], PaymentStep::Output(0)));
    }

    #[test]
    fn automatic_prefix_lower_bound_keeps_the_first_witness() {
        let mut model = forests(4);
        Arc::make_mut(&mut model.sources).push(modal_source(4, &["U", "R"], "1"));
        Arc::make_mut(&mut model.sources)[0].fixed = Some(Mana::parse("UU"));
        Arc::make_mut(&mut model.sources)[1].waterbend = true;
        for tapped in [false, true] {
            Arc::make_mut(&mut model.sources)[0].untapped = !tapped;
            for pool in [Mana::default(), Mana::parse("U"), Mana::parse("2")] {
                for cost in ["U", "3U", "5U", "R", "6"] {
                    for obligations in [
                        vec![pay(cost)],
                        waterbend(&ManaCost::parse(cost), 2),
                        vec![pay(cost), pay("G")],
                    ] {
                        let reference = (0..=model.sources.len()).find_map(|budget| {
                            let mut state = search_state(&model, obligations.clone());
                            state.pool = pool.clone();
                            state.prefix = Some(budget);
                            let mut search = PaymentSearch::new(&obligations);
                            model.witness(state, SourcePolicy::AutomaticPrefix, &mut search)
                        });
                        assert_eq!(
                            model.search(pool.clone(), obligations, SourcePolicy::AutomaticPrefix),
                            reference
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn limited_mana_uses_preserve_grouped_witnesses() {
        let abilities = ["U", "R"].map(|output| ManaActivation {
            tap_source: false,
            limited: true,
            ..tapping_ability(vec![Mana::parse(output)])
        });
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(
                (0..3)
                    .map(|index| source(index, None, abilities.to_vec()))
                    .collect(),
            ),
            life: 20,
        };
        for mask in 0..64 {
            let mut state = search_state(&model, vec![pay("UR")]);
            for source in 0..3 {
                for ability in 0..2 {
                    if mask & (1 << (source * 2 + ability)) != 0 {
                        state.resources.use_ability(source, ability);
                    }
                }
            }
            let mut identities = PaymentSearch {
                groups: Some(vec![]),
                max_states: Some(2000),
                ..Default::default()
            };
            let reference = model.witness(state.clone(), SourcePolicy::All, &mut identities);
            let mut grouped = PaymentSearch::new(&state.obligations);
            grouped.groups = Some(grouped.groups(&model));
            grouped.max_states = Some(2000);
            assert_eq!(
                model.witness(state, SourcePolicy::All, &mut grouped),
                reference
            );
        }
    }

    #[test]
    fn projected_output_bound_keeps_class_maxima_and_saturation() {
        let lesson = ManaRestriction::spell(["Lesson"]);
        let ally = ManaRestriction::spell(["Ally"]);
        let outputs = [
            Mana::parse("UU"),
            Mana::restricted_to(lesson.clone(), [0, 3, 0, 0, 0, 0]),
            Mana::restricted_to(lesson, [0, 1, 0, 4, 0, 0]),
            Mana::restricted_to(ally, [0, 5, 0, 0, 0, 0]),
            Mana::of_class(
                crate::state::mana::ManaClass {
                    combat: true,
                    restriction: None,
                    effect: None,
                },
                [0, 250, 0, 0, 0, 0],
            ),
        ];
        for purpose in [
            SpendingContext::Resolving,
            SpendingContext::cast(["Lesson"], []),
            SpendingContext::cast(["Ally"], []),
        ] {
            for text in ["U", "7U", "R", "{U/R}R", "255"] {
                let cost = ManaCost {
                    purpose: purpose.clone(),
                    ..ManaCost::parse(text)
                };
                for count in 0..=outputs.len() {
                    let modes = &outputs[..count];
                    let mut reference = Mana::default();
                    for mode in modes {
                        reference.max_with(mode);
                    }
                    let projected = super::optimistic_pool(modes, &cost);
                    assert!(projected.classes().is_empty());
                    assert_eq!(projected.can_pay(&cost), reference.can_pay(&cost));
                    // Adding repeated outputs preserves the same saturation
                    // even when their allowed classes differ.
                    for mode in &outputs {
                        let mut combined = projected.clone();
                        combined.add(&super::optimistic_pool(std::slice::from_ref(mode), &cost));
                        let mut single_reference = reference.clone();
                        single_reference.add(mode);
                        assert_eq!(combined.can_pay(&cost), single_reference.can_pay(&cost));
                    }
                }
            }
        }
    }

    #[test]
    fn forced_payment_transitions_keep_complete_witnesses() {
        let model = forests(1);
        let obligations = vec![
            Obligation::Tap {
                candidates: vec![PermanentId(0)].into(),
                remaining: 0,
            },
            pay(""),
            Obligation::Output {
                choices: vec![Mana::parse("U")].into(),
                life_cost: 2,
            },
            pay("1U"),
        ];
        assert_eq!(
            model.plan(Mana::default(), obligations.clone(), SourcePolicy::All),
            Some(vec![
                PaymentStep::Spend,
                PaymentStep::Output(0),
                PaymentStep::Fixed(PermanentId(0)),
                PaymentStep::Spend
            ])
        );
        let mut poor = model;
        poor.life = 1;
        assert!(poor
            .plan(Mana::default(), obligations, SourcePolicy::All)
            .is_none());
    }

    #[test]
    fn payment_net_bound_counts_filter_fees_and_saved_outputs() {
        for output in ["U", "UU", "UUU"] {
            for fee in ["1", "2", "3"] {
                let model = PaymentModel {
                    cache: None,
                    sources: Arc::new(vec![
                        modal_source(0, &[output], fee),
                        modal_source(1, &["R"], "U"),
                    ]),
                    life: 3,
                };
                for pool in [Mana::default(), Mana::parse("G"), Mana::parse("GG")] {
                    for saved in [false, true] {
                        for cost in ["U", "UU", "UR", "2U"] {
                            let mut obligations = vec![];
                            if saved {
                                obligations.push(Obligation::Output {
                                    choices: vec![Mana::parse("G")].into(),
                                    life_cost: 1,
                                });
                            }
                            obligations.push(pay(cost));
                            let mut state = search_state(&model, obligations);
                            state.pool = pool.clone();
                            bounded_search(&model, state, SourcePolicy::All, 200, true);
                        }
                    }
                }
            }
        }
        let model = PaymentModel {
            cache: None,
            sources: Arc::new(vec![modal_source(0, &["UU"], "2")]),
            life: 3,
        };
        let mut state = search_state(&model, vec![pay("UU")]);
        state.pool = Mana::parse("G");
        assert!(bounded_search(&model, state, SourcePolicy::All, 0, true).is_none());
    }

    fn modal_source(index: usize, outputs: &[&str], cost: &str) -> ManaSource {
        let mut ability = tapping_ability(outputs.iter().map(|text| Mana::parse(text)).collect());
        ability.cost = ManaCost::parse(cost);
        source(index, None, vec![ability])
    }

    /// The same search without the color-set bound, as a reference.
    fn total_bound_search(
        model: &PaymentModel,
        pool: &Mana,
        obligations: &[Obligation],
    ) -> Option<Vec<PaymentStep>> {
        let mut reference = PaymentSearch::new(obligations);
        reference.total_bound_only = true;
        reference.gross_bound_only = true;
        let mut state = search_state(model, obligations.to_vec());
        state.pool = pool.clone();
        model.witness(state, SourcePolicy::All, &mut reference)
    }

    /// One source that makes U or R cannot pay {U}{R}, however many Forests
    /// stand beside it: the root is rejected without expanding a state.
    #[test]
    fn payment_color_bound_rejects_unpayable_colors_without_searching() {
        const ANY: [&str; 5] = ["W", "U", "B", "R", "G"];
        for count in [8, 24] {
            let mut dual = forests(count);
            Arc::make_mut(&mut dual.sources).push(modal_source(count, &["U", "R"], ""));
            let mut filter = forests(count);
            Arc::make_mut(&mut filter.sources).push(modal_source(count, &ANY, "1"));
            let mut any = forests(count);
            Arc::make_mut(&mut any.sources).push(modal_source(count, &ANY, ""));
            for (model, obligations) in [
                (&dual, vec![pay("UR")]),
                (&dual, vec![pay("{U/R}R")]),
                (&filter, vec![pay("UR")]),
                (&any, vec![pay("U"), pay("U")]),
                (&any, vec![pay("1U"), pay("R")]),
                (&any, vec![pay("CU")]),
            ] {
                let state = search_state(model, obligations.clone());
                assert!(
                    bounded_search(model, state, SourcePolicy::All, 0, false).is_none(),
                    "{obligations:?}"
                );
                if count == 8 {
                    assert!(total_bound_search(model, &Mana::default(), &obligations).is_none());
                }
            }
        }
    }

    /// Payable costs that meet the bound with nothing to spare keep their
    /// first witness: hybrid pips, restricted mana for its purpose, a modal
    /// two-mana output and a paid filter.
    #[test]
    fn payment_color_bound_keeps_payable_costs_at_the_bound() {
        let lesson = |text: &str| Obligation::Pay {
            cost: ManaCost {
                purpose: SpendingContext::cast(["Lesson"], []),
                ..ManaCost::parse(text)
            },
            waterbend: false,
        };
        let mut herbalist = modal_source(9, &["U", "R"], "");
        Arc::make_mut(&mut herbalist.abilities[0].outputs).push(Mana::restricted_to(
            ManaRestriction::spell(["Lesson"]),
            [0, 2, 0, 0, 0, 0],
        ));
        let cases: Vec<(Vec<ManaSource>, Mana, Vec<Obligation>)> = vec![
            // Two U/R sources and a Forest pay two hybrid pips and green.
            (
                vec![
                    modal_source(8, &["U", "R"], ""),
                    modal_source(9, &["U", "R"], ""),
                ],
                Mana::default(),
                vec![pay("{U/R}{U/R}G")],
            ),
            // The hybrid pip takes the color the colored pip leaves.
            (
                vec![modal_source(9, &["U", "R"], "")],
                Mana::parse("R"),
                vec![pay("{U/R}R")],
            ),
            // Lesson-only mana in the pool pays a Lesson's blue pip.
            (
                vec![modal_source(9, &["U", "R"], "")],
                Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 1, 0, 0, 0, 0]),
                vec![lesson("UR")],
            ),
            // One tap for two Lesson mana pays both blue pips of a Lesson.
            (
                vec![herbalist.clone()],
                Mana::default(),
                vec![lesson("UUG")],
            ),
            // A filter paid by one Forest makes the only blue mana.
            (
                vec![modal_source(9, &["W", "U", "B", "R", "G"], "1")],
                Mana::default(),
                vec![pay("UG")],
            ),
            // A nested filter cost and its parent share the supply.
            (
                vec![modal_source(9, &["W", "U", "B", "R", "G"], "G")],
                Mana::default(),
                vec![pay("G"), pay("U")],
            ),
        ];
        for (extra, pool, obligations) in cases {
            let mut model = forests(2);
            Arc::make_mut(&mut model.sources).extend(extra);
            let reference = total_bound_search(&model, &pool, &obligations);
            assert!(reference.is_some(), "{obligations:?}");
            let mut state = search_state(&model, obligations.clone());
            state.pool = pool;
            assert_eq!(
                bounded_search(&model, state, SourcePolicy::All, 200, false),
                reference,
                "{obligations:?}"
            );
        }
        // The same two Lesson mana cannot pay a spell that is not a Lesson.
        let mut model = forests(8);
        Arc::make_mut(&mut model.sources).push(herbalist);
        let state = search_state(&model, vec![pay("UUG")]);
        assert!(bounded_search(&model, state, SourcePolicy::All, 0, true).is_none());
    }

    /// On every small board of a varied palette, the bound returns exactly
    /// the reference search's first witness: it rejects nothing payable.
    #[test]
    fn payment_color_bound_matches_the_search_without_it() {
        let lesson_two =
            Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 2, 0, 0, 0, 0]);
        let palette = |kind: usize, index: usize| -> ManaSource {
            let mut source = modal_source(index, &[], "");
            match kind {
                0 => source.abilities.clear(),
                1 => source.abilities[0].outputs = vec![Mana::parse("U"), Mana::parse("R")].into(),
                2 => {
                    source.abilities[0].outputs =
                        ["W", "U", "B", "R", "G"].map(Mana::parse).to_vec().into();
                    source.abilities[0].cost = ManaCost::parse("1");
                }
                3 => {
                    source.abilities[0].outputs =
                        vec![Mana::parse("U"), Mana::parse("G"), lesson_two.clone()].into();
                    source.waterbend = true;
                }
                4 => {
                    // A repeatable-per-turn ability that does not tap.
                    source.abilities[0].tap_source = false;
                    source.abilities[0].limited = true;
                    source.abilities[0].outputs = vec![Mana::parse("R"), Mana::parse("C")].into();
                }
                _ => source.abilities[0].outputs = vec![Mana::parse("UR")].into(),
            }
            source.fixed = match kind {
                0 => Some(Mana::parse("G")),
                4 => Some(Mana::parse("U")),
                _ => None,
            };
            source
        };
        let purposes = [
            SpendingContext::default(),
            SpendingContext::cast(["Lesson"], []),
        ];
        let costs = [
            vec!["UR"],
            vec!["UG"],
            vec!["{U/R}R"],
            vec!["{U/R}{U/G}"],
            vec!["UU1"],
            vec!["CU"],
            vec!["{U/R}{U/R}G"],
            vec!["2UR"],
            vec!["U", "U"],
            vec!["1", "UR"],
            vec!["G", "{U/R}"],
        ];
        let pools = [Mana::default(), Mana::parse("R"), lesson_two.clone()];
        let (mut payable, mut rejected) = (0, 0);
        // Every multiset of up to four sources from the six kinds.
        for code in 0..7_usize.pow(4) {
            let kinds = (0..4)
                .map(|slot| code / 7_usize.pow(slot) % 7)
                .collect::<Vec<_>>();
            if kinds.windows(2).any(|pair| pair[0] > pair[1]) {
                continue;
            }
            let model = PaymentModel {
                cache: None,
                sources: Arc::new(
                    kinds
                        .iter()
                        .enumerate()
                        .filter(|(_, kind)| **kind < 6)
                        .map(|(index, kind)| palette(*kind, index))
                        .collect::<Vec<_>>(),
                ),
                life: 20,
            };
            for cost in &costs {
                for purpose in &purposes {
                    for pool in &pools {
                        let mut obligations = cost
                            .iter()
                            .map(|text| Obligation::Pay {
                                cost: ManaCost {
                                    purpose: purpose.clone(),
                                    ..ManaCost::parse(text)
                                },
                                waterbend: cost.len() == 1,
                            })
                            .collect::<Vec<_>>();
                        if cost.len() == 2 {
                            // A started activation: its cost, its output, the parent.
                            obligations.insert(
                                1,
                                Obligation::Output {
                                    choices: vec![Mana::parse("U"), Mana::parse("R")].into(),
                                    life_cost: 0,
                                },
                            );
                        }
                        let reference = total_bound_search(&model, pool, &obligations);
                        let mut state = search_state(&model, obligations.clone());
                        state.pool = pool.clone();
                        let mut search = PaymentSearch::new(&obligations);
                        let path = model.witness(state, SourcePolicy::All, &mut search);
                        assert_eq!(path, reference, "{kinds:?} {cost:?} {pool:?}");
                        payable += usize::from(path.is_some());
                        rejected += usize::from(path.is_none() && search.visited.is_empty());
                    }
                }
            }
        }
        assert!(payable > 1_000 && rejected > 1_000, "{payable} {rejected}");
    }

    #[test]
    fn registered_mana_activations_consume_finite_payment_resources() {
        for definition in ContentPack::default().definitions() {
            for ability in &definition.activated_abilities {
                if ability.mana_choices().is_some() {
                    assert!(
                        ability.tap_source
                            || ability.sacrifice_source
                            || ability.once_each_turn
                            || ability.exhaust
                            || ability.tap_cost.as_ref().is_some_and(|cost| cost.count > 0),
                        "{} has no finite payment resource cost",
                        definition.name
                    );
                    assert!(
                        ability.sacrifice.is_none() && !ability.waterbend,
                        "{} needs its additional upfront cost represented",
                        definition.name
                    );
                }
            }
        }
    }
}
