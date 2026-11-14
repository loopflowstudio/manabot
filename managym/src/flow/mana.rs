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
// player's sources across the questions of a `PaymentScan`. Pool arithmetic
// and spend-purpose restrictions belong to state/mana.rs.

use std::num::NonZeroUsize;

use crate::{
    agent::action::{Action, AgentError},
    flow::{
        decision::{
            Decision, EffectFrame, ManaProduction, PendingManaBonus, SpendCompletion,
            SuspendedResolution, TappedForMana,
        },
        game::{ActivationSource, Game, Payment, PendingChoice},
        statics::Characteristics,
    },
    state::{
        ability::Effect,
        card::{
            ActivatedAbilityDefinition, ReplacementEffect, TriggeredManaAbility,
            TriggeredManaOutput,
        },
        game_object::{ObjectId, PermanentId, PlayerId},
        mana::{Color, Mana, ManaCost, ManaSpendChoice, ManaSpent},
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
        choices: Vec<Mana>,
        life_cost: u32,
    },
    Tap {
        candidates: Vec<PermanentId>,
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
    tap_candidates: Vec<PermanentId>,
    limited: bool,
    sacrifice_source: bool,
    ability_index: usize,
    cost: ManaCost,
    outputs: Vec<Mana>,
    life_cost: u32,
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
    used: Vec<Vec<bool>>,
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
#[derive(Clone, Copy, PartialEq, Eq)]
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
    root_taps: Vec<Vec<PermanentId>>,
    /// Classes of interchangeable sources, built when the search grows.
    groups: Option<Vec<Vec<usize>>>,
    visited: std::collections::HashSet<PaymentSearchState>,
    #[cfg(test)]
    max_states: Option<usize>,
    /// Reference search for tests: skip the color-set bound.
    #[cfg(test)]
    total_bound_only: bool,
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
    fn canonical(groups: &[Vec<usize>], state: &PaymentSearchState) -> PaymentSearchState {
        let mut key = state.clone();
        let resources = &state.resources;
        let resource_key = |index: usize| {
            (
                resources.present[index],
                resources.untapped[index],
                &resources.used[index],
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
            for (&to, from) in group.iter().zip(order) {
                key.resources.present[to] = resources.present[from];
                key.resources.untapped[to] = resources.untapped[from];
                key.resources.used[to].clone_from(&resources.used[from]);
            }
        }
        key
    }

    fn insert(&mut self, model: &PaymentModel, state: &PaymentSearchState) -> bool {
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

struct PaymentModel {
    sources: Vec<ManaSource>,
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
    sources: [Option<Vec<ManaSource>>; 2],
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

impl PaymentModel {
    fn reserve(&mut self, reservation: SourceReservation) {
        if let Some(source) = self
            .sources
            .iter_mut()
            .find(|source| source.permanent == reservation.permanent)
        {
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
    }

    fn resources(&self) -> PaymentResources {
        PaymentResources {
            present: vec![true; self.sources.len()],
            untapped: self.sources.iter().map(|source| source.untapped).collect(),
            used: self
                .sources
                .iter()
                .map(|source| vec![false; source.abilities.len()])
                .collect(),
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
        if resources.used[index][ability_index]
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
        let mut total = usable_total(&state.pool);
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            // Fixed mana, tapping abilities and waterbending compete for the
            // same tap. Non-tapping activations can still use this object later.
            let mut tap_total = 0;
            if state.resources.untapped[index] {
                tap_total = source.fixed.as_ref().map_or(0, &usable_total);
                if waterbend && source.waterbend {
                    tap_total = tap_total.max(1);
                }
            }
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                let output = self.uses(&state.resources, index, ability_index) as u64
                    * ability.outputs.iter().map(&usable_total).max().unwrap_or(0);
                if ability.tap_source {
                    tap_total = tap_total.max(output);
                } else {
                    total += output;
                }
            }
            total += tap_total;
        }
        for obligation in &state.obligations {
            if let Obligation::Output { choices, .. } = obligation {
                total += choices.iter().map(&usable_total).max().unwrap_or(0);
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
        // Only states that need exploration need an owned visited key.
        if total < owed
            || (visited.bounds_colors() && !self.colors_coverable(&state, visited))
            || !visited.insert(self, &state)
        {
            return None;
        }
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::state();
        let first = state.obligations[0].clone();
        let rest = state.obligations[1..].to_vec();
        match first {
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                if remaining == 0 {
                    return self.witness(
                        PaymentSearchState {
                            obligations: rest,
                            ..state
                        },
                        policy,
                        visited,
                    );
                }
                for index in self.tap_candidates(&state.resources, &candidates) {
                    let mut next = state.clone();
                    next.resources.untapped[index] = false;
                    next.obligations[0] = Obligation::Tap {
                        candidates: candidates.clone(),
                        remaining: remaining - 1,
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
                let life = state.life.checked_sub(life_cost)?;
                for (mode, output) in choices.iter().enumerate() {
                    let mut pool = state.pool.clone();
                    pool.add(output);
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            pool,
                            life,
                            obligations: rest.clone(),
                            ..state.clone()
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
                if cost.mana_value == 0 {
                    return self
                        .witness(
                            PaymentSearchState {
                                obligations: rest,
                                prefix: None,
                                ..state
                            },
                            policy,
                            visited,
                        )
                        .map(|mut path| {
                            path.insert(0, PaymentStep::Spend);
                            path
                        });
                }
                // Color/restriction upper bound: count each repeatable output,
                // relaxing shared resources and choosing every color together.
                let mut optimistic = state.pool.clone();
                let mut usable = u64::from(state.pool.usable_total(&cost));
                for (index, source) in self.sources.iter().enumerate() {
                    if !state.resources.present[index] {
                        continue;
                    }
                    if state.resources.untapped[index] {
                        if let Some(output) = &source.fixed {
                            optimistic.add(output);
                            usable += u64::from(output.usable_total(&cost));
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
                        let mut upper = Mana::default();
                        for output in &ability.outputs {
                            upper.max_with(output);
                        }
                        for _ in 0..uses {
                            optimistic.add(&upper);
                        }
                        usable += uses as u64
                            * ability
                                .outputs
                                .iter()
                                .map(|m| u64::from(m.usable_total(&cost)))
                                .max()
                                .unwrap_or(0);
                    }
                }
                if usable < u64::from(cost.mana_value) || !optimistic.can_pay(&cost) {
                    return None;
                }
                for choice in state.pool.spending_choices(&cost) {
                    let mut next = state.clone();
                    next.pool.spend(&choice.unit, &cost.purpose);
                    // Pool the whole current cost before spending starts;
                    // production cannot interleave with its unit choices.
                    if !next.pool.can_pay(&choice.remaining) {
                        continue;
                    }
                    next.obligations[0] = Obligation::Pay {
                        cost: choice.remaining,
                        waterbend,
                    };
                    // The zero-cost arm owns completion and the Spend boundary,
                    // including when this was the last unit of a nonzero cost.
                    if let Some(path) = self.witness(next, policy, visited) {
                        return Some(path);
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
                            let mut next = state.clone();
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
                        let mut next = state.clone();
                        if ability.tap_source {
                            next.resources.untapped[index] = false;
                        }
                        if ability.sacrifice_source {
                            next.resources.present[index] = false;
                        }
                        if ability.limited {
                            next.resources.used[index][ability_index] = true;
                        }
                        next.prefix = None;
                        next.obligations = Self::activation_obligations(ability);
                        next.obligations.extend(state.obligations.clone());
                        if let Some(mut path) = self.witness(next, policy, visited) {
                            path.insert(
                                0,
                                PaymentStep::Activate(source.permanent, ability.ability_index),
                            );
                            return Some(path);
                        }
                    }
                    if waterbend
                        && source.waterbend
                        && state.resources.untapped[index]
                        && cost.generic() > 0
                    {
                        let mut next = state.clone();
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
        let path = self.search_after_tap(pool, obligations, permanent);
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
        let path = match self.direct(&obligations, policy) {
            Some(direct) => {
                let path = self.direct_witness(pool.clone(), &direct);
                debug_assert_eq!(
                    path,
                    self.search(pool, obligations.clone(), policy),
                    "direct payment witness differs from the search"
                );
                path
            }
            None => self.search(pool, obligations, policy),
        };
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::end_plan(&path);
        path
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
        for budget in 0..=self.sources.len() {
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

    fn payment_model(&self, player: PlayerId, sacrifice: Option<PermanentId>) -> PaymentModel {
        let _scan = self.board_scan();
        let mut reserved = Vec::new();
        if let Some(source) = sacrifice {
            reserved.push(SourceReservation {
                permanent: source,
                tap: false,
                sacrifice: true,
            });
        }
        if let Some(PendingChoice::ManaPayment {
            payment, sacrifice, ..
        }) = &self.pending_choice
        {
            if let Some(reservation) = self.payment_reservation(payment) {
                reserved.push(reservation);
            }
            if let Some(source) = sacrifice {
                reserved.push(SourceReservation {
                    permanent: *source,
                    tap: false,
                    sacrifice: true,
                });
            }
        }
        if let Some(PendingChoice::SacrificePayment { payment, .. }) = &self.pending_choice {
            if let Some(reservation) = self.payment_reservation(payment) {
                reserved.push(reservation);
            }
        }
        fn reserve(decision: &Decision, game: &Game, reserved: &mut Vec<SourceReservation>) {
            match decision {
                Decision::ManaSpend { completion, .. } => {
                    if let SpendCompletion::Upfront(next) = completion.as_ref() {
                        reserve(next, game, reserved);
                    }
                    if let SpendCompletion::Announced {
                        payment, sacrifice, ..
                    } = completion.as_ref()
                    {
                        if let Some(r) = game.payment_reservation(payment) {
                            reserved.push(r);
                        }
                        if let Some(source) = sacrifice {
                            reserved.push(SourceReservation {
                                permanent: *source,
                                tap: false,
                                sacrifice: true,
                            });
                        }
                    }
                }
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    resume,
                    ..
                } => {
                    let ability =
                        &game.characteristics(*permanent).activated_abilities[*ability_index];
                    reserved.push(SourceReservation {
                        permanent: *permanent,
                        tap: ability.tap_source,
                        sacrifice: ability.tap_source || ability.sacrifice_source,
                    });
                    if let Some(next) = resume {
                        reserve(next, game, reserved);
                    }
                }
                Decision::ManaChoice {
                    resume: Some(next), ..
                } => reserve(next, game, reserved),
                _ => {}
            }
        }
        if let Some(suspended) = &self.state.suspended_decision {
            reserve(&suspended.decision, self, &mut reserved);
        }
        let sources = self.shared_payment_sources(player);
        let mut life = self.payable_life(player).max(0) as u32;
        if let Some(
            PendingChoice::ManaPayment { payment, .. }
            | PendingChoice::SacrificePayment { payment, .. },
        ) = &self.pending_choice
        {
            if let Payment::Activation {
                source,
                ability_index,
                ..
            } = payment.as_ref()
            {
                if let Some(ability) = self.activation_ability(*source, *ability_index) {
                    life = life.saturating_sub(ability.life_cost);
                }
            }
        }
        let mut model = PaymentModel { sources, life };
        for reservation in reserved {
            model.reserve(reservation);
        }
        model
    }

    /// Begin one legal-action computation; see `PaymentScan`.
    pub(crate) fn payment_scan(&self) -> PaymentScan<'_> {
        self.payment_sources.borrow_mut().scans += 1;
        PaymentScan { game: self }
    }

    /// `player`'s unreserved sources, reused inside a `PaymentScan`.
    fn shared_payment_sources(&self, player: PlayerId) -> Vec<ManaSource> {
        if let Some(sources) = &self.payment_sources.borrow().sources[player.0] {
            debug_assert_eq!(
                *sources,
                self.payment_sources(player),
                "stale shared payment sources"
            );
            return sources.clone();
        }
        let sources = self.payment_sources(player);
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
                            outputs = production_alternatives(&outputs, &bonuses);
                        }
                        Some(ManaActivation {
                            tap_source: ability.tap_source,
                            tap_count: ability.tap_cost.as_ref().map_or(0, |cost| cost.count),
                            tap_candidates: self.tap_cost_candidates(player, ability),
                            limited: ability.once_each_turn || ability.exhaust,
                            sacrifice_source: ability.sacrifice_source,
                            ability_index: index,
                            cost: ability.mana_cost.clone(),
                            outputs,
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

    fn payment_reservation(&self, payment: &Payment) -> Option<SourceReservation> {
        match payment {
            Payment::Activation {
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
                        outputs = production_alternatives(&outputs, &bonuses);
                    }
                    if *tap_remaining > 0 {
                        result.push(Obligation::Tap {
                            candidates: game.tap_cost_candidates(decision.player(), &ability),
                            remaining: *tap_remaining,
                        });
                    }
                    result.push(Obligation::Pay {
                        cost: ability.mana_cost,
                        waterbend: false,
                    });
                    result.push(Obligation::Output {
                        choices: outputs,
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
        match &self.pending_choice {
            Some(PendingChoice::ManaPayment {
                payment,
                alternative,
                ..
            }) => result.push(Obligation::Pay {
                cost: self.payment_mana_cost(player, payment, alternative.as_ref()),
                waterbend: false,
            }),
            Some(PendingChoice::Waterbend {
                payment,
                remaining_generic,
                tapped,
                ..
            }) => {
                let (owed, waterbend) =
                    self.waterbend_owed(player, payment, *remaining_generic, *tapped);
                result.extend(waterbend_obligations(&owed, waterbend));
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
        let mut obligations = PaymentModel::activation_obligations(&ability);
        obligations.extend(self.payment_obligations(player));
        if ability.limited {
            for source in &mut model.sources {
                if source.permanent == permanent {
                    source
                        .abilities
                        .retain(|a| a.ability_index != ability_index);
                }
            }
        }
        let mut path = model.plan(self.pooled_mana(player), obligations, SourcePolicy::All)?;
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

    pub(crate) fn mana_ability_actions(
        &self,
        player: PlayerId,
        sacrifice: Option<PermanentId>,
    ) -> Vec<Action> {
        self.payment_model(player, sacrifice)
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

    /// Feasibility of the chosen-object tap cost and its remaining activation
    /// mana together. `remaining` includes the hypothetical `selected` tap;
    /// the shared transition checks and consumes it before searching the rest.
    pub(crate) fn tap_activation_payment_possible(
        &self,
        player: PlayerId,
        payment: &Payment,
        remaining: usize,
        selected: Option<PermanentId>,
    ) -> bool {
        let Payment::Activation {
            source,
            ability_index,
            ..
        } = payment
        else {
            return false;
        };
        let Some(ability) = self.activation_ability(*source, *ability_index) else {
            return false;
        };
        let mut model = self.payment_model(player, None);
        if let Some(reservation) = self.payment_reservation(payment) {
            model.reserve(reservation);
        }
        let Some(life) = model.life.checked_sub(ability.life_cost) else {
            return false;
        };
        model.life = life;
        let obligations = vec![
            Obligation::Tap {
                candidates: self.tap_cost_candidates(player, &ability),
                remaining,
            },
            Obligation::Pay {
                cost: ability.mana_cost,
                waterbend: ability.waterbend,
            },
        ];
        let pool = self.pooled_mana(player);
        match selected {
            Some(permanent) => model.plan_after_tap(pool, obligations, permanent).is_some(),
            None => model.payable(pool, obligations, SourcePolicy::All),
        }
    }

    pub(crate) fn can_pay_announced_activation(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        payment: &Payment,
        sacrifice: Option<PermanentId>,
    ) -> bool {
        let mut model = self.payment_model(player, sacrifice);
        if let Some(reservation) = self.payment_reservation(payment) {
            model.reserve(reservation);
        }
        if let Payment::Activation {
            source,
            ability_index,
            ..
        } = payment
        {
            let Some(ability) = self.activation_ability(*source, *ability_index) else {
                return false;
            };
            let Some(remaining) = model.life.checked_sub(ability.life_cost) else {
                return false;
            };
            model.life = remaining;
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
                    model.sources[index] = source;
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
                        choices: after,
                        life_cost: 0,
                    },
                );
            }
        }
        self.payment_model(player, None)
            .payable(pool, obligations, SourcePolicy::All)
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
        for permanent in candidates {
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
        let obligations = self.payment_obligations(player);
        // Only the prefix through the first Spend must be automatic; later
        // obligations may still require visible mana activations and outputs.
        let model = self.payment_model(player, None);
        let Some(path) = model.plan(
            self.pooled_mana(player),
            obligations,
            SourcePolicy::AutomaticPrefix,
        ) else {
            return Ok(());
        };
        if path
            .iter()
            .take_while(|step| !matches!(step, PaymentStep::Spend))
            .any(|step| step.action(player).is_some())
        {
            return Ok(());
        }
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
            child.source = self.state.permanents[permanent].as_ref().map(|p| p.object);
            child.source_ref = source_ref;
            self.run_before_decision(child, caller);
        } else {
            *self.state.suspended_decision_mut() = Some(caller);
        }
    }

    /// Tap a source of fixed mana for mana and resolve its triggered mana
    /// abilities. The planner offers only sources whose bonuses need no
    /// choice (`fixed_source_mana`), so this never suspends.
    fn tap_fixed_mana_source(&mut self, player: PlayerId, permanent: PermanentId) {
        let produced = fixed_output(&self.characteristics(permanent));
        self.tap_permanent(permanent, true);
        self.state.mana_pool_mut(player).add(&produced);
        let tapped = self.tapped_for_mana(permanent);
        let bonuses = self.production_bonuses(player, &tapped);
        let source = self
            .permanent_object(permanent)
            .expect("a tapped fixed source is on the battlefield");
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
        match production_alternatives(&[fixed_output(view)], &bonuses).as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        }
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
                self.state.mana_pool_mut(player).add(&mana);
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
        self.payment_model(player, None).payable(
            self.pooled_mana(player),
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
            SourcePolicy::FixedOnly,
        )
    }

    /// Both pools combined (regular + until-end-of-combat).
    pub(crate) fn pooled_mana(&self, player: PlayerId) -> Mana {
        let mut total = self.state.players[player.0].mana_pool.clone();
        total.add_combat(&self.state.players[player.0].combat_mana_pool);
        total
    }

    pub(crate) fn produce_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        let path = self
            .payment_model(player, None)
            .plan(
                self.pooled_mana(player),
                vec![Obligation::Pay {
                    cost: cost.clone(),
                    waterbend: false,
                }],
                SourcePolicy::FixedOnly,
            )
            .ok_or_else(|| AgentError("not enough automatic mana".into()))?;
        for step in path {
            if let PaymentStep::Fixed(source) = step {
                self.tap_fixed_mana_source(player, source);
            }
        }

        Ok(())
    }

    /// Commit precisely the selected unit, retaining its expiry identity.
    fn spend_mana_unit(&mut self, player: PlayerId, unit: &Mana, cost: &ManaCost) -> ManaSpent {
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
            regular.spend(unit, &cost.purpose)
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
                remaining.spend(&choice.unit, &cost.purpose);
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
                spent: ManaSpent::default(),
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
                for (sum, amount) in spent.amounts.iter_mut().zip(receipt.amounts) {
                    *sum += amount;
                }
                spent.entry_counters += receipt.entry_counters;
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
                SpendCompletion::Announced {
                    payment,
                    sacrifice,
                    waterbend,
                } => {
                    self.complete_announced_payment(player, payment, sacrifice, waterbend, spent)?;
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
            *self.state.mana_pool_mut(player) = self.unspent_mana_retained(player, pool);
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
            let kept = self.unspent_mana_retained(player, combat);
            self.state.mana_pool_mut(player).add(&kept);
        }
    }

    /// What survives of `player`'s unspent `pool` as it would empty: all of
    /// it as red under Ozai (CR 614.1a), its red mana while they keep red
    /// mana this turn, otherwise nothing.
    fn unspent_mana_retained(&self, player: PlayerId, mut pool: Mana) -> Mana {
        if self.unspent_mana_becomes_red(player) {
            pool.become_red();
            pool
        } else if self.keeps_unspent_mana(player, Color::Red) {
            pool.red_only()
        } else {
            Mana::default()
        }
    }

    /// Whether `player` controls a permanent replacing mana loss (CR 614.1a).
    fn unspent_mana_becomes_red(&self, player: PlayerId) -> bool {
        self.iter_battlefield_permanents(player).any(|id| {
            self.state.permanents[id].as_ref().is_some_and(|permanent| {
                self.state.objects[permanent.object]
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

/// Each distinct pool addition of a mana ability with `outputs` followed by
/// the triggered mana abilities it triggers. The planner reads these as one
/// output choice; play resolves the bonuses' own choices afterward.
fn production_alternatives(outputs: &[Mana], bonuses: &[TriggeredManaOutput]) -> Vec<Mana> {
    let mut totals = Vec::new();
    for output in outputs {
        for bonus in bonus_alternatives(output, bonuses) {
            let mut sum = output.clone();
            sum.add(&bonus);
            if !totals.contains(&sum) {
                totals.push(sum);
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

    fn tapping_ability(outputs: Vec<Mana>) -> ManaActivation {
        ManaActivation {
            tap_source: true,
            tap_count: 0,
            tap_candidates: Vec::new(),
            limited: false,
            sacrifice_source: false,
            ability_index: 0,
            cost: ManaCost::default(),
            outputs,
            life_cost: 0,
        }
    }

    fn source(index: usize, fixed: Option<&str>, abilities: Vec<ManaActivation>) -> ManaSource {
        ManaSource {
            permanent: PermanentId(index),
            object: ObjectId::in_slot(index, 0),
            untapped: true,
            fixed: fixed.map(Mana::parse),
            abilities,
            waterbend: false,
        }
    }

    fn forests(count: usize) -> PaymentModel {
        PaymentModel {
            sources: (0..count)
                .map(|index| source(index, Some("G"), Vec::new()))
                .collect(),
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
                max_states: Some(20_000),
                ..Default::default()
            };
            let original = model.witness(state, policy, &mut identities);
            assert_eq!(path, original);
        }
        path
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
                sources: (0..rng.gen_range(0..7))
                    .map(|index| {
                        let untapped = rng.gen_bool(0.85);
                        ManaSource {
                            permanent: PermanentId(index),
                            object: ObjectId::in_slot(index, 0),
                            untapped,
                            fixed: outputs.choose(&mut rng).unwrap().clone(),
                            abilities: Vec::new(),
                            waterbend: untapped && rng.gen_bool(0.4),
                        }
                    })
                    .collect(),
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
                if let Some(source) = model.sources.first_mut() {
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
            candidates: vec![PermanentId(0)],
            remaining: 1,
        };
        assert!(model
            .direct(&[counted, pay("U")], SourcePolicy::All)
            .is_none());
        let output = Obligation::Output {
            choices: vec![Mana::parse("U")],
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
            model.sources.push(source(3, None, vec![ability]));
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
            model.sources.pop();
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
    }

    #[test]
    fn direct_payment_taps_in_source_order_and_keeps_waterbend_taps_it_needs() {
        // Source order, not usefulness: the Forest ahead of the Islands is tapped.
        let model = PaymentModel {
            sources: vec![
                source(0, Some("G"), Vec::new()),
                source(1, Some("U"), Vec::new()),
                source(2, Some("U"), Vec::new()),
            ],
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
            sources: vec![creature, source(1, Some("U"), Vec::new())],
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
            model.sources.push(ManaSource {
                permanent: PermanentId(count),
                object: ObjectId::in_slot(count, 0),
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
                            .collect(),
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
                model
                    .sources
                    .push(source(index, None, vec![filter.clone()]));
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
        for source in &mut model.sources {
            source.abilities.push(sacrifice.clone());
            source.waterbend = true;
        }
        // Only the first land can pay a later counted tap. A failed branch
        // spending it must not hide the successful branch spending land #1.
        let obligations = vec![
            pay("1"),
            Obligation::Tap {
                candidates: vec![PermanentId(0)],
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
        filter.tap_candidates = vec![PermanentId(0)];
        model.sources.push(source(4, None, vec![filter]));
        let state = search_state(&model, vec![pay("U")]);
        let path = bounded_search(&model, state, SourcePolicy::All, 200, true).unwrap();
        assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(1))));

        let mut model = forests(3);
        let mut limited = tapping_ability(vec![Mana::parse("U")]);
        limited.tap_source = false;
        limited.limited = true;
        limited.life_cost = 1;
        for source in &mut model.sources {
            source.abilities.push(limited.clone());
        }
        for mask in 0..8 {
            let mut state = search_state(&model, vec![pay("UU")]);
            state.life = 1; // Output life costs, too, survive canonicalization.
            for index in 0..3 {
                state.resources.used[index][0] = mask & (1 << index) != 0;
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
            for source in &mut model.sources {
                source.fixed = None;
            }
            let mut counted = tapping_ability(vec![Mana::parse("U")]);
            counted.tap_source = false;
            counted.tap_count = 2;
            counted.tap_candidates = model
                .sources
                .iter()
                .map(|source| source.permanent)
                .collect();
            for index in count..count + 2 {
                model
                    .sources
                    .push(source(index, None, vec![counted.clone()]));
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
                object: ObjectId::in_slot(index, 0),
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
        let model = PaymentModel { sources, life: 20 };
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
            sources: vec![ManaSource {
                permanent: PermanentId(0),
                object: ObjectId::in_slot(0, 0),
                untapped: true,
                fixed: Some(Mana::parse("G")),
                abilities: vec![sacrifice],
                waterbend: true,
            }],
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
            sources: Vec::new(),
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
            sources: Vec::new(),
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
            sources: vec![source(0, None, vec![filter])],
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
            dual.sources.push(modal_source(count, &["U", "R"], ""));
            let mut filter = forests(count);
            filter.sources.push(modal_source(count, &ANY, "1"));
            let mut any = forests(count);
            any.sources.push(modal_source(count, &ANY, ""));
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
        herbalist.abilities[0].outputs.push(Mana::restricted_to(
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
            model.sources.extend(extra);
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
        model.sources.push(herbalist);
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
                1 => source.abilities[0].outputs = vec![Mana::parse("U"), Mana::parse("R")],
                2 => {
                    source.abilities[0].outputs =
                        ["W", "U", "B", "R", "G"].map(Mana::parse).to_vec();
                    source.abilities[0].cost = ManaCost::parse("1");
                }
                3 => {
                    source.abilities[0].outputs =
                        vec![Mana::parse("U"), Mana::parse("G"), lesson_two.clone()];
                    source.waterbend = true;
                }
                4 => {
                    // A repeatable-per-turn ability that does not tap.
                    source.abilities[0].tap_source = false;
                    source.abilities[0].limited = true;
                    source.abilities[0].outputs = vec![Mana::parse("R"), Mana::parse("C")];
                }
                _ => source.abilities[0].outputs = vec![Mana::parse("UR")],
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
                sources: kinds
                    .iter()
                    .enumerate()
                    .filter(|(_, kind)| **kind < 6)
                    .map(|(index, kind)| palette(*kind, index))
                    .collect(),
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
                                    choices: vec![Mana::parse("U"), Mana::parse("R")],
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
