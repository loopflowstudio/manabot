// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel searches identity-preserving resource steps;
// its witness search prunes with optimistic whole-payment and color-set
// bounds and memoizes feasibility across interchangeable sources
// (PaymentSearch), always returning the first concrete path in source order.
// Game owns their events and explicit Decision continuations. Pool arithmetic
// and spend-purpose restrictions belong to state/mana.rs.

use std::sync::Arc;

use crate::{
    agent::action::{Action, AgentError},
    flow::{
        decision::{Decision, EffectFrame, SuspendedResolution},
        game::{ActivationSource, Game, Payment, PendingChoice},
        statics::Characteristics,
    },
    state::{
        ability::Effect,
        card::{ActivatedAbilityDefinition, ReplacementEffect},
        game_object::{CardId, PermanentId, PlayerId},
        mana::{Color, Mana, ManaCost},
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

#[derive(Clone)]
struct ManaSource {
    permanent: PermanentId,
    card: CardId,
    untapped: bool,
    fixed: Option<Mana>,
    abilities: Vec<ManaActivation>,
    waterbend: bool,
}

#[derive(Clone, PartialEq, Eq)]
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
enum PaymentStep {
    Fixed(PermanentId),
    Activate(PermanentId, usize),
    Spend(Mana),
    Output(usize),
    Waterbend(PermanentId),
    Tap(CardId),
}

impl PaymentStep {
    fn action(&self, player: PlayerId) -> Option<Action> {
        match self {
            Self::Activate(permanent, ability_index) => Some(Action::ActivateAbility {
                player,
                permanent: *permanent,
                ability_index: *ability_index,
            }),
            Self::Tap(card) => Some(Action::SelectCard {
                player,
                card: *card,
            }),
            Self::Output(mode) => Some(Action::ChooseMode {
                player,
                mode: *mode,
            }),
            Self::Waterbend(permanent) => Some(Action::WaterbendTap {
                player,
                permanent: *permanent,
            }),
            Self::Fixed(_) | Self::Spend(_) => None,
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
        for (restriction, restricted) in mana.restricted() {
            if self.obligations.iter().any(|obligation| {
                matches!(obligation, Obligation::Pay { cost, .. }
                    if restriction.allows(&cost.purpose))
            }) {
                for (amount, more) in usable.iter_mut().zip(restricted) {
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
        // No counted tap cost leaves this bound open (usize::MAX) for the
        // finite-resource limits below.
        let mut uses = self
            .tap_candidates(resources, &ability.tap_candidates)
            .len()
            .checked_div(ability.tap_count)
            .unwrap_or(usize::MAX);
        if ability.tap_source || ability.sacrifice_source || ability.limited {
            uses = uses.min(1);
        }
        assert_ne!(
            uses,
            usize::MAX,
            "mana activation must consume a finite resource"
        );
        uses
    }

    fn tap_candidates(
        &self,
        resources: &PaymentResources,
        candidates: &[PermanentId],
    ) -> Vec<usize> {
        self.sources
            .iter()
            .enumerate()
            .filter_map(|(index, source)| {
                (resources.present[index]
                    && resources.untapped[index]
                    && candidates.contains(&source.permanent))
                .then_some(index)
            })
            .collect()
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
                let candidates_now = self.tap_candidates(&state.resources, &candidates);
                if candidates_now.len() < remaining {
                    return None;
                }
                for index in candidates_now {
                    let mut next = state.clone();
                    next.resources.untapped[index] = false;
                    next.obligations[0] = Obligation::Tap {
                        candidates: candidates.clone(),
                        remaining: remaining - 1,
                    };
                    if let Some(mut path) = self.witness(next, policy, visited) {
                        path.insert(0, PaymentStep::Tap(self.sources[index].card));
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
                for remainder in state.pool.payment_options(&cost) {
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            pool: remainder.clone(),
                            obligations: rest.clone(),
                            prefix: None,
                            ..state.clone()
                        },
                        policy,
                        visited,
                    ) {
                        path.insert(0, PaymentStep::Spend(remainder));
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
    /// spent as another color during payment (`Mana::payment_remainders`), so
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
        crate::benchmark_payments::end_plan();
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
            .into_iter()
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

    fn plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::begin_plan(self.sources.len());
        let path = self.search(pool, obligations, policy);
        #[cfg(feature = "benchmark")]
        crate::benchmark_payments::end_plan();
        path
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
        let bonus_sources = self.tap_bonus_sources(player);
        let sources = self
            .characteristics_many(self.iter_battlefield_permanents(player))
            .map(|(permanent, characteristics)| {
                let source = self.state.permanents[permanent].as_ref().unwrap();
                // A source that lost its abilities offers no zero-mana tap.
                let can_tap = self.permanent_can_tap(permanent);
                let prohibited = self.activations_prohibited(permanent);
                let fixed = (can_tap && !prohibited && !characteristics.mana_abilities.is_empty())
                    .then(|| self.fixed_source_mana(permanent, &characteristics, &bonus_sources));
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
                            let bonus = self.tap_mana_bonus_from(&bonus_sources, permanent);
                            for output in &mut outputs {
                                output.add(&bonus);
                            }
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
                    card: source.card,
                    untapped: !source.tapped,
                    fixed,
                    abilities,
                    waterbend: !source.tapped
                        && (characteristics.types.is_artifact() || characteristics.is_creature()),
                }
            })
            .collect();
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
                        let player = decision.player();
                        for output in &mut outputs {
                            output.add(&game.tap_mana_bonus(*permanent, player));
                        }
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
        self.payment_model(player, sacrifice)
            .plan(
                self.pooled_mana(player),
                vec![Obligation::Pay {
                    cost: cost.clone(),
                    waterbend: false,
                }],
                SourcePolicy::All,
            )
            .is_some()
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
            Some(permanent) => model.plan_after_tap(pool, obligations, permanent),
            None => model.plan(pool, obligations, SourcePolicy::All),
        }
        .is_some()
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
        model
            .plan(
                self.pooled_mana(player),
                vec![Obligation::Pay {
                    cost: cost.clone(),
                    waterbend: false,
                }],
                SourcePolicy::All,
            )
            .is_some()
    }

    /// Can `player` pay `cost`, `waterbend` of its generic with mana or
    /// waterbend taps and the rest with mana?
    pub(crate) fn waterbend_payment_possible(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
    ) -> bool {
        self.payment_model(player, None)
            .plan(
                self.pooled_mana(player),
                waterbend_obligations(cost, waterbend),
                SourcePolicy::All,
            )
            .is_some()
    }

    /// Test a waterbend payment after reserving taps, without copying the game.
    /// Non-mana taps reserve resources but produce no mana. The debug oracle
    /// retains the full fork calculation if future rules add a tap dependency.
    pub(crate) fn waterbend_payment_after_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
        taps: impl Iterator<Item = PermanentId> + Clone,
    ) -> bool {
        let mut model = self.payment_model(player, None);
        for permanent in taps.clone() {
            model.reserve(SourceReservation {
                permanent,
                tap: true,
                sacrifice: false,
            });
        }
        let possible = model
            .plan(
                self.pooled_mana(player),
                waterbend_obligations(cost, waterbend),
                SourcePolicy::All,
            )
            .is_some();
        debug_assert_eq!(possible, {
            let mut probe = self.clone();
            for permanent in taps {
                probe.tap_permanent(permanent, false);
            }
            probe.waterbend_payment_possible(player, cost, waterbend)
        });
        possible
    }

    pub(crate) fn mana_choice_completes_payment(&self, player: PlayerId, mana: &Mana) -> bool {
        let mut pool = self.pooled_mana(player);
        pool.add(mana);
        self.payment_model(player, None)
            .plan(pool, self.payment_obligations(player), SourcePolicy::All)
            .is_some()
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
                    card: self.state.permanents[*permanent].as_ref().unwrap().card,
                });
            }
        }
        actions
    }

    pub(crate) fn pay_mana_tap_cost(
        &mut self,
        player: PlayerId,
        card: CardId,
    ) -> Result<(), AgentError> {
        if !self
            .mana_tap_cost_actions(player)
            .contains(&Action::SelectCard { player, card })
        {
            return Err(AgentError("illegal mana tap cost selection".into()));
        }
        let permanent = self.state.card_to_permanent[card].unwrap();
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

    /// Automatic fixed sources and spending commit the exact witness prefix.
    /// Chosen-output sources suspend instead; no witness selects a player's color.
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
            .take_while(|step| !matches!(step, PaymentStep::Spend(_)))
            .any(|step| step.action(player).is_some())
        {
            return Ok(());
        }
        for step in path {
            match step {
                PaymentStep::Fixed(source) => self.tap_fixed_mana_source(player, source),
                PaymentStep::Spend(remainder) => {
                    self.spend_mana_to_remainder(player, remainder);
                    break;
                }
                _ => unreachable!("validated automatic prefix"),
            }
        }
        let ability = self.characteristics(permanent).activated_abilities[ability_index].clone();
        if ability.tap_source {
            self.tap_permanent(permanent, true);
        }
        if ability.life_cost > 0 {
            self.change_life(player, -(ability.life_cost as i32));
        }
        // Retain the output and original payment frame before any cost
        // effect can suspend. Resumption must not spend mana, tap or pay life
        // again, nor expose this output before sacrifice/zone work finishes.
        let mut caller = self.state.suspended_decision_mut().take().unwrap();
        let Decision::ManaUpfront { resume, .. } = caller.decision else {
            unreachable!()
        };
        caller.decision = Decision::ManaChoice {
            player,
            choices: ability.mana_choices().unwrap(),
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
            let source_ref = self.permanent_object_ref(permanent);
            let mut child = EffectFrame::new(player, effects);
            child.source = self.state.permanents[permanent].as_ref().map(|p| p.card);
            child.source_ref = source_ref;
            self.run_before_decision(child, caller);
        } else {
            *self.state.suspended_decision_mut() = Some(caller);
        }
        Ok(())
    }

    fn tap_fixed_mana_source(&mut self, player: PlayerId, permanent: PermanentId) {
        let produced = self
            .characteristics(permanent)
            .mana_abilities
            .iter()
            .map(|ability| ability.mana.clone())
            .collect::<Vec<_>>();
        self.tap_permanent(permanent, true);
        for mana in produced {
            self.state.mana_pool_mut(player).add(&mana);
        }
    }

    /// Fixed tap output plus immediate triggered bonuses; chosen outputs stay
    /// in the payment search until the player selects a color. Callers check
    /// that the source has mana abilities, can tap and has not lost its
    /// abilities, reading its characteristics once.
    fn fixed_source_mana(
        &self,
        permanent: PermanentId,
        view: &Characteristics,
        bonus_sources: &[Arc<Characteristics>],
    ) -> Mana {
        let mut output = Mana::default();
        for ability in &view.mana_abilities {
            output.add(&ability.mana);
        }
        output.add(&self.tap_mana_bonus_from(bonus_sources, permanent));
        output
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
        self.payment_model(player, None)
            .plan(
                self.pooled_mana(player),
                vec![Obligation::Pay {
                    cost: cost.clone(),
                    waterbend: false,
                }],
                SourcePolicy::FixedOnly,
            )
            .is_some()
    }

    /// Both pools combined (regular + until-end-of-combat).
    pub(crate) fn pooled_mana(&self, player: PlayerId) -> Mana {
        let mut total = self.state.players[player.0].mana_pool.clone();
        total.add(&self.state.players[player.0].combat_mana_pool);
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

    /// Pay `cost` from eligible pools. Within each color, restricted mana is
    /// spent first; combat mana is spent first within each restriction class.
    pub(crate) fn spend_mana(
        &mut self,
        player: PlayerId,
        cost: &ManaCost,
    ) -> Result<(), AgentError> {
        let combined = self.pooled_mana(player);
        if !combined.can_pay(cost) {
            return Err(AgentError("insufficient mana in pool".to_string()));
        }
        let mut remaining = combined;
        remaining.pay(cost);

        self.spend_mana_to_remainder(player, remaining);
        Ok(())
    }

    /// Commit an already validated pool payment. Feasibility probes and real
    /// activations retain the same remainder, including restriction classes.
    fn spend_mana_to_remainder(&mut self, player: PlayerId, remaining: Mana) {
        // Rebuild the pools by restriction class and color, consuming the
        // expiring combat pool before the regular pool within each class.
        let (mana_pool, combat_mana_pool) = self.state.mana_pools_mut(player);
        let mut classes = mana_pool
            .restricted()
            .iter()
            .chain(combat_mana_pool.restricted())
            .map(|(class, _)| class.clone())
            .collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        for class in &classes {
            let mut regular = mana_pool.restricted_of(class);
            let mut combat = combat_mana_pool.restricted_of(class);
            let left = remaining.restricted_of(class);
            for slot in 0..left.len() {
                let spent = regular[slot] + combat[slot] - left[slot];
                let from_combat = spent.min(combat[slot]);
                combat[slot] -= from_combat;
                regular[slot] -= spent - from_combat;
            }
            mana_pool.set_restricted(class, regular);
            combat_mana_pool.set_restricted(class, combat);
        }
        for slot in 0..remaining.mana.len() {
            let spent = mana_pool.mana[slot] + combat_mana_pool.mana[slot] - remaining.mana[slot];
            let combat = spent.min(combat_mana_pool.mana[slot]);
            combat_mana_pool.mana[slot] -= combat;
            mana_pool.mana[slot] -= spent - combat;
        }
    }

    pub(crate) fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        let bonus_sources = self.tap_bonus_sources(player);
        for (permanent_id, view) in self.layer_views(self.iter_battlefield_permanents(player)) {
            if view.mana_abilities.is_empty()
                || !self.permanent_can_tap(permanent_id)
                || self.activations_prohibited(permanent_id)
            {
                continue;
            }
            total.add(&self.fixed_source_mana(permanent_id, &view, &bonus_sources));
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
                self.state.cards[permanent.card]
                    .replacement_effects
                    .contains(&ReplacementEffect::UnspentManaBecomesRed)
            }) && !self.characteristics(id).abilities_lost
        })
    }
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
        ManaActivation, ManaSource, Obligation, PaymentModel, PaymentSearch, PaymentSearchState,
        PaymentStep, SourcePolicy,
    };
    use crate::cards::content_pack::ContentPack;
    use crate::state::{
        game_object::{CardId, PermanentId},
        mana::{Mana, ManaCost, ManaRestriction, SpendingContext},
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
            card: CardId(index),
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
    /// sources, so a smaller limit binds from there; zero still asserts that
    /// the root is rejected.
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
                card: CardId(count),
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
                card: CardId(index),
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
        assert!(matches!(&path[4], PaymentStep::Spend(pool) if pool.total() == 0));
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
                card: CardId(0),
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
        assert!(matches!(path[1], PaymentStep::Spend(_)));
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
