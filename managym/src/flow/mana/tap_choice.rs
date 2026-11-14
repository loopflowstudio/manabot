//! Which fixed sources the automatic payer taps. `PaymentModel::choose_taps`
//! starts from
//! the first witness (`plan`, `FixedOnly`), drops every tap the cost does not
//! need (`is_surplus`) and then, among the payments with no surplus tap,
//! prefers in order:
//! 1. the most follow-up costs kept affordable (`Game::payment_followups`),
//! 2. the most hand cards among them,
//! 3. the most instant-timing hand cards,
//! 4. the least reluctant sources (`Reluctance`): the fewest taps of the
//!    most reluctant kind, then of the next kind down, so a basic land pays
//!    before a land with another ability, a mana rock or a creature,
//! 5. the fewest taps whose output has a color, keeping colored sources,
//! 6. the pruned witness, then the earlier candidate in class order.
//!
//! The kind of a source is a tiebreak, never an avoidance: a reluctant
//! source is tapped whenever that keeps a follow-up another payment loses.
//! Mana already in the pool is part of every candidate, so it is spent
//! before any source is tapped for the same mana.
//!
//! Legality stays with `PaymentModel`: a candidate is a set of untapped fixed
//! sources whose outputs, with the pool, satisfy `Mana::can_pay`, the
//! witness's own acceptance test. The preference is arithmetic only. A
//! follow-up is affordable when the pool the payment leaves plus the fixed
//! outputs still untapped pay it, which is the direct answer
//! (`PaymentModel::direct`) to a `FixedOnly` question; debug builds assert
//! the two agree. Mana abilities that must be activated are not counted
//! towards a follow-up, so on a board with such sources the preference is a
//! lower bound of the offline probe's, which asks `SourcePolicy::All`.
//!
//! Work per payment is bounded, never a subset enumeration over sources:
//! - Saved obligations (a non-empty tail): prune only. At most one
//!   `is_surplus` search per tap of the witness, none when the pool holds no
//!   more than the cost. No follow-up or source kind is read.
//! - Otherwise no search runs at all. Pruning is at most one check per tap.
//!   Kinds and follow-ups are read only when an untapped source could
//!   replace a tap. The kind-preferred payment (`Chooser::preferred`) is one
//!   pass that drops surplus sources from the most reluctant down: at most
//!   one check per untapped fixed source, and none when no untapped source
//!   is less reluctant than a tap of the pruned witness. It is the least
//!   reluctant payment when every source adds one mana; with larger outputs
//!   it is a payment with no surplus tap, used only when it is less
//!   reluctant than the pruned witness. The bound and the scores of those
//!   two payments cost at most
//!   `FOLLOWUPS + 2 * (SPEND_STATES + LEFTOVERS * FOLLOWUPS)` checks. The
//!   search over classes of sources with equal output and reluctance then
//!   spends at most `CHECKS` further checks and keeps the best payment seen
//!   when they run out (`TapChoice::exhausted`). A check is one
//!   `Mana::can_pay`, `is_surplus` without a tail, or pooled-unit spending
//!   state.

use std::collections::HashSet;

use super::{
    outcomes::{PaymentProbe, FOLLOWUPS},
    Color, Mana, ManaCost, Obligation, PaymentModel, PaymentStep, SourcePolicy, SourceReservation,
};

/// How reluctantly a fixed source is tapped, the least reluctant first
/// (`Game::source_reluctance`). No fixed source carries a drawback: a fixed
/// output is mana only, and abilities with life, damage, sacrifice or other
/// instructions are activated, never tapped by the automatic payer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Reluctance {
    /// A basic land with no activated ability that is not a creature.
    #[default]
    BasicLand,
    /// Another land with no activated ability that is not a creature.
    Land,
    /// A noncreature land with an activated ability, which tapping it for
    /// mana can give up.
    UtilityLand,
    /// A permanent that is neither a land nor a creature.
    Permanent,
    /// A creature, a land that is currently one included.
    Creature,
}

impl Reluctance {
    const TIERS: usize = 5;
}

/// A source's place in the kind preference: its reluctance, then whether its
/// output has a color.
type Kind = (Reluctance, bool);

/// The sources a payment taps, compared so that the smaller is preferred:
/// taps at each reluctance from the most reluctant down, then taps whose
/// output has a color.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Kinds {
    tiers: [u16; Reluctance::TIERS],
    colored: u16,
}

impl Kinds {
    fn tap(&mut self, kind: Kind, count: usize) {
        self.tiers[Reluctance::TIERS - 1 - kind.0 as usize] += count as u16;
        self.colored += u16::from(kind.1) * count as u16;
    }
}

fn colored(output: &Mana) -> bool {
    std::iter::once(&output.mana)
        .chain(output.classes().iter().map(|(_, amounts)| amounts))
        .any(|amounts| amounts[..Color::Colorless as usize].iter().any(|n| *n > 0))
}

/// Checks the search over source classes may spend on one payment.
pub(super) const CHECKS: usize = 1024;
/// Pooled-unit spending states read to find what one payment leaves floating.
const SPEND_STATES: usize = 32;
/// Distinct floating remainders one payment is scored with.
const LEFTOVERS: usize = 8;

/// The fixed sources to tap, as ascending indices into the model's sources,
/// and what choosing them cost.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct TapChoice {
    pub taps: Vec<usize>,
    /// Arithmetic checks: no payment search runs for any of them.
    pub checks: usize,
    /// `is_surplus` questions asked with saved obligations, each a search.
    pub searches: usize,
    /// Follow-up costs read; zero when the choice never needed them.
    pub followups: usize,
    /// The class search ran out of `CHECKS` and kept the best payment seen.
    pub exhausted: bool,
    /// Some payment taps less reluctant sources than the pruned witness.
    pub preferred: bool,
    pub stage: Stage,
}

/// Where a choice ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Stage {
    /// Saved obligations: pruned only.
    #[default]
    Tail,
    /// Nothing tapped, or no untapped source could replace a tap.
    NoReplacement,
    /// The follow-ups were read and there are none.
    NoFollowups,
    /// The kind-preferred payment, or the pruned witness when no payment is
    /// less reluctant, keeps every follow-up any payment could.
    WitnessAtBound,
    /// The class search ran.
    ClassSearch,
}

/// Follow-ups a payment leaves affordable, compared in preference order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Kept {
    all: u32,
    hand: u32,
    instant: u32,
}

#[derive(Default)]
struct Budget {
    used: usize,
    limit: Option<usize>,
    exhausted: bool,
}

impl Budget {
    fn take(&mut self) -> bool {
        if self.limit.is_some_and(|limit| self.used >= limit) {
            self.exhausted = true;
            return false;
        }
        self.used += 1;
        true
    }
}

/// Untapped fixed sources with one output and reluctance, in source order. A
/// source some follow-up taps for itself is a class of its own.
struct Class<'a> {
    output: &'a Mana,
    kind: Kind,
    members: Vec<usize>,
}

struct Chooser<'a> {
    model: &'a PaymentModel,
    pool: &'a Mana,
    cost: &'a ManaCost,
    probe: PaymentProbe,
    /// Every untapped fixed source.
    fixed: Vec<usize>,
    /// Parallel to the model's sources; empty until the kinds are read.
    reluctance: Vec<Reluctance>,
}

impl Chooser<'_> {
    fn output(&self, index: usize) -> &Mana {
        self.model.sources[index]
            .fixed
            .as_ref()
            .expect("a fixed source has an output")
    }

    fn kind(&self, index: usize) -> Kind {
        (self.reluctance[index], colored(self.output(index)))
    }

    fn kinds(&self, taps: &[usize]) -> Kinds {
        let mut kinds = Kinds::default();
        for index in taps {
            kinds.tap(self.kind(*index), 1);
        }
        kinds
    }

    /// The payment the kind preference picks when no follow-up matters, or
    /// `None` when it is no less reluctant than `taps`, the pruned witness.
    /// Every usable source starts tapped and each surplus one is dropped,
    /// the most reluctant first and the later of equals first: the reverse
    /// of a greedy choice, exact when every source adds one mana.
    fn preferred(&self, taps: &[usize], budget: &mut Budget) -> Option<Vec<usize>> {
        let worst = taps.iter().map(|index| self.kind(*index)).max()?;
        let mut order: Vec<usize> = self
            .fixed
            .iter()
            .copied()
            .filter(|index| self.output(*index).usable_total(self.cost) > 0)
            .collect();
        if !order
            .iter()
            .any(|index| !taps.contains(index) && self.kind(*index) < worst)
        {
            return None;
        }
        order.sort_unstable_by_key(|index| std::cmp::Reverse((self.kind(*index), *index)));
        let owed = u16::from(self.cost.mana_value);
        let mut chosen = order.clone();
        for index in order {
            let pooled = self.pooled(&chosen);
            if pooled.usable_total(self.cost) <= owed {
                break;
            }
            budget.take();
            if self
                .model
                .is_surplus(&pooled, self.output(index), self.cost, &[])
            {
                chosen.retain(|kept| *kept != index);
            }
        }
        chosen.sort_unstable();
        (self.kinds(&chosen) < self.kinds(taps)).then_some(chosen)
    }

    fn pooled(&self, taps: &[usize]) -> Mana {
        let mut pooled = self.pool.clone();
        for index in taps {
            pooled.add(self.output(*index));
        }
        pooled
    }

    /// `leftover` with the output of every fixed source left untapped,
    /// except `reserved`, which a follow-up taps for itself.
    fn supply(&self, leftover: &Mana, taps: &[usize], reserved: Option<usize>) -> Mana {
        let mut supply = leftover.clone();
        for index in &self.fixed {
            if !taps.contains(index) && reserved != Some(*index) {
                supply.add(self.output(*index));
            }
        }
        supply
    }

    /// The pools paying the cost from `pooled` can leave, at most
    /// `LEFTOVERS` of them from `SPEND_STATES` spending states. The player
    /// chooses the units, so a follow-up counts when any of them pays it, as
    /// in the offline probe. Never empty: when the states run out first, the
    /// empty pool is a lower bound of every remainder.
    fn leftovers(&self, pooled: &Mana, budget: &mut Budget) -> Vec<Mana> {
        if pooled.total() == u16::from(self.cost.mana_value) {
            return vec![Mana::default()];
        }
        let mut found = Vec::new();
        let mut visited = HashSet::new();
        let mut pending = vec![(pooled.clone(), self.cost.clone())];
        let mut states = 0;
        while let Some((pool, remaining)) = pending.pop() {
            if states == SPEND_STATES || found.len() == LEFTOVERS || !budget.take() {
                break;
            }
            states += 1;
            if !visited.insert((pool.clone(), remaining.clone())) {
                continue;
            }
            if remaining.mana_value == 0 {
                if !found.contains(&pool) {
                    found.push(pool);
                }
                continue;
            }
            for choice in pool.spending_choices(&remaining) {
                let mut next = pool.clone();
                next.spend(&choice.unit, &remaining.purpose);
                if next.can_pay(&choice.remaining) {
                    pending.push((next, choice.remaining));
                }
            }
        }
        if found.is_empty() {
            found.push(Mana::default());
        }
        found
    }

    /// Bit `i` is set when follow-up `i` is affordable after tapping `taps`,
    /// which pool `pooled`. `None` when the budget ran out first.
    fn affordable(&self, taps: &[usize], pooled: &Mana, budget: &mut Budget) -> Option<u64> {
        let leftovers = self.leftovers(pooled, budget);
        let supplies: Vec<Mana> = leftovers
            .iter()
            .map(|leftover| self.supply(leftover, taps, None))
            .collect();
        let mut set = 0;
        for (i, followup) in self.probe.followups.iter().enumerate() {
            let reserved = self.probe.tapped_by(i);
            if reserved.is_some_and(|source| taps.contains(&source)) {
                continue;
            }
            for (leftover, supply) in leftovers.iter().zip(&supplies) {
                if !budget.take() {
                    return None;
                }
                let pays = match reserved {
                    None => supply.can_pay(followup),
                    Some(_) => self.supply(leftover, taps, reserved).can_pay(followup),
                };
                debug_assert_eq!(
                    pays,
                    self.authority(leftover, taps, reserved, followup),
                    "follow-up arithmetic differs from the payment model"
                );
                if pays {
                    set |= 1 << i;
                    break;
                }
            }
        }
        Some(set)
    }

    /// The payment model's own answer to what `affordable` computes; debug
    /// builds only.
    fn authority(
        &self,
        leftover: &Mana,
        taps: &[usize],
        reserved: Option<usize>,
        followup: &ManaCost,
    ) -> bool {
        self.model
            .tapped(taps.iter().copied().chain(reserved))
            .payable(
                leftover.clone(),
                vec![Obligation::Pay {
                    cost: followup.clone(),
                    waterbend: false,
                }],
                SourcePolicy::FixedOnly,
            )
    }

    /// Follow-ups no payment of this cost can keep are outside this set: the
    /// whole supply must pay the follow-up and hold its mana beside the cost's.
    fn bound(&self, budget: &mut Budget) -> u64 {
        let owed = u16::from(self.cost.mana_value);
        let mut set = 0;
        for (i, followup) in self.probe.followups.iter().enumerate() {
            budget.take();
            let supply = self.supply(self.pool, &[], self.probe.tapped_by(i));
            if supply.total() >= owed + u16::from(followup.mana_value) && supply.can_pay(followup) {
                set |= 1 << i;
            }
        }
        set
    }

    fn kept(&self, set: u64) -> Kept {
        Kept {
            all: set.count_ones(),
            hand: (set & self.probe.hand).count_ones(),
            instant: (set & self.probe.instant).count_ones(),
        }
    }

    /// Sources that can pay towards the cost, by output and reluctance, the
    /// least reluctant class first and the earlier in source order among
    /// equals. Members of a class are interchangeable for every question
    /// asked here, so a payment is a count per class and taps each class's
    /// earliest sources.
    fn classes(&self) -> Vec<Class<'_>> {
        let mut classes: Vec<(bool, Class)> = Vec::new();
        for index in &self.fixed {
            let output = self.output(*index);
            if output.usable_total(self.cost) == 0 {
                continue;
            }
            let kind = self.kind(*index);
            let own = self.probe.tapping.contains(&Some(*index));
            match classes.iter_mut().find(|(single, class)| {
                !own && !single && class.kind == kind && class.output == output
            }) {
                Some((_, class)) => class.members.push(*index),
                None => classes.push((
                    own,
                    Class {
                        output,
                        kind,
                        members: vec![*index],
                    },
                )),
            }
        }
        classes.sort_by_key(|(_, class)| class.kind);
        classes.into_iter().map(|(_, class)| class).collect()
    }
}

/// Depth-first over tap counts per class, earlier classes first and more of
/// each first, so the first of equally good payments is the one nearest the
/// class order.
struct ClassSearch<'a, 'b> {
    chooser: &'a Chooser<'b>,
    classes: Vec<Class<'a>>,
    /// Usable mana the classes from each position on can still add.
    ahead: Vec<u16>,
    counts: Vec<usize>,
    bound: Kept,
    /// The kind-preferred payment's sources: no payment taps less when every
    /// source adds one mana.
    floor: Kinds,
    best: Kept,
    best_kinds: Kinds,
    taps: Vec<usize>,
    budget: Budget,
}

impl ClassSearch<'_, '_> {
    /// `kinds` counts the taps chosen so far. Taps only add to it, so once
    /// the best payment keeps every follow-up a branch at or past its kinds
    /// holds nothing better.
    fn visit(&mut self, position: usize, pooled: Mana, tapped: usize, kinds: Kinds) {
        if self.best == self.bound && (self.best_kinds <= self.floor || kinds >= self.best_kinds) {
            return;
        }
        if !self.budget.take() {
            return;
        }
        let cost = self.chooser.cost;
        // A payment that already pays the cost makes every further tap
        // surplus, so the remaining classes stay untapped.
        if pooled.can_pay(cost) {
            self.candidate(position, &pooled, kinds);
            return;
        }
        // Every tap of a payment without surplus pays at least one mana.
        let owed = usize::from(cost.mana_value);
        if position == self.classes.len()
            || tapped == owed
            || pooled.usable_total(cost) + self.ahead[position] < u16::from(cost.mana_value)
        {
            return;
        }
        let most = self.classes[position].members.len().min(owed - tapped);
        for count in (0..=most).rev() {
            let mut next = pooled.clone();
            for _ in 0..count {
                next.add(self.classes[position].output);
            }
            let mut chosen = kinds;
            chosen.tap(self.classes[position].kind, count);
            self.counts[position] = count;
            self.visit(position + 1, next, tapped + count, chosen);
        }
        self.counts[position] = 0;
    }

    fn candidate(&mut self, position: usize, pooled: &Mana, kinds: Kinds) {
        let chooser = self.chooser;
        for class in (0..position).filter(|class| self.counts[*class] > 0) {
            if !self.budget.take()
                || chooser
                    .model
                    .is_surplus(pooled, self.classes[class].output, chooser.cost, &[])
            {
                return;
            }
        }
        let mut taps: Vec<usize> = (0..position)
            .flat_map(|class| &self.classes[class].members[..self.counts[class]])
            .copied()
            .collect();
        taps.sort_unstable();
        let Some(set) = chooser.affordable(&taps, pooled, &mut self.budget) else {
            return;
        };
        let kept = chooser.kept(set);
        if (kept, std::cmp::Reverse(kinds)) > (self.best, std::cmp::Reverse(self.best_kinds)) {
            self.best = kept;
            self.best_kinds = kinds;
            self.taps = taps;
        }
    }
}

impl PaymentModel {
    /// This board after tapping `taps`.
    fn tapped(&self, taps: impl Iterator<Item = usize>) -> PaymentModel {
        let mut after = PaymentModel {
            sources: self.sources.clone(),
            life: self.life,
        };
        for index in taps {
            after.reserve(SourceReservation {
                permanent: self.sources[index].permanent,
                tap: true,
                sacrifice: false,
            });
        }
        after
    }

    /// The fixed sources to tap for `cost` in place of `witness`'s, the first
    /// `FixedOnly` witness for it from `pool`. `tail` is the saved callers'
    /// obligations; `reluctance`, parallel to the sources, and `followups`
    /// are each asked at most once, and only when another payment exists to
    /// prefer.
    pub(super) fn choose_taps(
        &self,
        pool: &Mana,
        cost: &ManaCost,
        tail: &[Obligation],
        witness: &[PaymentStep],
        reluctance: impl FnOnce() -> Vec<Reluctance>,
        followups: impl FnOnce() -> PaymentProbe,
    ) -> TapChoice {
        #[cfg(feature = "benchmark")]
        let mut clock = crate::benchmark_payments::SmartTapClock::start();
        let mut chooser = Chooser {
            model: self,
            pool,
            cost,
            probe: PaymentProbe::default(),
            fixed: (0..self.sources.len())
                .filter(|index| {
                    self.sources[*index].untapped && self.sources[*index].fixed.is_some()
                })
                .collect(),
            reluctance: Vec::new(),
        };
        let mut taps: Vec<usize> = witness
            .iter()
            .filter_map(|step| match step {
                PaymentStep::Fixed(id) => self
                    .sources
                    .iter()
                    .position(|source| source.permanent == *id),
                _ => None,
            })
            .collect();
        let mut budget = Budget::default();
        let mut searches = 0;

        // Drop surplus taps, the latest first so the earlier sources stay.
        // Without a tail one pass is exact: a tap the cost needs from a
        // larger pool it needs from a smaller one. With a tail each answer is
        // a search on the board as tapped so far.
        let owed = u16::from(cost.mana_value);
        for position in (0..taps.len()).rev() {
            let pooled = chooser.pooled(&taps);
            if pooled.usable_total(cost) <= owed {
                break;
            }
            let output = chooser.output(taps[position]);
            let surplus = if tail.is_empty() {
                budget.take();
                self.is_surplus(&pooled, output, cost, &[])
            } else {
                searches += 1;
                self.tapped(taps.iter().copied())
                    .is_surplus(&pooled, output, cost, tail)
            };
            if surplus {
                taps.remove(position);
            }
        }
        taps.sort_unstable();
        #[cfg(feature = "benchmark")]
        clock.lap(|metrics| &mut metrics.prune_ns);
        let choice = |taps: Vec<usize>, budget: Budget, followups: usize, stage: Stage| TapChoice {
            taps,
            checks: budget.used,
            searches,
            followups,
            exhausted: budget.exhausted,
            preferred: false,
            stage,
        };

        // Another payment needs an untapped source that can pay towards the
        // cost; with none, or nothing tapped, the pruned witness is the only
        // payment without surplus.
        let replaceable = chooser
            .fixed
            .iter()
            .any(|index| !taps.contains(index) && chooser.output(*index).usable_total(cost) > 0);
        if !tail.is_empty() {
            return choice(taps, budget, 0, Stage::Tail);
        }
        if taps.is_empty() || !replaceable {
            return choice(taps, budget, 0, Stage::NoReplacement);
        }
        chooser.reluctance = reluctance();
        debug_assert_eq!(chooser.reluctance.len(), self.sources.len());
        let preferred = chooser.preferred(&taps, &mut budget);
        #[cfg(feature = "benchmark")]
        clock.lap(|metrics| &mut metrics.kinds_ns);
        let choice = |taps, budget, followups, stage| TapChoice {
            preferred: preferred.is_some(),
            ..choice(taps, budget, followups, stage)
        };
        chooser.probe = followups();
        chooser.probe.followups.truncate(FOLLOWUPS);
        #[cfg(feature = "benchmark")]
        clock.lap(|metrics| &mut metrics.followups_ns);
        let read = chooser.probe.followups.len();
        if read == 0 {
            let taps = preferred.clone().unwrap_or(taps);
            return choice(taps, budget, 0, Stage::NoFollowups);
        }
        let bound = chooser.bound(&mut budget);
        let score = |taps: &[usize], budget: &mut Budget| {
            chooser
                .affordable(taps, &chooser.pooled(taps), budget)
                .expect("no limit is set before the class search")
        };
        // The kind-preferred payment is scored first: at the bound nothing
        // beats it. Short of the bound it stays the floor of the search and
        // the pruned witness competes with it.
        let floor = chooser.kinds(preferred.as_deref().unwrap_or(&taps));
        let mut best = score(preferred.as_deref().unwrap_or(&taps), &mut budget);
        if best == bound {
            #[cfg(feature = "benchmark")]
            clock.lap(|metrics| &mut metrics.score_ns);
            let taps = preferred.clone().unwrap_or(taps);
            return choice(taps, budget, read, Stage::WitnessAtBound);
        }
        if let Some(preferred) = &preferred {
            let witness = score(&taps, &mut budget);
            if chooser.kept(witness) > chooser.kept(best) {
                best = witness;
            } else {
                taps.clone_from(preferred);
            }
        }
        #[cfg(feature = "benchmark")]
        clock.lap(|metrics| &mut metrics.score_ns);

        budget.limit = Some(budget.used + CHECKS);
        let classes = chooser.classes();
        let mut ahead = vec![0; classes.len() + 1];
        for (position, class) in classes.iter().enumerate().rev() {
            ahead[position] =
                ahead[position + 1] + class.output.usable_total(cost) * class.members.len() as u16;
        }
        let mut search = ClassSearch {
            chooser: &chooser,
            counts: vec![0; classes.len()],
            classes,
            ahead,
            bound: chooser.kept(bound),
            floor,
            best: chooser.kept(best),
            best_kinds: chooser.kinds(&taps),
            taps,
            budget,
        };
        search.visit(0, pool.clone(), 0, Kinds::default());
        debug_assert!(
            chooser.pooled(&search.taps).can_pay(cost),
            "the chosen sources pay the cost"
        );
        #[cfg(feature = "benchmark")]
        clock.lap(|metrics| &mut metrics.search_ns);
        choice(search.taps, search.budget, read, Stage::ClassSearch)
    }
}

#[cfg(test)]
mod tests {
    use super::{Reluctance, Stage, TapChoice, CHECKS, LEFTOVERS, SPEND_STATES};
    use crate::flow::mana::{
        outcomes::PaymentProbe,
        tests::{source, tapping_ability, SEARCHED},
        ManaActivation, Obligation, PaymentModel, SourcePolicy,
    };
    use crate::state::{
        game_object::PermanentId,
        mana::{Mana, ManaCost, SpendingContext},
    };

    fn lands(outputs: &[&str]) -> PaymentModel {
        PaymentModel {
            sources: outputs
                .iter()
                .enumerate()
                .map(|(index, output)| source(index, Some(output), Vec::new()))
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

    /// Hand cards at these costs, none at instant timing.
    fn hand(costs: &[&str]) -> PaymentProbe {
        PaymentProbe {
            followups: costs.iter().map(|cost| ManaCost::parse(cost)).collect(),
            hand: (1 << costs.len()) - 1,
            tapping: vec![None; costs.len()],
            ..PaymentProbe::default()
        }
    }

    /// The smart choice for `cost` from an empty pool on basic lands, with
    /// the questions it answered by searching.
    fn choose(
        model: &PaymentModel,
        cost: &str,
        tail: &[Obligation],
        probe: Option<PaymentProbe>,
    ) -> (TapChoice, usize) {
        choose_among(model, &[], "", cost, tail, probe)
    }

    /// `choose` from `pool`, with each source's reluctance; sources past the
    /// end of `kinds` are basic lands.
    fn choose_among(
        model: &PaymentModel,
        kinds: &[Reluctance],
        pool: &str,
        cost: &str,
        tail: &[Obligation],
        probe: Option<PaymentProbe>,
    ) -> (TapChoice, usize) {
        let cost = ManaCost::parse(cost);
        let pool = Mana::parse(pool);
        let witness = model
            .plan(pool.clone(), vec![pay_cost(&cost)], SourcePolicy::FixedOnly)
            .unwrap();
        let mut reluctance = kinds.to_vec();
        reluctance.resize(model.sources.len(), Reluctance::BasicLand);
        SEARCHED.with(|searched| searched.set(0));
        let choice = model.choose_taps(
            &pool,
            &cost,
            tail,
            &witness,
            || reluctance,
            || probe.expect("this payment reads no follow-up"),
        );
        let mut pooled = pool;
        for index in &choice.taps {
            pooled.add(model.sources[*index].fixed.as_ref().unwrap());
        }
        assert!(pooled.can_pay(&cost));
        (choice, SEARCHED.with(|searched| searched.get()))
    }

    fn pay_cost(cost: &ManaCost) -> Obligation {
        Obligation::Pay {
            cost: cost.clone(),
            waterbend: false,
        }
    }

    /// The module's stated ceiling for a payment without saved obligations.
    fn ceiling(witness_taps: usize, sources: usize, followups: usize) -> usize {
        witness_taps + sources + followups + 2 * (SPEND_STATES + LEFTOVERS * followups) + CHECKS
    }

    #[test]
    fn taps_the_sources_that_keep_a_follow_up_affordable() {
        // Wherever the Mountain sits, both Forests pay {1}{G} and it keeps {R}.
        for (order, taps) in [
            (["G", "G", "R"], [0, 1]),
            (["G", "R", "G"], [0, 2]),
            (["R", "G", "G"], [1, 2]),
        ] {
            let (choice, searched) = choose(&lands(&order), "1G", &[], Some(hand(&["R"])));
            assert_eq!(choice.taps, taps);
            assert_eq!((choice.searches, searched), (0, 0));
            assert!(!choice.exhausted);
        }
        // Today's witness for the middle board strands it.
        let witness = lands(&["G", "R", "G"])
            .plan(Mana::default(), vec![pay("1G")], SourcePolicy::FixedOnly)
            .unwrap();
        assert_eq!(witness.len(), 3);
        assert_eq!(
            witness[..2],
            [0, 1].map(|index| super::PaymentStep::Fixed(PermanentId(index)))
        );
    }

    #[test]
    fn never_taps_a_source_the_cost_does_not_need() {
        // Mountain, Mountain, Forest for {1}{G}: the witness taps all three.
        let model = lands(&["R", "R", "G"]);
        let (choice, _) = choose(&model, "1G", &[], Some(PaymentProbe::default()));
        assert_eq!(choice.taps, [0, 2]);
        // A source the cost cannot spend is never tapped on the way.
        let (choice, _) = choose(&lands(&["R", "G"]), "G", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1]);
        // Indivisible surplus stays: one doubled Forest is needed for {G}.
        let (choice, _) = choose(&lands(&["GG", "GG"]), "G", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [0]);
    }

    #[test]
    fn saved_obligations_stay_payable_and_read_no_follow_up() {
        // The saved caller needs the second red: without it, {1}{G} would
        // drop a Mountain; with it, every tap is needed across both costs.
        let model = lands(&["R", "R", "G"]);
        let tail = [pay("R")];
        let (choice, searched) = choose(&model, "1G", &tail, None);
        assert_eq!(choice.taps, [0, 1, 2]);
        assert_eq!((choice.checks, choice.followups), (0, 0));
        assert!(choice.searches <= 3 && searched <= 3);
        assert!(model.tapped(choice.taps.iter().copied()).payable(
            Mana::parse("RRG"),
            vec![pay("1G"), pay("R")],
            SourcePolicy::All,
        ));
        // A tail the pool already covers lets the surplus go, and the saved
        // cost is still payable from the board that leaves.
        let tail = [pay("G")];
        let model = lands(&["R", "R", "G", "G"]);
        let (choice, _) = choose(&model, "1G", &tail, None);
        assert_eq!(choice.taps, [0, 2]);
        assert!(model.tapped(choice.taps.iter().copied()).payable(
            Mana::parse("RG"),
            vec![pay("1G"), pay("G")],
            SourcePolicy::All,
        ));
        // An exact pool asks nothing.
        let (choice, searched) = choose(&lands(&["G", "G", "G"]), "G", &[pay("GG")], None);
        assert_eq!((choice.taps, choice.searches, searched), (vec![0], 0, 0));
    }

    #[test]
    fn prefers_follow_ups_then_hand_cards_then_instants_then_source_order() {
        let model = lands(&["G", "R"]);
        let probe = |hand, instant| PaymentProbe {
            followups: vec![ManaCost::parse("G"), ManaCost::parse("R")],
            hand,
            instant,
            tapping: vec![None; 2],
            ..PaymentProbe::default()
        };
        // {1} keeps one of the two. Equal counts: the hand card wins, then
        // the instant, then the witness's Forest tap.
        assert_eq!(choose(&model, "1", &[], Some(probe(0b01, 0))).0.taps, [1]);
        assert_eq!(choose(&model, "1", &[], Some(probe(0b10, 0))).0.taps, [0]);
        assert_eq!(
            choose(&model, "1", &[], Some(probe(0b11, 0b01))).0.taps,
            [1]
        );
        assert_eq!(
            choose(&model, "1", &[], Some(probe(0b11, 0b10))).0.taps,
            [0]
        );
        assert_eq!(choose(&model, "1", &[], Some(probe(0b11, 0))).0.taps, [0]);
        assert_eq!(choose(&model, "1", &[], Some(probe(0, 0))).0.taps, [0]);
        // More follow-ups beat a hand card: two abilities at {G}{G}{G}
        // against one instant at {R}.
        let mut ability = ManaCost::parse("GGG");
        ability.purpose = SpendingContext::ability(["Elf"]);
        let abilities = PaymentProbe {
            followups: vec![ManaCost::parse("GGG"), ManaCost::parse("R"), ability],
            hand: 0b10,
            instant: 0b10,
            tapping: vec![None; 3],
            ..PaymentProbe::default()
        };
        let model = lands(&["G", "G", "G", "R"]);
        assert_eq!(choose(&model, "1", &[], Some(abilities)).0.taps, [3]);
    }

    #[test]
    fn an_ability_that_taps_its_source_loses_it_to_a_payment_that_taps_it() {
        // Source 0 is a mana creature whose {G}, {T} ability is the follow-up.
        // Paying {G} with it loses the ability; its own mana never pays for
        // the ability either, so the second Forest must be left for it.
        let model = lands(&["G", "G", "G"]);
        let probe = PaymentProbe {
            followups: vec![ManaCost::parse("G")],
            tapping: vec![Some(0)],
            ..PaymentProbe::default()
        };
        let (choice, searched) = choose(&model, "G", &[], Some(probe.clone()));
        assert_eq!((choice.taps, searched), (vec![1], 0));
        // {G}{G} must spare the creature and leaves nothing to pay with.
        let (choice, _) = choose(&model, "GG", &[], Some(probe));
        assert_eq!(choice.taps, [0, 1]);
    }

    #[test]
    fn floating_mana_counts_towards_a_follow_up() {
        // Doubled lands: {1} from a doubled Mountain floats {R} for the
        // follow-up {R}{G}, which the remaining doubled Forest completes.
        let model = lands(&["GG", "RR"]);
        let (choice, searched) = choose(&model, "1", &[], Some(hand(&["RG"])));
        assert_eq!(choice.taps, [0]);
        assert_eq!(searched, 0);
        let (choice, _) = choose(&model, "1", &[], Some(hand(&["RRG"])));
        assert_eq!(choice.taps, [0]);
        let (choice, _) = choose(&model, "1", &[], Some(hand(&["GGR"])));
        assert_eq!(choice.taps, [1]);
    }

    #[test]
    fn taps_the_least_reluctant_sources_among_equal_follow_ups() {
        use Reluctance::{BasicLand, Creature, Land, Permanent, UtilityLand};
        // Llanowar Elves ahead of a Forest for {G}: the witness taps the
        // Elves, the choice the Forest, with one check and no follow-up.
        let model = lands(&["G", "G"]);
        let (choice, searched) = choose_among(&model, &[Creature], "", "G", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1]);
        assert_eq!((choice.stage, choice.preferred), (Stage::NoFollowups, true));
        assert_eq!((choice.checks, searched), (1, 0));
        // A Plains before a land with another ability for a generic pip,
        // both keeping the {W} in hand.
        let model = lands(&["W", "W"]);
        let probe = Some(hand(&["W"]));
        let (choice, _) = choose_among(&model, &[UtilityLand], "", "1", &[], probe);
        assert_eq!(choice.taps, [1]);
        assert_eq!(choice.stage, Stage::WitnessAtBound);
        // The earliest sources of the least reluctant kinds.
        let kinds = [Creature, Land, Land, BasicLand];
        let model = lands(&["G"; 4]);
        let (choice, _) = choose_among(&model, &kinds, "", "2", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1, 3]);
        // The worst tap decides first: two basic lands before one mana rock,
        // and two mana rocks before a creature with a basic land.
        let model = lands(&["CC", "G", "G"]);
        let (choice, _) = choose_among(&model, &[Permanent], "", "2", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1, 2]);
        let kinds = [Creature, Permanent, Permanent, BasicLand];
        let model = lands(&["G", "R", "R", "G"]);
        let (choice, _) = choose_among(&model, &kinds, "", "RR", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1, 2]);
        let (choice, _) = choose_among(&model, &kinds, "", "1R", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1, 3]);
        // No less reluctant source: the pruned witness, with no check.
        let kinds = [BasicLand, Creature];
        let (choice, _) = choose_among(&lands(&["G"; 2]), &kinds, "", "G", &[], Some(hand(&[])));
        assert_eq!((choice.taps, choice.checks), (vec![0], 0));
        assert!(!choice.preferred);
    }

    #[test]
    fn taps_a_reluctant_source_that_keeps_a_follow_up() {
        use Reluctance::{Creature, UtilityLand};
        // A Mountain and a land with another ability that adds {G}, for {1}.
        let model = lands(&["R", "G"]);
        let kinds = [Reluctance::BasicLand, UtilityLand];
        let taps = |costs: &[&str]| {
            choose_among(&model, &kinds, "", "1", &[], Some(hand(costs)))
                .0
                .taps
        };
        assert_eq!(taps(&[]), [0]);
        assert_eq!(taps(&["G"]), [0]);
        assert_eq!(taps(&["R"]), [1]);
        // Elves, Mountain, Forest for {1} with {R} in hand: the witness's
        // Elves keep it, the least reluctant payment taps the Mountain and
        // loses it, and the Forest is the least reluctant that keeps it.
        let model = lands(&["G", "R", "G"]);
        let probe = Some(hand(&["R"]));
        let (choice, searched) = choose_among(&model, &[Creature], "", "1", &[], probe);
        assert_eq!(choice.taps, [2]);
        assert_eq!((choice.stage, searched), (Stage::ClassSearch, 0));
        assert!(!choice.exhausted);
        assert!(choice.checks <= ceiling(1, 3, 1));
        // Only the Elves keep {R}{G}.
        let model = lands(&["G", "R", "G"]);
        let kinds = [Reluctance::BasicLand, Reluctance::BasicLand, Creature];
        let probe = Some(hand(&["RG"]));
        let (choice, _) = choose_among(&model, &kinds, "", "G", &[], probe);
        assert_eq!(choice.taps, [0]);
        let kinds = [Creature];
        let probe = Some(hand(&["RG"]));
        let (choice, _) = choose_among(&model, &kinds, "", "G", &[], probe);
        assert_eq!(choice.taps, [2]);
    }

    #[test]
    fn keeps_colored_sources_among_equally_reluctant_ones() {
        use Reluctance::{BasicLand, Creature, Land};
        // A Forest ahead of a colorless land of its kind, for {1}.
        let model = lands(&["G", "C"]);
        let (choice, _) = choose_among(&model, &[], "", "1", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1]);
        let (choice, _) = choose_among(&model, &[], "", "1", &[], Some(hand(&["1"])));
        assert_eq!(choice.taps, [1]);
        // Reluctance comes first: a colored basic before a colorless
        // nonbasic land or creature.
        for kind in [Land, Creature] {
            let kinds = [BasicLand, kind];
            let (choice, _) = choose_among(&model, &kinds, "", "1", &[], Some(hand(&[])));
            assert_eq!(choice.taps, [0]);
        }
        // A follow-up comes before both: {2} in hand needs every source left
        // to add mana, and {G} needs the Forest.
        let model = lands(&["G", "C", "C"]);
        let (choice, _) = choose_among(&model, &[], "", "1", &[], Some(hand(&["G"])));
        assert_eq!(choice.taps, [1]);
        let model = lands(&["C", "G", "G"]);
        let (choice, _) = choose_among(&model, &[], "", "G", &[], Some(hand(&["GG"])));
        assert_eq!(choice.taps, [1]);
    }

    #[test]
    fn spends_floating_mana_before_tapping_a_source() {
        // {G} floating pays half of {1}{G}: one Forest, not two.
        let model = lands(&["G", "G", "G"]);
        let (choice, _) = choose_among(&model, &[], "G", "1G", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [0]);
        // A pool that pays the whole cost taps nothing and reads nothing.
        let (choice, _) = choose_among(&model, &[], "GG", "1G", &[], None);
        assert_eq!((choice.taps, choice.checks), (vec![], 0));
        assert_eq!(choice.stage, Stage::NoReplacement);
        // Floating {R} pays the generic pip, whichever source adds the {G}.
        let kinds = [Reluctance::Creature];
        let (choice, _) = choose_among(&model, &kinds, "R", "1G", &[], Some(hand(&[])));
        assert_eq!(choice.taps, [1]);
        // The class search starts from the pool as well.
        let model = lands(&["G", "R", "G"]);
        let (choice, _) = choose_among(&model, &kinds, "G", "1G", &[], Some(hand(&["R"])));
        assert_eq!(choice.taps, [2]);
        assert_eq!(choice.stage, Stage::ClassSearch);
    }

    // Blow-up positions. Each asserts counted work, never time.

    #[test]
    fn smart_tap_bounds_many_kinds_of_sources_whose_follow_ups_cannot_be_kept() {
        // Five colors in each of five kinds pay {15}; each follow-up needs
        // all five of one color. Every source is a class of its own.
        use Reluctance::{BasicLand, Creature, Land, Permanent, UtilityLand};
        let outputs: Vec<&str> = (0..25)
            .map(|index| ["W", "U", "B", "R", "G"][index % 5])
            .collect();
        let kinds: Vec<Reluctance> = (0..25)
            .map(|index| [Creature, Permanent, UtilityLand, Land, BasicLand][index / 5])
            .collect();
        let model = lands(&outputs);
        let followups = ["WWWWW", "UUUUU", "BBBBB", "RRRRR", "GGGGG"];
        let probe = Some(hand(&followups));
        let (choice, searched) = choose_among(&model, &kinds, "", "15", &[], probe);
        assert_eq!(searched, 0);
        assert!(choice.exhausted && choice.preferred);
        assert!(choice.checks <= ceiling(15, 25, followups.len()));
        // Ten drops for the least reluctant payment, the bound, two scores,
        // then the whole class budget.
        assert_eq!(choice.checks, 10 + 5 + 5 + 5 + CHECKS);
        // No follow-up can be kept, so the three least reluctant kinds pay.
        assert_eq!(choice.taps, (10..25).collect::<Vec<_>>());
    }

    #[test]
    fn smart_tap_bounds_many_lands_whose_follow_ups_cannot_all_be_kept() {
        // Five each of five colors pay {15}; each follow-up needs all five of
        // one color, so two at most survive and the bound of five is out of
        // reach: the class search runs until its checks are spent.
        let outputs: Vec<&str> = (0..25)
            .map(|index| ["W", "U", "B", "R", "G"][index % 5])
            .collect();
        let model = lands(&outputs);
        let followups = ["WWWWW", "UUUUU", "BBBBB", "RRRRR", "GGGGG"];
        let (choice, searched) = choose(&model, "15", &[], Some(hand(&followups)));
        assert_eq!(searched, 0);
        assert!(choice.exhausted);
        assert!(choice.checks <= ceiling(15, 25, followups.len()));
        // The bound and the witness's score, then the whole class budget.
        assert_eq!(choice.checks, 5 + 5 + CHECKS);
        // The first candidate taps three whole colors and keeps the other two.
        assert_eq!(choice.taps.len(), 15);
        assert!(choice.taps.iter().all(|index| index % 5 < 3));
    }

    #[test]
    fn smart_tap_stops_at_the_first_payment_that_keeps_every_follow_up() {
        let outputs: Vec<&str> = (0..25)
            .map(|index| ["W", "U", "B", "R", "G"][index % 5])
            .collect();
        let model = lands(&outputs);
        // Four of a color survive the witness's one tap of each: the bound
        // and the witness's score are the only work.
        let kept = ["WWWW", "UUUU", "BBBB", "RRRR", "GGGG"];
        let (choice, searched) = choose(&model, "5", &[], Some(hand(&kept)));
        assert_eq!((choice.checks, searched, choice.exhausted), (10, 0, false));
        assert_eq!(choice.taps, [0, 1, 2, 3, 4]);
        // Five of every color but white do not; the first candidate, all five
        // Plains, keeps them all and ends the search.
        let whole = ["UUUUU", "BBBBB", "RRRRR", "GGGGG"];
        let (choice, searched) = choose(&model, "5", &[], Some(hand(&whole)));
        assert_eq!(choice.taps, [0, 5, 10, 15, 20]);
        assert_eq!((choice.checks, searched, choice.exhausted), (15, 0, false));
        // Thirty Forests have one payment up to identity.
        let (choice, _) = choose(&lands(&["G"; 30]), "7", &[], Some(hand(&["G"])));
        assert_eq!((choice.checks, choice.followups), (2, 1));
        assert_eq!(choice.taps, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn smart_tap_bounds_doubled_lands_that_float_mana() {
        // Heartbeat of Spring: twelve doubled Forests and Mountains pay {5}
        // with one mana floating, so every candidate also reads the pools its
        // spending can leave.
        let outputs: Vec<&str> = (0..24).map(|index| ["GG", "RR"][index % 2]).collect();
        let model = lands(&outputs);
        let followups: Vec<String> = (17..=22)
            .flat_map(|red| ["R".repeat(red), "G".repeat(red)])
            .collect();
        let followups: Vec<&str> = followups.iter().map(String::as_str).collect();
        let (choice, searched) = choose(&model, "5", &[], Some(hand(&followups)));
        assert_eq!(searched, 0);
        assert!(choice.checks <= ceiling(3, 24, followups.len()));
        assert_eq!((choice.checks, choice.exhausted), (191, false));
        assert_eq!(choice.taps, [0, 1, 2]);
    }

    #[test]
    fn smart_tap_bounds_a_full_list_of_follow_ups_and_abilities() {
        // Sixty-four follow-ups, forty-four of them abilities and half of
        // those tapping a land of their own, on twenty lands of four outputs.
        let outputs: Vec<&str> = (0..20)
            .map(|index| ["W", "U", "R", "G"][index % 4])
            .collect();
        let model = lands(&outputs);
        let colors = ["W", "U", "R", "G"];
        let followups: Vec<ManaCost> = (0..64)
            .map(|index| {
                let color = colors[index % 4];
                ManaCost::parse(&format!("{}{}", index / 4, color.repeat(1 + index % 3)))
            })
            .collect();
        let probe = PaymentProbe {
            hand: (1 << 20) - 1,
            instant: 0b1010,
            tapping: (0..64)
                .map(|index| (index >= 20 && index % 2 == 0).then_some(index % 20))
                .collect(),
            followups,
            ..PaymentProbe::default()
        };
        let (choice, searched) = choose(&model, "8", &[], Some(probe));
        assert_eq!(searched, 0);
        assert_eq!(choice.followups, 64);
        assert!(choice.checks <= ceiling(8, 20, 64));
        assert_eq!((choice.checks, choice.exhausted), (1142, true));
        assert_eq!(choice.taps.len(), 8);
    }

    #[test]
    fn smart_tap_bounds_saved_obligations_to_one_search_per_witness_tap() {
        // Twelve Mountains ahead of twelve Forests: the witness for {G} taps
        // thirteen sources, and the saved {R}{R} is a search per pruned tap.
        let outputs: Vec<&str> = (0..24)
            .map(|index| if index < 12 { "R" } else { "G" })
            .collect();
        let model = lands(&outputs);
        let (choice, searched) = choose(&model, "G", &[pay("RR")], None);
        // Every pruned Mountain is untapped again for the saved cost.
        assert_eq!(choice.taps, [12]);
        assert_eq!((choice.checks, choice.followups), (0, 0));
        assert_eq!(choice.searches, 13);
        assert!(searched <= 13, "{searched}");
    }

    #[test]
    fn smart_tap_never_searches_a_board_outside_the_direct_class() {
        // Modal sources, a filter with a mana cost, a life cost, a counted
        // tap and waterbend sources put every `SourcePolicy::All` question on
        // this board in the search. The choice asks none.
        let mut model = lands(&["G", "R", "G", "R", "U", "U"]);
        let modal = || tapping_ability(vec![Mana::parse("G"), Mana::parse("R")]);
        for index in 6..12 {
            model.sources.push(source(index, None, vec![modal()]));
        }
        model.sources.push(source(
            12,
            None,
            vec![ManaActivation {
                cost: ManaCost::parse("1"),
                life_cost: 1,
                ..tapping_ability(vec![Mana::parse("UU")])
            }],
        ));
        model.sources.push(source(
            13,
            None,
            vec![ManaActivation {
                tap_source: false,
                tap_count: 2,
                tap_candidates: (6..12).map(PermanentId).collect(),
                ..tapping_ability(vec![Mana::parse("RRR")])
            }],
        ));
        for index in 14..18 {
            let mut creature = source(index, None, Vec::new());
            creature.waterbend = true;
            model.sources.push(creature);
        }
        assert!(model.direct(&[pay("1G")], SourcePolicy::All).is_none());
        let followups = ["RR", "UU", "3", "RRR"];
        let (choice, searched) = choose(&model, "1G", &[], Some(hand(&followups)));
        assert_eq!((choice.searches, searched), (0, 0));
        // Both Forests keep the Mountains for {R}{R}.
        assert_eq!(choice.taps, [0, 2]);
        assert!(choice.checks <= ceiling(2, 18, followups.len()));
        assert_eq!((choice.checks, choice.exhausted), (15, false));
        // With a saved obligation the same board prunes only.
        let (choice, searched) = choose(&model, "1G", &[pay("RR")], None);
        assert_eq!(choice.taps, [0, 1]);
        assert_eq!((choice.searches, searched), (0, 0));
    }
}
