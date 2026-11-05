// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel searches identity-preserving resource steps;
// Game owns their events and explicit Decision continuations. Pool arithmetic
// and spend-purpose restrictions belong to state/mana.rs.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{
        ability::{Effect, EffectValue},
        card::{ActivatedAbilityDefinition, ReplacementEffect},
        game_object::{PermanentId, PlayerId},
        mana::{Color, Mana, ManaCost, SpendingContext},
    },
};

/// Opt-in, thread-local payment diagnostics. Normal builds compile out every
/// counter write; measurements include all probes in the measured operation.
#[cfg(feature = "benchmark")]
pub mod measurement {
    use super::*;
    use std::cell::Cell;

    #[derive(Clone, Copy, Default, Debug, serde::Serialize)]
    pub struct Counters {
        pub explored: u64,
        pub deduplicated: u64,
        pub pruned: u64,
        pub dynamic_evaluations: u64,
        pub game_clones: u64,
        pub max_obligations: usize,
        pub payment_remainders: usize,
        pub max_remainders: usize,
    }

    thread_local! {
        static COUNTERS: Cell<Counters> = Cell::new(Counters::default());
    }

    pub(super) fn update(f: impl FnOnce(&mut Counters)) {
        COUNTERS.with(|cell| {
            let mut counters = cell.get();
            f(&mut counters);
            cell.set(counters);
        });
    }

    pub fn take() -> Counters {
        COUNTERS.with(|cell| cell.replace(Counters::default()))
    }

    /// Search the ordinary payment model, retaining its exact last remainder.
    pub fn payment(game: &Game, player: PlayerId, cost: &ManaCost) -> Option<Mana> {
        let path = game.payment_model(player, None).plan(
            game.pooled_mana(player),
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
            SourcePolicy::All,
        )?;
        match path.last() {
            Some(PaymentStep::Spend(remainder)) => Some(remainder.clone()),
            _ => unreachable!("payment witness ends in spending"),
        }
    }
}

/// One finite resource transition model. An activation is reserved before its
/// upfront obligation, so filters cannot finance themselves or form seedless
/// cycles. Taps, departures, selected-object costs and usage limits remain
/// distinct. Witnesses retain identities and concrete typed pool remainders.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Obligation {
    Pay {
        cost: ManaCost,
        waterbend: bool,
    },
    Tap {
        candidates: Vec<PermanentId>,
        remaining: u8,
    },
    Output {
        choices: Vec<Mana>,
        life_cost: u32,
        source: PermanentId,
        ability_index: usize,
        sacrifice: bool,
    },
}

struct ManaSource {
    permanent: PermanentId,
    fixed: Option<Mana>,
    abilities: Vec<ManaActivation>,
    waterbend: bool,
}

struct ManaActivation {
    tap_source: bool,
    sacrifice_source: bool,
    ability_index: usize,
    cost: ManaCost,
    outputs: Vec<Mana>,
    dynamic: Option<Effect>,
    life_cost: u32,
    limited: bool,
    tap_cost: Option<(Vec<PermanentId>, u8)>,
}

impl ManaActivation {
    /// Optimistic uses from non-mana resources. A mana-only filter is bounded
    /// by the available mana instead and cannot contribute net production.
    fn resource_uses(&self, life: u32) -> Option<u64> {
        if self.tap_source || self.sacrifice_source || self.limited {
            Some(1)
        } else if let Some((candidates, count)) = &self.tap_cost {
            Some(candidates.len() as u64 / u64::from(*count))
        } else if let Some(uses) = life.checked_div(self.life_cost) {
            Some(u64::from(uses))
        } else {
            debug_assert!(self
                .outputs
                .iter()
                .all(|output| { u16::from(self.cost.mana_value) > output.total() }));
            None
        }
    }

    /// Selected taps and mana are paid before costs commit and output exists.
    fn obligations(&self, permanent: PermanentId) -> Vec<Obligation> {
        let mut obligations = Vec::new();
        if let Some((candidates, remaining)) = &self.tap_cost {
            obligations.push(Obligation::Tap {
                candidates: candidates.clone(),
                remaining: *remaining,
            });
        }
        obligations.push(Obligation::Pay {
            cost: self.cost.clone(),
            waterbend: false,
        });
        obligations.push(Obligation::Output {
            choices: self.outputs.clone(),
            life_cost: self.life_cost,
            source: permanent,
            ability_index: self.ability_index,
            sacrifice: self.sacrifice_source,
        });
        obligations
    }
}

/// Reserve incompatible uses, not the whole object. A tapped mana creature
/// can still be sacrificed later, but an announced tap cannot be spent twice.
#[derive(Clone, Copy)]
struct SourceReservation {
    permanent: PermanentId,
    tap: bool,
    sacrifice: bool,
    ability: Option<usize>,
}

#[derive(Clone, Debug)]
enum PaymentStep {
    Fixed(PermanentId),
    Activate(PermanentId, usize),
    Spend(Mana),
    Output(usize),
    Waterbend(PermanentId),
    TapCost(PermanentId),
}

impl PaymentStep {
    fn action(&self, game: &Game, player: PlayerId) -> Option<crate::agent::action::Action> {
        use crate::agent::action::Action;
        match self {
            Self::Activate(permanent, ability_index) => Some(Action::ActivateAbility {
                player,
                permanent: *permanent,
                ability_index: *ability_index,
            }),
            Self::Output(mode) => Some(Action::ChooseMode {
                player,
                mode: *mode,
            }),
            Self::Waterbend(permanent) => Some(Action::WaterbendTap {
                player,
                permanent: *permanent,
            }),
            Self::TapCost(permanent) => Some(Action::SelectCard {
                player,
                card: game.state.permanents[*permanent].as_ref()?.card,
            }),
            Self::Fixed(_) | Self::Spend(_) => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PaymentSearchState {
    resources: Vec<SourceResources>,
    pool: Mana,
    life: u32,
    obligations: Vec<Obligation>,
    /// Fixed-source budget until the first Spend; later obligations are unrestricted.
    prefix: Option<usize>,
}

/// Costs reserve only what they consume. An activation cannot recursively
/// finance itself while its upfront obligation is pending, but the same
/// ability can be used again after output if it has no usage limit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct SourceResources {
    tapped: bool,
    departed: bool,
    sacrifice_reserved: bool,
    /// Fixed sources carry no activation history. Keep their search keys small;
    /// only a reserved activation allocates this source's unbounded index lists.
    activations: Option<Box<ActivationUsage>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct ActivationUsage {
    used: Vec<usize>,
    pending: Vec<usize>,
}

impl SourceResources {
    /// An upper bound includes reuse after a pending activation resolves.
    fn has_resources(&self, ability: &ManaActivation) -> bool {
        !self.departed
            && !(self.tapped && ability.tap_source)
            && !(self.sacrifice_reserved && ability.sacrifice_source)
            && !self
                .activations
                .as_ref()
                .is_some_and(|usage| usage.used.contains(&ability.ability_index))
    }

    fn admits(&self, ability: &ManaActivation) -> bool {
        self.has_resources(ability)
            && !self.activations.as_ref().is_some_and(|usage| {
                (!usage.pending.is_empty() && ability.sacrifice_source)
                    || usage.pending.contains(&ability.ability_index)
            })
    }

    fn reserve(&mut self, ability: &ManaActivation) {
        self.tapped |= ability.tap_source;
        self.sacrifice_reserved |= ability.sacrifice_source;
        let usage = self.activations.get_or_insert_with(Default::default);
        usage.pending.push(ability.ability_index);
        if ability.limited {
            usage.used.push(ability.ability_index);
        }
    }

    fn finish_activation(&mut self, ability_index: usize) {
        if let Some(usage) = &mut self.activations {
            usage.pending.retain(|index| *index != ability_index);
            // Canonical empty history lets equivalent resource paths deduplicate.
            if usage.pending.is_empty() && usage.used.is_empty() {
                self.activations = None;
            }
        }
    }
}

/// Which transitions may be automatic at the caller's current boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourcePolicy {
    All,
    FixedOnly,
    AutomaticPrefix,
}

impl SourcePolicy {
    fn allows_activations(self, prefix: Option<usize>) -> bool {
        match self {
            Self::All => true,
            Self::FixedOnly => false,
            Self::AutomaticPrefix => prefix.is_none(),
        }
    }
}

struct PaymentModel<'a> {
    game: &'a Game,
    player: PlayerId,
    sources: Vec<ManaSource>,
    life: u32,
    reserved: Vec<SourceResources>,
}

impl PaymentModel<'_> {
    /// Each probe owns its changing resources and borrows this decision's inventory.
    fn initial_state(&self, pool: Mana, obligations: Vec<Obligation>) -> PaymentSearchState {
        PaymentSearchState {
            resources: self.reserved.clone(),
            pool,
            life: self.life,
            obligations,
            prefix: None,
        }
    }

    fn activation_witness(
        &self,
        permanent: PermanentId,
        ability_index: usize,
    ) -> Option<Vec<PaymentStep>> {
        let index = self
            .sources
            .iter()
            .position(|source| source.permanent == permanent)?;
        let ability = self.sources[index]
            .abilities
            .iter()
            .find(|ability| ability.ability_index == ability_index)?;
        if !self.reserved[index].admits(ability) {
            return None;
        }
        let mut obligations = ability.obligations(permanent);
        obligations.extend(self.game.payment_obligations(self.player));
        let mut state = self.initial_state(self.game.pooled_mana(self.player), obligations);
        state.resources[index].reserve(ability);
        let mut path = self.witness(state, SourcePolicy::All, &mut Default::default())?;
        path.insert(0, PaymentStep::Activate(permanent, ability_index));
        Some(path)
    }

    fn tap_completes_payment(&self, permanent: PermanentId) -> bool {
        let Some(index) = self
            .sources
            .iter()
            .position(|source| source.permanent == permanent)
        else {
            return false;
        };
        if self.reserved[index].tapped || self.reserved[index].departed {
            return false;
        }
        let mut obligations = self.game.payment_obligations(self.player);
        let Some(Obligation::Tap {
            remaining,
            candidates,
        }) = obligations.first_mut()
        else {
            return false;
        };
        if *remaining == 0 || !candidates.contains(&permanent) {
            return false;
        }
        *remaining -= 1;
        let mut state = self.initial_state(self.game.pooled_mana(self.player), obligations);
        state.resources[index].tapped = true;
        self.witness(state, SourcePolicy::All, &mut Default::default())
            .is_some()
    }

    fn reserve(&mut self, reservation: SourceReservation) {
        if let Some(index) = self
            .sources
            .iter()
            .position(|source| source.permanent == reservation.permanent)
        {
            let resources = &mut self.reserved[index];
            resources.tapped |= reservation.tap;
            resources.sacrifice_reserved |= reservation.sacrifice;
            if let Some(ability) = reservation.ability {
                resources
                    .activations
                    .get_or_insert_with(Default::default)
                    .pending
                    .push(ability);
            }
        }
    }

    /// Evaluate variable output after upstream costs, against the same layered
    /// authority as execution. Ordinary sources never clone the game here.
    fn transition_outputs(
        &self,
        permanent: PermanentId,
        ability_index: usize,
        resources: &[SourceResources],
        pool: &Mana,
        life: u32,
    ) -> Option<Vec<Mana>> {
        let source = self
            .sources
            .iter()
            .find(|source| source.permanent == permanent)?;
        let ability = source
            .abilities
            .iter()
            .find(|ability| ability.ability_index == ability_index)?;
        let effect = ability.dynamic.as_ref()?;
        #[cfg(feature = "benchmark")]
        measurement::update(|c| {
            c.dynamic_evaluations += 1;
            c.game_clones += 1;
        });
        let mut probe = self.game.clone();
        for (source, remaining) in self.sources.iter().zip(resources) {
            if remaining.departed {
                probe.sacrifice_permanent(source.permanent);
            } else if remaining.tapped {
                // Taps don't invalidate characteristics. Keep the clone's
                // existing layer cache, with debug recomputation as its oracle.
                probe.state.tap(source.permanent);
            }
        }
        // `self.life` excludes life reserved for an outer activation, whose
        // cost has not committed yet. Only the searched prefix's expenditure
        // changes characteristics while this output resolves.
        if life != self.life {
            probe.state.players[self.player.0].life -= (self.life - life) as i32;
        }
        let (regular, combat) = probe.state.mana_pools_mut(self.player);
        *regular = pool.clone();
        combat.clear();
        let mut choices = probe.current_mana_choices(effect, self.player)?;
        if ability.tap_source {
            let bonus = probe.tap_mana_bonus(permanent, self.player);
            for output in &mut choices {
                output.add(&bonus);
            }
        }
        Some(choices)
    }

    /// Reject only impossible payments; exact transitions establish success.
    fn can_cover(&self, state: &PaymentSearchState, policy: SourcePolicy) -> bool {
        // A variable producer can gain output when another resource leaves
        // (for example, a negative static modifier). Its current amount is not
        // an admissible upper bound; exact transitions below decide feasibility.
        if self.sources.iter().zip(&state.resources).any(|(source, resources)| {
            source.abilities.iter().any(|ability| {
                ability.dynamic.is_some()
                    && (resources.has_resources(ability)
                        || state.obligations.iter().any(|obligation| matches!(
                            obligation,
                            Obligation::Output { source: pending, ability_index, .. }
                                if *pending == source.permanent && *ability_index == ability.ability_index
                        )))
            })
        }) {
            return true;
        }
        // An admissible resource bound. Independent abilities may share costs;
        // summing their optimistic gains overestimates supply, which is safe.
        // Strictly consuming filters contribute no gain, even when repeatable.
        let waterbend = state.obligations.iter().any(|obligation| {
            matches!(
                obligation,
                Obligation::Pay {
                    waterbend: true,
                    ..
                }
            )
        });
        let mut total = u64::from(state.pool.total());
        for (source, resources) in self.sources.iter().zip(&state.resources) {
            if resources.departed {
                continue;
            }
            if !resources.tapped {
                total += source
                    .fixed
                    .as_ref()
                    .map_or(0, |mana| u64::from(mana.total()));
                if waterbend && source.waterbend {
                    total += 1;
                }
            }
            for ability in &source.abilities {
                if !resources.has_resources(ability) {
                    continue;
                }
                let output = ability.outputs.iter().map(Mana::total).max().unwrap_or(0);
                total += ability.resource_uses(state.life).unwrap_or(0)
                    * u64::from(output.saturating_sub(u16::from(ability.cost.mana_value)));
            }
        }
        let mut owed = 0_u64;
        for obligation in &state.obligations {
            match obligation {
                Obligation::Tap { .. } => {}
                Obligation::Pay { cost, .. } => owed += u64::from(cost.mana_value),
                Obligation::Output { choices, .. } => {
                    total += u64::from(choices.iter().map(Mana::total).max().unwrap_or(0))
                }
            }
        }
        if total < owed {
            return false;
        }
        if let Some(Obligation::Pay { cost, waterbend }) = state.obligations.first() {
            let mut upper = state.pool.clone();
            for (source, resources) in self.sources.iter().zip(&state.resources) {
                if resources.departed {
                    continue;
                }
                if !resources.tapped {
                    if let Some(fixed) = &source.fixed {
                        upper.add(fixed);
                    }
                    if *waterbend && source.waterbend {
                        upper.add(&Mana::single(Color::Colorless));
                    }
                }
                if !policy.allows_activations(state.prefix) {
                    continue;
                }
                for ability in &source.abilities {
                    if !resources.has_resources(ability) {
                        continue;
                    }
                    // Every mana-only repeat consumes at least one mana.
                    let uses = ability.resource_uses(state.life).unwrap_or(total);
                    let mut best = Mana::default();
                    for output in &ability.outputs {
                        best.max_with(output);
                    }
                    for amount in &mut best.mana {
                        *amount = (u64::from(*amount) * uses).min(u64::from(u8::MAX)) as u8;
                    }
                    if let Some(classes) = &mut best.restricted {
                        for (_, amounts) in classes.iter_mut() {
                            for amount in amounts {
                                *amount = (u64::from(*amount) * uses).min(u64::from(u8::MAX)) as u8;
                            }
                        }
                    }
                    upper.add(&best);
                }
            }
            if !upper.can_pay(cost) {
                return false;
            }
        }

        true
    }

    fn witness(
        &self,
        mut state: PaymentSearchState,
        policy: SourcePolicy,
        visited: &mut std::collections::HashSet<PaymentSearchState>,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(feature = "benchmark")]
        measurement::update(|c| {
            c.explored += 1;
            c.max_obligations = c.max_obligations.max(state.obligations.len());
        });
        if state.obligations.is_empty() {
            return Some(Vec::new());
        }
        // A payable final obligation needs no more resource search. Keep the
        // ordinary class-aware spending order without cloning/hashing the
        // inventory or constructing an optimistic bound for a known success.
        if let [Obligation::Pay { cost, .. }] = state.obligations.as_slice() {
            if state.pool.can_pay(cost) {
                state.pool.pay(cost);
                #[cfg(feature = "benchmark")]
                measurement::update(|c| {
                    c.payment_remainders += 1;
                    c.max_remainders = c.max_remainders.max(1);
                });
                return Some(vec![PaymentStep::Spend(state.pool)]);
            }
        }
        if !self.can_cover(&state, policy) {
            #[cfg(feature = "benchmark")]
            measurement::update(|c| c.pruned += 1);
            return None;
        }
        // Only states that need exploration need an owned visited key. Terminal
        // success and the resource bound above require neither cloning nor hashing.
        if !visited.insert(state.clone()) {
            #[cfg(feature = "benchmark")]
            measurement::update(|c| c.deduplicated += 1);
            return None;
        }
        let PaymentSearchState {
            resources,
            pool,
            life,
            obligations,
            prefix,
        } = state;
        let first = &obligations[0];
        let rest = &obligations[1..];
        match first {
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                if *remaining == 0 {
                    return self.witness(
                        PaymentSearchState {
                            resources,
                            pool,
                            life,
                            obligations: rest.to_vec(),
                            prefix,
                        },
                        policy,
                        visited,
                    );
                }
                let available = candidates
                    .iter()
                    .filter(|permanent| {
                        self.sources
                            .iter()
                            .zip(&resources)
                            .any(|(source, resource)| {
                                source.permanent == **permanent
                                    && !resource.tapped
                                    && !resource.departed
                            })
                    })
                    .take(usize::from(*remaining))
                    .count();
                if available < usize::from(*remaining) {
                    #[cfg(feature = "benchmark")]
                    measurement::update(|c| c.pruned += 1);
                    return None;
                }
                for (candidate_index, permanent) in candidates.iter().enumerate() {
                    let Some(index) = self
                        .sources
                        .iter()
                        .position(|source| source.permanent == *permanent)
                    else {
                        continue;
                    };
                    if resources[index].tapped || resources[index].departed {
                        continue;
                    }
                    let mut next = resources.clone();
                    next[index].tapped = true;
                    let mut obligations = vec![Obligation::Tap {
                        // This obligation consumes a set. Explore each set once;
                        // the real chooser still receives every legal first tap.
                        candidates: candidates[candidate_index + 1..].to_vec(),
                        remaining: remaining - 1,
                    }];
                    obligations.extend_from_slice(rest);
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            resources: next,
                            pool: pool.clone(),
                            life,
                            obligations,
                            prefix,
                        },
                        policy,
                        visited,
                    ) {
                        path.insert(0, PaymentStep::TapCost(*permanent));
                        return Some(path);
                    }
                }
            }
            Obligation::Output {
                choices,
                life_cost,
                source,
                ability_index,
                sacrifice,
            } => {
                let mut after_cost = resources;
                if let Some(index) = self
                    .sources
                    .iter()
                    .position(|candidate| candidate.permanent == *source)
                {
                    after_cost[index].departed |= sacrifice;
                    after_cost[index].finish_activation(*ability_index);
                }
                if *life_cost > life {
                    return None;
                }
                let dynamic = self.transition_outputs(
                    *source,
                    *ability_index,
                    &after_cost,
                    &pool,
                    life - life_cost,
                );
                let choices = dynamic.as_deref().unwrap_or(choices);
                for (mode, output) in choices.iter().enumerate() {
                    let mut next = pool.clone();
                    next.add(output);
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            resources: after_cost.clone(),
                            pool: next,
                            life: life - life_cost,
                            obligations: rest.to_vec(),
                            prefix,
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
                // A payable terminal cost returned above. Nested payments
                // retain every allocation that could preserve their parent.
                let remainders = if rest.is_empty() {
                    Vec::new()
                } else {
                    pool.payment_options(cost)
                };
                #[cfg(feature = "benchmark")]
                measurement::update(|c| {
                    c.payment_remainders += remainders.len();
                    c.max_remainders = c.max_remainders.max(remainders.len());
                });
                for remainder in remainders {
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            resources: resources.clone(),
                            pool: remainder.clone(),
                            life,
                            obligations: rest.to_vec(),
                            prefix: None,
                        },
                        policy,
                        visited,
                    ) {
                        path.insert(0, PaymentStep::Spend(remainder));
                        return Some(path);
                    }
                }
                if prefix == Some(0) {
                    return None;
                }
                let after_source = prefix.map(|budget| budget - 1);
                for (index, source) in self.sources.iter().enumerate() {
                    if resources[index].departed {
                        continue;
                    }
                    if let Some(output) = source.fixed.as_ref().filter(|_| !resources[index].tapped)
                    {
                        let mut tapped = resources.clone();
                        tapped[index].tapped = true;
                        let mut next = pool.clone();
                        next.add(output);
                        if let Some(mut path) = self.witness(
                            PaymentSearchState {
                                resources: tapped,
                                pool: next,
                                life,
                                obligations: obligations.clone(),
                                prefix: after_source,
                            },
                            policy,
                            visited,
                        ) {
                            path.insert(0, PaymentStep::Fixed(source.permanent));
                            return Some(path);
                        }
                    }
                    // Chosen activations cannot nest before the automatic
                    // prefix's first Spend, so its budget needs no depth.
                    if !policy.allows_activations(prefix) {
                        continue;
                    }
                    for ability in &source.abilities {
                        if !resources[index].admits(ability) {
                            continue;
                        }
                        let mut remaining = resources.clone();
                        remaining[index].reserve(ability);
                        let mut nested = ability.obligations(source.permanent);
                        nested.extend_from_slice(&obligations);
                        if let Some(mut path) = self.witness(
                            PaymentSearchState {
                                resources: remaining,
                                pool: pool.clone(),
                                life,
                                obligations: nested,
                                prefix: None,
                            },
                            policy,
                            visited,
                        ) {
                            path.insert(
                                0,
                                PaymentStep::Activate(source.permanent, ability.ability_index),
                            );
                            return Some(path);
                        }
                    }
                    if *waterbend
                        && source.waterbend
                        && !resources[index].tapped
                        && cost.generic() > 0
                    {
                        let mut tapped = resources.clone();
                        tapped[index].tapped = true;
                        let mut next = vec![Obligation::Pay {
                            cost: cost.reduced_generic(1),
                            waterbend: true,
                        }];
                        next.extend_from_slice(rest);
                        if let Some(mut path) = self.witness(
                            PaymentSearchState {
                                resources: tapped,
                                pool: pool.clone(),
                                life,
                                obligations: next,
                                prefix: after_source,
                            },
                            policy,
                            visited,
                        ) {
                            path.insert(0, PaymentStep::Waterbend(source.permanent));
                            return Some(path);
                        }
                    }
                }
            }
        }
        None
    }

    fn plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
        // Offers need one executable witness, not a proof of the fewest taps.
        // Ordinary fixed production retains source-order automatic payment.
        // Only a nested automatic prefix minimizes early commitments, where
        // retaining a parent's colored route requires considering alternatives.
        if policy != SourcePolicy::AutomaticPrefix {
            return self.witness(
                self.initial_state(pool, obligations),
                policy,
                &mut Default::default(),
            );
        }
        // Minimize resources committed before the current obligation is paid.
        // Later costs remain part of the same witness, but cannot force an
        // unnecessary early tap just because a source was enumerated first.
        for budget in 0..=self.sources.len() {
            if let Some(path) = self.witness(
                PaymentSearchState {
                    prefix: Some(budget),
                    ..self.initial_state(pool.clone(), obligations.clone())
                },
                policy,
                &mut Default::default(),
            ) {
                return Some(path);
            }
        }
        None
    }
}

impl Game {
    /// Read a variable output from current state (CR 605.3a). Callers evaluate
    /// after costs; characteristics retain the printed effect, never a preview.
    pub(crate) fn current_mana_choices(
        &self,
        effect: &Effect,
        controller: PlayerId,
    ) -> Option<Vec<Mana>> {
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
                match &branch[0] {
                    Effect::AddMana {
                        mana,
                        amount: EffectValue::Fixed(1),
                        ..
                    } => Some(vec![mana.clone()]),
                    _ => None,
                }
            }
            Effect::AddOneColorManaValue { amount } => {
                let amount = u8::try_from(self.controller_value(amount, controller).max(0))
                    .expect("mana output exceeds pool capacity");
                if amount == 0 {
                    return Some(vec![Mana::default()]);
                }
                Some(
                    (0..5)
                        .map(|color| {
                            let mut mana = Mana::default();
                            mana.mana[color] = amount;
                            mana
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    }

    pub(crate) fn mana_outputs(
        &self,
        ability: &ActivatedAbilityDefinition,
        player: PlayerId,
    ) -> Option<Vec<Mana>> {
        if let Some(choices) = ability.mana_choices() {
            return Some(choices);
        }
        let [effect] = ability.effects.as_slice() else {
            return None;
        };
        self.current_mana_choices(effect, player)
    }

    /// The activation's mana cost as paid now: its purpose names the source
    /// (for restricted mana, CR 106.6) and per-permanent reductions apply to
    /// its generic part (CR 601.2f, 602.2b).
    pub(crate) fn price_activation(
        &self,
        controller: PlayerId,
        purpose: SpendingContext,
        ability: &mut ActivatedAbilityDefinition,
    ) {
        ability.mana_cost.purpose = purpose;
        if let Some(predicate) = &ability.cost_reduction_per {
            let count = self.count_battlefield_matching(controller, predicate);
            ability.mana_cost = ability
                .mana_cost
                .reduced_generic(u8::try_from(count).unwrap_or(u8::MAX));
        }
    }

    fn payment_model(&self, player: PlayerId, sacrifice: Option<PermanentId>) -> PaymentModel<'_> {
        let mut reserved = Vec::new();
        if let Some(source) = sacrifice {
            reserved.push(SourceReservation {
                permanent: source,
                tap: false,
                sacrifice: true,
                ability: None,
            });
        }
        if let Some(crate::flow::game::PendingChoice::ManaPayment {
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
                    ability: None,
                });
            }
        }
        if let Some(crate::flow::game::PendingChoice::SacrificePayment { payment, .. }) =
            &self.pending_choice
        {
            if let Some(reservation) = self.payment_reservation(payment) {
                reserved.push(reservation);
            }
        }
        fn reserve(
            decision: &crate::flow::decision::Decision,
            game: &Game,
            reserved: &mut Vec<SourceReservation>,
        ) {
            use crate::flow::decision::Decision;
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
                        sacrifice: ability.sacrifice_source,
                        ability: Some(*ability_index),
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
        let sources: Vec<ManaSource> = self
            .iter_battlefield_permanents(player)
            .map(|permanent| {
                let source = self.state.permanents[permanent].as_ref().unwrap();
                let card = &self.state.cards[source.card];
                let characteristics = self.characteristics(permanent);
                // A source that lost its abilities offers no zero-mana tap.
                let can_tap = self.permanent_can_tap(permanent);
                let fixed =
                    (can_tap && !card.mana_abilities.is_empty() && !characteristics.abilities_lost)
                        .then(|| self.fixed_source_mana(permanent));
                let abilities = characteristics
                    .activated_abilities
                    .iter()
                    .enumerate()
                    .filter(|(index, ability)| {
                        ability.zone == crate::state::card::ActivationZone::Battlefield
                            && (!ability.tap_source || can_tap)
                            && self.check_activation_terms(player, ability).is_ok()
                            && !source.activation_used(*index, ability)
                    })
                    .filter_map(|(index, ability)| {
                        // Classify once: fixed outputs need no retained effect;
                        // variable outputs are evaluated again after costs.
                        let (mut outputs, dynamic) = match ability.mana_choices() {
                            Some(outputs) => (outputs, None),
                            None => {
                                let [effect] = ability.effects.as_slice() else {
                                    return None;
                                };
                                (
                                    self.current_mana_choices(effect, player)?,
                                    Some(effect.clone()),
                                )
                            }
                        };
                        if ability.tap_source {
                            let bonus = self.tap_mana_bonus(permanent, player);
                            for output in &mut outputs {
                                output.add(&bonus);
                            }
                        }
                        Some(ManaActivation {
                            tap_source: ability.tap_source,
                            sacrifice_source: ability.sacrifice_source,
                            ability_index: index,
                            cost: ability.mana_cost.clone(),
                            outputs,
                            dynamic,
                            life_cost: ability.life_cost,
                            limited: ability.once_each_turn || ability.exhaust,
                            tap_cost: ability.tap_cost.as_ref().map(|cost| {
                                (self.tap_cost_candidates(player, ability), cost.count)
                            }),
                        })
                    })
                    .collect();
                ManaSource {
                    permanent,
                    fixed,
                    abilities,
                    waterbend: !source.tapped
                        && (characteristics.types.is_artifact()
                            || characteristics.types.is_creature()),
                }
            })
            .collect();
        let mut life = self.state.players[player.0].life.max(0) as u32;
        if let Some(
            crate::flow::game::PendingChoice::ManaPayment { payment, .. }
            | crate::flow::game::PendingChoice::SacrificePayment { payment, .. },
        ) = &self.pending_choice
        {
            if let crate::flow::game::Payment::Activation {
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
        let resources = vec![SourceResources::default(); sources.len()];
        let mut model = PaymentModel {
            game: self,
            player,
            sources,
            life,
            reserved: resources,
        };
        for reservation in reserved {
            model.reserve(reservation);
        }
        model
    }

    fn payment_reservation(
        &self,
        payment: &crate::flow::game::Payment,
    ) -> Option<SourceReservation> {
        match payment {
            crate::flow::game::Payment::Activation {
                source: crate::flow::game::ActivationSource::Permanent(permanent),
                ability_index,
                ..
            } => {
                let ability = &self.characteristics(*permanent).activated_abilities[*ability_index];
                Some(SourceReservation {
                    permanent: *permanent,
                    tap: ability.tap_source,
                    sacrifice: ability.sacrifice_source,
                    ability: Some(*ability_index),
                })
            }
            _ => None,
        }
    }

    fn payment_obligations(&self, player: PlayerId) -> Vec<Obligation> {
        use crate::flow::{decision::Decision, game::PendingChoice};
        fn append(decision: &Decision, game: &Game, result: &mut Vec<Obligation>) {
            match decision {
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    taps_remaining,
                    resume,
                    ..
                } => {
                    let view = game.characteristics(*permanent);
                    let ability = &view.activated_abilities[*ability_index];
                    let mut outputs = game.mana_outputs(ability, decision.player()).unwrap();
                    if ability.tap_source {
                        let bonus = game.tap_mana_bonus(*permanent, decision.player());
                        for output in &mut outputs {
                            output.add(&bonus);
                        }
                    }
                    if *taps_remaining > 0 {
                        result.push(Obligation::Tap {
                            candidates: game.tap_cost_candidates(decision.player(), ability),
                            remaining: *taps_remaining,
                        });
                    }
                    result.push(Obligation::Pay {
                        cost: ability.mana_cost.clone(),
                        waterbend: false,
                    });
                    result.push(Obligation::Output {
                        choices: outputs,
                        life_cost: ability.life_cost,
                        source: *permanent,
                        ability_index: *ability_index,
                        sacrifice: ability.sacrifice_source,
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

    pub(crate) fn can_activate_mana_ability(
        &self,
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
    ) -> bool {
        self.payment_model(player, None)
            .activation_witness(permanent, ability_index)
            .is_some()
    }

    pub(crate) fn mana_ability_actions(
        &self,
        player: PlayerId,
        sacrifice: Option<PermanentId>,
    ) -> Vec<crate::agent::action::Action> {
        let inventory = self.payment_model(player, sacrifice);
        inventory
            .sources
            .iter()
            .flat_map(|source| {
                source.abilities.iter().filter_map(|ability| {
                    let ability_index = ability.ability_index;
                    inventory
                        .activation_witness(source.permanent, ability_index)
                        .map(|_| crate::agent::action::Action::ActivateAbility {
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

    pub(crate) fn can_pay_announced_activation(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        payment: &crate::flow::game::Payment,
        sacrifice: Option<PermanentId>,
    ) -> bool {
        let mut model = self.payment_model(player, sacrifice);
        if let Some(reservation) = self.payment_reservation(payment) {
            model.reserve(reservation);
        }
        if let crate::flow::game::Payment::Activation {
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
                ability: None,
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
        let old = self.state.suspended_decision_mut().take();
        self.suspend_rule_decision(crate::flow::decision::Decision::ManaUpfront {
            taps_remaining: ability.tap_cost.as_ref().map_or(0, |cost| cost.count),
            player,
            permanent,
            ability_index,
            resume: old.as_ref().map(|s| Box::new(s.decision.clone())),
        });
        if let Some(old) = old {
            self.state.suspended_decision_mut().as_mut().unwrap().frame = old.frame;
        }
        self.advance_mana_upfront()
    }

    pub(crate) fn mana_upfront_actions(
        &self,
        player: PlayerId,
    ) -> Vec<crate::agent::action::Action> {
        use crate::{agent::action::Action, flow::decision::Decision};
        let Some(suspended) = &self.state.suspended_decision else {
            return Vec::new();
        };
        let Decision::ManaUpfront {
            permanent,
            ability_index,
            taps_remaining,
            ..
        } = &suspended.decision
        else {
            return Vec::new();
        };
        if *taps_remaining == 0 {
            return self.mana_ability_actions(player, None);
        }
        let ability = &self.characteristics(*permanent).activated_abilities[*ability_index];
        let inventory = self.payment_model(player, None);
        self.tap_cost_candidates(player, ability)
            .into_iter()
            .filter(|candidate| inventory.tap_completes_payment(*candidate))
            .map(|permanent| Action::SelectCard {
                player,
                card: self.state.permanents[permanent].as_ref().unwrap().card,
            })
            .collect()
    }

    pub(crate) fn pay_mana_tap(
        &mut self,
        player: PlayerId,
        card: crate::state::game_object::CardId,
    ) -> Result<(), AgentError> {
        let Some(permanent) = self.state.card_to_permanent[card] else {
            return Err(AgentError("tap cost source is absent".into()));
        };
        if !self
            .payment_model(player, None)
            .tap_completes_payment(permanent)
        {
            return Err(AgentError("tap cannot complete the payment".into()));
        }
        self.tap_permanent(permanent, false);
        let suspended = self.state.suspended_decision_mut().as_mut().unwrap();
        let crate::flow::decision::Decision::ManaUpfront { taps_remaining, .. } =
            &mut suspended.decision
        else {
            unreachable!()
        };
        *taps_remaining -= 1;
        self.advance_mana_upfront()
    }

    /// Automatic fixed sources and spending commit the exact witness prefix.
    /// Chosen-output sources suspend instead; no witness selects a player's color.
    pub(crate) fn advance_mana_upfront(&mut self) -> Result<(), AgentError> {
        use crate::flow::decision::Decision;
        let Some(suspended) = &self.state.suspended_decision else {
            return Ok(());
        };
        let Decision::ManaUpfront {
            taps_remaining: 0,
            player,
            permanent,
            ability_index,
            ..
        } = suspended.decision
        else {
            return Ok(());
        };
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
            .any(|step| step.action(self, player).is_some())
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
        if ability.sacrifice_source {
            self.sacrifice_permanent(permanent);
        }
        let choices = self.mana_outputs(&ability, player).unwrap();
        let suspended = self.state.suspended_decision_mut().as_mut().unwrap();
        let Decision::ManaUpfront { resume, .. } = &mut suspended.decision else {
            unreachable!()
        };
        suspended.decision = Decision::ManaChoice {
            player,
            choices,
            resume: resume.take(),
        };
        Ok(())
    }

    fn tap_fixed_mana_source(&mut self, player: PlayerId, permanent: PermanentId) {
        let source = self.state.permanents[permanent].as_ref().unwrap();
        let produced = self.state.cards[source.card]
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
    fn fixed_source_mana(&self, permanent: PermanentId) -> Mana {
        let source = self.state.permanents[permanent].as_ref().unwrap();
        let card = &self.state.cards[source.card];
        let mut output = Mana::default();
        for ability in &card.mana_abilities {
            output.add(&ability.mana);
        }
        output.add(&self.tap_mana_bonus(permanent, source.controller));
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
        for class in classes {
            let mut regular = mana_pool.restricted_of(&class);
            let mut combat = combat_mana_pool.restricted_of(&class);
            let left = remaining.restricted_of(&class);
            for slot in 0..left.len() {
                let spent = regular[slot] + combat[slot] - left[slot];
                let from_combat = spent.min(combat[slot]);
                combat[slot] -= from_combat;
                regular[slot] -= spent - from_combat;
            }
            mana_pool.set_restricted(class.clone(), regular);
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
        for permanent_id in self.iter_battlefield_permanents(player) {
            let source = self.state.permanents[permanent_id].as_ref().unwrap();
            if self.state.cards[source.card].mana_abilities.is_empty()
                || !self.permanent_can_tap(permanent_id)
                || self.characteristics(permanent_id).abilities_lost
            {
                continue;
            }
            total.add(&self.fixed_source_mana(permanent_id));
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
    use crate::cards::content_pack::ContentPack;

    #[test]
    fn finite_resources_allow_repeats_and_keep_tap_separate_from_departure() {
        use super::{
            ManaActivation, ManaSource, Obligation, PaymentModel, PaymentStep, SourcePolicy,
            SourceResources,
        };
        use crate::{
            flow::game::Game,
            state::{
                game_object::{PermanentId, PlayerId},
                mana::{Mana, ManaCost},
                player::PlayerConfig,
            },
        };
        let game = Game::new(
            vec![
                PlayerConfig::new("a", [("Mountain".into(), 40)].into()),
                PlayerConfig::new("b", [("Mountain".into(), 40)].into()),
            ],
            1,
            false,
        );
        let ability = ManaActivation {
            tap_source: false,
            sacrifice_source: false,
            ability_index: 0,
            cost: ManaCost::default(),
            outputs: vec![Mana::parse("R")],
            dynamic: None,
            life_cost: 1,
            limited: false,
            tap_cost: None,
        };
        let mut model = PaymentModel {
            game: &game,
            player: PlayerId(0),
            life: 3,
            sources: vec![ManaSource {
                permanent: PermanentId(0),
                fixed: None,
                abilities: vec![ability],
                waterbend: false,
            }],
            reserved: vec![SourceResources::default()],
        };
        let cost = || {
            vec![Obligation::Pay {
                cost: ManaCost::parse("RRR"),
                waterbend: false,
            }]
        };
        let path = model
            .plan(Mana::default(), cost(), SourcePolicy::All)
            .unwrap();
        assert_eq!(
            path.iter()
                .filter(|step| matches!(step, PaymentStep::Activate(_, _)))
                .count(),
            3
        );
        model.life = 2;
        assert!(model
            .plan(Mana::default(), cost(), SourcePolicy::All)
            .is_none());
        model.sources[0].fixed = Some(Mana::parse("RR"));
        model.sources[0].abilities[0].life_cost = 0;
        model.sources[0].abilities[0].sacrifice_source = true;
        let path = model
            .plan(Mana::default(), cost(), SourcePolicy::All)
            .unwrap();
        assert!(matches!(path[0], PaymentStep::Fixed(PermanentId(0))));
        assert_eq!(
            path.iter()
                .filter(|step| matches!(step, PaymentStep::Activate(_, _)))
                .count(),
            1
        );
        assert!(matches!(path.last(), Some(PaymentStep::Spend(pool)) if pool.total() == 0));
    }

    #[test]
    fn registered_mana_activations_have_finite_source_costs() {
        for definition in ContentPack::default().definitions() {
            for ability in &definition.activated_abilities {
                if ability.is_mana_ability() {
                    // Each repeat consumes a resource represented by the witness.
                    assert!(
                        ability.tap_source
                            || ability.sacrifice_source
                            || ability.once_each_turn
                            || ability.exhaust
                            || ability.life_cost > 0
                            || ability.tap_cost.as_ref().is_some_and(|cost| cost.count > 0)
                            || ability
                                .mana_choices()
                                .is_some_and(|outputs| outputs.iter().all(|output| output.total()
                                    < u16::from(ability.mana_cost.mana_value))),
                        "{} needs a bounded resource transition",
                        definition.name
                    );
                    assert!(
                        ability.zone == crate::state::card::ActivationZone::Battlefield
                            && ability.sacrifice.is_none()
                            && !ability.waterbend
                            && !ability.discard_source
                            && !ability.discard_card
                            && ability.crew.is_none(),
                        "{} needs its additional upfront cost represented",
                        definition.name
                    );
                }
            }
        }
    }
}
