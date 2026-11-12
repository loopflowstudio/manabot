// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel searches identity-preserving resource steps;
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
        mana::{Color, Mana, ManaCost, ManaUse},
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

#[derive(Clone)]
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

#[derive(Clone, Debug)]
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
        visited: &mut std::collections::HashSet<PaymentSearchState>,
    ) -> Option<Vec<PaymentStep>> {
        if state.obligations.is_empty() {
            return Some(Vec::new());
        }
        // Gross production is deliberately optimistic: nested upfront costs,
        // mutually exclusive uses and competing tap costs can only reduce it.
        // A full witness, never this bound, establishes affordability.
        let mut total = u64::from(state.pool.total());
        for (index, source) in self.sources.iter().enumerate() {
            if !state.resources.present[index] {
                continue;
            }
            if state.resources.untapped[index] {
                total += source.fixed.as_ref().map_or(0, |m| u64::from(m.total()));
                if source.waterbend {
                    total += 1;
                }
            }
            for (ability_index, ability) in source.abilities.iter().enumerate() {
                total += self.uses(&state.resources, index, ability_index) as u64
                    * ability
                        .outputs
                        .iter()
                        .map(|m| u64::from(m.total()))
                        .max()
                        .unwrap_or(0);
            }
        }
        for obligation in &state.obligations {
            if let Obligation::Output { choices, .. } = obligation {
                total += choices
                    .iter()
                    .map(|m| u64::from(m.total()))
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
        // Only states that need exploration need an owned visited key.
        if total < owed || !visited.insert(state.clone()) {
            return None;
        }
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
        self.witness(
            PaymentSearchState {
                resources,
                pool,
                life: self.life,
                obligations,
                prefix: None,
            },
            SourcePolicy::All,
            &mut Default::default(),
        )
    }

    fn plan(
        &self,
        pool: Mana,
        obligations: Vec<Obligation>,
        policy: SourcePolicy,
    ) -> Option<Vec<PaymentStep>> {
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
                &mut Default::default(),
            );
        }
        // Minimize taps before the first Spend without starving a saved caller.
        for budget in 0..=self.sources.len() {
            if let Some(path) = self.witness(
                PaymentSearchState {
                    resources: self.resources(),
                    pool: pool.clone(),
                    life: self.life,
                    obligations: obligations.clone(),
                    prefix: Some(budget),
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

    /// The activation's mana cost as paid now: its purpose names the source
    /// (for restricted mana, CR 106.6) and per-permanent reductions apply to
    /// its generic part (CR 601.2f, 602.2b).
    pub(crate) fn price_activation(
        &self,
        controller: PlayerId,
        purpose: ManaUse,
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
            .map(|(class, _)| *class)
            .collect::<Vec<_>>();
        classes.sort();
        classes.dedup();
        for class in classes {
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
                purpose: cost.purpose,
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
