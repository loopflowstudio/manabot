// mana.rs
// One payment transition model for offers, automatic production and nested
// mana activations. PaymentModel searches identity-preserving resource steps;
// Game owns their events and explicit Decision continuations. Pool arithmetic
// and spend-purpose restrictions belong to state/mana.rs.

use crate::{
    agent::action::AgentError,
    flow::game::Game,
    state::{
        ability::Effect,
        card::{ActivatedAbilityDefinition, ReplacementEffect},
        game_object::{PermanentId, PlayerId},
        mana::{Mana, ManaCost},
    },
};

/// Basic land types and the mana their intrinsic ability adds (CR 305.6).
const BASIC_LAND_TYPES: [(&str, &str); 5] = [
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
];

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
        source: PermanentId,
    },
}

#[derive(Clone)]
struct ManaSource {
    permanent: PermanentId,
    fixed: Option<Mana>,
    abilities: Vec<ManaActivation>,
    waterbend: bool,
}

#[derive(Clone)]
struct ManaActivation {
    tap_source: bool,
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
}

impl PaymentStep {
    fn action(&self, player: PlayerId) -> Option<crate::agent::action::Action> {
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
            Self::Fixed(_) | Self::Spend(_) => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PaymentSearchState {
    available: Vec<bool>,
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
            }
            source.abilities.retain(|ability| {
                !(reservation.tap && ability.tap_source
                    || reservation.sacrifice && ability.sacrifice_source)
            });
        }
    }

    fn witness(
        &self,
        state: PaymentSearchState,
        policy: SourcePolicy,
        visited: &mut std::collections::HashSet<PaymentSearchState>,
    ) -> Option<Vec<PaymentStep>> {
        if !visited.insert(state.clone()) {
            return None;
        }
        let PaymentSearchState {
            available,
            pool,
            life,
            obligations,
            prefix,
        } = state;
        let Some(first) = obligations.first() else {
            return Some(Vec::new());
        };
        let rest = &obligations[1..];
        // An admissible total-resource bound, not a payment answer: filters
        // cannot increase total mana by recycling their own upfront cost.
        // Ignore colors, restrictions, sequencing and life here; relaxing each
        // constraint can only overestimate what the remaining identities fund.
        let waterbend = obligations.iter().any(|obligation| {
            matches!(
                obligation,
                Obligation::Pay {
                    waterbend: true,
                    ..
                }
            )
        });
        let mut total = u32::from(pool.total());
        for (index, source) in self.sources.iter().enumerate() {
            if !available[index] {
                continue;
            }
            let mut net = source
                .fixed
                .as_ref()
                .map_or(0, |mana| u32::from(mana.total()));
            for ability in &source.abilities {
                let output = ability
                    .outputs
                    .iter()
                    .map(|mana| u32::from(mana.total()))
                    .max()
                    .unwrap_or(0);
                net = net.max(output.saturating_sub(u32::from(ability.cost.mana_value)));
            }
            if waterbend && source.waterbend {
                net = net.max(1);
            }
            total += net;
        }
        for obligation in &obligations {
            match obligation {
                Obligation::Pay { cost, .. } => {
                    total = total.checked_sub(u32::from(cost.mana_value))?;
                }
                Obligation::Output { choices, .. } => {
                    total += choices
                        .iter()
                        .map(|mana| u32::from(mana.total()))
                        .max()
                        .unwrap_or(0);
                }
            }
        }
        match first {
            Obligation::Output {
                choices,
                life_cost,
                source,
            } => {
                let mut after_cost = available.clone();
                if let Some(index) = self
                    .sources
                    .iter()
                    .position(|candidate| candidate.permanent == *source)
                {
                    after_cost[index] = false;
                }
                if *life_cost > life {
                    return None;
                }
                for (mode, output) in choices.iter().enumerate() {
                    let mut next = pool.clone();
                    next.add(output);
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            available: after_cost.clone(),
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
                let mut usable_total = u32::from(pool.usable_total(cost));
                let mut optimistic = pool.clone();
                for (index, source) in self.sources.iter().enumerate() {
                    if !available[index] {
                        continue;
                    }
                    let mut net = source
                        .fixed
                        .as_ref()
                        .map_or(0, |mana| u32::from(mana.usable_total(cost)));
                    for ability in &source.abilities {
                        let output = ability
                            .outputs
                            .iter()
                            .map(|mana| u32::from(mana.usable_total(cost)))
                            .max()
                            .unwrap_or(0);
                        let upfront = if !ability.cost.spend_on_lesson || cost.spend_on_lesson {
                            u32::from(ability.cost.mana_value)
                        } else {
                            0
                        };
                        net = net.max(output.saturating_sub(upfront));
                    }
                    if *waterbend && source.waterbend {
                        net = net.max(1);
                    }
                    usable_total += net;
                    let mut upper = source.fixed.clone().unwrap_or_default();
                    for ability in &source.abilities {
                        for output in &ability.outputs {
                            for slot in 0..6 {
                                upper.mana[slot] = upper.mana[slot].max(output.mana[slot]);
                                upper.lesson[slot] = upper.lesson[slot].max(output.lesson[slot]);
                            }
                        }
                    }
                    if *waterbend && source.waterbend {
                        upper.mana[5] = upper.mana[5].max(1);
                    }
                    for slot in 0..6 {
                        optimistic.mana[slot] =
                            optimistic.mana[slot].saturating_add(upper.mana[slot]);
                        optimistic.lesson[slot] =
                            optimistic.lesson[slot].saturating_add(upper.lesson[slot]);
                    }
                }
                if usable_total < u32::from(cost.mana_value) || !optimistic.can_pay(cost) {
                    return None;
                }
                for remainder in pool.payment_options(cost) {
                    if let Some(mut path) = self.witness(
                        PaymentSearchState {
                            available: available.clone(),
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
                    if !available[index] {
                        continue;
                    }
                    let mut remaining = available.clone();
                    remaining[index] = false;
                    if let Some(output) = &source.fixed {
                        let mut next = pool.clone();
                        next.add(output);
                        if let Some(mut path) = self.witness(
                            PaymentSearchState {
                                available: remaining.clone(),
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
                    if policy == SourcePolicy::FixedOnly
                        || (policy == SourcePolicy::AutomaticPrefix && prefix.is_some())
                    {
                        continue;
                    }
                    for ability in &source.abilities {
                        let mut nested = vec![
                            Obligation::Pay {
                                cost: ability.cost.clone(),
                                waterbend: false,
                            },
                            Obligation::Output {
                                choices: ability.outputs.clone(),
                                life_cost: ability.life_cost,
                                source: source.permanent,
                            },
                        ];
                        nested.extend(obligations.clone());
                        if let Some(mut path) = self.witness(
                            PaymentSearchState {
                                available: remaining.clone(),
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
                    if *waterbend && source.waterbend && cost.generic() > 0 {
                        let mut next = vec![Obligation::Pay {
                            cost: cost.reduced_generic(1),
                            waterbend: true,
                        }];
                        next.extend_from_slice(rest);
                        if let Some(mut path) = self.witness(
                            PaymentSearchState {
                                available: remaining,
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
                PaymentSearchState {
                    available: vec![true; self.sources.len()],
                    pool,
                    life: self.life,
                    obligations,
                    prefix: None,
                },
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
                    available: vec![true; self.sources.len()],
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
    /// Printed abilities, then currently granted abilities in source order,
    /// then the intrinsic mana ability of basic land types gained by an effect.
    pub(crate) fn activated_abilities_for(
        &self,
        permanent: PermanentId,
    ) -> Vec<ActivatedAbilityDefinition> {
        let Some(source) = self.state.permanents[permanent].as_ref() else {
            return Vec::new();
        };
        if source.abilities_lost {
            return Vec::new();
        }
        let mut abilities = self.state.cards[source.card].activated_abilities.clone();
        for ability in &mut abilities {
            if let Some(effect) = self.current_mana_output(&ability.effect, source.controller) {
                ability.effect = effect;
            }
        }
        for grant in self.battlefield_permanents(source.controller) {
            let grant = self.state.permanents[grant].as_ref().unwrap();
            if self.state.cards[grant.card]
                .mana_grant
                .iter()
                .any(|predicate| self.permanent_matches_predicate(permanent, predicate))
            {
                abilities.push(ActivatedAbilityDefinition {
                    tap_source: true,
                    effect: Effect::any_color_mana(),
                    ..Default::default()
                });
            }
        }
        if let Some(effect) = self.gained_land_type_mana(permanent) {
            abilities.push(ActivatedAbilityDefinition {
                tap_source: true,
                effect,
                ..Default::default()
            });
        }
        abilities
    }

    /// A land with a basic land type has "{T}: Add [its color]" (CR 305.6).
    /// Printed types already carry that ability; types gained until end of
    /// turn (Energybending) add one ability choosing among their colors.
    fn gained_land_type_mana(&self, permanent: PermanentId) -> Option<Effect> {
        let source = self.state.permanents[permanent].as_ref()?;
        if source.temp_subtypes.is_empty() || !self.permanent_is_land(permanent) {
            return None;
        }
        let card = &self.state.cards[source.card];
        let colors: Vec<Mana> = BASIC_LAND_TYPES
            .iter()
            .filter(|(name, _)| {
                source.temp_subtypes.iter().any(|s| s == name) && !card.has_subtype(name)
            })
            .map(|(_, mana)| Mana::parse(mana))
            .collect();
        match colors.as_slice() {
            [] => None,
            [mana] => Some(Effect::AddMana {
                mana: mana.clone(),
                until_end_of_combat: false,
            }),
            _ => Some(Effect::mana_choice(colors)),
        }
    }

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
            Effect::IfCondition {
                condition,
                effects,
                otherwise,
            } if [effects, otherwise].iter().all(|branch| {
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
                    effects
                } else {
                    otherwise
                };
                Some(branch[0].clone())
            }
            Effect::AddOneColorManaValue { amount } => {
                let amount = self.effect_value(amount, controller, 0).max(0) as usize;
                if amount == 0 {
                    return Some(Effect::AddMana {
                        mana: Mana::default(),
                        until_end_of_combat: false,
                    });
                }
                Some(Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(|color| Mana::parse(&color.repeat(amount))),
                ))
            }
            _ => None,
        }
    }

    fn payment_model(&self, player: PlayerId, sacrifice: Option<PermanentId>) -> PaymentModel {
        let mut reserved = Vec::new();
        if let Some(source) = sacrifice {
            reserved.push(SourceReservation {
                permanent: source,
                tap: false,
                sacrifice: true,
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
                    let ability = &game.activated_abilities_for(*permanent)[*ability_index];
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
        let sources = self
            .battlefield_permanents(player)
            .into_iter()
            .map(|permanent| {
                let source = self.state.permanents[permanent].as_ref().unwrap();
                let card = &self.state.cards[source.card];
                let fixed = (source.can_tap(card) && !card.mana_abilities.is_empty())
                    .then(|| self.fixed_source_mana(permanent));
                let abilities = self
                    .activated_abilities_for(permanent)
                    .iter()
                    .enumerate()
                    .filter(|(_, ability)| {
                        (!ability.tap_source || source.can_tap(card))
                            && (!ability.own_turn_only || self.active_player() == player)
                    })
                    .filter_map(|(index, ability)| {
                        let mut outputs = ability.mana_choices()?;
                        if ability.tap_source {
                            for output in &mut outputs {
                                output.add(&self.tap_mana_bonus(permanent, player));
                            }
                        }
                        Some(ManaActivation {
                            tap_source: ability.tap_source,
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
                    fixed,
                    abilities,
                    waterbend: !source.tapped
                        && (card.types.is_artifact() || card.types.is_creature()),
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
                permanent,
                ability_index,
                ..
            } = payment.as_ref()
            {
                let source = self.state.permanents[*permanent].as_ref().unwrap();
                life = life.saturating_sub(
                    self.state.cards[source.card].activated_abilities[*ability_index].life_cost,
                );
            }
        }
        let mut model = PaymentModel { sources, life };
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
                permanent,
                ability_index,
                ..
            } => {
                let ability = &self.activated_abilities_for(*permanent)[*ability_index];
                Some(SourceReservation {
                    permanent: *permanent,
                    tap: ability.tap_source,
                    sacrifice: ability.tap_source || ability.sacrifice_source,
                })
            }
            crate::flow::game::Payment::Spell { .. } => None,
        }
    }

    fn payment_obligations(&self, player: PlayerId) -> Vec<Obligation> {
        use crate::flow::{decision::Decision, game::PendingChoice};
        fn append(decision: &Decision, game: &Game, result: &mut Vec<Obligation>) {
            match decision {
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    resume,
                    ..
                } => {
                    let ability = game.activated_abilities_for(*permanent)[*ability_index].clone();
                    let mut outputs = ability.mana_choices().unwrap();
                    if ability.tap_source {
                        let player = decision.player();
                        for output in &mut outputs {
                            output.add(&game.tap_mana_bonus(*permanent, player));
                        }
                    }
                    result.push(Obligation::Pay {
                        cost: ability.mana_cost,
                        waterbend: false,
                    });
                    result.push(Obligation::Output {
                        choices: outputs,
                        life_cost: ability.life_cost,
                        source: *permanent,
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
                permanent,
                ability_index,
                remaining_generic,
                ..
            }) => {
                let source = self.state.permanents[*permanent].as_ref().unwrap();
                result.push(Obligation::Pay {
                    cost: self.state.cards[source.card].activated_abilities[*ability_index]
                        .mana_cost
                        .with_generic(*remaining_generic),
                    waterbend: true,
                });
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
        let mut obligations = vec![
            Obligation::Pay {
                cost: ability.cost,
                waterbend: false,
            },
            Obligation::Output {
                choices: ability.outputs,
                life_cost: ability.life_cost,
                source: permanent,
            },
        ];
        obligations.extend(self.payment_obligations(player));
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
    ) -> Vec<crate::agent::action::Action> {
        self.payment_model(player, sacrifice)
            .sources
            .iter()
            .flat_map(|source| {
                source.abilities.iter().filter_map(|ability| {
                    let ability_index = ability.ability_index;
                    self.activation_witness(player, source.permanent, ability_index)
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
            permanent,
            ability_index,
            ..
        } = payment
        {
            let source = self.state.permanents[*permanent].as_ref().unwrap();
            let life = self.state.cards[source.card].activated_abilities[*ability_index].life_cost;
            let Some(remaining) = model.life.checked_sub(life) else {
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

    pub(crate) fn waterbend_payment_possible(&self, player: PlayerId, cost: &ManaCost) -> bool {
        self.payment_model(player, None)
            .plan(
                self.pooled_mana(player),
                vec![Obligation::Pay {
                    cost: cost.clone(),
                    waterbend: true,
                }],
                SourcePolicy::All,
            )
            .is_some()
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
        let old = self.state.suspended_decision.take();
        self.suspend_rule_decision(crate::flow::decision::Decision::ManaUpfront {
            player,
            permanent,
            ability_index,
            resume: old.as_ref().map(|s| Box::new(s.decision.clone())),
        });
        if let Some(old) = old {
            self.state.suspended_decision.as_mut().unwrap().frame = old.frame;
        }
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
        let ability = self.activated_abilities_for(permanent)[ability_index].clone();
        if ability.tap_source {
            self.tap_permanent(permanent, true);
        }
        if ability.life_cost > 0 {
            self.change_life(player, -(ability.life_cost as i32));
        }
        if ability.sacrifice_source {
            self.sacrifice_permanent(permanent);
        }
        let suspended = self.state.suspended_decision.as_mut().unwrap();
        let Decision::ManaUpfront { resume, .. } = &mut suspended.decision else {
            unreachable!()
        };
        suspended.decision = Decision::ManaChoice {
            player,
            choices: ability.mana_choices().unwrap(),
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
        self.journal_player(player);
        for mana in produced {
            self.state.players[player.0].mana_pool.add(&mana);
        }
    }

    /// Fixed tap output plus immediate triggered bonuses; chosen outputs stay
    /// in the payment search until the player selects a color.
    fn fixed_source_mana(&self, permanent: PermanentId) -> Mana {
        let source = self.state.permanents[permanent].as_ref().unwrap();
        let card = &self.state.cards[source.card];
        let mut output = source.producible_mana(card);
        if source.can_tap(card) && !card.mana_abilities.is_empty() {
            output.add(&self.tap_mana_bonus(permanent, source.controller));
        }
        output
    }

    pub fn cached_producible_mana(&mut self, player: PlayerId) -> Mana {
        if let Some(cached) = &self.state.mana_cache[player.0] {
            return cached.clone();
        }
        let mana = self.producible_mana(player);
        self.journal_mana_cache(player);
        self.state.mana_cache[player.0] = Some(mana.clone());
        mana
    }

    pub fn invalidate_mana_cache(&mut self, player: PlayerId) {
        self.journal_mana_cache(player);
        self.state.mana_cache[player.0] = None;
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
        self.journal_player(player);
        let player_state = &mut self.state.players[player.0];
        for slot in 0..remaining.mana.len() {
            let restricted_spent = player_state.mana_pool.lesson[slot]
                + player_state.combat_mana_pool.lesson[slot]
                - remaining.lesson[slot];
            let combat = restricted_spent.min(player_state.combat_mana_pool.lesson[slot]);
            player_state.combat_mana_pool.lesson[slot] -= combat;
            player_state.mana_pool.lesson[slot] -= restricted_spent - combat;
            let spent = player_state.mana_pool.mana[slot]
                + player_state.combat_mana_pool.mana[slot]
                - remaining.mana[slot];
            let combat = spent.min(player_state.combat_mana_pool.mana[slot]);
            player_state.combat_mana_pool.mana[slot] -= combat;
            player_state.mana_pool.mana[slot] -= spent - combat;
        }
    }

    pub(crate) fn producible_mana(&self, player: PlayerId) -> Mana {
        let mut total = Mana::default();
        for permanent_id in self.battlefield_permanents(player) {
            total.add(&self.fixed_source_mana(permanent_id));
        }
        total
    }

    /// CR 106.4 — regular pools empty as each step/phase ends. The combat
    /// pool (until-end-of-combat mana) survives; see
    /// `clear_combat_mana_pools`.
    pub(crate) fn clear_mana_pools(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            self.journal_player(player);
            let becomes_red = self.unspent_mana_becomes_red(player);
            let keeps_red = self.state.players[player.0].keeps_red_mana;
            let pool = &mut self.state.players[player.0].mana_pool;
            if becomes_red {
                pool.become_red();
            } else if keeps_red {
                *pool = pool.red_only();
            } else {
                pool.clear();
            }
        }
    }

    /// Until-end-of-combat mana (firebending) empties as combat ends. Under
    /// Ozai it stays as ordinary red mana instead; red mana a player keeps
    /// this turn joins their ordinary pool.
    pub(crate) fn clear_combat_mana_pools(&mut self) {
        for player in [PlayerId(0), PlayerId(1)] {
            self.journal_player(player);
            let mut combat = std::mem::take(&mut self.state.players[player.0].combat_mana_pool);
            if self.unspent_mana_becomes_red(player) {
                combat.become_red();
            } else if self.state.players[player.0].keeps_red_mana {
                combat = combat.red_only();
            } else {
                continue;
            }
            self.state.players[player.0].mana_pool.add(&combat);
        }
    }

    /// Whether `player` controls a permanent replacing mana loss (CR 614.1a).
    fn unspent_mana_becomes_red(&self, player: PlayerId) -> bool {
        self.battlefield_permanents(player).into_iter().any(|id| {
            self.state.permanents[id].as_ref().is_some_and(|permanent| {
                self.state.cards[permanent.card]
                    .replacement_effects
                    .contains(&ReplacementEffect::UnspentManaBecomesRed)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::cards::content_pack::ContentPack;

    #[test]
    fn registered_mana_activations_have_finite_source_costs() {
        for definition in ContentPack::default().definitions() {
            for ability in &definition.activated_abilities {
                if ability.mana_choices().is_some() {
                    assert!(
                        ability.tap_source,
                        "{} needs separate tap/sacrifice resource transitions before admission",
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
