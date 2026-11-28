//! Exact payment transitions when an activation changes what another step reads.

use super::*;

#[derive(Clone)]
pub(super) struct CoupledPayment<'a> {
    game: &'a Game,
    player: PlayerId,
    profile: BoardArithmeticProfile,
    pub(super) reservations: Vec<SourceReservation>,
    pub(super) help: Help,
    pub(super) delve: Option<DelveCost>,
}

#[derive(Clone)]
pub(super) struct DelveCost {
    pub(super) cost: ManaCost,
    pub(super) permissions: CostPermissionContext,
    pub(super) life: u32,
}

#[derive(Clone)]
pub(super) enum PaymentCandidate {
    Produce(PaymentStep),
    CountedTap(PermanentId),
    HelpTap(PermanentId),
    Mode(usize),
    UpfrontFee,
    Spend(ManaSpendChoice),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PaymentCoupling {
    Invariant,
    Sensitive,
    Mutator,
}

impl PaymentModel<'_> {
    pub(super) fn coupled(&self) -> Option<&CoupledPayment<'_>> {
        let coupled = self.coupled.as_ref()?;
        (self.sources.iter().any(|source| {
            let role = |ability| coupling(coupled.profile, coupled.help, ability);
            source.untapped && source.fixed.is_some() && coupled.profile.stateful_tap()
                || source
                    .abilities
                    .iter()
                    .any(|ability| role(ability) == PaymentCoupling::Mutator)
        }) || coupled.profile.stateful_tap()
            || coupled.has_mutating_caller())
        .then_some(coupled)
    }

    /// The synchronous prefix uses only simple taps. It is safe when
    /// those taps cannot alter the board that prices or produces mana.
    pub(super) fn exact_tap_model(&self) -> Option<Self> {
        let coupled = self.coupled()?;
        if coupled.profile.stateful_tap() || coupled.has_mutating_caller() {
            return None;
        }
        let mut exact = self.clone();
        exact.cache = None;
        exact.coupled = None;
        Some(exact)
    }

    pub(super) fn fixed_tap_payable(
        &self,
        permanent: PermanentId,
        output: &Mana,
        mut pool: Mana,
        obligations: Vec<Obligation>,
    ) -> bool {
        if let Some(coupled) = self.coupled() {
            return coupled
                .plan(
                    pool,
                    obligations,
                    SourcePolicy::All,
                    Some(PaymentCandidate::Produce(PaymentStep::Fixed(permanent))),
                )
                .is_some();
        }
        pool.add(output);
        self.payable_after_tap(permanent, pool, obligations)
    }
}

fn coupling(
    profile: BoardArithmeticProfile,
    help: Help,
    ability: &ManaActivation,
) -> PaymentCoupling {
    if profile.reads_mana
        || profile.stateful_tap() && (ability.tap_source || ability.tap_count > 0)
        || profile.activation_mutates(ability.facts, help)
    {
        PaymentCoupling::Mutator
    } else if ability.facts.reads || ability.tap_source && profile.sensitive_tap() {
        PaymentCoupling::Sensitive
    } else {
        PaymentCoupling::Invariant
    }
}

impl<'a> CoupledPayment<'a> {
    pub(super) fn new(game: &'a Game, player: PlayerId, profile: BoardArithmeticProfile) -> Self {
        Self {
            game,
            player,
            profile,
            reservations: Vec::new(),
            help: Help::default(),
            delve: None,
        }
    }

    pub(super) fn taps_change_board(&self) -> bool {
        self.profile.stateful_tap()
    }

    fn has_mutating_caller(&self) -> bool {
        self.game
            .state
            .suspended_decision
            .iter()
            .flat_map(SuspendedResolution::decisions)
            .any(|decision| match decision {
                Decision::ManaUpfront {
                    activation,
                    permanent,
                    ..
                } => {
                    let source = self.game.state.permanents[*permanent].as_ref();
                    let unverified_suffix = source.is_some_and(|source| {
                        self.game
                            .mana_suffix_life_delta(self.player, source.object, &activation.suffix)
                            .is_none()
                    });
                    let rule_source = activation.sacrifice_source
                        && source.is_some_and(|source| {
                            let view = self.game.layer_view(*permanent);
                            !view.static_abilities.is_empty()
                                || !view.triggered_mana_abilities.is_empty()
                                || !self.game.state[source.object]
                                    .replacement_effects
                                    .is_empty()
                                || self
                                    .game
                                    .state
                                    .exile_links
                                    .iter()
                                    .any(|link| link.source == source.object)
                        });
                    self.profile.activation_mutates(
                        ActivationFacts {
                            counters: activation.add_counters.is_some()
                                || activation.remove_counters.is_some(),
                            life: activation.life_cost > 0 || !activation.suffix.is_empty(),
                            sacrifices: activation.sacrifice_source,
                            rule_source,
                            unverified_suffix,
                            ..Default::default()
                        },
                        self.help,
                    )
                }
                _ => false,
            })
    }

    pub(super) fn plan(
        &self,
        pool: Mana,
        mut obligations: Vec<Obligation>,
        policy: SourcePolicy,
        candidate: Option<PaymentCandidate>,
    ) -> Option<Vec<PaymentStep>> {
        #[cfg(test)]
        {
            self.game.payment_cache.borrow_mut().coupled_searches += 1;
        }
        let mut fork = self.game.clone();
        fork.pending_choice = None;
        let native = self.game.native_mana_obligations(self.player);
        if obligations.starts_with(&native) {
            obligations.drain(..native.len());
        }
        let delve = self.delve.as_ref().filter(|delve| {
            let modeled =
                self.game
                    .help_obligations(self.player, &delve.cost, delve.permissions, delve.life);
            if !obligations.ends_with(&modeled) {
                return false;
            }
            obligations.truncate(obligations.len() - modeled.len());
            true
        });
        if !matches!(
            fork.state.suspended_decision.as_ref().map(|s| &s.decision),
            Some(
                Decision::ManaChoice { .. }
                    | Decision::ManaUpfront { .. }
                    | Decision::OrderReplacement { .. }
                    | Decision::ManaSpend { .. }
            )
        ) {
            *fork.state.suspended_decision_mut() = None;
        }
        if pool != fork.pooled_mana(self.player) {
            let current = fork.state.mana_pool_mut(self.player);
            *current = ManaPool::default();
            current.add(&pool, None);
        }
        let mut search = CoupledSearch {
            payment: self,
            policy,
            failed: FastHashSet::default(),
        };
        search.run(fork, obligations, delve, candidate)
    }
}

/// Between native transitions, rules can read these payment resources. Keep
/// the continuation for equality, but hash the structural board and pool so
/// fork ids and event order never enter the memo hash.
#[derive(PartialEq, Eq)]
struct CoupledMemoKey {
    permanents: Vec<Option<crate::state::permanent::Permanent>>,
    life: [i32; 2],
    history: [crate::flow::turn::PlayerTurnHistory; 2],
    arrivals: Vec<(ObjectId, bool)>,
    damage: crate::flow::turn::DamageHistory,
    effects: crate::state::effect::EffectRecords,
    zones: [[Vec<ObjectId>; 2]; 2],
    pool: Mana,
    obligations: Vec<Obligation>,
    continuation: Option<Box<SuspendedResolution>>,
    delve: bool,
}

impl CoupledMemoKey {
    fn new(game: &Game, player: PlayerId, obligations: &[Obligation], delve: bool) -> Self {
        let state = &game.state;
        let zones = [ZoneType::Graveyard, ZoneType::Exile].map(|zone| {
            [PlayerId(0), PlayerId(1)].map(|seat| {
                let mut objects: Vec<_> = state
                    .zones
                    .zone_objects(zone, seat)
                    .iter()
                    .copied()
                    .collect();
                objects.sort_unstable();
                objects
            })
        });
        let mut arrivals = state.turn.graveyard_arrivals.clone();
        arrivals.sort_unstable();
        let mut permanents = state.permanents.0.clone();
        for permanent in permanents.iter_mut().flatten() {
            permanent.activated_this_turn.sort_unstable();
            permanent.spent_abilities.sort_unstable();
        }
        Self {
            permanents,
            life: [state.players[0].life, state.players[1].life],
            history: state.turn.history,
            arrivals,
            damage: state.turn.damage_pairs.clone(),
            effects: state.effects.clone(),
            zones,
            pool: game.pooled_mana(player),
            obligations: obligations.to_vec(),
            continuation: state.suspended_decision.clone().map(Box::new),
            delve,
        }
    }
}

impl std::hash::Hash for CoupledMemoKey {
    fn hash<H: std::hash::Hasher>(&self, hash: &mut H) {
        self.permanents.len().hash(hash);
        for permanent in &self.permanents {
            if let Some(p) = permanent {
                p.object.hash(hash);
                p.tapped.hash(hash);
                for counter in p.counters.iter() {
                    counter.hash(hash);
                }
                p.spent_abilities.hash(hash);
                p.activated_this_turn.hash(hash);
            }
        }
        self.life.hash(hash);
        self.history.hash(hash);
        self.arrivals.hash(hash);
        self.damage.hash(hash);
        self.zones.hash(hash);
        self.pool.hash(hash);
        self.obligations.hash(hash);
        self.delve.hash(hash);
    }
}

struct CoupledSearch<'a, 'game> {
    payment: &'a CoupledPayment<'game>,
    policy: SourcePolicy,
    failed: FastHashSet<CoupledMemoKey>,
}

impl CoupledSearch<'_, '_> {
    fn run(
        &mut self,
        mut game: Game,
        mut obligations: Vec<Obligation>,
        delve: Option<&DelveCost>,
        candidate: Option<PaymentCandidate>,
    ) -> Option<Vec<PaymentStep>> {
        if game.is_game_over() {
            return None;
        }
        let player = self.payment.player;
        if let Some(candidate) = candidate {
            match candidate {
                PaymentCandidate::Produce(PaymentStep::Activate(permanent, ability)) => {
                    game.begin_mana_activation(player, permanent, ability);
                }
                PaymentCandidate::Produce(PaymentStep::Fixed(permanent)) => {
                    game.produce_fixed_tap(player, permanent);
                }
                PaymentCandidate::Produce(_) => unreachable!("expected a production step"),
                PaymentCandidate::CountedTap(permanent) => {
                    if matches!(
                        game.state.suspended_decision.as_ref().map(|s| &s.decision),
                        Some(Decision::ManaUpfront {
                            tap_remaining: 1..,
                            ..
                        })
                    ) {
                        game.apply_mana_cost_tap(permanent);
                    } else {
                        let Some(Obligation::Tap {
                            candidates,
                            remaining,
                        }) = obligations.first_mut()
                        else {
                            return None;
                        };
                        if !candidates.contains(&permanent) {
                            return None;
                        }
                        *remaining = remaining.checked_sub(1)?;
                        game.tap_permanent(permanent, false);
                    }
                }
                PaymentCandidate::HelpTap(permanent) => {
                    game.tap_permanent(permanent, false);
                }
                PaymentCandidate::Mode(mode) => {
                    let caller = game.state.suspended_decision_mut().take()?;
                    match caller.decision {
                        Decision::ManaChoice { .. } => {
                            game.apply_mana_choice(caller, mode);
                        }
                        Decision::OrderReplacement { .. } => {
                            game.apply_replacement_order(caller, mode).ok()?;
                        }
                        _ => return None,
                    }
                }
                PaymentCandidate::UpfrontFee => {
                    let caller = game.state.suspended_decision_mut().take()?;
                    return self.fund_upfront(game, caller, obligations, delve);
                }
                PaymentCandidate::Spend(choice) => {
                    let caller = game.state.suspended_decision_mut().take()?;
                    let Decision::ManaSpend {
                        cost, completion, ..
                    } = caller.decision
                    else {
                        return None;
                    };
                    game.spend_mana_unit(player, &choice.unit, &cost);
                    return self.spend(game, choice.remaining, &mut |search, mut game| {
                        if let SpendCompletion::Upfront(activation) = completion.as_ref() {
                            game.complete_mana_upfront(SuspendedResolution {
                                frame: caller.frame.clone(),
                                decision: (**activation).clone(),
                            });
                        }
                        search.visit(game, obligations.clone(), delve)
                    });
                }
            }
        }
        self.visit(game, obligations, delve)
    }

    fn visit(
        &mut self,
        mut game: Game,
        obligations: Vec<Obligation>,
        delve: Option<&DelveCost>,
    ) -> Option<Vec<PaymentStep>> {
        if game.is_game_over() {
            return None;
        }
        if matches!(
            game.state.suspended_decision.as_ref().map(|s| &s.decision),
            Some(Decision::Payment(_))
        ) {
            *game.state.suspended_decision_mut() = None;
        }
        let key = CoupledMemoKey::new(&game, self.payment.player, &obligations, delve.is_some());
        if self.failed.contains(&key) {
            #[cfg(test)]
            {
                self.payment
                    .game
                    .payment_cache
                    .borrow_mut()
                    .coupled_memo_hits += 1;
            }
            return None;
        }
        #[cfg(test)]
        {
            self.payment.game.payment_cache.borrow_mut().coupled_states += 1;
        }
        let result = self.expand(game, obligations, delve);
        if result.is_none() {
            self.failed.insert(key);
        }
        result
    }

    fn expand(
        &mut self,
        mut game: Game,
        mut obligations: Vec<Obligation>,
        delve: Option<&DelveCost>,
    ) -> Option<Vec<PaymentStep>> {
        let player = self.payment.player;
        let mut model = game.pure_payment_model(player);
        model.cache = None;
        for &reservation in &self.payment.reservations {
            model.reserve_source(reservation);
        }
        game.reserve_mana_callers(&mut model);
        game.mark_help_taps(&mut model, self.payment.help);
        let profile = game.with_layer_board(|board| board.arithmetic);
        let pool = game.pooled_mana(player);
        let mut funding = false;
        let mut fee_purpose: &[SpendingContext] = &[];
        if let Some(suspended) = &game.state.suspended_decision {
            match &suspended.decision {
                Decision::ManaChoice { choices, .. } => {
                    if self.policy == SourcePolicy::FixedOnly {
                        return None;
                    }
                    return (0..choices.len()).find_map(|mode| {
                        self.run(
                            game.clone(),
                            obligations.clone(),
                            delve,
                            Some(PaymentCandidate::Mode(mode)),
                        )
                        .map(|path| prepend(PaymentStep::Output(mode), path))
                    });
                }
                Decision::OrderReplacement {
                    candidates,
                    player: chooser,
                } => {
                    let modes = (0..candidates.len()).map(|mode| {
                        self.run(
                            game.clone(),
                            obligations.clone(),
                            delve,
                            Some(PaymentCandidate::Mode(mode)),
                        )
                    });
                    return if *chooser == player {
                        modes.flatten().next()
                    } else {
                        // An opponent remains free to choose any legal order.
                        modes.collect::<Option<Vec<_>>>()?.into_iter().next()
                    };
                }
                Decision::ManaUpfront {
                    permanent,
                    activation,
                    tap_remaining,
                    ..
                } => {
                    if *tap_remaining > 0 {
                        let tap = activation.tap_cost.as_ref().expect("counted tap cost");
                        return game
                            .tap_candidates(player, &tap.predicate)
                            .into_iter()
                            .find_map(|candidate| {
                                let source = model.sources.iter().find(|source| {
                                    source.permanent == candidate && source.untapped
                                })?;
                                self.run(
                                    game.clone(),
                                    obligations.clone(),
                                    delve,
                                    Some(PaymentCandidate::CountedTap(candidate)),
                                )
                                .map(|path| prepend(PaymentStep::Tap(source.object), path))
                            });
                    }
                    if pool.can_pay(&activation.mana_cost)
                        && game.state.permanents[*permanent].is_some()
                    {
                        if let Some(path) = self.fund_upfront(
                            game.clone(),
                            suspended.clone(),
                            obligations.clone(),
                            delve,
                        ) {
                            return Some(path);
                        }
                    }
                    if activation.mana_cost.mana_value == 0 {
                        return None;
                    }
                    fee_purpose = std::slice::from_ref(&activation.mana_cost.purpose);
                    funding = true;
                }
                _ => return None,
            }
        }
        let mut pure = model.clone();
        let loses_rules = model
            .sources
            .iter()
            .flat_map(|s| &s.abilities)
            .any(|a| a.facts.sacrifices && a.facts.rule_source);
        let roles: Vec<_> = model
            .sources
            .iter()
            .map(|source| {
                let abilities: Vec<_> = source
                    .abilities
                    .iter()
                    .map(|a| coupling(profile, self.payment.help, a))
                    .collect();
                let fixed = if profile.stateful_tap() {
                    PaymentCoupling::Mutator
                } else if profile.sensitive_tap()
                    || loses_rules
                    || abilities.contains(&PaymentCoupling::Mutator)
                {
                    PaymentCoupling::Sensitive
                } else {
                    PaymentCoupling::Invariant
                };
                (fixed, abilities)
            })
            .collect();
        let mut concrete = obligations.clone();
        if let Some(delve) = delve {
            let revised = game.help_obligations(player, &delve.cost, delve.permissions, delve.life);
            concrete.extend(revised);
        }
        if !funding && !profile.reads_mana {
            for (source, (fixed, roles)) in Arc::make_mut(&mut pure.sources).iter_mut().zip(&roles)
            {
                if *fixed == PaymentCoupling::Mutator {
                    source.fixed = None;
                    source.waterbend = false;
                    source.keyword = false;
                    source.convoke_colors = ManaTypeSet::EMPTY;
                }
                let mut roles = roles.iter();
                source
                    .abilities
                    .retain(|_| *roles.next().unwrap() != PaymentCoupling::Mutator);
            }
            if let Some(path) = pure.cached_plan(pool.clone(), concrete.clone(), self.policy, None)
            {
                return Some(path);
            }
        }
        if !funding && obligations.is_empty() && delve.is_none() {
            return Some(Vec::new());
        }

        // An invariant source only needs a native transition while funding a
        // native activation. Else the pure finish accounts for its supply.
        let mut purposes = None;
        let tap_sets =
            model.tap_candidate_sets(concrete.iter().filter_map(Obligation::tap_candidates));
        let mut invariant_tried = Vec::new();
        let all_sources = self.policy == SourcePolicy::All;
        let production_allowed = funding
            || matches!(concrete.first(), Some(Obligation::Pay { cost, .. }) if cost.mana_value > 0)
            || obligations.is_empty() && delve.is_some_and(|delve| delve.cost.mana_value > 0);
        for (index, (source, (fixed, roles))) in model
            .sources
            .iter()
            .zip(&roles)
            .enumerate()
            .filter(|_| production_allowed)
        {
            let invariant = *fixed == PaymentCoupling::Invariant
                && roles.iter().all(|role| *role == PaymentCoupling::Invariant);
            if invariant && funding {
                if invariant_tried.iter().any(|&previous| {
                    PurposeProjection::sources_equivalent(
                        || {
                            Some(purposes.get_or_insert_with(|| {
                                PurposeProjection::for_model(&model, &concrete, fee_purpose)
                            }))
                        },
                        &model.sources[previous],
                        source,
                        &tap_sets,
                    )
                }) {
                    continue;
                }
                invariant_tried.push(index);
            }
            let steps = source
                .fixed
                .iter()
                .map(|_| (*fixed, PaymentStep::Fixed(source.permanent)))
                .chain(
                    source
                        .abilities
                        .iter()
                        .zip(roles)
                        .filter_map(|(ability, &role)| {
                            all_sources.then(|| (role, ability.kind.step(source.permanent)))
                        }),
                );
            for (role, step) in steps {
                if role == PaymentCoupling::Invariant && !funding {
                    continue;
                }
                if let Some(path) = self.run(
                    game.clone(),
                    obligations.clone(),
                    delve,
                    Some(PaymentCandidate::Produce(step.clone())),
                ) {
                    return Some(prepend(step, path));
                }
            }
        }
        if funding {
            return None;
        }

        let mut delve = delve;
        if obligations.is_empty() {
            // Spending the entire last cost commits it before anything else
            // can change how many cards are available for delve.
            obligations = concrete;
            delve = None;
        }
        let first = obligations.first()?.clone();
        let tail = obligations[1..].to_vec();
        match first {
            Obligation::Pay { cost, taps } => {
                if pool.can_pay(&cost) {
                    if let Some(path) =
                        self.spend(game.clone(), cost.clone(), &mut |search, game| {
                            search.visit(game, tail.clone(), delve)
                        })
                    {
                        return Some(prepend(PaymentStep::Spend, path));
                    }
                }
                self.try_help_taps(&game, &model, obligations, delve, cost, taps)
            }
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                if remaining == 0 {
                    return self.visit(game, tail, delve);
                }
                model
                    .tap_candidates(&model.resources(), &candidates)
                    .find_map(|index| {
                        let source = &model.sources[index];
                        self.run(
                            game.clone(),
                            obligations.clone(),
                            delve,
                            Some(PaymentCandidate::CountedTap(source.permanent)),
                        )
                        .map(|path| prepend(PaymentStep::Tap(source.object), path))
                    })
            }
            Obligation::PayLife(amount) => {
                game.life_budget(player).pay(amount)?;
                game.change_life(player, -(amount as i32));
                self.visit(game, tail, delve)
            }
            Obligation::Output { .. } => unreachable!("the native activation owns its output"),
        }
    }

    fn try_help_taps(
        &mut self,
        game: &Game,
        model: &PaymentModel<'_>,
        obligations: Vec<Obligation>,
        delve: Option<&DelveCost>,
        cost: ManaCost,
        taps: Taps,
    ) -> Option<Vec<PaymentStep>> {
        if self.policy != SourcePolicy::All || taps == Taps::None {
            return None;
        }
        for source in model.sources.iter().filter(|source| source.untapped) {
            let colors = match taps {
                Taps::Convoke => source
                    .convoke_colors
                    .iter()
                    .filter(|&color| cost.cost[color] > 0)
                    .map(|color| Color::COLORS[color])
                    .collect::<Vec<_>>(),
                Taps::Keyword if source.keyword && cost.generic() > 0 => vec![Color::Generic],
                Taps::Any if source.waterbend && cost.generic() > 0 => vec![Color::Generic],
                _ => continue,
            };
            for color in colors {
                let step = if taps == Taps::Any {
                    PaymentStep::Waterbend(source.permanent)
                } else {
                    PaymentStep::Convoke(source.permanent)
                };
                let mut rest = obligations.clone();
                rest[0] = Obligation::Pay {
                    cost: cost.less_pip(color),
                    taps,
                };
                if let Some(path) = self.run(
                    game.clone(),
                    rest,
                    delve,
                    Some(PaymentCandidate::HelpTap(source.permanent)),
                ) {
                    return Some(prepend(step, path));
                }
            }
        }
        None
    }

    fn fund_upfront(
        &mut self,
        game: Game,
        caller: SuspendedResolution,
        obligations: Vec<Obligation>,
        delve: Option<&DelveCost>,
    ) -> Option<Vec<PaymentStep>> {
        let Decision::ManaUpfront {
            activation,
            player,
            tap_remaining: 0,
            ..
        } = &caller.decision
        else {
            return None;
        };
        let player = *player;
        let life_cost = activation.life_cost;
        let mut game = game;
        *game.state.suspended_decision_mut() = None;
        let path = self.spend(
            game,
            activation.mana_cost.clone(),
            &mut |search, mut game| {
                if !game.life_budget(player).can_pay(life_cost) {
                    return None;
                }
                game.complete_mana_upfront(caller.clone());
                search.visit(game, obligations.clone(), delve)
            },
        )?;
        Some(prepend(PaymentStep::Spend, path))
    }

    fn spend<F>(&mut self, game: Game, cost: ManaCost, finish: &mut F) -> Option<Vec<PaymentStep>>
    where
        F: FnMut(&mut Self, Game) -> Option<Vec<PaymentStep>>,
    {
        self.spend_inner(game, cost, finish, &mut FastHashSet::default())
    }

    fn spend_inner<F>(
        &mut self,
        game: Game,
        cost: ManaCost,
        finish: &mut F,
        visited: &mut FastHashSet<(Mana, ManaCost)>,
    ) -> Option<Vec<PaymentStep>>
    where
        F: FnMut(&mut Self, Game) -> Option<Vec<PaymentStep>>,
    {
        if cost.mana_value == 0 {
            return finish(self, game);
        }
        let player = self.payment.player;
        let pool = game.pooled_mana(player);
        if !pool.can_pay(&cost) || !visited.insert((pool.clone(), cost.clone())) {
            return None;
        }
        pool.spending_choices(&cost).into_iter().find_map(|choice| {
            let mut fork = game.clone();
            fork.spend_mana_unit(player, &choice.unit, &cost);
            self.spend_inner(fork, choice.remaining, finish, visited)
        })
    }
}

fn prepend(step: PaymentStep, mut path: Vec<PaymentStep>) -> Vec<PaymentStep> {
    path.insert(0, step);
    path
}
