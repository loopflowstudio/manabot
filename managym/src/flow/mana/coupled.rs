//! Exact payment transitions when an activation changes what another step reads.

use super::*;

#[derive(Clone)]
pub(super) struct CoupledPayment<'a> {
    game: &'a Game,
    player: PlayerId,
    profile: BoardArithmeticProfile,
    mutating_caller: bool,
    pub(super) reservations: Vec<SourceReservation>,
    help: Help,
    pub(super) delve: Option<DelveCost>,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct DelveCost {
    pub(super) cost: ManaCost,
    pub(super) permissions: CostPermissionContext,
    pub(super) life: u32,
}

impl DelveCost {
    fn after_pip(&self, color: Color, waterbend: bool) -> Self {
        let mut rest = self.clone();
        rest.cost = rest.cost.less_pip(color);
        rest.permissions.waterbend = rest
            .permissions
            .waterbend
            .saturating_sub(u8::from(waterbend))
            .min(rest.cost.generic());
        rest
    }
}

#[derive(Clone)]
pub(super) enum PaymentCandidate {
    Produce(PaymentStep),
    CountedTap(PermanentId),
    HelpTap(PermanentId),
    DelveExile(ObjectId),
    Mode(usize),
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
        self.coupled.as_ref()
    }

    /// The synchronous prefix uses only simple taps. It is safe when
    /// those taps cannot alter the board that prices or produces mana.
    pub(super) fn exact_tap_model(&self) -> Option<Self> {
        let coupled = self.coupled()?;
        if coupled.profile.stateful_tap() || coupled.mutating_caller {
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
    loses_rules: bool,
    ability: &ManaActivation,
) -> PaymentCoupling {
    if profile.reads_mana
        || profile.stateful_tap() && (ability.tap_source || ability.tap_count > 0)
        || profile.activation_mutates(ability.facts, help)
    {
        PaymentCoupling::Mutator
    } else if ability.facts.reads
        || profile.writes_non_pt
        || loses_rules
        || ability.tap_source && profile.stateful_triggers
    {
        PaymentCoupling::Sensitive
    } else {
        PaymentCoupling::Invariant
    }
}

impl<'a> CoupledPayment<'a> {
    pub(super) fn new(
        game: &'a Game,
        player: PlayerId,
        profile: BoardArithmeticProfile,
        help: Help,
        sources: &[ManaSource],
    ) -> Option<Self> {
        // The resource filters can remove sources after this check. Keeping a
        // coupled classification is safe even when a reservation does so.
        let mutating_caller = !profile.stateful_tap()
            && game
                .state
                .suspended_decision
                .as_ref()
                .is_some_and(|suspended| {
                    matches!(suspended.decision, Decision::OrderReplacement { .. })
                        || suspended
                            .decisions()
                            .any(|decision| upfront_mutates(game, profile, help, decision))
                });
        (profile.stateful_tap()
            || mutating_caller
            || help.delve.is_some() && profile.stateful_delve()
            || sources
                .iter()
                .flat_map(|source| &source.abilities)
                .any(|ability| profile.activation_mutates(ability.facts, help)))
        .then(|| Self {
            game,
            player,
            profile,
            mutating_caller,
            reservations: Vec::new(),
            help,
            delve: None,
        })
    }

    pub(super) fn taps_change_board(&self) -> bool {
        self.profile.stateful_tap()
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
        let automatic_prefix = policy == SourcePolicy::AutomaticPrefix;
        let mut search = CoupledSearch {
            payment: self,
            policy: if automatic_prefix {
                SourcePolicy::All
            } else {
                policy
            },
            prefix_taps: None,
            failed: FastHashSet::default(),
        };
        if automatic_prefix {
            // A tap can change later outputs, so the frozen supply cannot
            // bound the shortfall. Try the fewest exact taps before the fee.
            let taps = fork.iter_battlefield_permanents(self.player).count();
            return (0..=taps).find_map(|budget| {
                search.prefix_taps = Some(budget);
                search.run(fork.clone(), obligations.clone(), delve, None)
            });
        }
        search.run(fork, obligations, delve, candidate)
    }
}

fn upfront_mutates(
    game: &Game,
    profile: BoardArithmeticProfile,
    help: Help,
    decision: &Decision,
) -> bool {
    let Decision::ManaUpfront {
        player,
        permanent,
        activation,
        ..
    } = decision
    else {
        return false;
    };
    let Some(source) = game.state.permanents[*permanent].as_ref() else {
        return true;
    };
    let rule_source = activation.sacrifice_source && {
        let view = game.layer_view(*permanent);
        !view.static_abilities.is_empty()
            || !view.triggered_mana_abilities.is_empty()
            || !game.state[source.object].replacement_effects.is_empty()
            || game
                .state
                .exile_links
                .iter()
                .any(|link| link.source == source.object)
    };
    let facts = ActivationFacts {
        counters: activation.add_counters.is_some() || activation.remove_counters.is_some(),
        life: activation.life_cost > 0 || !activation.suffix.is_empty(),
        sacrifices: activation.sacrifice_source,
        rule_source,
        ..Default::default()
    };
    profile.activation_mutates(facts, help)
        || game
            .mana_suffix_life_delta(*player, source.object, &activation.suffix)
            .is_none()
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
    delve: Option<DelveCost>,
    prefix_taps: Option<usize>,
}

impl CoupledMemoKey {
    fn new(
        game: &Game,
        player: PlayerId,
        obligations: &[Obligation],
        delve: Option<&DelveCost>,
        prefix_taps: Option<usize>,
    ) -> Self {
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
            delve: delve.cloned(),
            prefix_taps,
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
        self.delve.as_ref().map(|delve| &delve.cost).hash(hash);
        self.prefix_taps.hash(hash);
    }
}

struct CoupledSearch<'a, 'game> {
    payment: &'a CoupledPayment<'game>,
    policy: SourcePolicy,
    /// Remaining exact taps before the initial fee. Its callers may use
    /// any legal production once that fee is spent.
    prefix_taps: Option<usize>,
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
                PaymentCandidate::DelveExile(object) => {
                    let delve = delve?;
                    if delve.cost.generic() == 0 {
                        return None;
                    }
                    game.move_object(object, ZoneType::Exile);
                    return self.visit(
                        game,
                        obligations,
                        Some(&delve.after_pip(Color::Generic, false)),
                    );
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
                            complete_upfront(
                                &mut game,
                                SuspendedResolution {
                                    frame: caller.frame.clone(),
                                    decision: (**activation).clone(),
                                },
                            )?;
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
        let key = CoupledMemoKey::new(
            &game,
            self.payment.player,
            &obligations,
            delve,
            self.prefix_taps,
        );
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
            if self.prefix_taps.is_some()
                && !matches!(
                    suspended.decision,
                    Decision::ManaUpfront {
                        tap_remaining: 0,
                        ..
                    }
                )
            {
                return None;
            }
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
                    if pool.can_pay(&activation.mana_cost) {
                        if let Some(path) = self.fund_upfront(
                            game.clone(),
                            suspended.clone(),
                            obligations.clone(),
                            delve,
                        ) {
                            return Some(path);
                        }
                    }
                    if activation.mana_cost.mana_value == 0 || self.prefix_taps == Some(0) {
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
                    .map(|a| coupling(profile, self.payment.help, loses_rules, a))
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
            let mut permissions = delve.permissions;
            if self.payment.profile.stateful_delve() {
                permissions.help.delve = None;
            }
            concrete.extend(game.help_obligations(player, &delve.cost, permissions, delve.life));
        }
        if !funding && !profile.reads_mana {
            for (source, (fixed, roles)) in Arc::make_mut(&mut pure.sources).iter_mut().zip(&roles)
            {
                if *fixed == PaymentCoupling::Mutator {
                    source.untapped = false;
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
        let prefix_taps = self.prefix_taps;
        let all_sources = self.policy == SourcePolicy::All && prefix_taps.is_none();
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
                self.prefix_taps = prefix_taps.map(|budget| budget - 1);
                let path = self.run(
                    game.clone(),
                    obligations.clone(),
                    delve,
                    Some(PaymentCandidate::Produce(step.clone())),
                );
                self.prefix_taps = prefix_taps;
                if let Some(path) = path {
                    return Some(prepend(step, path));
                }
            }
        }
        if funding {
            return None;
        }

        if obligations.is_empty()
            && self.policy == SourcePolicy::All
            && self.payment.profile.stateful_delve()
        {
            if let Some(delve) = delve.filter(|delve| delve.cost.generic() > 0) {
                let spell = delve.permissions.help.delve.expect("tracked delve cost");
                for object in game.graveyard_payers(player, spell) {
                    if let Some(path) = self.run(
                        game.clone(),
                        Vec::new(),
                        Some(delve),
                        Some(PaymentCandidate::DelveExile(object)),
                    ) {
                        return Some(prepend(PaymentStep::DelveExile(object), path));
                    }
                }
            }
        }

        let paying_delve_cost = obligations.is_empty();
        if paying_delve_cost {
            // Spending the entire last cost commits it before anything else
            // can change how many cards are available for delve.
            obligations = concrete;
        }
        let saved_delve = if paying_delve_cost { None } else { delve };
        let first = obligations.first()?.clone();
        let tail = obligations[1..].to_vec();
        match first {
            Obligation::Pay { cost, .. } => {
                if pool.can_pay(&cost) {
                    if let Some(path) =
                        self.spend(game.clone(), cost.clone(), &mut |search, game| {
                            search.visit(game, tail.clone(), saved_delve)
                        })
                    {
                        return Some(prepend(PaymentStep::Spend, path));
                    }
                }
                self.try_help_taps(&game, &model, obligations, delve, paying_delve_cost)
            }
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                if remaining == 0 {
                    return self.visit(game, tail, saved_delve);
                }
                model
                    .tap_candidates(&model.resources(), &candidates)
                    .find_map(|index| {
                        let source = &model.sources[index];
                        self.run(
                            game.clone(),
                            obligations.clone(),
                            saved_delve,
                            Some(PaymentCandidate::CountedTap(source.permanent)),
                        )
                        .map(|path| prepend(PaymentStep::Tap(source.object), path))
                    })
            }
            Obligation::PayLife(amount) => {
                game.life_budget(player).pay(amount)?;
                game.change_life(player, -(amount as i32));
                self.visit(game, tail, saved_delve)
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
        paying_delve_cost: bool,
    ) -> Option<Vec<PaymentStep>> {
        let Some(Obligation::Pay { cost, taps }) = obligations.first() else {
            return None;
        };
        let taps = *taps;
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
                let next_delve = delve
                    .filter(|_| paying_delve_cost && self.payment.profile.stateful_delve())
                    .map(|tracked| tracked.after_pip(color, taps == Taps::Any));
                let (rest, delve) = if let Some(tracked) = &next_delve {
                    (Vec::new(), Some(tracked))
                } else {
                    let mut rest = obligations.clone();
                    rest[0] = Obligation::Pay {
                        cost: cost.less_pip(color),
                        taps,
                    };
                    (rest, if paying_delve_cost { None } else { delve })
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
        let Decision::ManaUpfront { activation, .. } = &caller.decision else {
            return None;
        };
        let mut game = game;
        *game.state.suspended_decision_mut() = None;
        let prefix_taps = self.prefix_taps.take();
        let path = self.spend(
            game,
            activation.mana_cost.clone(),
            &mut |search, mut game| {
                complete_upfront(&mut game, caller.clone())?;
                search.visit(game, obligations.clone(), delve)
            },
        );
        self.prefix_taps = prefix_taps;
        path.map(|path| prepend(PaymentStep::Spend, path))
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

fn complete_upfront(game: &mut Game, caller: SuspendedResolution) -> Option<()> {
    let Decision::ManaUpfront {
        player,
        permanent,
        activation,
        tap_remaining: 0,
        ..
    } = &caller.decision
    else {
        return None;
    };
    if game.state.permanents[*permanent].is_none()
        || !game.life_budget(*player).can_pay(activation.life_cost)
    {
        return None;
    }
    game.complete_mana_upfront(caller);
    Some(())
}

fn prepend(step: PaymentStep, mut path: Vec<PaymentStep>) -> Vec<PaymentStep> {
    path.insert(0, step);
    path
}
