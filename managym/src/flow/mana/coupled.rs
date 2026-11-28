//! Exact payment transitions when an activation changes what another step reads.

use super::*;
use crate::flow::payment::TapPayment;
use crate::state::payment::Toward;
use std::borrow::Cow;

#[derive(Clone)]
pub(super) struct CoupledPayment<'a> {
    game: &'a Game,
    player: PlayerId,
    ordered_taps: bool,
    pub(super) reservations: Vec<SourceReservation>,
    help: Help,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct HelpedCost {
    pub(super) cost: ManaCost,
    pub(super) permissions: CostPermissionContext,
    pub(super) life: u32,
}

impl HelpedCost {
    fn pay_pip(&mut self, color: Color) {
        self.cost = self.cost.less_pip(color);
        self.permissions.waterbend = self.permissions.waterbend.min(self.cost.generic());
    }
}

pub(super) enum PaymentCandidate {
    Produce(PermanentId, ManaActivationKind),
    CountedTap(PermanentId),
    HelpTap(PermanentId, TapPayment),
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
    /// The synchronous prefix uses only simple taps. It is safe when
    /// those taps cannot alter the board that prices or produces mana.
    pub(super) fn exact_tap_model(&self) -> Option<Self> {
        if self.coupled.as_ref()?.ordered_taps {
            return None;
        }
        Some(Self {
            coupled: None,
            cache: None,
            sources: self.sources.clone(),
            life: self.life,
        })
    }

    pub(super) fn fixed_tap_payable(
        &self,
        permanent: PermanentId,
        output: &Mana,
        mut pool: Mana,
        obligations: Vec<Obligation>,
    ) -> bool {
        if let Some(coupled) = &self.coupled {
            return coupled
                .plan(
                    pool,
                    obligations,
                    SourcePolicy::All,
                    Some(PaymentCandidate::Produce(
                        permanent,
                        ManaActivationKind::FixedTap,
                    )),
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
    if profile.reads.mana
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
        let ordered_taps = profile.stateful_tap()
            || game
                .state
                .suspended_decision
                .as_ref()
                .is_some_and(|suspended| {
                    matches!(suspended.decision, Decision::OrderReplacement { .. })
                        || suspended
                            .decisions()
                            .any(|decision| upfront_mutates(game, profile, help, decision))
                });
        (ordered_taps
            || help.delve.is_some() && profile.stateful_delve()
            || sources
                .iter()
                .flat_map(|source| &source.abilities)
                .any(|ability| profile.activation_mutates(ability.facts, help)))
        .then(|| Self {
            game,
            player,
            ordered_taps,
            reservations: Vec::new(),
            help,
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
        if policy == SourcePolicy::FixedOnly
            && obligations
                .iter()
                .any(|obligation| matches!(obligation, Obligation::Helped(_)))
        {
            obligations = obligations
                .into_iter()
                .flat_map(|obligation| match obligation {
                    Obligation::Helped(helped) => {
                        Obligation::mana_and_life(&helped.cost, helped.life)
                    }
                    other => vec![other],
                })
                .collect();
        }
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
                search.visit(fork.clone(), &obligations)
            });
        }
        match candidate {
            Some(step) => search.after(fork, &obligations, step),
            None => search.visit(fork, &obligations),
        }
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
    let facts = ActivationFacts {
        counters: activation.add_counters.is_some() || activation.remove_counters.is_some(),
        life: activation.life_cost > 0 || !activation.suffix.is_empty(),
        sacrifices: activation.sacrifice_source,
        rule_source: activation.sacrifice_source
            && game.supplies_payment_rules(source.object, &game.layer_view(*permanent)),
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
    prefix_taps: Option<usize>,
}

impl CoupledMemoKey {
    fn new(
        game: &Game,
        player: PlayerId,
        obligations: &[Obligation],
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
    fn after(
        &mut self,
        mut game: Game,
        obligations: &[Obligation],
        candidate: PaymentCandidate,
    ) -> Option<Vec<PaymentStep>> {
        if game.is_game_over() {
            return None;
        }
        let player = self.payment.player;
        let mut obligations = Cow::Borrowed(obligations);
        match candidate {
            PaymentCandidate::Produce(permanent, ManaActivationKind::Activated(ability)) => {
                game.begin_mana_activation(player, permanent, ability);
            }
            PaymentCandidate::Produce(permanent, ManaActivationKind::FixedTap) => {
                game.produce_fixed_tap(player, permanent);
            }
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
                    }) = obligations.to_mut().first_mut()
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
            PaymentCandidate::HelpTap(permanent, payment) => {
                let Some(Obligation::Helped(helped)) = obligations.to_mut().first_mut() else {
                    return None;
                };
                game.tap_permanent(permanent, false);
                helped.permissions = helped.permissions.after_tap(payment);
                helped.pay_pip(payment.color);
            }
            PaymentCandidate::DelveExile(object) => {
                let Some(Obligation::Helped(helped)) = obligations.to_mut().first_mut() else {
                    return None;
                };
                if helped.cost.generic() == 0 {
                    return None;
                }
                game.move_object(object, ZoneType::Exile);
                helped.pay_pip(Color::Generic);
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
                    search.visit(game, &obligations)
                });
            }
        }
        self.visit(game, &obligations)
    }

    fn visit(&mut self, mut game: Game, obligations: &[Obligation]) -> Option<Vec<PaymentStep>> {
        if game.is_game_over() {
            return None;
        }
        if matches!(
            game.state.suspended_decision.as_ref().map(|s| &s.decision),
            Some(Decision::Payment(_))
        ) {
            *game.state.suspended_decision_mut() = None;
        }
        let key = CoupledMemoKey::new(&game, self.payment.player, obligations, self.prefix_taps);
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
        let result = self.expand(game, obligations);
        if result.is_none() {
            self.failed.insert(key);
        }
        result
    }

    fn expand(&mut self, mut game: Game, obligations: &[Obligation]) -> Option<Vec<PaymentStep>> {
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
                        self.after(game.clone(), obligations, PaymentCandidate::Mode(mode))
                            .map(|path| prepend(PaymentStep::Output(mode), path))
                    });
                }
                Decision::OrderReplacement {
                    candidates,
                    player: chooser,
                } => {
                    let mut modes = (0..candidates.len()).map(|mode| {
                        self.after(game.clone(), obligations, PaymentCandidate::Mode(mode))
                    });
                    return if *chooser == player {
                        modes.flatten().next()
                    } else {
                        // An opponent remains free to choose any legal order.
                        let first = modes.next()??;
                        modes.all(|path| path.is_some()).then_some(first)
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
                                self.after(
                                    game.clone(),
                                    obligations,
                                    PaymentCandidate::CountedTap(candidate),
                                )
                                .map(|path| prepend(PaymentStep::Tap(source.object), path))
                            });
                    }
                    if pool.can_pay(&activation.mana_cost) {
                        if let Some(path) =
                            self.fund_upfront(game.clone(), suspended.clone(), obligations)
                        {
                            return Some(path);
                        }
                    }
                    if activation.mana_cost.mana_value == 0 || self.prefix_taps == Some(0) {
                        return None;
                    }
                    fee_purpose = std::slice::from_ref(&activation.mana_cost.purpose);
                }
                _ => return None,
            }
        }
        let funding = !fee_purpose.is_empty();
        if !funding && obligations.is_empty() {
            return Some(Vec::new());
        }
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
        if !funding && !profile.reads.mana {
            if let Some(path) =
                self.pure_finish(&game, &model, profile, &roles, pool.clone(), obligations)
            {
                return Some(path);
            }
        }

        // An invariant source only needs a native transition while funding a
        // native activation. Else the pure finish accounts for its supply.
        let mut purposes = None;
        let tap_sets =
            model.tap_candidate_sets(obligations.iter().filter_map(Obligation::tap_candidates));
        let mut invariant_tried = Vec::new();
        let prefix_taps = self.prefix_taps;
        let all_sources = self.policy == SourcePolicy::All && prefix_taps.is_none();
        let production_allowed = funding
            || match obligations.first() {
                Some(Obligation::Pay { cost, .. }) => cost.mana_value > 0,
                Some(Obligation::Helped(helped)) => helped.cost.mana_value > 0,
                _ => false,
            };
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
                                PurposeProjection::for_model(&model, obligations, fee_purpose)
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
                .map(|_| (*fixed, ManaActivationKind::FixedTap))
                .chain(
                    source
                        .abilities
                        .iter()
                        .zip(roles)
                        .filter_map(|(ability, &role)| all_sources.then_some((role, ability.kind))),
                );
            for (role, kind) in steps {
                if role == PaymentCoupling::Invariant && !funding {
                    continue;
                }
                self.prefix_taps = prefix_taps.map(|budget| budget - 1);
                let path = self.after(
                    game.clone(),
                    obligations,
                    PaymentCandidate::Produce(source.permanent, kind),
                );
                self.prefix_taps = prefix_taps;
                if let Some(path) = path {
                    return Some(prepend(kind.step(source.permanent), path));
                }
            }
        }
        if funding {
            return None;
        }

        let (first, tail) = obligations.split_first()?;
        match first {
            Obligation::Helped(helped) => {
                self.pay_helped(&game, &model, profile, helped, obligations)
            }
            Obligation::Pay {
                cost,
                taps: Taps::None,
            } => {
                if !pool.can_pay(cost) {
                    return None;
                }
                self.spend(game, cost.clone(), &mut |search, game| {
                    search.visit(game, tail)
                })
                .map(|path| prepend(PaymentStep::Spend, path))
            }
            Obligation::Pay { .. } => unreachable!("native search receives unsplit help"),
            Obligation::Tap {
                candidates,
                remaining,
            } => {
                if *remaining == 0 {
                    return self.visit(game, tail);
                }
                model
                    .tap_candidates(&model.resources(), candidates)
                    .find_map(|index| {
                        let source = &model.sources[index];
                        self.after(
                            game.clone(),
                            obligations,
                            PaymentCandidate::CountedTap(source.permanent),
                        )
                        .map(|path| prepend(PaymentStep::Tap(source.object), path))
                    })
            }
            Obligation::PayLife(amount) => {
                game.life_budget(player).pay(*amount)?;
                game.change_life(player, -(*amount as i32));
                self.visit(game, tail)
            }
            Obligation::Output { .. } => unreachable!("the native activation owns its output"),
        }
    }

    /// Try paying from the current board without another mutating step. Delve
    /// stays on the fork when exiling a card could change the remaining sources.
    fn pure_finish(
        &self,
        game: &Game,
        model: &PaymentModel,
        profile: BoardArithmeticProfile,
        roles: &[(PaymentCoupling, Vec<PaymentCoupling>)],
        pool: Mana,
        obligations: &[Obligation],
    ) -> Option<Vec<PaymentStep>> {
        let mut concrete = Vec::new();
        for obligation in obligations {
            if let Obligation::Helped(helped) = obligation {
                let mut permissions = helped.permissions;
                if profile.stateful_delve() {
                    permissions.help.delve = None;
                }
                concrete.extend(game.help_obligations(
                    self.payment.player,
                    &helped.cost,
                    permissions,
                    helped.life,
                ));
            } else {
                concrete.push(obligation.clone());
            }
        }
        let mut pure = model.clone();
        for (source, (fixed, roles)) in Arc::make_mut(&mut pure.sources).iter_mut().zip(roles) {
            if *fixed == PaymentCoupling::Mutator {
                source.reserve_tap();
            }
            let mut roles = roles.iter();
            source
                .abilities
                .retain(|_| *roles.next().unwrap() != PaymentCoupling::Mutator);
        }
        pure.cached_plan(pool, concrete, self.policy, None)
    }

    fn pay_helped(
        &mut self,
        game: &Game,
        model: &PaymentModel<'_>,
        profile: BoardArithmeticProfile,
        helped: &HelpedCost,
        obligations: &[Obligation],
    ) -> Option<Vec<PaymentStep>> {
        let player = self.payment.player;
        if profile.stateful_delve() && helped.cost.generic() > 0 {
            if let Some(spell) = helped.permissions.help.delve {
                for object in game.graveyard_payers(player, spell) {
                    if let Some(path) = self.after(
                        game.clone(),
                        obligations,
                        PaymentCandidate::DelveExile(object),
                    ) {
                        return Some(prepend(PaymentStep::DelveExile(object), path));
                    }
                }
            }
        }
        let cost = if profile.stateful_delve() {
            helped.cost.clone()
        } else {
            game.cost_after_delve(player, &helped.cost, helped.permissions.help)
        };
        if game.pooled_mana(player).can_pay(&cost) {
            let mut after_spend: Vec<_> = Obligation::life(helped.life).into_iter().collect();
            after_spend.extend_from_slice(&obligations[1..]);
            if let Some(path) = self.spend(game.clone(), cost, &mut |search, game| {
                search.visit(game, &after_spend)
            }) {
                return Some(prepend(PaymentStep::Spend, path));
            }
        }
        for source in model
            .sources
            .iter()
            .filter(|source| source.untapped && source.waterbend)
        {
            for payment in game.tap_payments(&helped.cost, helped.permissions, source.permanent) {
                let step = if payment.toward == Toward::Waterbend {
                    PaymentStep::Waterbend(source.permanent)
                } else {
                    PaymentStep::Convoke(source.permanent)
                };
                if let Some(path) = self.after(
                    game.clone(),
                    obligations,
                    PaymentCandidate::HelpTap(source.permanent, payment),
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
        obligations: &[Obligation],
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
                search.visit(game, obligations)
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
