// help.rs
// Whether a cost is payable when part of it is paid another way: convoke,
// improvise, delve and waterbend beside the mana of `PaymentModel`, and what
// one direct tap of an artifact or creature may pay. This is the one owner
// of direct-tap eligibility for casts, activations and payments made during
// resolution: offers, labels and execution all read `help_taps`.
//
// A cost's help splits its generic mana into tiers (`help_obligations`):
// convoke and improvise taps may pay any of it, any artifact's or creature's
// tap the announced waterbend part, and mana all of it. A convoking
// creature may also tap to pay an owed pip of one of its colors.

use super::*;
use crate::flow::payment::TapPayment;
use crate::state::payment::Toward;

impl Game {
    /// Mark what each untapped creature or artifact may pay by tapping
    /// under this cost's convoke or improvise permissions.
    pub(super) fn mark_help_taps(&self, model: &mut PaymentModel, help: Help) {
        if let Some(coupled) = &mut model.coupled {
            coupled.help = help;
        }
        let tap_permissions = |source: &ManaSource| {
            let mut colors = ManaTypeSet::EMPTY;
            if !source.waterbend || !(help.convoke || help.improvise) {
                return (false, colors);
            }
            let view = self.characteristics(source.permanent);
            let convoke = help.convoke && view.is_creature();
            if convoke {
                for &color in &view.colors {
                    colors.insert(ManaTypeSet::of(color));
                }
            }
            (
                convoke || help.improvise && view.types.is_artifact(),
                colors,
            )
        };
        if model
            .sources
            .iter()
            .all(|source| (source.keyword, source.convoke_colors) == tap_permissions(source))
        {
            return;
        }
        for source in Arc::make_mut(&mut model.sources) {
            (source.keyword, source.convoke_colors) = tap_permissions(source);
        }
    }

    pub(super) fn announced_obligations<A>(
        &self,
        player: PlayerId,
        payment: &Payment<A>,
    ) -> Vec<Obligation> {
        self.help_obligations(
            player,
            payment.mana().0,
            payment.mana_permissions(),
            payment.life(),
        )
    }

    /// `cost` as the payment model's obligations under `permissions`: delve
    /// pays as much generic as the graveyard holds, convoke and improvise taps
    /// may pay the rest of the generic, and mana pays everything.
    pub(super) fn help_obligations(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        permissions: CostPermissionContext,
        life: u32,
    ) -> Vec<Obligation> {
        let cost = match permissions.help.delve {
            Some(spell) => {
                let cards = self.graveyard_payers(player, spell).len();
                cost.reduced_generic(cards.min(usize::from(u8::MAX)) as u8)
            }
            None => cost.clone(),
        };
        let waterbend = permissions.waterbend.min(cost.generic());
        let keyword = if permissions.help.convoke || permissions.help.improvise {
            cost.generic() - waterbend
        } else {
            0
        };
        let mut obligations = tap_obligations(&cost, keyword, waterbend, permissions.help.convoke);
        obligations.extend(Obligation::life(life));
        obligations
    }

    /// Can `player` pay `cost` with mana and `permissions`, `pool` in their pool?
    pub(crate) fn payable_with_help(
        &self,
        player: PlayerId,
        pool: &Mana,
        cost: &ManaCost,
        permissions: CostPermissionContext,
    ) -> bool {
        self.payable_with_help_and_life(
            player,
            pool,
            cost,
            permissions,
            self.payment_life_owed(player),
        )
    }

    pub(in crate::flow) fn payable_with_help_and_life(
        &self,
        player: PlayerId,
        pool: &Mana,
        cost: &ManaCost,
        permissions: CostPermissionContext,
        life: u32,
    ) -> bool {
        let mut model = self.payment_model(player, None);
        if let Some(permanent) = permissions.reserved_tap {
            model.reserve(SourceReservation::tap(permanent));
        }
        self.mark_help_taps(&mut model, permissions.help);
        model.track_delve(cost, permissions, life);
        model.payable(
            pool.clone(),
            self.help_obligations(player, cost, permissions, life),
            SourcePolicy::All,
        )
    }

    /// What tapping `tapped` may pay of `cost` under `permissions`, before
    /// asking whether the rest stays payable: by convoke an owed pip of one of
    /// its colors or {1}, by improvise {1}, and {1} of the waterbend part while
    /// `permissions.waterbend` is left. Each reason is its own payment, since
    /// the receipt keeps it: a creature that could convoke or waterbend {1}
    /// offers both.
    fn tap_payments(
        &self,
        cost: &ManaCost,
        permissions: CostPermissionContext,
        tapped: PermanentId,
    ) -> Vec<TapPayment> {
        let view = self.characteristics(tapped);
        let (creature, artifact) = (view.is_creature(), view.types.is_artifact());
        let generic = cost.generic() > 0;
        let mut payments = Vec::new();
        let mut pay = |toward: Toward, color: Color| payments.push(TapPayment { toward, color });
        if permissions.help.convoke && creature {
            for color in view.colors.iter() {
                if cost.cost[*color as usize] > 0 {
                    pay(Toward::Convoke, *color);
                }
            }
            if generic {
                pay(Toward::Convoke, Color::Generic);
            }
        }
        if permissions.help.improvise && artifact && generic {
            pay(Toward::Improvise, Color::Generic);
        }
        if permissions.waterbend > 0 && (creature || artifact) && generic {
            pay(Toward::Waterbend, Color::Generic);
        }
        payments
    }

    /// What each of `candidates` may pay of `cost` by one direct tap under
    /// `permissions` (`tap_payments`) such that the rest stays payable. A
    /// candidate with no such payment is left out.
    ///
    /// One decision's candidates are judged against one model, each tap
    /// reserved for its own question and then restored. The debug oracle
    /// asks each question of a model built for it alone.
    pub(crate) fn help_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        permissions: CostPermissionContext,
        candidates: &[PermanentId],
    ) -> Vec<(PermanentId, Vec<TapPayment>)> {
        let _scan = self.board_scan();
        let model_for = |tapped: Option<PermanentId>| {
            let mut model = self.payment_model(player, None);
            for permanent in permissions.reserved_tap.into_iter().chain(tapped) {
                model.reserve(SourceReservation::tap(permanent));
            }
            self.mark_help_taps(&mut model, permissions.help);
            model
        };
        let mut model = model_for(None);
        let pool = self.pooled_mana(player);
        let life = self.payment_life_owed(player);
        // The payments of `asked` that leave the rest payable from `model`.
        let keeping_payable = |model: &PaymentModel, asked: &[TapPayment]| {
            asked
                .iter()
                .copied()
                .filter(|payment| {
                    let next_permissions = permissions.after_tap(*payment);
                    let rest = cost.less_pip(payment.color);
                    model.payable(
                        pool.clone(),
                        self.help_obligations(player, &rest, next_permissions, life),
                        SourcePolicy::All,
                    )
                })
                .collect::<Vec<_>>()
        };
        candidates
            .iter()
            .filter_map(|&tapped| {
                let asked = self.tap_payments(cost, permissions, tapped);
                if asked.is_empty() {
                    return None;
                }
                if model.coupled().is_some() {
                    let options: Vec<_> = asked
                        .into_iter()
                        .filter(|&payment| {
                            let rest = cost.less_pip(payment.color);
                            let next = permissions.after_tap(payment);
                            let mut model = model.clone();
                            model.track_delve(&rest, next, life);
                            model
                                .coupled()
                                .unwrap()
                                .plan(
                                    pool.clone(),
                                    self.help_obligations(player, &rest, next, life),
                                    SourcePolicy::All,
                                    Some(PaymentCandidate::HelpTap(tapped)),
                                )
                                .is_some()
                        })
                        .collect();
                    return (!options.is_empty()).then_some((tapped, options));
                }
                // Tapped, the permanent is no longer a source of the model,
                // so it pays its own pip and not a second one.
                let options = model.with_tapped(tapped, |model| keeping_payable(model, &asked));
                debug_assert_eq!(options, {
                    keeping_payable(&model_for(Some(tapped)), &asked)
                });
                (!options.is_empty()).then_some((tapped, options))
            })
            .collect()
    }
}

impl PaymentModel<'_> {
    /// `ask` of this board with `permanent` tapped for something other than
    /// mana; the board is as it was afterwards.
    fn with_tapped<T>(&mut self, permanent: PermanentId, ask: impl FnOnce(&mut Self) -> T) -> T {
        let index = self
            .sources
            .iter()
            .position(|source| source.permanent == permanent);
        let untouched = index.map(|index| self.sources[index].clone());
        self.reserve(SourceReservation::tap(permanent));
        let answer = ask(self);
        if let (Some(index), Some(source)) = (index, untouched) {
            Arc::make_mut(&mut self.sources)[index] = source;
        }
        answer
    }
}
