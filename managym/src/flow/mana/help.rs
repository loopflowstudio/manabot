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
// creature paying a colored pip is tried by `convoke_search`.

use super::*;
use crate::flow::payment::TapPayment;
use crate::state::payment::Toward;

impl Game {
    /// Mark the sources whose tap `help`'s convoke or improvise lets pay any
    /// generic mana (`ManaSource::keyword`). Every untapped artifact and
    /// creature may still pay the waterbend part.
    pub(super) fn mark_help_taps(&self, model: &mut PaymentModel, help: Help) {
        let keyword = |source: &ManaSource| {
            source.waterbend && {
                let view = self.characteristics(source.permanent);
                help.convoke && view.is_creature() || help.improvise && view.types.is_artifact()
            }
        };
        if model
            .sources
            .iter()
            .all(|source| source.keyword == keyword(source))
        {
            return;
        }
        for source in Arc::make_mut(&mut model.sources) {
            source.keyword = keyword(source);
        }
    }

    /// `cost` as the payment model's obligations under `help`, `waterbend`
    /// of its generic still payable by waterbending: delve pays as much
    /// generic as the graveyard holds, convoke and improvise taps may pay the
    /// rest of the generic, and mana pays everything.
    pub(super) fn help_obligations(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
    ) -> Vec<Obligation> {
        let cost = match help.delve {
            Some(spell) => {
                let cards = self.graveyard_payers(player, spell).len();
                cost.reduced_generic(cards.min(usize::from(u8::MAX)) as u8)
            }
            None => cost.clone(),
        };
        let waterbend = waterbend.min(cost.generic());
        let keyword = if help.convoke || help.improvise {
            cost.generic() - waterbend
        } else {
            0
        };
        tap_obligations(&cost, keyword, waterbend)
    }

    /// The creatures that may pay colored pips under `help` and `model`,
    /// each with its colors: none unless the cost has convoke.
    fn convokers(&self, model: &PaymentModel, help: Help) -> Vec<(PermanentId, Vec<Color>)> {
        if !help.convoke {
            return Vec::new();
        }
        model
            .sources
            .iter()
            .filter(|source| source.keyword)
            .filter_map(|source| {
                let view = self.characteristics(source.permanent);
                view.is_creature()
                    .then(|| (source.permanent, view.colors.iter().copied().collect()))
            })
            .collect()
    }

    /// Can `cost` be paid from `model` with mana and `help`, `waterbend` of
    /// its generic still payable by waterbending?
    #[allow(clippy::too_many_arguments)]
    fn helped_payable(
        &self,
        player: PlayerId,
        pool: &Mana,
        model: &mut PaymentModel,
        convokers: &[(PermanentId, Vec<Color>)],
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
    ) -> bool {
        self.convoke_search(model, cost, convokers, &mut |model, cost| {
            model
                .payable(
                    pool.clone(),
                    self.help_obligations(player, cost, help, waterbend),
                    SourcePolicy::All,
                )
                .then_some(())
        })
        .is_some()
    }

    /// Can `player` pay `cost` with mana and `help`, `pool` in their pool
    /// and `waterbend` of its generic payable by waterbending? `tapped` is a
    /// permanent the operation taps anyway, which pays nothing else.
    pub(crate) fn payable_with_help(
        &self,
        player: PlayerId,
        pool: &Mana,
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
        tapped: Option<PermanentId>,
    ) -> bool {
        let mut model = self.payment_model(player, None);
        if let Some(permanent) = tapped {
            model.reserve(SourceReservation::tap(permanent));
        }
        self.mark_help_taps(&mut model, help);
        let convokers = self.convokers(&model, help);
        self.helped_payable(player, pool, &mut model, &convokers, cost, help, waterbend)
    }

    /// What tapping `tapped` may pay of `cost` under `help`, before asking
    /// whether the rest stays payable: by convoke an owed pip of one of its
    /// colors or {1}, by improvise {1}, and {1} of the waterbend part while
    /// `waterbend` of it is left. Each reason is its own payment, since the
    /// receipt keeps it: a creature that could convoke or waterbend {1}
    /// offers both.
    fn tap_payments(
        &self,
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
        tapped: PermanentId,
    ) -> Vec<TapPayment> {
        let view = self.characteristics(tapped);
        let (creature, artifact) = (view.is_creature(), view.types.is_artifact());
        let generic = cost.generic() > 0;
        let mut payments = Vec::new();
        let mut pay = |toward: Toward, color: Color| payments.push(TapPayment { toward, color });
        if help.convoke && creature {
            for color in view.colors.iter() {
                if cost.cost[*color as usize] > 0 {
                    pay(Toward::Convoke, *color);
                }
            }
            if generic {
                pay(Toward::Convoke, Color::Generic);
            }
        }
        if help.improvise && artifact && generic {
            pay(Toward::Improvise, Color::Generic);
        }
        if waterbend > 0 && (creature || artifact) && generic {
            pay(Toward::Waterbend, Color::Generic);
        }
        payments
    }

    /// What each of `candidates` may pay of `cost` by one direct tap under
    /// `help` (`tap_payments`) such that the rest stays payable, `waterbend`
    /// of its generic still payable by waterbending. `also_tapped` is a
    /// permanent the operation taps anyway. A candidate with no such payment
    /// is left out.
    ///
    /// One decision's candidates are judged against one model, each tap
    /// reserved for its own question and then restored. The debug oracle
    /// asks each question of a model built for it alone.
    pub(crate) fn help_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
        candidates: &[PermanentId],
        also_tapped: Option<PermanentId>,
    ) -> Vec<(PermanentId, Vec<TapPayment>)> {
        let model_for = |tapped: Option<PermanentId>| {
            let mut model = self.payment_model(player, None);
            for permanent in also_tapped.into_iter().chain(tapped) {
                model.reserve(SourceReservation::tap(permanent));
            }
            self.mark_help_taps(&mut model, help);
            let convokers = self.convokers(&model, help);
            (model, convokers)
        };
        let (mut model, convokers) = model_for(None);
        let pool = self.pooled_mana(player);
        // The payments of `asked` that leave the rest payable from `model`.
        let keeping_payable = |model: &mut PaymentModel,
                               convokers: &[(PermanentId, Vec<Color>)],
                               asked: &[TapPayment]| {
            asked
                .iter()
                .copied()
                .filter(|payment| {
                    let left = waterbend - u8::from(payment.toward == Toward::Waterbend);
                    let rest = cost.less_pip(payment.color);
                    self.helped_payable(player, &pool, model, convokers, &rest, help, left)
                })
                .collect::<Vec<_>>()
        };
        candidates
            .iter()
            .filter_map(|&tapped| {
                let asked = self.tap_payments(cost, help, waterbend, tapped);
                if asked.is_empty() {
                    return None;
                }
                // Tapped, the permanent is no longer a source of the model,
                // so it pays its own pip and not a second one.
                let options =
                    model.with_tapped(tapped, |model| keeping_payable(model, &convokers, &asked));
                debug_assert_eq!(options, {
                    let (mut alone, convokers) = model_for(Some(tapped));
                    keeping_payable(&mut alone, &convokers, &asked)
                });
                (!options.is_empty()).then_some((tapped, options))
            })
            .collect()
    }

    /// `attempt` against the obligations `player` has pending. A convoke
    /// payment in progress is asked as it stands and then with each set of
    /// creatures paying colored pips of it, since the payment model itself
    /// only taps for generic mana.
    pub(super) fn while_paying<T>(
        &self,
        player: PlayerId,
        model: &mut PaymentModel,
        attempt: &mut dyn FnMut(&PaymentModel, Vec<Obligation>) -> Option<T>,
    ) -> Option<T> {
        match self.paying() {
            Some((payer, payment, CostComponent::Mana { .. }))
                if payer == player && payment.help().convoke =>
            {
                let convokers = self.convokers(model, payment.help());
                let owed = payment.mana().0.clone();
                self.convoke_search(model, &owed, &convokers, &mut |model, owed| {
                    attempt(model, self.payment_obligations_owing(player, Some(owed)))
                })
            }
            _ => attempt(model, self.payment_obligations(player)),
        }
    }

    /// `attempt` to pay `cost`, then `cost` less each set of colored pips
    /// that `convokers` can pay, those creatures tapped. A convoker the
    /// model already has tapped pays nothing. The search is exhaustive over
    /// those assignments, so it grows with the number of colored creatures
    /// when no attempt succeeds. Creatures are tried in one order, so each
    /// set of taps is visited once.
    fn convoke_search<T>(
        &self,
        model: &mut PaymentModel,
        cost: &ManaCost,
        convokers: &[(PermanentId, Vec<Color>)],
        attempt: &mut dyn FnMut(&PaymentModel, &ManaCost) -> Option<T>,
    ) -> Option<T> {
        if let Some(found) = attempt(model, cost) {
            return Some(found);
        }
        for (position, (permanent, colors)) in convokers.iter().enumerate() {
            if !model
                .sources
                .iter()
                .any(|source| source.permanent == *permanent && source.keyword)
            {
                continue;
            }
            for color in colors
                .iter()
                .filter(|color| cost.cost[**color as usize] > 0)
            {
                let found = model.with_tapped(*permanent, |model| {
                    self.convoke_search(
                        model,
                        &cost.less_pip(*color),
                        &convokers[position + 1..],
                        attempt,
                    )
                });
                if found.is_some() {
                    return found;
                }
            }
        }
        None
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
