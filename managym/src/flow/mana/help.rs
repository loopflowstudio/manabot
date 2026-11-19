// help.rs
// Whether a cost is payable when part of it is paid another way: convoke,
// improvise, delve and waterbend taps beside the mana of `PaymentModel`.

use super::*;
use crate::state::payment::Toward;

impl Game {
    /// Convoke and improvise name which taps may pay generic mana: creatures
    /// and artifacts respectively, where waterbend alone takes either. A
    /// cost with both waterbend and one of them keeps only the keyword's
    /// taps, which may hide a legal waterbend tap and never offers an
    /// illegal one.
    pub(super) fn restrict_help_taps(&self, model: &mut PaymentModel, help: Help) {
        if !help.taps_any_generic() || !model.sources.iter().any(|source| source.waterbend) {
            return;
        }
        for source in Arc::make_mut(&mut model.sources)
            .iter_mut()
            .filter(|source| source.waterbend)
        {
            source.waterbend = self.help_tap_toward(help, source.permanent).is_some();
        }
    }

    /// `cost` as the payment model sees it under `help`: delve pays as much
    /// generic as the graveyard holds, and the second value is the generic
    /// that a tap may pay instead of mana.
    pub(super) fn helped_cost(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        waterbend: u8,
    ) -> (ManaCost, u8) {
        let cost = match help.delve {
            Some(spell) => {
                let cards = self.graveyard_payers(player, spell).len();
                cost.reduced_generic(cards.min(usize::from(u8::MAX)) as u8)
            }
            None => cost.clone(),
        };
        let split = if help.taps_any_generic() {
            cost.generic()
        } else {
            waterbend.min(cost.generic())
        };
        (cost, split)
    }

    /// The creatures that may convoke under `model`, each with its colors.
    fn convokers(&self, model: &PaymentModel) -> Vec<(PermanentId, Vec<Color>)> {
        model
            .sources
            .iter()
            .filter(|source| source.waterbend)
            .map(|source| {
                let colors = &self.characteristics(source.permanent).colors;
                (source.permanent, colors.iter().copied().collect())
            })
            .collect()
    }

    /// The creatures that may pay colored pips under `help` and `model`.
    fn help_convokers(&self, model: &PaymentModel, help: Help) -> Vec<(PermanentId, Vec<Color>)> {
        if help.convoke {
            self.convokers(model)
        } else {
            Vec::new()
        }
    }

    /// Can `cost` be paid from `model` with mana and `help`? Delve and taps
    /// for generic mana are part of the payment model's question; a
    /// convoking creature paying a colored pip is tried by `convoke_search`.
    fn helped_payable(
        &self,
        player: PlayerId,
        model: &mut PaymentModel,
        convokers: &[(PermanentId, Vec<Color>)],
        cost: &ManaCost,
        help: Help,
    ) -> bool {
        let pool = self.pooled_mana(player);
        self.convoke_search(model, cost, convokers, &mut |model, cost| {
            let (relaxed, split) = self.helped_cost(player, cost, help, help.waterbend);
            model
                .payable(
                    pool.clone(),
                    waterbend_obligations(&relaxed, split),
                    SourcePolicy::All,
                )
                .then_some(())
        })
        .is_some()
    }

    /// Can `player` pay `cost` with mana and `help`?
    pub(crate) fn payable_with_help(&self, player: PlayerId, cost: &ManaCost, help: Help) -> bool {
        let mut model = self.payment_model(player, None);
        self.restrict_help_taps(&mut model, help);
        let convokers = self.help_convokers(&model, help);
        self.helped_payable(player, &mut model, &convokers, cost, help)
    }

    /// The pips each of `candidates` may pay of `cost` by convoke or
    /// improvise such that the rest stays payable: each color it has that
    /// the cost still owes, or {1} (`Color::Generic`) when it has none of
    /// them. A colored pip is never worse for the payer than {1}: whatever
    /// mana would have paid that pip can pay the {1} instead. A candidate
    /// with no such pip is left out.
    ///
    /// One decision's candidates are judged against one model, each tap
    /// reserved for its own question and then restored. The debug oracle
    /// asks each question of a model built for it alone.
    pub(crate) fn help_taps(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        help: Help,
        candidates: &[PermanentId],
    ) -> Vec<(PermanentId, Vec<Color>)> {
        let tap = |permanent| SourceReservation {
            permanent,
            tap: true,
            sacrifice: false,
        };
        let mut model = self.payment_model(player, None);
        self.restrict_help_taps(&mut model, help);
        let mut convokers = self.help_convokers(&model, help);
        candidates
            .iter()
            .filter_map(|&tapped| {
                let convokes = self.help_tap_toward(help, tapped)? == Toward::Convoke;
                let mut options: Vec<Color> = self
                    .characteristics(tapped)
                    .colors
                    .iter()
                    .copied()
                    .filter(|color| convokes && cost.cost[*color as usize] > 0)
                    .collect();
                if options.is_empty() && cost.generic() > 0 {
                    options.push(Color::Generic);
                }
                let asked = options.clone();
                let index = model
                    .sources
                    .iter()
                    .position(|source| source.permanent == tapped);
                let untouched = index.map(|index| model.sources[index].clone());
                // The tapped creature pays its own pip, not a second one.
                let own = convokers
                    .iter()
                    .position(|(convoker, _)| *convoker == tapped)
                    .map(|position| (position, convokers.remove(position)));
                model.reserve(tap(tapped));
                options.retain(|color| {
                    self.helped_payable(
                        player,
                        &mut model,
                        &convokers,
                        &cost.less_pip(*color),
                        help,
                    )
                });
                if let (Some(index), Some(source)) = (index, untouched) {
                    Arc::make_mut(&mut model.sources)[index] = source;
                }
                if let Some((position, convoker)) = own {
                    convokers.insert(position, convoker);
                }
                debug_assert_eq!(options, {
                    let mut alone = self.payment_model(player, None);
                    alone.reserve(tap(tapped));
                    self.restrict_help_taps(&mut alone, help);
                    let convokers = self.help_convokers(&alone, help);
                    let mut asked = asked;
                    asked.retain(|color| {
                        let rest = cost.less_pip(*color);
                        self.helped_payable(player, &mut alone, &convokers, &rest, help)
                    });
                    asked
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
                let convokers = self.convokers(model);
                let owed = payment.mana().0.clone();
                self.convoke_search(model, &owed, &convokers, &mut |model, owed| {
                    attempt(model, self.payment_obligations_owing(player, Some(owed)))
                })
            }
            _ => attempt(model, self.payment_obligations(player)),
        }
    }

    /// `attempt` to pay `cost`, then `cost` less each set of colored pips
    /// that `convokers` can pay, those creatures tapped. The search is
    /// exhaustive over those assignments, so it grows with the number of
    /// colored creatures when no attempt succeeds. Creatures are tried in
    /// one order, so each set of taps is visited once.
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
            let Some(index) = model
                .sources
                .iter()
                .position(|source| source.permanent == *permanent)
            else {
                continue;
            };
            for color in colors
                .iter()
                .filter(|color| cost.cost[**color as usize] > 0)
            {
                let untouched = model.sources[index].clone();
                model.reserve(SourceReservation {
                    permanent: *permanent,
                    tap: true,
                    sacrifice: false,
                });
                let found = self.convoke_search(
                    model,
                    &cost.less_pip(*color),
                    &convokers[position + 1..],
                    attempt,
                );
                Arc::make_mut(&mut model.sources)[index] = untouched;
                if found.is_some() {
                    return found;
                }
            }
        }
        None
    }

    /// Can `player` pay `cost`, `waterbend` of its generic with mana or
    /// waterbend taps and the rest with mana?
    pub(crate) fn waterbend_payment_possible(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
    ) -> bool {
        self.payment_model(player, None).payable(
            self.pooled_mana(player),
            waterbend_obligations(cost, waterbend),
            SourcePolicy::All,
        )
    }

    /// The `candidates` whose waterbend tap, beside `also_tapped`, leaves
    /// `cost` payable, without copying the game. A tap reserves its source
    /// and produces no mana. One decision's candidates are judged against one
    /// model, each tap reserved for its own question and then restored. The
    /// debug oracle retains the full fork calculation if future rules add a
    /// tap dependency.
    pub(crate) fn waterbend_taps_keeping_payable(
        &self,
        player: PlayerId,
        cost: &ManaCost,
        waterbend: u8,
        candidates: Vec<PermanentId>,
        also_tapped: Option<PermanentId>,
    ) -> Vec<PermanentId> {
        let tap = |permanent| SourceReservation {
            permanent,
            tap: true,
            sacrifice: false,
        };
        let mut model = self.payment_model(player, None);
        if let Some(permanent) = also_tapped {
            model.reserve(tap(permanent));
        }
        let pool = self.pooled_mana(player);
        let obligations = waterbend_obligations(cost, waterbend);
        candidates
            .into_iter()
            .filter(|&candidate| {
                let index = model
                    .sources
                    .iter()
                    .position(|source| source.permanent == candidate);
                let untouched = index.map(|index| model.sources[index].clone());
                model.reserve(tap(candidate));
                let possible = model.payable(pool.clone(), obligations.clone(), SourcePolicy::All);
                if let (Some(index), Some(source)) = (index, untouched) {
                    Arc::make_mut(&mut model.sources)[index] = source;
                }
                debug_assert_eq!(possible, {
                    let mut probe = self.clone();
                    for permanent in std::iter::once(candidate).chain(also_tapped) {
                        probe.tap_permanent(permanent, false);
                    }
                    probe.waterbend_payment_possible(player, cost, waterbend)
                });
                possible
            })
            .collect()
    }
}
