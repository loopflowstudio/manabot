//! Offline counterfactuals over PaymentModel's fixed-source boundary. This is
//! deliberately opt-in: ordinary play neither enumerates nor stores outcomes.
//! Source subsets supply modeled outputs; the existing payment authority checks
//! affordability and native unit transitions produce exact pools and receipts.

use super::{Mana, ManaCost, ManaSpent, Obligation, PaymentModel, SourcePolicy, SourceReservation};
use std::collections::{HashMap, HashSet};

/// The most follow-up costs one payment is judged against.
pub(crate) const FOLLOWUPS: usize = 64;

/// What a payment's source choice is judged against
/// (`Game::payment_followups`): the distinct mana costs of the payer's other
/// castable cards in hand, then of the payer's battlefield abilities, at most
/// `FOLLOWUPS` together. Bit `i` of `hand` marks cost `i` as a hand card and
/// of `instant` as one castable at instant timing. `tapping[i]` is the index
/// in the model's sources of the permanent an ability taps as its own cost:
/// that source cannot pay for it, and a payment tapping it loses the ability.
/// `automatic` is the fixed sources the automatic payer tapped, by index.
#[derive(Clone, Debug, Default)]
pub(crate) struct PaymentProbe {
    pub followups: Vec<ManaCost>,
    pub hand: u64,
    pub instant: u64,
    pub tapping: Vec<Option<usize>>,
    pub automatic: Option<Vec<usize>>,
    /// Permanents whose activated abilities were read to build this list:
    /// only those that have any.
    pub ability_sources: usize,
}

impl PaymentProbe {
    /// The source follow-up `index` taps for itself, if any.
    pub(super) fn tapped_by(&self, index: usize) -> Option<usize> {
        self.tapping.get(index).copied().flatten()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PaymentShapeCapture {
    pub cap: usize,
    pub rows: Vec<PaymentShapeCounts>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct PaymentShapeCounts {
    pub spend_offers: Vec<usize>,
    pub cost: u8,
    pub cast: bool,
    pub saved_obligations: usize,
    pub fixed_sources: usize,
    pub activated_sources: usize,
    pub untapped_outcomes: usize,
    pub pool_outcomes: usize,
    pub source_outcomes: usize,
    pub receipt_outcomes: usize,
    pub outcomes: usize,
    pub identity_outcomes: usize,
    pub non_redundant_outcomes: usize,
    pub non_redundant_identity_outcomes: usize,
    pub work: usize,
    pub cap_hit: bool,
    /// Distinct follow-up costs probed: hand cards and battlefield abilities
    /// (`Game::payment_followups`). A follow-up is affordable after an
    /// outcome when its mana cost alone is payable from the sources left
    /// untapped and the remaining pool, and the outcome did not tap the
    /// source its own {T} cost needs; each follow-up is tested by itself.
    pub followups: usize,
    /// How many of them are hand cards, the only follow-ups in rows written
    /// before abilities were probed.
    pub hand_followups: usize,
    /// The automatic source set was reached before the cap.
    pub automatic_known: bool,
    /// That set taps a source the cost does not need.
    pub automatic_redundant: bool,
    /// Follow-ups affordable after the automatic source set, under its best
    /// pooled-unit spending.
    pub automatic_affordable: usize,
    /// The most follow-ups any non-redundant outcome leaves affordable.
    pub best_affordable: usize,
    /// Some non-redundant outcome affords a follow-up the automatic set does
    /// not; `_hand` restricts that to hand cards and `_instant` to
    /// instant-timing hand cards.
    pub stranded: bool,
    pub stranded_hand: bool,
    pub stranded_instant: bool,
    /// Maximal affordable sets among non-redundant outcomes. More than one
    /// means no outcome affords everything another does: keeping one
    /// follow-up gives up another.
    pub maximal_affordable_sets: usize,
    pub maximal_hand_sets: usize,
    pub maximal_instant_sets: usize,
}

#[derive(Default)]
struct Outcomes {
    probe: PaymentProbe,
    /// Affordable follow-ups by exact untapped sources and remaining pool.
    affordable: HashMap<(Vec<usize>, Mana), u64>,
    non_redundant_affordable: HashSet<u64>,
    automatic_affordable: u64,
    automatic_seen: bool,
    automatic_redundant: bool,
    untapped: HashSet<Vec<String>>,
    pools: HashSet<Mana>,
    sources: HashSet<(Vec<String>, Mana)>,
    receipts: HashSet<ManaSpent>,
    combined: HashSet<(Vec<String>, Mana, ManaSpent)>,
    identities: HashSet<(Vec<usize>, Mana, ManaSpent)>,
    non_redundant: HashSet<(Vec<String>, Mana, ManaSpent)>,
    non_redundant_identities: HashSet<(Vec<usize>, Mana, ManaSpent)>,
    work: usize,
    cap_hit: bool,
}

impl Outcomes {
    fn tick(&mut self, cap: usize) -> bool {
        if self.work == cap {
            self.cap_hit = true;
            return false;
        }
        self.work += 1;
        true
    }
}

impl PaymentModel {
    /// Enumerate fixed-source subsets, including deliberate surplus production.
    /// Equal fixed outputs collapse by capability; identity_outcomes separately
    /// exposes what that collapse loses. Activated abilities remain outside this
    /// boundary, just as they do for produce_mana's FixedOnly witness.
    #[cfg(test)]
    pub(super) fn fixed_outcomes(
        &self,
        pool: Mana,
        cost: &ManaCost,
        cap: usize,
        tail: &[Obligation],
    ) -> PaymentShapeCounts {
        self.probed_outcomes(pool, cost, cap, tail, PaymentProbe::default())
    }

    /// `fixed_outcomes`, also testing each non-redundant outcome and the
    /// automatic source set against `probe`'s follow-up costs.
    pub(super) fn probed_outcomes(
        &self,
        pool: Mana,
        cost: &ManaCost,
        cap: usize,
        tail: &[Obligation],
        mut probe: PaymentProbe,
    ) -> PaymentShapeCounts {
        probe.followups.truncate(FOLLOWUPS);
        if let Some(automatic) = &mut probe.automatic {
            automatic.sort_unstable();
        }
        let followups = probe.followups.len();
        let (hand, instant) = (probe.hand, probe.instant);
        let indices: Vec<usize> = self
            .sources
            .iter()
            .enumerate()
            .filter_map(|(i, source)| (source.untapped && source.fixed.is_some()).then_some(i))
            .collect();
        let mut result = Outcomes {
            probe,
            ..Outcomes::default()
        };
        self.outcome_subsets(
            &indices,
            0,
            &mut Vec::new(),
            pool,
            cost,
            cap,
            &mut result,
            tail,
        );
        PaymentShapeCounts {
            spend_offers: Vec::new(),
            cost: cost.mana_value,
            cast: matches!(cost.purpose, crate::state::mana::SpendingContext::Cast(_)),
            saved_obligations: tail.len(),
            fixed_sources: indices.len(),
            activated_sources: self
                .sources
                .iter()
                .filter(|source| !source.abilities.is_empty())
                .count(),
            untapped_outcomes: result.untapped.len(),
            pool_outcomes: result.pools.len(),
            source_outcomes: result.sources.len(),
            receipt_outcomes: result.receipts.len(),
            outcomes: result.combined.len(),
            identity_outcomes: result.identities.len(),
            non_redundant_outcomes: result.non_redundant.len(),
            non_redundant_identity_outcomes: result.non_redundant_identities.len(),
            work: result.work,
            cap_hit: result.cap_hit,
            followups,
            hand_followups: (0..followups).filter(|i| hand >> i & 1 == 1).count(),
            automatic_known: result.automatic_seen,
            automatic_redundant: result.automatic_redundant,
            automatic_affordable: result.automatic_affordable.count_ones() as usize,
            best_affordable: result
                .non_redundant_affordable
                .iter()
                .map(|set| set.count_ones() as usize)
                .max()
                .unwrap_or(0),
            stranded: result
                .non_redundant_affordable
                .iter()
                .any(|set| set & !result.automatic_affordable != 0),
            stranded_hand: result
                .non_redundant_affordable
                .iter()
                .any(|set| set & hand & !result.automatic_affordable != 0),
            stranded_instant: result
                .non_redundant_affordable
                .iter()
                .any(|set| set & instant & !result.automatic_affordable != 0),
            maximal_affordable_sets: maximal_sets(&result.non_redundant_affordable, u64::MAX),
            maximal_hand_sets: maximal_sets(&result.non_redundant_affordable, hand),
            maximal_instant_sets: maximal_sets(&result.non_redundant_affordable, instant),
        }
    }

    /// Bit `i` is set when follow-up `i` alone is payable from `after`'s
    /// untapped sources and `pool`. An ability that taps its own source
    /// needs it still untapped in `after` and takes no mana from it.
    fn affordable_followups(&self, after: &PaymentModel, pool: &Mana, probe: &PaymentProbe) -> u64 {
        let payable = |board: &PaymentModel, cost: &ManaCost| {
            board.payable(
                pool.clone(),
                vec![Obligation::Pay {
                    cost: cost.clone(),
                    waterbend: false,
                }],
                SourcePolicy::All,
            )
        };
        probe
            .followups
            .iter()
            .enumerate()
            .filter(|(i, cost)| match probe.tapped_by(*i) {
                None => payable(after, cost),
                Some(source) if !after.sources[source].untapped => false,
                Some(source) => {
                    let mut board = PaymentModel {
                        sources: after.sources.clone(),
                        life: after.life,
                    };
                    board.reserve(SourceReservation {
                        permanent: self.sources[source].permanent,
                        tap: true,
                        sacrifice: true,
                    });
                    payable(&board, cost)
                }
            })
            .fold(0, |set, (i, _)| set | 1 << i)
    }

    #[allow(clippy::too_many_arguments)]
    fn outcome_subsets(
        &self,
        indices: &[usize],
        position: usize,
        selected: &mut Vec<usize>,
        pool: Mana,
        cost: &ManaCost,
        cap: usize,
        result: &mut Outcomes,
        tail: &[Obligation],
    ) {
        if !result.tick(cap) {
            return;
        }
        if position < indices.len() {
            self.outcome_subsets(
                indices,
                position + 1,
                selected,
                pool.clone(),
                cost,
                cap,
                result,
                tail,
            );
            if result.cap_hit || cost.mana_value == 0 {
                return;
            }
            let index = indices[position];
            let mut next = pool;
            next.add(self.sources[index].fixed.as_ref().unwrap());
            selected.push(index);
            self.outcome_subsets(
                indices,
                position + 1,
                selected,
                next,
                cost,
                cap,
                result,
                tail,
            );
            selected.pop();
            return;
        }
        // FixedOnly with no available sources asks the single payment authority
        // whether the selected pooled supply suffices, including restrictions.
        let pooled = PaymentModel {
            sources: Vec::new(),
            life: self.life,
        };
        if !pooled.payable(
            pool.clone(),
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }],
            SourcePolicy::FixedOnly,
        ) {
            return;
        }
        let available: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|i| !selected.contains(i))
            .collect();
        let mut capabilities: Vec<String> = available
            .iter()
            .map(|i| serde_json::to_string(self.sources[*i].fixed.as_ref().unwrap()).unwrap())
            .collect();
        capabilities.sort();
        let mut after = PaymentModel {
            sources: self.sources.clone(),
            life: self.life,
        };
        for index in selected.iter() {
            after.reserve(SourceReservation {
                permanent: self.sources[*index].permanent,
                tap: true,
                sacrifice: false,
            });
        }
        // record_mana_payment's attribution, classified once per subset before
        // its receipts. Sharing the traversal and cap preserves the
        // surplus-inclusive lower bounds.
        let redundant = pool.usable_total(cost) > u16::from(cost.mana_value)
            && selected.iter().any(|index| {
                let output = self.sources[*index].fixed.as_ref().unwrap();
                after.is_surplus(&pool, output, cost, tail)
            });
        let automatic = result.probe.automatic.as_deref() == Some(selected.as_slice());
        if automatic {
            result.automatic_seen = true;
            result.automatic_redundant = redundant;
        }
        let mut pending = vec![(pool, cost.clone(), ManaSpent::default())];
        let mut visited = HashSet::new();
        while let Some((pool, remaining, spent)) = pending.pop() {
            if !result.tick(cap) {
                return;
            }
            if !visited.insert((pool.clone(), remaining.clone(), spent)) {
                continue;
            }
            if remaining.mana_value == 0 {
                if !tail.is_empty()
                    && !after.payable(pool.clone(), tail.to_vec(), SourcePolicy::All)
                {
                    continue;
                }
                if (automatic || !redundant) && !result.probe.followups.is_empty() {
                    let key = (available.clone(), pool.clone());
                    let set = match result.affordable.get(&key) {
                        Some(set) => *set,
                        None => {
                            let set = self.affordable_followups(&after, &pool, &result.probe);
                            result.affordable.insert(key, set);
                            set
                        }
                    };
                    if automatic {
                        result.automatic_affordable |= set;
                    }
                    if !redundant {
                        result.non_redundant_affordable.insert(set);
                    }
                }
                result.untapped.insert(capabilities.clone());
                result.pools.insert(pool.clone());
                result.sources.insert((capabilities.clone(), pool.clone()));
                result.receipts.insert(spent);
                result
                    .combined
                    .insert((capabilities.clone(), pool.clone(), spent));
                if !redundant {
                    result
                        .non_redundant
                        .insert((capabilities.clone(), pool.clone(), spent));
                    result.non_redundant_identities.insert((
                        available.clone(),
                        pool.clone(),
                        spent,
                    ));
                }
                result.identities.insert((available.clone(), pool, spent));
                continue;
            }
            for choice in pool.spending_choices(&remaining) {
                let mut next = pool.clone();
                let receipt = next.spend(&choice.unit, &remaining.purpose);
                if !next.can_pay(&choice.remaining) {
                    continue;
                }
                let mut total = spent;
                for (sum, amount) in total.amounts.iter_mut().zip(receipt.amounts) {
                    *sum += amount;
                }
                total.entry_counters += receipt.entry_counters;
                pending.push((next, choice.remaining, total));
            }
        }
    }
}

/// Sets, restricted to `within`, that no other restricted set strictly contains.
fn maximal_sets(sets: &HashSet<u64>, within: u64) -> usize {
    let restricted: HashSet<u64> = sets.iter().map(|set| set & within).collect();
    restricted
        .iter()
        .filter(|set| {
            !restricted
                .iter()
                .any(|other| other != *set && *other & **set == **set)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::PaymentProbe;
    use crate::flow::mana::{tests::forests, Obligation, PaymentModel};
    use crate::state::mana::{Mana, ManaCost};

    #[test]
    fn fixed_outcomes_collapse_equal_lands_but_retain_identity_and_cap() {
        let model = forests(3);
        let outcomes = model.fixed_outcomes(Mana::default(), &ManaCost::parse("G"), 1000, &[]);
        // Pay using one, two or three Forests, retaining 0, 1 or 2 floating G.
        assert_eq!(outcomes.outcomes, 3);
        assert_eq!(outcomes.source_outcomes, 3);
        assert_eq!(outcomes.receipt_outcomes, 1);
        assert_eq!(outcomes.identity_outcomes, 7);
        assert_eq!(outcomes.non_redundant_outcomes, 1);
        assert_eq!(outcomes.non_redundant_identity_outcomes, 3);
        assert!(!outcomes.cap_hit);
        assert!(
            model
                .fixed_outcomes(Mana::default(), &ManaCost::parse("G"), 2, &[])
                .cap_hit
        );
    }

    #[test]
    fn fixed_outcomes_keep_saved_callers_payable() {
        let model = forests(2);
        let tail = [Obligation::Pay {
            cost: ManaCost::parse("GG"),
            waterbend: false,
        }];
        let outcomes = model.fixed_outcomes(Mana::default(), &ManaCost::parse("G"), 1000, &tail);
        assert_eq!(outcomes.outcomes, 0);
        assert_eq!(outcomes.non_redundant_outcomes, 0);
        assert!(!outcomes.cap_hit);
    }

    #[test]
    fn fixed_outcomes_distinguish_pool_and_receipt_colors() {
        let model = PaymentModel {
            sources: vec![],
            life: 20,
        };
        let outcomes = model.fixed_outcomes(Mana::parse("RG"), &ManaCost::parse("1"), 100, &[]);
        assert_eq!(outcomes.outcomes, 2);
        assert_eq!(outcomes.source_outcomes, 2);
        assert_eq!(outcomes.receipt_outcomes, 2);
        assert_eq!(outcomes.non_redundant_outcomes, 2);
        assert!(!outcomes.cap_hit);
    }

    #[test]
    fn non_redundant_outcomes_retain_indivisible_surplus_and_zero_cost_pool() {
        let mut model = forests(2);
        for source in &mut model.sources {
            source.fixed = Some(Mana::parse("GG"));
        }
        let outcomes = model.fixed_outcomes(Mana::default(), &ManaCost::parse("G"), 1000, &[]);
        assert_eq!(outcomes.outcomes, 2);
        assert_eq!(outcomes.non_redundant_outcomes, 1);
        assert_eq!(outcomes.non_redundant_identity_outcomes, 2);
        let free = model.fixed_outcomes(Mana::parse("RG"), &ManaCost::parse("0"), 1000, &[]);
        assert_eq!(free.non_redundant_outcomes, 1);
        assert_eq!(free.non_redundant_identity_outcomes, 1);
        let floating = model.fixed_outcomes(Mana::parse("G"), &ManaCost::parse("G"), 1000, &[]);
        assert_eq!(floating.non_redundant_outcomes, 1);
        assert_eq!(floating.non_redundant_identity_outcomes, 1);
    }

    #[test]
    fn non_redundant_outcomes_preserve_saved_obligations_on_post_tap_board() {
        let model = forests(3);
        let tail = [Obligation::Pay {
            cost: ManaCost::parse("GG"),
            waterbend: false,
        }];
        let outcomes = model.fixed_outcomes(Mana::default(), &ManaCost::parse("G"), 1000, &tail);
        // All three units are needed across the current and saved costs.
        // Removing any tapped output while retaining the post-tap board makes
        // the saved cost infeasible, even when the current cost still fits.
        assert_eq!(outcomes.non_redundant_outcomes, 3);
        assert_eq!(outcomes.non_redundant_identity_outcomes, 7);
        assert!(!outcomes.cap_hit);
    }

    fn forests_and_mountain() -> PaymentModel {
        let mut model = forests(3);
        model.sources[2].fixed = Some(Mana::parse("R"));
        model
    }

    #[test]
    fn automatic_sources_can_strand_a_follow_up_another_payment_keeps() {
        let model = forests_and_mountain();
        let probe = |automatic: Vec<usize>| PaymentProbe {
            followups: vec![ManaCost::parse("R")],
            hand: 1,
            instant: 1,
            automatic: Some(automatic),
            ..PaymentProbe::default()
        };
        // Forest and Mountain for {1}{G} leave no red; both Forests keep it.
        let stranded = model.probed_outcomes(
            Mana::default(),
            &ManaCost::parse("1G"),
            1000,
            &[],
            probe(vec![2, 0]),
        );
        assert!(stranded.automatic_known && !stranded.automatic_redundant);
        assert_eq!(stranded.followups, 1);
        assert_eq!(stranded.automatic_affordable, 0);
        assert_eq!(stranded.best_affordable, 1);
        assert!(stranded.stranded && stranded.stranded_instant);
        assert_eq!(stranded.maximal_affordable_sets, 1);

        let kept = model.probed_outcomes(
            Mana::default(),
            &ManaCost::parse("1G"),
            1000,
            &[],
            probe(vec![0, 1]),
        );
        assert_eq!(kept.automatic_affordable, 1);
        assert!(!kept.stranded && !kept.stranded_instant);

        // All three tapped: redundant, but the floating red still pays.
        let floated = model.probed_outcomes(
            Mana::default(),
            &ManaCost::parse("1G"),
            1000,
            &[],
            probe(vec![0, 1, 2]),
        );
        assert!(floated.automatic_redundant);
        assert_eq!(floated.automatic_affordable, 1);
    }

    #[test]
    fn follow_ups_no_single_payment_keeps_are_separate_maximal_sets() {
        let mut model = forests(2);
        model.sources[1].fixed = Some(Mana::parse("R"));
        let outcomes = model.probed_outcomes(
            Mana::default(),
            &ManaCost::parse("1"),
            1000,
            &[],
            PaymentProbe {
                followups: vec![ManaCost::parse("G"), ManaCost::parse("R")],
                hand: 0b11,
                instant: 0b10,
                automatic: Some(vec![0]),
                ..PaymentProbe::default()
            },
        );
        // Tapping the Forest keeps {R}; tapping the Mountain keeps {G}.
        assert_eq!(outcomes.maximal_affordable_sets, 2);
        assert_eq!(outcomes.maximal_instant_sets, 1);
        assert_eq!(outcomes.automatic_affordable, 1);
        assert!(outcomes.stranded && !outcomes.stranded_instant);
    }
}
