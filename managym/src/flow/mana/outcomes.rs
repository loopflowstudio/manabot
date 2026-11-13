//! Offline counterfactuals over PaymentModel's fixed-source boundary. This is
//! deliberately opt-in: ordinary play neither enumerates nor stores outcomes.
//! Source subsets supply modeled outputs; the existing payment authority checks
//! affordability and native unit transitions produce exact pools and receipts.

use super::{Mana, ManaCost, ManaSpent, Obligation, PaymentModel, SourcePolicy, SourceReservation};
use std::collections::HashSet;

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
    pub work: usize,
    pub cap_hit: bool,
}

#[derive(Default)]
struct Outcomes {
    untapped: HashSet<Vec<String>>,
    pools: HashSet<Mana>,
    sources: HashSet<(Vec<String>, Mana)>,
    receipts: HashSet<ManaSpent>,
    combined: HashSet<(Vec<String>, Mana, ManaSpent)>,
    identities: HashSet<(Vec<usize>, Mana, ManaSpent)>,
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
    pub(super) fn fixed_outcomes(
        &self,
        pool: Mana,
        cost: &ManaCost,
        cap: usize,
        tail: &[Obligation],
    ) -> PaymentShapeCounts {
        let indices: Vec<usize> = self
            .sources
            .iter()
            .enumerate()
            .filter_map(|(i, source)| (source.untapped && source.fixed.is_some()).then_some(i))
            .collect();
        let mut result = Outcomes::default();
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
            work: result.work,
            cap_hit: result.cap_hit,
        }
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
                result.untapped.insert(capabilities.clone());
                result.pools.insert(pool.clone());
                result.sources.insert((capabilities.clone(), pool.clone()));
                result.receipts.insert(spent);
                result
                    .combined
                    .insert((capabilities.clone(), pool.clone(), spent));
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

#[cfg(test)]
mod tests {
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
        assert!(!outcomes.cap_hit);
    }
}
