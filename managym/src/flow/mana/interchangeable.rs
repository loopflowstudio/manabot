//! Manual payment may exchange two taps only when the rules cannot tell
//! which source stayed untapped. The affordability search has a weaker
//! equivalence: it only needs to know whether the mana can be produced.

use super::{Game, Mana, ManaCost, ManaSource, Obligation, PaymentModel, UsefulTap};

impl Game {
    /// Every tap supplies exactly one unit. Removing a representative tests
    /// whether every member of its class is needed. Once those taps are
    /// accounted for, only one class can supply the balance unambiguously.
    ///
    /// Checks cheap profile optionality before scanning board bindings or
    /// computing layered characteristics: for a determined count to exist,
    /// partitioning `taps` by payment profile must yield one optional profile
    /// of size `>= spare + 1` (with every other profile mandatory), and every
    /// member of that optional profile must be mutually interchangeable under
    /// the shared rules-state predicate (`Game::all_interchangeable`).
    pub(super) fn determined_source_counts<'a>(
        &self,
        model: &PaymentModel,
        pool: &Mana,
        cost: &ManaCost,
        needed: usize,
        taps: Vec<UsefulTap<'a>>,
        need_taps: bool,
    ) -> Option<Vec<UsefulTap<'a>>> {
        if self.state.cast_caller.is_some() {
            return None;
        }
        let tap_count = taps.len();
        let spare = tap_count.checked_sub(needed)?;
        if spare == 0 {
            return None;
        }
        // The unique optional class leaves `spare` untapped members and
        // contributes at least one tap (`count >= 1`), so it must have at
        // least `spare + 1` interchangeable members sharing a payment profile.
        let min_opt = spare + 1;
        let mut profile_groups: Vec<Vec<usize>> = Vec::new();
        for (index, tap) in taps.iter().enumerate() {
            if let Some(group) = profile_groups
                .iter_mut()
                .find(|group| taps[group[0]].source.same_payment_profile(tap.source))
            {
                group.push(index);
            } else {
                profile_groups.push(vec![index]);
            }
        }
        if !profile_groups.iter().any(|group| group.len() >= min_opt) {
            return None;
        }

        // Fast arithmetic affordability check when every source is fixed-only.
        let fixed_total = model
            .sources
            .iter()
            .all(|s| s.abilities.is_empty())
            .then(|| model.fixed_total(pool));
        let payable_without = |source: &ManaSource| -> bool {
            let obligations = || vec![Obligation::mana(cost.clone())];
            if let Some(total) = &fixed_total {
                let mut without = total.clone();
                if let Some(output) = &source.fixed {
                    without.spend(output);
                }
                let fast = without.can_pay(cost);
                debug_assert_eq!(
                    fast,
                    model.payable_after_tap(source.permanent, pool.clone(), obligations()),
                    "direct determined_source_counts differs from model"
                );
                fast
            } else {
                model.payable_after_tap(source.permanent, pool.clone(), obligations())
            }
        };

        // Check small (mandatory-candidate) groups first so a common board
        // like `Forest + Plains × 3` exits on its first arithmetic check.
        let mut optional: Option<Vec<usize>> = None;
        for group in profile_groups.iter().filter(|group| group.len() < min_opt) {
            if payable_without(taps[group[0]].source) {
                return None;
            }
        }
        for group in profile_groups
            .into_iter()
            .filter(|group| group.len() >= min_opt)
        {
            if payable_without(taps[group[0]].source) {
                if optional.replace(group).is_some() {
                    return None;
                }
            }
        }
        let opt_indices = optional?;

        // Every member of the unique optional profile must be mutually
        // interchangeable under the shared fail-closed predicate.
        let opt_objects: Vec<_> = opt_indices.iter().map(|&i| taps[i].source.object).collect();
        if !self.all_interchangeable(&opt_objects) {
            return None;
        }
        if !need_taps {
            return Some(Vec::new());
        }

        // `opt_indices.len() >= spare + 1`, so `1 <= count < opt_indices.len()`.
        let count = opt_indices.len() - spare;
        let skip = &opt_indices[count..];
        let mut result: Vec<_> = taps
            .into_iter()
            .enumerate()
            .filter_map(|(index, tap)| (!skip.contains(&index)).then_some(tap))
            .collect();
        result.sort_unstable_by_key(|tap| tap.source.permanent);
        Some(result)
    }
}
