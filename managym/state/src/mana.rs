// mana.rs
// Mana costs and pools, including exact two-color hybrid payment.

use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Blue = 1,
    Black = 2,
    Red = 3,
    Green = 4,
    Colorless = 5,
    Generic = 6,
}

impl Color {
    pub fn symbol(self) -> &'static str {
        match self {
            Color::White => "W",
            Color::Blue => "U",
            Color::Black => "B",
            Color::Red => "R",
            Color::Green => "G",
            Color::Colorless => "C",
            Color::Generic => "*",
        }
    }
}

pub type Colors = BTreeSet<Color>;

/// A positive spending predicate. Unknown subtype names remain predicates;
/// they cannot disappear into an unrestricted pool. Alternatives are ORed.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum SpendPredicate {
    CastSubtype(String),
    ActivateSubtype(String),
}

/// One restriction class carried by mana until it is spent or lost (CR 106.6).
/// The empty class admits no payment. Unrestricted mana lives in `Mana::mana`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ManaRestriction {
    pub alternatives: BTreeSet<SpendPredicate>,
}

impl ManaRestriction {
    pub fn new<'a>(
        spell_subtypes: impl IntoIterator<Item = &'a str>,
        ability_source_subtypes: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        Self {
            alternatives: spell_subtypes
                .into_iter()
                .map(|name| SpendPredicate::CastSubtype(name.into()))
                .chain(
                    ability_source_subtypes
                        .into_iter()
                        .map(|name| SpendPredicate::ActivateSubtype(name.into())),
                )
                .collect(),
        }
    }

    pub fn spell<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::new(subtypes, [])
    }

    pub fn ability<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::new([], subtypes)
    }

    pub fn allows(&self, context: &SpendingContext) -> bool {
        self.alternatives
            .iter()
            .any(|predicate| match (predicate, context) {
                (
                    SpendPredicate::CastSubtype(name),
                    SpendingContext::Cast {
                        characteristics, ..
                    },
                )
                | (
                    SpendPredicate::ActivateSubtype(name),
                    SpendingContext::Activate {
                        characteristics, ..
                    },
                ) => characteristics.subtypes.contains(name),
                _ => false,
            })
    }

    fn breadth(&self) -> usize {
        self.alternatives.len()
    }

    /// Compatibility projection and presentation use the same typed predicates.
    pub fn subtypes(&self, activation: bool) -> Vec<String> {
        self.alternatives
            .iter()
            .filter_map(|predicate| match predicate {
                SpendPredicate::CastSubtype(name) if !activation => Some(name.clone()),
                SpendPredicate::ActivateSubtype(name) if activation => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        let spells = self.subtypes(false);
        let abilities = self.subtypes(true);
        if !spells.is_empty() {
            parts.push(format!("{} spells", spells.join(" or ")));
        }
        if !abilities.is_empty() {
            parts.push(format!("abilities of {} sources", abilities.join(" or ")));
        }
        if parts.is_empty() {
            return "no purposes".into();
        }
        parts.join(" or ")
    }
}

/// Characteristics announced for a spell or read from an activation's source.
/// These are a payment snapshot, not another authority for layered objects.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct SpendingCharacteristics {
    pub subtypes: BTreeSet<String>,
    pub types: BTreeSet<crate::card::CardType>,
    pub colors: Colors,
}

/// Why mana is being spent. Ordinary resolving payments never inherit a
/// spell's casting permission. Activations retain the exact source and zone.
/// Announced characteristics are immutable snapshots shared by payment costs,
/// witness branches and game forks; sharing never follows live characteristics.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum SpendingContext {
    #[default]
    Resolving,
    Cast {
        characteristics: Arc<SpendingCharacteristics>,
    },
    Activate {
        source: Option<crate::game_object::ObjectRef>,
        zone: crate::zone::ZoneType,
        characteristics: Arc<SpendingCharacteristics>,
    },
}

impl SpendingContext {
    pub fn spell<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::Cast {
            characteristics: Arc::new(SpendingCharacteristics {
                subtypes: subtypes.into_iter().map(String::from).collect(),
                ..Default::default()
            }),
        }
    }

    pub fn ability<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::Activate {
            source: None,
            zone: crate::zone::ZoneType::Battlefield,
            characteristics: Arc::new(SpendingCharacteristics {
                subtypes: subtypes.into_iter().map(String::from).collect(),
                ..Default::default()
            }),
        }
    }

    pub fn is_ordinary(&self) -> bool {
        matches!(self, Self::Resolving)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ManaCost {
    /// The payment's purpose, which decides the restricted mana it may use:
    /// set for a cast spell or an activation, never for resolving effects.
    #[serde(skip_serializing_if = "SpendingContext::is_ordinary")]
    pub purpose: SpendingContext,
    pub cost: [u8; 7],
    pub mana_value: u8,
    /// Each two-color hybrid pip is paid with exactly one of its colors.
    /// Omission keeps frozen ordinary-cost manifests byte-for-byte stable.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hybrid: Vec<[Color; 2]>,
    /// Number of {X} symbols. X is 0 outside the stack (CR 202.3e), so it
    /// adds nothing to `mana_value` until [`ManaCost::with_x`] fixes it.
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: u8,
}

impl ManaCost {
    pub fn parse(text: &str) -> Self {
        let mut out = ManaCost::default();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.peek().copied() {
            if ch.is_ascii_digit() {
                let mut value: u16 = 0;
                while let Some(digit) = chars.peek().copied() {
                    if !digit.is_ascii_digit() {
                        break;
                    }
                    value = value * 10 + (digit as u16 - b'0' as u16);
                    chars.next();
                }
                out.cost[Color::Generic as usize] = value.min(u8::MAX as u16) as u8;
                continue;
            }

            if ch == '{' {
                chars.next();
                let mut symbol = String::new();
                for next in chars.by_ref() {
                    if next == '}' {
                        break;
                    }
                    symbol.push(next);
                }
                if let Some((left, right)) = symbol.split_once('/') {
                    let color = |s: &str| match s {
                        "W" => Color::White,
                        "U" => Color::Blue,
                        "B" => Color::Black,
                        "R" => Color::Red,
                        "G" => Color::Green,
                        _ => panic!("unsupported hybrid mana symbol: {symbol}"),
                    };
                    out.hybrid.push([color(left), color(right)]);
                } else {
                    let part = Self::parse(&symbol);
                    for (slot, amount) in out.cost.iter_mut().zip(part.cost) {
                        *slot = slot.saturating_add(amount);
                    }
                    out.x = out.x.saturating_add(part.x);
                }
                continue;
            }

            if ch == 'X' {
                out.x += 1;
                chars.next();
                continue;
            }
            let slot = match ch {
                'W' => Color::White,
                'U' => Color::Blue,
                'B' => Color::Black,
                'R' => Color::Red,
                'G' => Color::Green,
                'C' => Color::Colorless,
                _ => panic!("invalid mana symbol: {ch}"),
            };
            out.cost[slot as usize] += 1;
            chars.next();
        }

        out.update_mana_value();
        out
    }

    /// Component-wise sum of two costs (base + kicker).
    pub fn plus(&self, other: &ManaCost) -> ManaCost {
        let mut out = self.clone();
        for i in 0..out.cost.len() {
            out.cost[i] = out.cost[i].saturating_add(other.cost[i]);
        }
        out.hybrid.extend_from_slice(&other.hybrid);
        out.x = out.x.saturating_add(other.x);
        out.update_mana_value();
        out
    }

    /// This cost less `reduction` (CR 118.7b, 118.7c): each colored symbol removes
    /// one of that color, or one generic when none of that color is left;
    /// generic in `reduction` removes generic.
    pub fn reduced_by(&self, reduction: &ManaCost) -> ManaCost {
        let mut out = self.clone();
        let generic = Color::Generic as usize;
        for (slot, &count) in reduction.cost.iter().enumerate() {
            let colored = if slot == generic {
                0
            } else {
                out.cost[slot].min(count)
            };
            out.cost[slot] -= colored;
            out.cost[generic] = out.cost[generic].saturating_sub(count - colored);
        }
        out.update_mana_value();
        out
    }

    /// Reduce the generic component by `amount`, floored at zero
    /// (affinity-style cost reduction, waterbend payments).
    pub fn reduced_generic(&self, amount: u8) -> ManaCost {
        self.with_generic(self.generic().saturating_sub(amount))
    }

    /// This cost with X announced as `value` (CR 601.2b): each {X} becomes
    /// `value` generic mana.
    pub fn with_x(&self, value: u8) -> ManaCost {
        let mut out = self.clone();
        out.cost[Color::Generic as usize] =
            out.generic().saturating_add(self.x.saturating_mul(value));
        out.x = 0;
        out.update_mana_value();
        out
    }

    /// The generic component of the cost.
    pub fn generic(&self) -> u8 {
        self.cost[Color::Generic as usize]
    }

    /// This cost with the generic component replaced by `generic`.
    pub fn with_generic(&self, generic: u8) -> ManaCost {
        let mut out = self.clone();
        out.cost[Color::Generic as usize] = generic;
        out.update_mana_value();
        out
    }

    fn update_mana_value(&mut self) {
        self.mana_value = self
            .cost
            .iter()
            .copied()
            .sum::<u8>()
            .saturating_add(self.hybrid.len() as u8);
    }

    pub fn colors(&self) -> Colors {
        let mut out = self.hybrid.iter().flatten().copied().collect::<Colors>();
        for color in [
            Color::White,
            Color::Blue,
            Color::Black,
            Color::Red,
            Color::Green,
        ] {
            if self.cost[color as usize] > 0 {
                out.insert(color);
            }
        }
        out
    }
}

/// One restriction and its pooled amounts by W/U/B/R/G/C.
pub type RestrictedClass = (ManaRestriction, [u8; 6]);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Mana {
    pub mana: [u8; 6],
    /// Mana that keeps a spending restriction until spent or lost (CR 106.6),
    /// by restriction and color. Canonical: sorted by use, no all-zero entry,
    /// `None` when empty. Boxed so ordinary pools stay small and
    /// allocation-free when cloned.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(clippy::box_collection)] // a thin pointer; `Vec` would widen every pool
    pub restricted: Option<Box<Vec<RestrictedClass>>>,
}

impl Mana {
    pub fn parse(text: &str) -> Self {
        let mut out = Mana::default();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.peek().copied() {
            if ch.is_ascii_digit() {
                let mut value: u16 = 0;
                while let Some(digit) = chars.peek().copied() {
                    if !digit.is_ascii_digit() {
                        break;
                    }
                    value = value * 10 + (digit as u16 - b'0' as u16);
                    chars.next();
                }
                out.mana[Color::Colorless as usize] += value.min(u8::MAX as u16) as u8;
                continue;
            }

            let slot = match ch {
                'W' => Color::White,
                'U' => Color::Blue,
                'B' => Color::Black,
                'R' => Color::Red,
                'G' => Color::Green,
                'C' => Color::Colorless,
                _ => panic!("invalid mana symbol: {ch}"),
            };
            out.mana[slot as usize] += 1;
            chars.next();
        }

        out
    }

    pub fn single(color: Color) -> Self {
        let mut out = Mana::default();
        if color == Color::Generic {
            return out;
        }
        out.mana[color as usize] = 1;
        out
    }

    /// Mana of `amounts` restricted to `restriction`.
    pub fn restricted_to(restriction: ManaRestriction, amounts: [u8; 6]) -> Self {
        let mut out = Mana::default();
        out.add_restricted(restriction, amounts);
        out
    }

    /// Restricted classes with their amounts by color, sorted by use.
    pub fn restricted(&self) -> &[RestrictedClass] {
        self.restricted.as_deref().map_or(&[], Vec::as_slice)
    }

    /// The restricted amounts of one restriction class, by color.
    pub fn restricted_of(&self, restriction: &ManaRestriction) -> [u8; 6] {
        self.restricted()
            .iter()
            .find(|(class, _)| class == restriction)
            .map_or([0; 6], |(_, amounts)| *amounts)
    }

    /// The class's amounts, inserting a zero entry; callers prune.
    fn class_mut(&mut self, restriction: ManaRestriction) -> &mut [u8; 6] {
        let classes = self.restricted.get_or_insert_with(Default::default);
        let index = match classes.binary_search_by(|(class, _)| class.cmp(&restriction)) {
            Ok(index) => index,
            Err(index) => {
                classes.insert(index, (restriction, [0; 6]));
                index
            }
        };
        &mut classes[index].1
    }

    /// Restore the canonical form after spending.
    fn prune(&mut self) {
        if let Some(classes) = &mut self.restricted {
            classes.retain(|(_, amounts)| *amounts != [0; 6]);
            if classes.is_empty() {
                self.restricted = None;
            }
        }
    }

    fn add_restricted(&mut self, restriction: ManaRestriction, amounts: [u8; 6]) {
        if amounts == [0; 6] {
            return;
        }
        let class = self.class_mut(restriction);
        for (slot, amount) in class.iter_mut().zip(amounts) {
            *slot = slot.saturating_add(amount);
        }
    }

    /// Set one class's amounts, keeping the canonical form.
    pub fn set_restricted(&mut self, restriction: ManaRestriction, amounts: [u8; 6]) {
        *self.class_mut(restriction) = amounts;
        self.prune();
    }

    /// All restricted mana by color, whatever it may be spent on.
    pub fn restricted_colors(&self) -> [u8; 6] {
        let mut out = [0_u8; 6];
        for (_, amounts) in self.restricted() {
            for (slot, amount) in out.iter_mut().zip(amounts) {
                *slot = slot.saturating_add(*amount);
            }
        }
        out
    }

    pub fn add(&mut self, other: &Mana) {
        for i in 0..self.mana.len() {
            self.mana[i] = self.mana[i].saturating_add(other.mana[i]);
        }
        for (restriction, amounts) in other.restricted() {
            self.add_restricted(restriction.clone(), *amounts);
        }
    }

    /// Per class and color, the larger of the two (an optimistic bound).
    pub fn max_with(&mut self, other: &Mana) {
        for i in 0..self.mana.len() {
            self.mana[i] = self.mana[i].max(other.mana[i]);
        }
        for (restriction, amounts) in other.restricted() {
            let class = self.class_mut(restriction.clone());
            for (slot, amount) in class.iter_mut().zip(amounts) {
                *slot = (*slot).max(*amount);
            }
        }
    }

    pub fn total(&self) -> u16 {
        self.mana
            .iter()
            .chain(self.restricted().iter().flat_map(|(_, amounts)| amounts))
            .map(|x| *x as u16)
            .sum()
    }

    /// Indices of eligible classes, narrowest first. Payment only changes
    /// amounts; indices stay valid until the final prune.
    fn usable_classes(&self, cost: &ManaCost) -> Vec<usize> {
        let mut classes = self
            .restricted()
            .iter()
            .enumerate()
            .filter(|(_, (class, _))| class.allows(&cost.purpose))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        // The canonical pool is already ordered by restriction, so a stable
        // breadth sort preserves its deterministic tie-break without cloning.
        classes.sort_by_key(|index| self.restricted()[*index].0.breadth());
        classes
    }

    fn usable_colors(&self, cost: &ManaCost) -> [u8; 6] {
        let mut colors = self.mana;
        for (class, amounts) in self.restricted() {
            if class.allows(&cost.purpose) {
                for (total, amount) in colors.iter_mut().zip(amounts) {
                    *total = total.saturating_add(*amount);
                }
            }
        }
        colors
    }

    /// Visit colored, hybrid, then generic remainders in spending order.
    /// Returning true stops enumeration. First-result queries keep their result
    /// on the stack; only nested payments collect all concrete alternatives.
    fn visit_payment_remainders(
        &self,
        cost: &ManaCost,
        visit: &mut impl FnMut([u8; 6]) -> bool,
    ) -> bool {
        fn generic(
            remaining: [u8; 6],
            slot: usize,
            needed: u8,
            visit: &mut impl FnMut([u8; 6]) -> bool,
        ) -> bool {
            if needed == 0 {
                return visit(remaining);
            }
            if slot == 6 {
                return false;
            }
            let order = [5, 0, 1, 2, 3, 4];
            if order[slot..]
                .iter()
                .map(|index| u16::from(remaining[*index]))
                .sum::<u16>()
                < u16::from(needed)
            {
                return false;
            }
            // Preserve the ordinary colorless-first spending order.
            let index = order[slot];
            for spent in (0..=needed.min(remaining[index])).rev() {
                let mut next = remaining;
                next[index] -= spent;
                if generic(next, slot + 1, needed - spent, visit) {
                    return true;
                }
            }
            false
        }
        fn assign(
            remaining: [u8; 6],
            pips: &[[Color; 2]],
            needed: u8,
            visit: &mut impl FnMut([u8; 6]) -> bool,
        ) -> bool {
            if let Some((pip, rest)) = pips.split_first() {
                for color in pip {
                    let index = *color as usize;
                    if remaining[index] > 0 {
                        let mut next = remaining;
                        next[index] -= 1;
                        if assign(next, rest, needed, visit) {
                            return true;
                        }
                    }
                }
                false
            } else {
                generic(remaining, 0, needed, visit)
            }
        }
        let mut remaining = self.usable_colors(cost);
        for (available, required) in remaining.iter_mut().zip(cost.cost) {
            let Some(left) = available.checked_sub(required) else {
                return false;
            };
            *available = left;
        }
        assign(remaining, &cost.hybrid, cost.generic(), visit)
    }

    pub fn can_pay(&self, cost: &ManaCost) -> bool {
        self.visit_payment_remainders(cost, &mut |_| true)
    }

    /// Commit the first legal remainder, leaving the pool unchanged on failure.
    pub fn try_pay(&mut self, cost: &ManaCost) -> bool {
        let mut remaining = None;
        self.visit_payment_remainders(cost, &mut |mana| {
            remaining = Some(mana);
            true
        });
        let Some(remaining) = remaining else {
            return false;
        };
        self.apply_payment_remainder(cost, remaining);
        true
    }

    pub fn pay(&mut self, cost: &ManaCost) {
        assert!(self.try_pay(cost), "insufficient mana");
    }

    /// Every eligible concrete remainder. Equal-colored mana with different
    /// restrictions is not interchangeable: a child must preserve the class
    /// that its parent can spend. Ordinary `pay` keeps deterministic ordering.
    pub fn payment_options(&self, cost: &ManaCost) -> Vec<Mana> {
        fn assign(
            pool: Mana,
            classes: &[usize],
            slot: usize,
            class_index: usize,
            spent: [u8; 6],
            results: &mut Vec<Mana>,
        ) {
            if slot == 6 {
                let mut pool = pool;
                pool.prune();
                if !results.contains(&pool) {
                    results.push(pool);
                }
                return;
            }
            if spent[slot] == 0 {
                assign(pool, classes, slot + 1, 0, spent, results);
                return;
            }
            let Some(&class) = classes.get(class_index) else {
                if pool.mana[slot] >= spent[slot] {
                    let mut pool = pool;
                    pool.mana[slot] -= spent[slot];
                    assign(pool, classes, slot + 1, 0, spent, results);
                }
                return;
            };
            let available = pool.restricted()[class].1[slot];
            for amount in (0..=spent[slot].min(available)).rev() {
                let mut next = pool.clone();
                next.restricted.as_mut().unwrap()[class].1[slot] -= amount;
                let mut remaining = spent;
                remaining[slot] -= amount;
                assign(next, classes, slot, class_index + 1, remaining, results);
            }
        }
        let mut results = Vec::new();
        // Ordinary pools have no class allocation to enumerate.
        if self.restricted.is_none() {
            self.visit_payment_remainders(cost, &mut |mana| {
                let pool = Mana { mana, restricted: None };
                if !results.contains(&pool) {
                    results.push(pool);
                }
                false
            });
            return results;
        }
        let classes = self.usable_classes(cost);
        let usable = self.usable_colors(cost);
        let mut seen = Vec::new();
        self.visit_payment_remainders(cost, &mut |remaining| {
            // Different hybrid assignments can reach the same color remainder;
            // expand its class allocations only once.
            if seen.contains(&remaining) {
                return false;
            }
            seen.push(remaining);
            let spent = std::array::from_fn(|slot| usable[slot] - remaining[slot]);
            assign(self.clone(), &classes, 0, 0, spent, &mut results);
            false
        });
        results
    }

    /// Spend down to `remaining` per color: usable restricted mana first,
    /// narrowest restriction first, then unrestricted mana.
    fn apply_payment_remainder(&mut self, cost: &ManaCost, remaining: [u8; 6]) {
        let classes = self.usable_classes(cost);
        let usable = self.usable_colors(cost);
        for (slot, left) in remaining.into_iter().enumerate() {
            let mut spent = usable[slot] - left;
            for &class in &classes {
                let amounts = &mut self.restricted.as_mut().unwrap()[class].1;
                let restricted = spent.min(amounts[slot]);
                amounts[slot] -= restricted;
                spent -= restricted;
            }
            self.mana[slot] -= spent;
        }
        self.prune();
    }

    /// All unspent mana becomes red, keeping its spending restrictions
    /// (Ozai, the Phoenix King).
    pub fn become_red(&mut self) {
        let red = Color::Red as usize;
        let all_red = |pool: &mut [u8; 6]| {
            let total = pool
                .iter()
                .fold(0u8, |sum, amount| sum.saturating_add(*amount));
            *pool = [0; 6];
            pool[red] = total;
        };
        all_red(&mut self.mana);
        if let Some(classes) = &mut self.restricted {
            for (_, amounts) in classes.iter_mut() {
                all_red(amounts);
            }
        }
    }

    /// Only red mana, with its spending restrictions.
    pub fn red_only(&self) -> Self {
        let red = Color::Red as usize;
        let mut kept = Self::default();
        kept.mana[red] = self.mana[red];
        for (restriction, amounts) in self.restricted() {
            let mut only_red = [0; 6];
            only_red[red] = amounts[red];
            kept.add_restricted(restriction.clone(), only_red);
        }
        kept
    }

    pub fn clear(&mut self) {
        // CR 106.4 — Unspent mana empties from pools as steps/phases end.
        self.mana = [0; 6];
        self.restricted = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{Mana, ManaCost, ManaRestriction, SpendingContext};

    #[test]
    fn spending_snapshot_keeps_forked_payment_independent() {
        let restriction = ManaRestriction::spell(["Citizen"]);
        let mut cost = ManaCost::parse("U");
        cost.purpose = SpendingContext::spell(["Citizen"]);
        let mut fork_cost = cost.clone();
        let SpendingContext::Cast { characteristics } = &mut fork_cost.purpose else {
            unreachable!();
        };
        std::sync::Arc::make_mut(characteristics).subtypes = ["Elf".into()].into();
        let pool = Mana::restricted_to(restriction, [0, 1, 0, 0, 0, 0]);
        assert!(pool.can_pay(&cost));
        assert!(!pool.can_pay(&fork_cost));
        assert_eq!(cost.purpose, SpendingContext::spell(["Citizen"]));
    }

    #[test]
    fn hybrid_payment_backtracks_and_preserves_required_colors() {
        let cost = ManaCost::parse("{1}{U/R}{U/B}{R}");
        assert_eq!(cost.mana_value, 4);
        let mut pool = Mana::parse("1URR");
        assert!(pool.can_pay(&cost));
        pool.pay(&cost);
        assert_eq!(pool.total(), 0);
        assert!(!Mana::parse("1UUU").can_pay(&cost));
        let cost = ManaCost::parse("{U/R}{U/B}");
        let mut pool = Mana::parse("UR");
        pool.pay(&cost);
        assert_eq!(pool.total(), 0);
    }

    #[test]
    fn hybrid_cost_arithmetic_keeps_pips_and_color_identity() {
        use super::{Color, Colors};
        let cost = ManaCost::parse("{2}{R/W}{R/W}{R/W}");
        assert_eq!(cost.mana_value, 5);
        assert_eq!(cost.colors(), Colors::from([Color::Red, Color::White]));
        let reduced = cost.reduced_generic(1);
        assert_eq!(reduced.mana_value, 4);
        assert!(Mana::parse("1RRW").can_pay(&reduced));
        let kicked = reduced.plus(&ManaCost::parse("1G"));
        assert_eq!(kicked.mana_value, 6);
        assert!(Mana::parse("2RRWG").can_pay(&kicked));
        assert_eq!(kicked.with_generic(0).mana_value, 4);
    }

    #[test]
    fn colored_reductions_spill_into_generic_but_never_other_colors() {
        for (cost, reduction, expected) in [
            ("{5}{W}{W}{U}{R}", "{W}{W}{W}{U}{B}{R}{G}", "{2}"),
            ("{1}{U}{U}", "{W}{W}{W}", "{U}{U}"),
            ("{3}{G}", "{2}{G}{G}", "{0}"),
            ("{X}{2}{R/W}{R}", "{R}{R}{R}", "{X}{R/W}"),
        ] {
            assert_eq!(
                ManaCost::parse(cost).reduced_by(&ManaCost::parse(reduction)),
                ManaCost::parse(expected),
                "{cost} less {reduction}"
            );
        }
    }

    #[test]
    fn x_counts_zero_until_its_value_is_chosen() {
        let cost = ManaCost::parse("{X}{R}{R}{R}");
        assert_eq!((cost.x, cost.mana_value), (1, 3));
        let paid = cost.with_x(2);
        assert_eq!((paid.x, paid.mana_value, paid.generic()), (0, 5, 2));
        assert_eq!(ManaCost::parse("{X}{X}{B}").with_x(3).generic(), 6);
    }

    #[test]
    fn activation_payment_options_preserve_colors_and_restrictions() {
        let lesson = ManaRestriction::spell(["Lesson"]);
        let mut pool = Mana::parse("WU");
        pool.add(&Mana::restricted_to(lesson.clone(), [0, 0, 0, 0, 2, 0]));
        let options = pool.payment_options(&ManaCost::parse("1"));
        assert_eq!(options.len(), 2);
        assert!(options.iter().any(|p| p.mana == Mana::parse("W").mana));
        assert!(options.iter().any(|p| p.mana == Mana::parse("U").mana));
        assert!(options.iter().all(|p| p.restricted() == pool.restricted()));
        assert!(pool.payment_options(&ManaCost::parse("G")).is_empty());
        let cost = ManaCost {
            purpose: SpendingContext::spell(["Lesson"]),
            ..ManaCost::parse("1G")
        };
        let options = pool.payment_options(&cost);
        assert!(options
            .iter()
            .any(|p| p.mana == pool.mana && p.restricted().is_empty()));
        let hybrid = Mana::parse("WU").payment_options(&ManaCost::parse("{W/U}"));
        assert_eq!(
            hybrid,
            Mana::parse("WU").payment_options(&ManaCost::parse("1"))
        );
    }

    #[test]
    fn parse_and_pay_mana() {
        let mut mana = Mana::parse("2RG");
        let cost = ManaCost::parse("1RG");
        assert!(mana.can_pay(&cost));
        mana.pay(&cost);
        assert_eq!(mana.total(), 1);
    }

    #[test]
    fn generic_cost_spends_colorless_first() {
        use super::Color;
        let mut mana = Mana::default();
        mana.mana[Color::Colorless as usize] = 2;
        mana.mana[Color::Green as usize] = 1;

        let cost = ManaCost::parse("2G");
        assert!(mana.can_pay(&cost));
        mana.pay(&cost);

        assert_eq!(mana.mana[Color::Colorless as usize], 0);
        assert_eq!(mana.mana[Color::Green as usize], 0);
    }

    #[test]
    fn generic_cost_preserves_colored_mana() {
        use super::Color;
        let mut mana = Mana::default();
        mana.mana[Color::Colorless as usize] = 1;
        mana.mana[Color::Red as usize] = 1;

        let cost = ManaCost::parse("1");
        mana.pay(&cost);

        // Colorless spent, red preserved
        assert_eq!(mana.mana[Color::Colorless as usize], 0);
        assert_eq!(mana.mana[Color::Red as usize], 1);
    }

    #[test]
    fn restricted_mana_pays_only_its_purposes_narrowest_first() {
        let lesson = ManaRestriction::spell(["Lesson"]);
        let lesson_or_shrine = ManaRestriction::spell(["Lesson", "Shrine"]);
        let ally = ManaRestriction::new(["Ally"], ["Ally"]);
        let mut pool = Mana::restricted_to(lesson_or_shrine.clone(), [0, 1, 0, 0, 0, 0]);
        pool.add(&Mana::restricted_to(lesson.clone(), [0, 1, 0, 0, 0, 0]));
        pool.add(&Mana::restricted_to(ally.clone(), [1, 0, 0, 0, 0, 0]));
        let cost = |text: &str, purpose: SpendingContext| ManaCost {
            purpose,
            ..ManaCost::parse(text)
        };
        assert!(!pool.can_pay(&ManaCost::parse("1")));
        assert!(!pool.can_pay(&cost("U", SpendingContext::spell(["Ally"]))));
        assert!(pool.can_pay(&cost("W", SpendingContext::ability(["Human", "Ally"]))));
        assert!(pool.can_pay(&cost("U", SpendingContext::spell(["Shrine"]))));
        assert!(!pool.can_pay(&cost("UU", SpendingContext::spell(["Shrine"]))));
        // A Lesson spends the Lesson-only mana and keeps the broader class.
        let mut paid = pool.clone();
        paid.pay(&cost("U", SpendingContext::spell(["Lesson"])));
        assert_eq!(paid.restricted_of(&lesson), [0; 6]);
        assert_eq!(paid.restricted_of(&lesson_or_shrine), [0, 1, 0, 0, 0, 0]);
        assert_eq!(paid.restricted().len(), 2);
        assert_eq!(pool.restricted_colors(), [1, 2, 0, 0, 0, 0]);
        assert_eq!(ally.describe(), "Ally spells or abilities of Ally sources");
    }
    #[test]
    fn open_subtypes_keep_cast_activation_and_resolving_payments_distinct() {
        for name in ["Citizen", "Scarecrow", "UnknownFutureSubtype"] {
            let restriction = ManaRestriction::spell([name]);
            let pool = Mana::restricted_to(restriction, [0, 1, 0, 0, 0, 0]);
            let mut cost = ManaCost::parse("U");
            assert!(!pool.can_pay(&cost));
            cost.purpose = SpendingContext::ability([name]);
            assert!(!pool.can_pay(&cost));
            cost.purpose = SpendingContext::spell([name]);
            assert!(pool.can_pay(&cost));
            let mut paid = pool;
            paid.pay(&cost);
            assert_eq!(paid.total(), 0);
        }
        assert!(!ManaRestriction::default().allows(&SpendingContext::spell(["Citizen"])));
    }

    #[test]
    fn nested_payment_preserves_a_parent_class_even_when_breadths_are_equal() {
        let parent = ManaRestriction::new(["Citizen"], ["Scarecrow"]);
        let other = ManaRestriction::new(["Elf"], ["Scarecrow"]);
        let mut pool = Mana::restricted_to(parent.clone(), [0, 1, 0, 0, 0, 0]);
        pool.add(&Mana::restricted_to(other, [0, 1, 0, 0, 0, 0]));
        let child = ManaCost {
            purpose: SpendingContext::ability(["Scarecrow"]),
            ..ManaCost::parse("1")
        };
        let parent_cost = ManaCost {
            purpose: SpendingContext::spell(["Citizen"]),
            ..ManaCost::parse("U")
        };
        let remainders = pool.payment_options(&child);
        assert_eq!(remainders.len(), 2);
        let remainder = remainders
            .iter()
            .find(|pool| pool.can_pay(&parent_cost))
            .unwrap();
        assert_eq!(remainder.restricted_of(&parent), [0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn terminal_payment_retains_first_class_aware_remainder() {
        for regular in 0..=2 {
            for citizen in 0..=2 {
                for shared in 0..=2 {
                    let mut pool = Mana::parse(&"U".repeat(regular));
                    pool.add(&Mana::restricted_to(ManaRestriction::spell(["Citizen"]), [0, citizen, 0, 0, 0, 0]));
                    pool.add(&Mana::restricted_to(ManaRestriction::spell(["Citizen", "Elf"]), [0, shared, 0, 0, 0, 0]));
                    for generic in 0..=5 {
                        let cost = ManaCost { purpose: SpendingContext::spell(["Citizen"]), ..ManaCost::parse("U").with_generic(generic) };
                        let options = pool.payment_options(&cost);
                        assert_eq!(pool.can_pay(&cost), !options.is_empty());
                        let mut attempted = pool.clone();
                        assert_eq!(attempted.try_pay(&cost), !options.is_empty());
                        assert_eq!(&attempted, options.first().unwrap_or(&pool));
                        if let Some(first) = options.first() {
                            let mut terminal = pool.clone();
                            terminal.pay(&cost);
                            assert_eq!(&terminal, first);
                        }
                    }
                }
            }
        }
    }
}
