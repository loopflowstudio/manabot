// mana.rs
// Mana costs and pools, including exact two-color hybrid payment.

use std::collections::BTreeSet;

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

/// Subtypes a spending restriction can name, by bit position in [`ManaUse`].
pub const SPEND_SUBTYPES: [&str; 3] = ["Lesson", "Shrine", "Ally"];

/// What mana is spent on (CR 106.6). As a payment's purpose it names the spell
/// being cast or the activated ability's source; as a restriction it names
/// every purpose the mana may pay for. Bit `i` of `spells` is "cast a spell
/// with subtype `SPEND_SUBTYPES[i]`"; of `abilities`, "activate an ability of a
/// source with that subtype". The empty use is an ordinary payment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ManaUse {
    pub spells: u8,
    pub abilities: u8,
}

impl ManaUse {
    fn mask<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> u8 {
        subtypes
            .into_iter()
            .filter_map(|subtype| SPEND_SUBTYPES.iter().position(|s| *s == subtype))
            .fold(0, |mask, bit| mask | 1 << bit)
    }

    pub fn new<'a>(
        spell_subtypes: impl IntoIterator<Item = &'a str>,
        ability_source_subtypes: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        Self {
            spells: Self::mask(spell_subtypes),
            abilities: Self::mask(ability_source_subtypes),
        }
    }

    /// Casting a spell with these subtypes.
    pub fn spell<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::new(subtypes, [])
    }

    /// Activating an ability of a source with these subtypes.
    pub fn ability<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::new([], subtypes)
    }

    pub fn is_ordinary(&self) -> bool {
        *self == Self::default()
    }

    /// Whether mana restricted to `self` may pay for `purpose`.
    pub fn allows(self, purpose: ManaUse) -> bool {
        self.spells & purpose.spells != 0 || self.abilities & purpose.abilities != 0
    }

    /// How many purposes a restriction admits; narrower mana is spent first.
    fn breadth(self) -> u32 {
        self.spells.count_ones() + self.abilities.count_ones()
    }

    /// "Lesson or Shrine spells", "Ally spells or abilities of Ally sources".
    pub fn describe(self) -> String {
        let names = |mask: u8| {
            SPEND_SUBTYPES
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & 1 << bit != 0)
                .map(|(_, name)| *name)
                .collect::<Vec<_>>()
                .join(" or ")
        };
        let mut parts = Vec::new();
        if self.spells != 0 {
            parts.push(format!("{} spells", names(self.spells)));
        }
        if self.abilities != 0 {
            parts.push(format!("abilities of {} sources", names(self.abilities)));
        }
        parts.join(" or ")
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ManaCost {
    /// The payment's purpose, which decides the restricted mana it may use:
    /// set for a cast spell or an activation, never for resolving effects.
    #[serde(skip_serializing_if = "ManaUse::is_ordinary")]
    pub purpose: ManaUse,
    pub cost: [u8; 7],
    pub mana_value: u8,
    /// Each two-color hybrid pip is paid with exactly one of its colors.
    /// Omission keeps frozen ordinary-cost manifests byte-for-byte stable.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hybrid: Vec<[Color; 2]>,
    /// Number of {X} symbols. X is 0 outside the stack (CR 202.3e), so it
    /// adds nothing to `mana_value` until [`ManaCost::with_x`] fixes it.
    #[serde(skip_serializing_if = "crate::state::is_zero")]
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
pub type RestrictedClass = (ManaUse, [u8; 6]);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Mana {
    pub mana: [u8; 6],
    /// Mana that keeps a spending restriction until spent or lost (CR 106.6),
    /// by restriction and color. Canonical: sorted by use, no all-zero entry,
    /// `None` when empty. Boxed so ordinary pools stay small and
    /// allocation-free in the undo journal.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(clippy::box_collection)] // a thin pointer; `Vec` would widen every pool
    pub(crate) restricted: Option<Box<Vec<RestrictedClass>>>,
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
    pub fn restricted_to(restriction: ManaUse, amounts: [u8; 6]) -> Self {
        let mut out = Mana::default();
        out.add_restricted(restriction, amounts);
        out
    }

    /// Restricted classes with their amounts by color, sorted by use.
    pub fn restricted(&self) -> &[RestrictedClass] {
        self.restricted.as_deref().map_or(&[], Vec::as_slice)
    }

    /// The restricted amounts of one restriction class, by color.
    pub fn restricted_of(&self, restriction: ManaUse) -> [u8; 6] {
        self.restricted()
            .iter()
            .find(|(class, _)| *class == restriction)
            .map_or([0; 6], |(_, amounts)| *amounts)
    }

    /// The class's amounts, inserting a zero entry; callers prune.
    fn class_mut(&mut self, restriction: ManaUse) -> &mut [u8; 6] {
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

    fn add_restricted(&mut self, restriction: ManaUse, amounts: [u8; 6]) {
        if amounts == [0; 6] {
            return;
        }
        let class = self.class_mut(restriction);
        for (slot, amount) in class.iter_mut().zip(amounts) {
            *slot = slot.saturating_add(amount);
        }
    }

    /// Set one class's amounts, keeping the canonical form.
    pub(crate) fn set_restricted(&mut self, restriction: ManaUse, amounts: [u8; 6]) {
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
            self.add_restricted(*restriction, *amounts);
        }
    }

    /// Per class and color, the larger of the two (an optimistic bound).
    pub(crate) fn max_with(&mut self, other: &Mana) {
        for i in 0..self.mana.len() {
            self.mana[i] = self.mana[i].max(other.mana[i]);
        }
        for (restriction, amounts) in other.restricted() {
            let class = self.class_mut(*restriction);
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

    /// Restriction classes that may pay for `cost`, narrowest first.
    fn usable_classes(&self, cost: &ManaCost) -> Vec<ManaUse> {
        let mut classes = self
            .restricted()
            .iter()
            .map(|(class, _)| *class)
            .filter(|class| class.allows(cost.purpose))
            .collect::<Vec<_>>();
        classes.sort_by_key(|class| (class.breadth(), *class));
        classes
    }

    /// Total usable for one spend purpose, without promising a color assignment.
    pub(crate) fn usable_total(&self, cost: &ManaCost) -> u16 {
        self.mana
            .iter()
            .chain(
                self.restricted()
                    .iter()
                    .filter(|(class, _)| class.allows(cost.purpose))
                    .flat_map(|(_, amounts)| amounts),
            )
            .map(|amount| u16::from(*amount))
            .sum()
    }

    /// Enumerate concrete remainders after colored, hybrid, then generic
    /// payment. Ordinary spending stops at the first result; activation search
    /// considers alternatives so generic costs cannot consume a parent's pip.
    fn payment_remainders(&self, cost: &ManaCost, all: bool) -> Vec<[u8; 6]> {
        fn generic(
            remaining: [u8; 6],
            slot: usize,
            needed: u8,
            results: &mut Vec<[u8; 6]>,
            all: bool,
        ) -> bool {
            if needed == 0 {
                if !results.contains(&remaining) {
                    results.push(remaining);
                }
                return !all;
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
                if generic(next, slot + 1, needed - spent, results, all) {
                    return true;
                }
            }
            false
        }
        fn assign(
            remaining: [u8; 6],
            pips: &[[Color; 2]],
            needed: u8,
            results: &mut Vec<[u8; 6]>,
            all: bool,
        ) -> bool {
            if let Some((pip, rest)) = pips.split_first() {
                for color in pip {
                    let index = *color as usize;
                    if remaining[index] > 0 {
                        let mut next = remaining;
                        next[index] -= 1;
                        if assign(next, rest, needed, results, all) {
                            return true;
                        }
                    }
                }
                false
            } else {
                generic(remaining, 0, needed, results, all)
            }
        }
        let mut remaining = self.mana;
        for class in self.usable_classes(cost) {
            for (available, restricted) in remaining.iter_mut().zip(self.restricted_of(class)) {
                *available = available.saturating_add(restricted);
            }
        }
        for (available, required) in remaining.iter_mut().zip(cost.cost) {
            let Some(left) = available.checked_sub(required) else {
                return Vec::new();
            };
            *available = left;
        }
        let mut results = Vec::new();
        assign(remaining, &cost.hybrid, cost.generic(), &mut results, all);
        results
    }

    pub fn can_pay(&self, cost: &ManaCost) -> bool {
        !self.payment_remainders(cost, false).is_empty()
    }

    pub fn pay(&mut self, cost: &ManaCost) {
        let remaining = self
            .payment_remainders(cost, false)
            .into_iter()
            .next()
            .expect("insufficient mana");
        self.apply_payment_remainder(cost, remaining);
    }

    /// Eligible concrete pool payments, retaining spending restrictions and
    /// the restricted-first policy within each color.
    pub(crate) fn payment_options(&self, cost: &ManaCost) -> Vec<Mana> {
        self.payment_remainders(cost, true)
            .into_iter()
            .map(|remaining| {
                let mut pool = self.clone();
                pool.apply_payment_remainder(cost, remaining);
                pool
            })
            .collect()
    }

    /// Spend down to `remaining` per color: usable restricted mana first,
    /// narrowest restriction first, then unrestricted mana.
    fn apply_payment_remainder(&mut self, cost: &ManaCost, remaining: [u8; 6]) {
        let classes = self.usable_classes(cost);
        for (slot, left) in remaining.into_iter().enumerate() {
            let before = classes.iter().fold(self.mana[slot], |sum, class| {
                sum.saturating_add(self.restricted_of(*class)[slot])
            });
            let mut spent = before - left;
            for class in &classes {
                let amounts = self.class_mut(*class);
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
            kept.add_restricted(*restriction, only_red);
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
    use super::{Mana, ManaCost, ManaUse};

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
    fn x_counts_zero_until_its_value_is_chosen() {
        let cost = ManaCost::parse("{X}{R}{R}{R}");
        assert_eq!((cost.x, cost.mana_value), (1, 3));
        let paid = cost.with_x(2);
        assert_eq!((paid.x, paid.mana_value, paid.generic()), (0, 5, 2));
        assert_eq!(ManaCost::parse("{X}{X}{B}").with_x(3).generic(), 6);
    }

    #[test]
    fn activation_payment_options_preserve_colors_and_restrictions() {
        let lesson = ManaUse::spell(["Lesson"]);
        let mut pool = Mana::parse("WU");
        pool.add(&Mana::restricted_to(lesson, [0, 0, 0, 0, 2, 0]));
        let options = pool.payment_options(&ManaCost::parse("1"));
        assert_eq!(options.len(), 2);
        assert!(options.iter().any(|p| p.mana == Mana::parse("W").mana));
        assert!(options.iter().any(|p| p.mana == Mana::parse("U").mana));
        assert!(options.iter().all(|p| p.restricted() == pool.restricted()));
        assert!(pool.payment_options(&ManaCost::parse("G")).is_empty());
        let cost = ManaCost {
            purpose: lesson,
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
        let lesson = ManaUse::spell(["Lesson"]);
        let lesson_or_shrine = ManaUse::spell(["Lesson", "Shrine"]);
        let ally = ManaUse::new(["Ally"], ["Ally"]);
        let mut pool = Mana::restricted_to(lesson_or_shrine, [0, 1, 0, 0, 0, 0]);
        pool.add(&Mana::restricted_to(lesson, [0, 1, 0, 0, 0, 0]));
        pool.add(&Mana::restricted_to(ally, [1, 0, 0, 0, 0, 0]));
        let cost = |text: &str, purpose: ManaUse| ManaCost {
            purpose,
            ..ManaCost::parse(text)
        };
        assert!(!pool.can_pay(&ManaCost::parse("1")));
        assert!(!pool.can_pay(&cost("U", ManaUse::spell(["Ally"]))));
        assert!(pool.can_pay(&cost("W", ManaUse::ability(["Human", "Ally"]))));
        assert!(pool.can_pay(&cost("U", ManaUse::spell(["Shrine"]))));
        assert!(!pool.can_pay(&cost("UU", ManaUse::spell(["Shrine"]))));
        // A Lesson spends the Lesson-only mana and keeps the broader class.
        let mut paid = pool.clone();
        paid.pay(&cost("U", ManaUse::spell(["Lesson"])));
        assert_eq!(paid.restricted_of(lesson), [0; 6]);
        assert_eq!(paid.restricted_of(lesson_or_shrine), [0, 1, 0, 0, 0, 0]);
        assert_eq!(paid.restricted().len(), 2);
        assert_eq!(pool.restricted_colors(), [1, 2, 0, 0, 0, 0]);
        assert_eq!(ally.describe(), "Ally spells or abilities of Ally sources");
    }
}
