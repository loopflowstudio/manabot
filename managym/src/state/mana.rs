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

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ManaCost {
    /// Payment context, set only for a cast Lesson (not its resolving effects).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub spend_on_lesson: bool,
    pub cost: [u8; 7],
    pub mana_value: u8,
    /// Each two-color hybrid pip is paid with exactly one of its colors.
    /// Omission keeps frozen ordinary-cost manifests byte-for-byte stable.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hybrid: Vec<[Color; 2]>,
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
                }
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
        out.update_mana_value();
        out
    }

    /// Reduce the generic component by `amount`, floored at zero
    /// (affinity-style cost reduction, waterbend payments).
    pub fn reduced_generic(&self, amount: u8) -> ManaCost {
        self.with_generic(self.generic().saturating_sub(amount))
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

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Mana {
    pub mana: [u8; 6],
    /// Hermitic Herbalist mana keeps its spending restriction until spent or lost.
    #[serde(skip_serializing_if = "is_empty_mana")]
    pub lesson: [u8; 6],
}

fn is_empty_mana(mana: &[u8; 6]) -> bool {
    *mana == [0; 6]
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

    pub fn add(&mut self, other: &Mana) {
        for i in 0..self.mana.len() {
            self.mana[i] = self.mana[i].saturating_add(other.mana[i]);
            self.lesson[i] = self.lesson[i].saturating_add(other.lesson[i]);
        }
    }

    pub fn total(&self) -> u16 {
        self.mana
            .iter()
            .chain(&self.lesson)
            .map(|x| *x as u16)
            .sum()
    }

    /// Find a concrete payment, exploring hybrid assignments before spending
    /// generic mana. Greedy hybrid payment can consume a later required color.
    fn payment_remainder(&self, cost: &ManaCost) -> Option<[u8; 6]> {
        fn assign(remaining: [u8; 6], pips: &[[Color; 2]], generic: u8) -> Option<[u8; 6]> {
            if let Some((pip, rest)) = pips.split_first() {
                for color in pip {
                    let index = *color as usize;
                    if remaining[index] > 0 {
                        let mut next = remaining;
                        next[index] -= 1;
                        if let Some(paid) = assign(next, rest, generic) {
                            return Some(paid);
                        }
                    }
                }
                return None;
            }
            let mut remaining = remaining;
            let mut needed = generic;
            // Spend colorless first, preserving the existing ordinary policy.
            for index in [5, 0, 1, 2, 3, 4] {
                let spent = needed.min(remaining[index]);
                remaining[index] -= spent;
                needed -= spent;
            }
            (needed == 0).then_some(remaining)
        }
        let mut remaining = self.mana;
        if cost.spend_on_lesson {
            for (available, restricted) in remaining.iter_mut().zip(self.lesson) {
                *available = available.saturating_add(restricted);
            }
        }
        for (available, required) in remaining.iter_mut().zip(cost.cost) {
            *available = available.checked_sub(required)?;
        }
        assign(remaining, &cost.hybrid, cost.generic())
    }

    pub fn can_pay(&self, cost: &ManaCost) -> bool {
        self.payment_remainder(cost).is_some()
    }

    /// Legal remainders for an intermediate payment, where a later cost may
    /// need a color that the ordinary greedy generic payment would consume.
    /// Restricted mana is still spent first within each eligible color.
    pub(crate) fn payment_options(&self, cost: &ManaCost) -> Vec<Self> {
        // Fixed-color costs have one remainder. Most source activations are
        // free, so avoid the allocation search on the ordinary land path.
        if cost.generic() == 0 && cost.hybrid.is_empty() {
            if !self.can_pay(cost) {
                return Vec::new();
            }
            let mut remaining = self.clone();
            remaining.pay(cost);
            return vec![remaining];
        }
        fn spend(pool: &Mana, slot: usize, amount: u8, lesson: bool) -> Option<Mana> {
            let mut next = pool.clone();
            let restricted = if lesson {
                amount.min(next.lesson[slot])
            } else {
                0
            };
            next.lesson[slot] -= restricted;
            next.mana[slot] = next.mana[slot].checked_sub(amount - restricted)?;
            Some(next)
        }
        fn generic(pool: Mana, cost: &ManaCost, slot: usize, needed: u8, out: &mut Vec<Mana>) {
            if needed == 0 {
                if !out.contains(&pool) {
                    out.push(pool);
                }
                return;
            }
            let Some(&color) = [5, 0, 1, 2, 3, 4].get(slot) else {
                return;
            };
            let available = pool.mana[color].saturating_add(if cost.spend_on_lesson {
                pool.lesson[color]
            } else {
                0
            });
            for amount in (0..=needed.min(available)).rev() {
                if let Some(next) = spend(&pool, color, amount, cost.spend_on_lesson) {
                    generic(next, cost, slot + 1, needed - amount, out);
                }
            }
        }
        fn hybrid(pool: Mana, cost: &ManaCost, pip: usize, out: &mut Vec<Mana>) {
            let Some(colors) = cost.hybrid.get(pip) else {
                generic(pool, cost, 0, cost.generic(), out);
                return;
            };
            for color in colors {
                if let Some(next) = spend(&pool, *color as usize, 1, cost.spend_on_lesson) {
                    hybrid(next, cost, pip + 1, out);
                }
            }
        }
        let mut pool = self.clone();
        for slot in 0..6 {
            let Some(next) = spend(&pool, slot, cost.cost[slot], cost.spend_on_lesson) else {
                return Vec::new();
            };
            pool = next;
        }
        let mut options = Vec::new();
        hybrid(pool, cost, 0, &mut options);
        options
    }

    pub fn pay(&mut self, cost: &ManaCost) {
        let remaining = self.payment_remainder(cost).expect("insufficient mana");
        for (slot, left) in remaining.into_iter().enumerate() {
            let before = self.mana[slot].saturating_add(if cost.spend_on_lesson {
                self.lesson[slot]
            } else {
                0
            });
            let spent = before - left;
            let restricted = if cost.spend_on_lesson {
                spent.min(self.lesson[slot])
            } else {
                0
            };
            self.lesson[slot] -= restricted;
            self.mana[slot] -= spent - restricted;
        }
    }

    pub fn clear(&mut self) {
        // CR 106.4 — Unspent mana empties from pools as steps/phases end.
        self.mana = [0; 6];
        self.lesson = [0; 6];
    }
}

#[cfg(test)]
mod tests {
    use super::{Mana, ManaCost};

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
    fn intermediate_payment_keeps_all_legal_generic_and_hybrid_remainders() {
        let pool = Mana::parse("UG");
        let options = pool.payment_options(&ManaCost::parse("1"));
        assert_eq!(options, vec![Mana::parse("G"), Mana::parse("U")]);
        assert_eq!(pool.payment_options(&ManaCost::parse("{U/G}")), options);
        assert!(pool.payment_options(&ManaCost::parse("3")).is_empty());
        let restricted = Mana {
            lesson: Mana::parse("G").mana,
            ..pool
        };
        let mut lesson = ManaCost::parse("G");
        lesson.spend_on_lesson = true;
        let remaining = restricted.payment_options(&lesson);
        assert_eq!(remaining, vec![Mana::parse("UG")]);
        assert_eq!(
            restricted.payment_options(&ManaCost::parse("G"))[0].lesson[4],
            1
        );
    }
}
