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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
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
    /// Number of {X} symbols. X is zero everywhere but the stack (CR 107.3,
    /// 202.3e), so it adds nothing to `mana_value`; `with_x` fixes its value.
    #[serde(skip_serializing_if = "is_zero_u8")]
    pub x: u8,
}

fn is_zero_u8(value: &u8) -> bool {
    *value == 0
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
                    out.x += part.x;
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

    /// This cost with each {X} paid as `value` generic mana (CR 601.2f).
    pub fn with_x(&self, value: u8) -> ManaCost {
        let mut out = self.clone();
        out.cost[Color::Generic as usize] =
            out.generic().saturating_add(self.x.saturating_mul(value));
        out.x = 0;
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
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

    /// Total usable for one spend purpose, without promising a color assignment.
    pub(crate) fn usable_total(&self, cost: &ManaCost) -> u16 {
        self.mana
            .iter()
            .map(|amount| u16::from(*amount))
            .sum::<u16>()
            + if cost.spend_on_lesson {
                self.lesson.iter().map(|amount| u16::from(*amount)).sum()
            } else {
                0
            }
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
        if cost.spend_on_lesson {
            for (available, restricted) in remaining.iter_mut().zip(self.lesson) {
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

    /// Eligible concrete pool payments, retaining Lesson restrictions and the
    /// restricted-first policy within each color.
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

    fn apply_payment_remainder(&mut self, cost: &ManaCost, remaining: [u8; 6]) {
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

    /// All unspent mana becomes red, keeping its spending restrictions
    /// (Ozai, the Phoenix King).
    pub fn become_red(&mut self) {
        let red = Color::Red as usize;
        for pool in [&mut self.mana, &mut self.lesson] {
            let total = pool
                .iter()
                .fold(0u8, |sum, amount| sum.saturating_add(*amount));
            *pool = [0; 6];
            pool[red] = total;
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
    fn x_counts_zero_until_its_value_is_chosen() {
        let cost = ManaCost::parse("{X}{R}{R}{R}");
        assert_eq!((cost.x, cost.mana_value), (1, 3));
        let paid = cost.with_x(2);
        assert_eq!((paid.x, paid.mana_value, paid.generic()), (0, 5, 2));
        assert_eq!(ManaCost::parse("{X}{X}{B}").with_x(3).generic(), 6);
    }

    #[test]
    fn activation_payment_options_preserve_colors_and_restrictions() {
        let mut pool = Mana::parse("WU");
        pool.lesson[4] = 2;
        let options = pool.payment_options(&ManaCost::parse("1"));
        assert_eq!(options.len(), 2);
        assert!(options.iter().any(|p| p.mana == Mana::parse("W").mana));
        assert!(options.iter().any(|p| p.mana == Mana::parse("U").mana));
        assert!(options.iter().all(|p| p.lesson == pool.lesson));
        assert!(pool.payment_options(&ManaCost::parse("G")).is_empty());
        let cost = ManaCost {
            spend_on_lesson: true,
            ..ManaCost::parse("1G")
        };
        let options = pool.payment_options(&cost);
        assert!(options
            .iter()
            .any(|p| p.mana == pool.mana && p.lesson == [0; 6]));
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
}
