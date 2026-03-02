use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ManaCost {
    pub cost: [u8; 7],
    pub mana_value: u8,
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

        out.mana_value = out.cost.iter().copied().sum();
        out
    }

    pub fn colors(&self) -> Colors {
        let mut out = Colors::new();
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Mana {
    pub mana: [u8; 6],
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
        }
    }

    pub fn total(&self) -> u16 {
        self.mana.iter().map(|x| *x as u16).sum()
    }

    pub fn can_pay(&self, cost: &ManaCost) -> bool {
        if self.total() < cost.mana_value as u16 {
            return false;
        }

        let mut remaining = self.mana;
        for color in [
            Color::White,
            Color::Blue,
            Color::Black,
            Color::Red,
            Color::Green,
            Color::Colorless,
        ] {
            let idx = color as usize;
            if remaining[idx] < cost.cost[idx] {
                return false;
            }
            remaining[idx] -= cost.cost[idx];
        }

        let generic_needed = cost.cost[Color::Generic as usize] as u16;
        let generic_available: u16 = remaining.iter().map(|x| *x as u16).sum();
        generic_available >= generic_needed
    }

    pub fn pay(&mut self, cost: &ManaCost) {
        assert!(self.can_pay(cost), "insufficient mana");

        for color in [
            Color::White,
            Color::Blue,
            Color::Black,
            Color::Red,
            Color::Green,
            Color::Colorless,
        ] {
            let idx = color as usize;
            self.mana[idx] -= cost.cost[idx];
        }

        let mut generic = cost.cost[Color::Generic as usize] as i32;
        // Spend colorless first, then colored (preserve colored for future casts)
        let priority = [
            Color::Colorless as usize,
            Color::White as usize,
            Color::Blue as usize,
            Color::Black as usize,
            Color::Red as usize,
            Color::Green as usize,
        ];
        while generic > 0 {
            for &idx in &priority {
                while generic > 0 && self.mana[idx] > 0 {
                    self.mana[idx] -= 1;
                    generic -= 1;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.mana = [0; 6];
    }
}

#[cfg(test)]
mod tests {
    use super::{Mana, ManaCost};

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
