// predicate.rs
// Reusable structural predicates over cards, shared by trigger subjects,
// blocking restrictions, and zone counting (no scattered string compares).

use super::card::{Card, CardType};

/// A structural predicate over cards. Every populated field must match.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CardPredicate {
    /// Card must have this card type.
    pub card_type: Option<CardType>,
    /// Card must have this subtype (e.g. "Ally", "Lesson").
    pub subtype: Option<String>,
    /// Card's power must be `max_power` or less ("power N or less").
    /// Checked against effective power when a permanent is in play.
    pub max_power: Option<i32>,
}

impl CardPredicate {
    pub fn creature() -> Self {
        Self {
            card_type: Some(CardType::Creature),
            ..Self::default()
        }
    }

    pub fn subtype(subtype: &str) -> Self {
        Self {
            subtype: Some(subtype.to_string()),
            ..Self::default()
        }
    }

    /// Match against a card's printed characteristics.
    pub fn matches_card(&self, card: &Card) -> bool {
        self.matches_card_with_power(card, card.power.unwrap_or(0))
    }

    /// Match against a card, substituting `power` (e.g. a permanent's
    /// effective power including counters and until-EOT modifiers).
    pub fn matches_card_with_power(&self, card: &Card, power: i32) -> bool {
        if let Some(card_type) = self.card_type {
            if !card.types.types.contains(&card_type) {
                return false;
            }
        }
        if let Some(subtype) = &self.subtype {
            if !card.has_subtype(subtype) {
                return false;
            }
        }
        if let Some(max_power) = self.max_power {
            if power > max_power {
                return false;
            }
        }
        true
    }
}
