// predicate.rs
// Reusable structural predicates over cards, shared by trigger subjects,
// blocking restrictions, and zone counting (no scattered string compares).

use super::card::{Card, CardType};

/// A structural predicate over cards. Every populated field must match.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CardPredicate {
    /// Card must have this card type.
    pub card_type: Option<CardType>,
    /// Card must have at least one of these card types ("artifact,
    /// creature, or enchantment").
    pub card_types_any: Vec<CardType>,
    /// Card must have none of these card types ("noncreature, nonland").
    pub not_card_types: Vec<CardType>,
    /// Card must have this subtype (e.g. "Ally", "Lesson").
    pub subtype: Option<String>,
    /// Card's power must be `max_power` or less ("power N or less").
    /// Checked against effective power when a permanent is in play.
    pub max_power: Option<i32>,
    /// Card's mana value must be `min_mana_value` or greater ("with mana
    /// value 3 or greater"). Cards without a mana cost have mana value 0.
    pub min_mana_value: Option<u8>,
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
        self.matches(card, power, false)
    }

    /// Full matching: `power` substitutes the card's printed power and
    /// `animated_creature` treats the card as having the creature type in
    /// addition to its printed types (earthbent lands).
    pub fn matches(&self, card: &Card, power: i32, animated_creature: bool) -> bool {
        let has_type = |card_type: CardType| {
            card.types.types.contains(&card_type)
                || (animated_creature && card_type == CardType::Creature)
        };
        if let Some(card_type) = self.card_type {
            if !has_type(card_type) {
                return false;
            }
        }
        if !self.card_types_any.is_empty()
            && !self.card_types_any.iter().any(|t| has_type(*t))
        {
            return false;
        }
        if self.not_card_types.iter().any(|t| has_type(*t)) {
            return false;
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
        if let Some(min_mana_value) = self.min_mana_value {
            let mana_value = card
                .mana_cost
                .as_ref()
                .map(|cost| cost.mana_value)
                .unwrap_or(0);
            if mana_value < min_mana_value {
                return false;
            }
        }
        true
    }
}
