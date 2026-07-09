// strixhaven.rs
// Strixhaven cards used by the Milestone-1 UR Lessons deck: the learn
// spells (learn = "you may discard a card; if you do, draw a card" — no
// sideboard in 1v1 constructed) and a ward proof card.

use crate::state::{
    ability::{Effect, TargetRequirement, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

use super::alpha::CardRegistry;

impl CardRegistry {
    pub(super) fn register_strixhaven(&mut self) {
        // Pop Quiz {2}{U} Instant
        // Draw a card. Learn.
        self.register_card(CardDefinition {
            name: "Pop Quiz".to_string(),
            mana_cost: Some(ManaCost::parse("2U")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::DrawCards { count: 1 }, Effect::Learn],
            text_box: "Draw a card.\nLearn. (You may discard a card. If you do, draw a card.)"
                .to_string(),
            ..Default::default()
        });

        // Igneous Inspiration {2}{R} Sorcery
        // Igneous Inspiration deals 3 damage to any target. Learn.
        self.register_card(CardDefinition {
            name: "Igneous Inspiration".to_string(),
            mana_cost: Some(ManaCost::parse("2R")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::DealDamage {
                    amount: 3,
                    target: TargetSpec::CreatureOrPlayer,
                },
                Effect::Learn,
            ],
            text_box: "Igneous Inspiration deals 3 damage to any target.\nLearn.".to_string(),
            ..Default::default()
        });

        // Divide by Zero {2}{U} Instant
        // Return target spell or permanent with mana value 1 or greater to
        // its owner's hand. Learn.
        self.register_card(CardDefinition {
            name: "Divide by Zero".to_string(),
            mana_cost: Some(ManaCost::parse("2U")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::SpellOrPermanent {
                min_mana_value: 1,
            })],
            spell_effects: vec![
                Effect::ReturnToHand {
                    target: TargetSpec::SpellOrPermanent { min_mana_value: 1 },
                },
                Effect::Learn,
            ],
            text_box: "Return target spell or permanent with mana value 1 or greater to its owner's hand.\nLearn.".to_string(),
            ..Default::default()
        });

        // Waterfall Aerialist {2}{U} 2/4 — Djinn
        // Flying, Ward {2}. (Ward-machinery proof card; Dragonfly Swarm
        // arrives with Stage 3's dynamic P/T.)
        self.register_card(CardDefinition {
            name: "Waterfall Aerialist".to_string(),
            mana_cost: Some(ManaCost::parse("2U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Djinn".into()],
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            ward: Some(ManaCost::parse("2")),
            text_box: "Flying\nWard {2} (Whenever this creature becomes the target of a spell an opponent controls, counter it unless that player pays {2}.)".to_string(),
            power: Some(2),
            toughness: Some(4),
            ..Default::default()
        });
    }
}
