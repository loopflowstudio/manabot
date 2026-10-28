// temur.rs
// Non-TLA cards retained alongside the shared typed TLA catalog.

use super::alpha::CardRegistry;
use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

impl CardRegistry {
    pub(super) fn register_temur(&mut self) {
        self.register_card(CardDefinition {
            name: "Prophetic Prism".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec![],
            subtypes: vec![],
            text_box:
                "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color."
                    .into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrawCards { count: 1 }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effect: Effect::any_color_mana(),
                ..Default::default()
            }],
            ..Default::default()
        });
    }
}
