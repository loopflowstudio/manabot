use crate::state::{
    ability::Effect,
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

use super::alpha::CardRegistry;

impl CardRegistry {
    pub(super) fn register_ice_age(&mut self) {
        self.register_card(CardDefinition {
            name: "Pyroclasm".to_string(),
            mana_cost: Some(ManaCost::parse("1R")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effect: Some(Effect::MassDamage { amount: 2 }),
            text_box: "Pyroclasm deals 2 damage to each creature.".to_string(),
            ..Default::default()
        });
    }
}
