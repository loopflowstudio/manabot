use crate::state::{
    ability::{Ability, Effect, TargetController, TargetSpec, TriggerCondition, TriggerSource},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

use super::alpha::CardRegistry;

impl CardRegistry {
    pub(super) fn register_visions(&mut self) {
        self.register_card(CardDefinition {
            name: "Man-o'-War".to_string(),
            mana_cost: Some(ManaCost::parse("2U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Jellyfish".to_string()],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    source: TriggerSource::This,
                },
                effect: Effect::ReturnToHand {
                    target: TargetSpec::Creature {
                        controller: TargetController::Any,
                    },
                },
                intervening_if: None,
            }],
            text_box: "When Man-o'-War enters the battlefield, return target creature to its owner's hand.".to_string(),
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        });
    }
}
