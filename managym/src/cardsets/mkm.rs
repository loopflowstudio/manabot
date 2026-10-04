// mkm.rs
// Murders at Karlov Manor cards admitted to custom decklists.

use super::alpha::CardRegistry;
use crate::state::{
    ability::{
        Ability, Effect, EffectValue, StaticCondition, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

impl CardRegistry {
    pub(super) fn register_mkm(&mut self) {
        let drawn_twice = StaticCondition::CardsDrawnAtLeast { count: 2 };
        self.register_card(CardDefinition {
            name: "Proft's Eidetic Memory".into(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Enchantment]),
            supertypes: vec!["Legendary".into()],
            no_maximum_hand_size: true,
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
                    effects: vec![Effect::DrawCards { count: 1 }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::ActiveIf {
                        active_if: drawn_twice.clone(),
                        condition: Box::new(TriggerCondition::BeginningOfYourCombat),
                    },
                    effects: vec![Effect::IfCondition {
                        condition: drawn_twice,
                        effects: vec![Effect::PutCountersValue {
                            count: EffectValue::CardsDrawnMinus { subtract: 1 },
                            target: TargetSpec::CreatureYouControl,
                        }],
                    }],
                },
            ],
            text_box: "When Proft's Eidetic Memory enters, draw a card.\nYou have no maximum hand size.\nAt the beginning of combat on your turn, if you've drawn more than one card this turn, put X +1/+1 counters on target creature you control, where X is the number of cards you've drawn this turn minus one.".into(),
            ..Default::default()
        });
    }
}
