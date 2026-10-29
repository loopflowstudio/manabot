//! Murders at Karlov Manor.
use crate::state::{
    ability::{
        Ability, Effect, EffectValue, StaticCondition, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![proft_s_eidetic_memory()]
}

fn proft_s_eidetic_memory() -> CardDefinition {
    let drawn_twice = StaticCondition::CardsDrawnAtLeast { count: 2 };
    CardDefinition {
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
                        otherwise: vec![],
                    }],
                },
            ],
            text_box: "When Proft's Eidetic Memory enters, draw a card.\nYou have no maximum hand size.\nAt the beginning of combat on your turn, if you've drawn more than one card this turn, put X +1/+1 counters on target creature you control, where X is the number of cards you've drawn this turn minus one.".into(),
            ..Default::default()
        }
}
