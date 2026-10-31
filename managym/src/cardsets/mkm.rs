//! Murders at Karlov Manor.
use crate::state::{
    ability::{
        Ability, Condition, Effect, EffectValue, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![proft_s_eidetic_memory()]
}

fn proft_s_eidetic_memory() -> CardDefinition {
    let text_box = "When Proft's Eidetic Memory enters, draw a card.\nYou have no maximum hand size.\nAt the beginning of combat on your turn, if you've drawn more than one card this turn, put X +1/+1 counters on target creature you control, where X is the number of cards you've drawn this turn minus one.";

    CardDefinition {
        key: "mkm.proft_s_eidetic_memory",
        scryfall_id: "af5b29b3-974c-4200-8df8-b072c11e1600",
        name: "Proft's Eidetic Memory".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        supertypes: vec!["Legendary".into()],
        statics: vec![StaticAbility::NoMaximumHandSize],
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            },
            Ability::Triggered {
                condition: TriggerCondition::BeginningOfYourCombat,
                intervening_if: Some(Condition::CardsDrawnAtLeast { count: 2 }),
                effects: vec![Effect::PutCountersValue {
                    count: EffectValue::CardsDrawnMinus { subtract: 1 },
                    target: TargetSpec::CreatureYouControl,
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
