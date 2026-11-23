//! Amonkhet.
use managym_state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, LayerValue, PrintedCharacteristics,
        StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![regal_caracal()]
}

fn regal_caracal() -> CardDefinition {
    CardDefinition {
        key: "akh.regal_caracal",
        scryfall_id: "cd349f95-3eae-4ef2-abf8-e911bb8e93e5",
        printed: PrintedCharacteristics {
            name: "Regal Caracal".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(3),
            toughness: Some(3),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::OtherYouControl(CardPredicate::subtype("Cat")),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt { power: LayerValue::Fixed(1), toughness: LayerValue::Fixed(1) }, Modification::AddKeywords(Keywords::of(&[Keyword::Lifelink]))],
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
                intervening_if: None,
                effects: vec![Effect::create_tokens("token.cat_1_1_w_lifelink", 2)],
            }],
            text_box: "Other Cats you control get +1/+1 and have lifelink.\nWhen this creature enters, create two 1/1 white Cat creature tokens with lifelink.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
