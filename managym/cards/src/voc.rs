//! Crimson Vow Commander.
use managym_state::{
    ability::{Ability, Effect, PaymentCost, TriggerCondition, TriggerSubject},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility, StaticKeywordGrant, StaticScope,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![crossway_troublemakers()]
}

fn crossway_troublemakers() -> CardDefinition {
    let vampire = CardPredicate::subtype("Vampire");
    let text_box = "Attacking Vampires you control have deathtouch and lifelink. (Any amount of damage they deal to a creature is enough to destroy it. Damage dealt by those creatures also causes their controller to gain that much life.)\nWhenever a Vampire you control dies, you may pay 2 life. If you do, draw a card.";
    CardDefinition {
        key: "voc.crossway_troublemakers",
        scryfall_id: "431711c5-c04f-4d34-97c9-5199cfbf9da9",
        printed: PrintedCharacteristics {
            name: "Crossway Troublemakers".into(),
            mana_cost: Some(ManaCost::parse("{5}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into()],
            power: Some(5),
            toughness: Some(5),
            statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::YouControl(CardPredicate {
                    attacking: true,
                    ..vampire.clone()
                }),
                condition: None,
                keywords: Keywords::of(&[Keyword::Deathtouch, Keyword::Lifelink]),
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(vampire),
                },
                intervening_if: None,
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Life(2),
                    if_paid: vec![Effect::DrawCards { count: 1.into() }],
                    if_declined: vec![],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
