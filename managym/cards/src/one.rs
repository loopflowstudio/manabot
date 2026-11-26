//! Phyrexia: All Will Be One.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PaymentCost, TargetSpec,
        TriggerCondition,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![ovika_enigma_goliath(), offer_immortality()]
}

fn ovika_enigma_goliath() -> CardDefinition {
    let text_box = "Flying\nWard—{3}, Pay 3 life.\nWhenever you cast a noncreature spell, create X 1/1 red Phyrexian Goblin creature tokens, where X is the mana value of that spell. They gain haste until end of turn.";
    CardDefinition {
        key: "one.ovika_enigma_goliath",
        scryfall_id: "b298cf34-7aa5-4f97-a86c-7f28d2113b87",
        printed: PrintedCharacteristics {
            name: "Ovika, Enigma Goliath".into(),
            mana_cost: Some(ManaCost::parse("{5}{U}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Phyrexian".into(), "Nightmare".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::Ward(PaymentCost::All(vec![
                PaymentCost::Mana(ManaCost::parse("{3}")),
                PaymentCost::Life(3),
            ]))],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                ),
                effects: vec![
                    Effect::create_tokens(
                        "token.phyrexian_goblin_1_1_r",
                        EffectValue::TriggeringManaValue,
                    ),
                    Effect::until_end_of_turn(
                        ObjectSelector::CreatedThisWay,
                        Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn offer_immortality() -> CardDefinition {
    let text_box = "Target creature gains deathtouch and indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";
    CardDefinition {
        key: "one.offer_immortality",
        scryfall_id: "b0aac10a-6d47-4a6c-8a10-2b7c06f3ff32",
        printed: PrintedCharacteristics {
            name: "Offer Immortality".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords::of(&[
                    Keyword::Deathtouch,
                    Keyword::Indestructible,
                ])),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
