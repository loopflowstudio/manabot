//! Guilds of Ravnica.
use managym_state::{
    ability::{Ability, Effect, EffectValue, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![healer_s_hawk(), midnight_reaper()]
}

fn midnight_reaper() -> CardDefinition {
    let text_box = "Whenever a nontoken creature you control dies, this creature deals 1 damage to you and you draw a card.";
    CardDefinition {
        key: "grn.midnight_reaper",
        scryfall_id: "5e122fe0-51c5-404d-a7b9-3d161a426c35",
        printed: PrintedCharacteristics {
            name: "Midnight Reaper".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into(), "Knight".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                // Includes its own death (CR 603.10a).
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate {
                        token: Some(false),
                        ..CardPredicate::creature()
                    }),
                },
                intervening_if: None,
                effects: vec![
                    Effect::deal_damage(1, ObjectSelector::You),
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn healer_s_hawk() -> CardDefinition {
    let text_box =
        "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)";

    CardDefinition {
        key: "grn.healer_s_hawk",
        scryfall_id: "3313bd5c-b657-47a3-822a-dd0d9165492a",
        printed: PrintedCharacteristics {
            name: "Healer's Hawk".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
