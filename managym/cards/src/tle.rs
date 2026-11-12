//! Avatar: The Last Airbender Eternal.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TriggerCondition},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, TapCost},
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![the_cabbage_merchant()]
}

fn the_cabbage_merchant() -> CardDefinition {
    let text_box = "Whenever an opponent casts a noncreature spell, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever a creature deals combat damage to you, sacrifice a Food token.\nTap two untapped Foods you control: Add one mana of any color.";
    CardDefinition {
        key: "tle.the_cabbage_merchant",
        scryfall_id: "2fea0356-6684-4730-9eb4-0262856bc1f9",
        name: "The Cabbage Merchant".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::OpponentCastsSpell {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::create_tokens("Food", 1)],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::CreatureDealsCombatDamageToYou,
                effects: vec![Effect::Sacrifice {
                    optional: false,
                    what: ObjectSelector::chosen(CardPredicate {
                        subtype: Some("Food".into()),
                        token: Some(true),
                        ..Default::default()
                    }),
                }],
            },
        ],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_cost: Some(TapCost {
                predicate: CardPredicate::subtype("Food"),
                count: 2,
            }),
            effects: vec![Effect::mana_choice(
                ["W", "U", "B", "R", "G"].map(Mana::parse),
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
