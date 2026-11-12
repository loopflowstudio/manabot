//! Avatar: The Last Airbender Eternal.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, ReplacementEffect, TapCost,
    },
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        capital_guard(),
        explosive_shot(),
        kindly_customer(),
        koala_sheep(),
        kyoshi_warrior_guard(),
        the_cabbage_merchant(),
        warship_scout(),
        wolf_cove_villager(),
        zukos_offense(),
    ]
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

fn capital_guard() -> CardDefinition {
    let text_box = "";
    CardDefinition {
        key: "tle.capital_guard",
        scryfall_id: "91cbed11-3b5c-4e7a-9b13-125c1fe5f22f",
        name: "Capital Guard".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(2),
        toughness: Some(2),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn explosive_shot() -> CardDefinition {
    let text_box = "Explosive Shot deals 4 damage to target creature.";
    CardDefinition {
        key: "tle.explosive_shot",
        scryfall_id: "4ba682a6-1045-4ed6-ae95-bd5af8bd7224",
        name: "Explosive Shot".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::deal_damage(
            4,
            ObjectSelector::Target(TargetSpec::Creature),
        )],
        targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kindly_customer() -> CardDefinition {
    let text_box = "When this creature enters, draw a card.";
    CardDefinition {
        key: "tle.kindly_customer",
        scryfall_id: "84332812-367a-4ac4-9be5-2adc57562c9d",
        name: "Kindly Customer".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        power: Some(1),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn koala_sheep() -> CardDefinition {
    let text_box = "When this creature enters, you gain 3 life.";
    CardDefinition {
        key: "tle.koala_sheep",
        scryfall_id: "7785ddfe-f8ae-473a-8064-549884c5aabc",
        name: "Koala-Sheep".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bear".into(), "Sheep".into()],
        power: Some(3),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::gain_life(3)],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kyoshi_warrior_guard() -> CardDefinition {
    let text_box = "";
    CardDefinition {
        key: "tle.kyoshi_warrior_guard",
        scryfall_id: "7a2acb28-c0e2-492b-a227-ef49e905e0fb",
        name: "Kyoshi Warrior Guard".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn warship_scout() -> CardDefinition {
    let text_box = "";
    CardDefinition {
        key: "tle.warship_scout",
        scryfall_id: "b1a95982-be16-465a-9c1b-1f4d875c0c40",
        name: "Warship Scout".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Scout".into()],
        power: Some(2),
        toughness: Some(1),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wolf_cove_villager() -> CardDefinition {
    let text_box = "This creature enters tapped.";
    CardDefinition {
        key: "tle.wolf_cove_villager",
        scryfall_id: "7dbeced9-d27d-476c-92d4-3c14d8a40458",
        name: "Wolf Cove Villager".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Peasant".into()],
        power: Some(2),
        toughness: Some(2),
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zukos_offense() -> CardDefinition {
    let text_box = "Zuko's Offense deals 2 damage to any target.";
    CardDefinition {
        key: "tle.zukos_offense",
        scryfall_id: "efc6eea1-9872-47e5-a3bc-3c8c1b9705be",
        name: "Zuko's Offense".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::deal_damage(
            2,
            ObjectSelector::Target(TargetSpec::AnyTarget),
        )],
        targeting: vec![TargetRequirement::one(TargetSpec::AnyTarget)],
        text_box: text_box.into(),
        ..Default::default()
    }
}
