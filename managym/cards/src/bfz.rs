//! Battle for Zendikar.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        ManaAbility, PrintedCharacteristics, SacrificeCost,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        sure_strike(),
        felidar_cub(),
        hedron_archive(),
        vampiric_rites(),
    ]
}

fn felidar_cub() -> CardDefinition {
    let text_box = "Sacrifice this creature: Destroy target enchantment.";
    CardDefinition {
        key: "bfz.felidar_cub",
        scryfall_id: "ea76a183-e15c-4968-b29d-91c074aa8681",
        printed: PrintedCharacteristics {
            name: "Felidar Cub".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Beast".into()],
            power: Some(2),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                sacrifice_source: true,
                effects: vec![Effect::destroy(ObjectSelector::Target(
                    TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_type: Some(CardType::Enchantment),
                            ..Default::default()
                        },
                    },
                ))],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hedron_archive() -> CardDefinition {
    let text_box = "{T}: Add {C}{C}.\n{2}, {T}, Sacrifice this artifact: Draw two cards.";
    CardDefinition {
        key: "bfz.hedron_archive",
        scryfall_id: "2519149c-f8a3-413f-b7b2-cd596970be4c",
        printed: PrintedCharacteristics {
            name: "Hedron Archive".into(),
            mana_cost: Some(ManaCost::parse("{4}")),
            types: CardTypes::new([CardType::Artifact]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("CC"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vampiric_rites() -> CardDefinition {
    let text_box = "{1}{B}, Sacrifice a creature: You gain 1 life and draw a card.";
    CardDefinition {
        key: "bfz.vampiric_rites",
        scryfall_id: "3416309a-5824-48f5-876e-00e0f180acf9",
        printed: PrintedCharacteristics {
            name: "Vampiric Rites".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{B}"),
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate::creature(),
                    another: false,
                    or_mana: None,
                }),
                effects: vec![
                    Effect::gain_life(1),
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sure_strike() -> CardDefinition {
    let text_box = "Target creature gets +3/+0 and gains first strike until end of turn. (It deals combat damage before creatures without first strike.)";
    CardDefinition {
        key: "bfz.sure_strike",
        scryfall_id: "074dd176-4608-42ca-8fc3-c7040cd7b32e",
        printed: PrintedCharacteristics {
            name: "Sure Strike".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(3, 0),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::AddKeywords(Keywords::of(&[Keyword::FirstStrike])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
