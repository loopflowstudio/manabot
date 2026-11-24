//! Rivals of Ixalan.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        zetalpa_primal_dawn(),
        swab_goblin(),
        gleaming_barrier(),
        fanatical_firebrand(),
        thrashing_brontodon(),
    ]
}

fn fanatical_firebrand() -> CardDefinition {
    let text_box = "Haste (This creature can attack and {T} as soon as it comes under your control.)\n{T}, Sacrifice this creature: It deals 1 damage to any target.";
    CardDefinition {
        key: "rix.fanatical_firebrand",
        scryfall_id: "5e5565de-028c-4799-a9f6-4dcd685639eb",
        printed: PrintedCharacteristics {
            name: "Fanatical Firebrand".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Pirate".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Haste]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::deal_damage(
                    1,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn thrashing_brontodon() -> CardDefinition {
    let text_box = "{1}, Sacrifice this creature: Destroy target artifact or enchantment.";
    let artifact_or_enchantment = CardPredicate {
        card_types_any: vec![CardType::Artifact, CardType::Enchantment],
        ..Default::default()
    };
    CardDefinition {
        key: "rix.thrashing_brontodon",
        scryfall_id: "0d9264ff-9f7c-46f3-862a-fee7ad213250",
        printed: PrintedCharacteristics {
            name: "Thrashing Brontodon".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(3),
            toughness: Some(4),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                sacrifice_source: true,
                effects: vec![Effect::destroy(ObjectSelector::Target(
                    TargetSpec::Permanent {
                        predicate: artifact_or_enchantment,
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

fn gleaming_barrier() -> CardDefinition {
    let text_box = "Defender\nWhen this creature dies, create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "rix.gleaming_barrier",
        scryfall_id: "62447a76-4aa7-4823-941e-84bc18eb672a",
        printed: PrintedCharacteristics {
            name: "Gleaming Barrier".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Wall".into()],
            power: Some(0),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Defender]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::create_tokens("token.treasure", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn swab_goblin() -> CardDefinition {
    CardDefinition {
        key: "rix.swab_goblin",
        scryfall_id: "0403b8e5-29c5-4a4a-b9e7-1e79a7452f14",
        printed: PrintedCharacteristics {
            name: "Swab Goblin".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn zetalpa_primal_dawn() -> CardDefinition {
    let text_box = "Flying, double strike, vigilance, trample, indestructible";
    CardDefinition {
        key: "rix.zetalpa_primal_dawn",
        scryfall_id: "3d10560f-199d-4d04-b573-90024f8aecc4",
        printed: PrintedCharacteristics {
            name: "Zetalpa, Primal Dawn".into(),
            mana_cost: Some(ManaCost::parse("{6}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elder".into(), "Dinosaur".into()],
            power: Some(4),
            toughness: Some(8),
            supertypes: vec!["Legendary".into()],
            keywords: Keywords::of(&[
                Keyword::Flying,
                Keyword::Indestructible,
                Keyword::Vigilance,
                Keyword::Trample,
                Keyword::DoubleStrike,
            ]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
