//! Rivals of Ixalan.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fanatical_firebrand(), thrashing_brontodon()]
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
