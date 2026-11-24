//! Rivals of Ixalan.
use managym_state::{
    ability::{Aggregate, Effect, ObjectProperty, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        brasss_bounty(),
        ghalta_primal_hunger(),
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

fn brasss_bounty() -> CardDefinition {
    let text_box = "For each land you control, create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "rix.brasss_bounty",
        scryfall_id: "13505866-6ce9-4d11-b3f3-fc2f09839b71",
        printed: PrintedCharacteristics {
            name: "Brass's Bounty".into(),
            mana_cost: Some(ManaCost::parse("{6}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens(
                "token.treasure",
                ObjectSelector::each_you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                })
                .count(),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ghalta_primal_hunger() -> CardDefinition {
    let text_box = "This spell costs {X} less to cast, where X is the total power of creatures you control.\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "rix.ghalta_primal_hunger",
        scryfall_id: "0104b5b3-9376-4ad7-9a77-3e564e9c42e6",
        printed: PrintedCharacteristics {
            name: "Ghalta, Primal Hunger".into(),
            mana_cost: Some(ManaCost::parse("{10}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Elder".into(), "Dinosaur".into()],
            power: Some(12),
            toughness: Some(12),
            keywords: Keywords::of(&[Keyword::Trample]),
            statics: vec![StaticAbility::CostsLess {
                amount: ObjectSelector::each_you_control(CardPredicate::creature())
                    .property(ObjectProperty::Power, Aggregate::Sum),
                condition: None,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
