//! Innistrad: Midnight Hunt.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics, SacrificeCost, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        cathar_commando(),
        consider(),
        eaten_alive(),
        join_the_dance(),
    ]
}

fn consider() -> CardDefinition {
    let text_box = "Surveil 1. (Look at the top card of your library. You may put it into your graveyard.)\nDraw a card.";
    CardDefinition {
        key: "mid.consider",
        scryfall_id: "a211d505-4d40-4914-a9da-220770d6ddbc",
        printed: PrintedCharacteristics {
            name: "Consider".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::Surveil { count: 1 },
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn join_the_dance() -> CardDefinition {
    let text_box = "Create two 1/1 white Human creature tokens.\nFlashback {3}{G}{W} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "mid.join_the_dance",
        scryfall_id: "56b30a99-601b-40b9-b012-30fa4be5fd3c",
        printed: PrintedCharacteristics {
            name: "Join the Dance".into(),
            mana_cost: Some(ManaCost::parse("{G}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens("token.human", 2)],
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{3}{G}{W}"))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cathar_commando() -> CardDefinition {
    let text_box = "Flash\n{1}, Sacrifice this creature: Destroy target artifact or enchantment.";
    let artifact_or_enchantment = CardPredicate {
        card_types_any: vec![CardType::Artifact, CardType::Enchantment],
        ..Default::default()
    };
    CardDefinition {
        key: "mid.cathar_commando",
        scryfall_id: "98cbc1c2-b76e-4da3-aa43-00e10b2ce532",
        printed: PrintedCharacteristics {
            name: "Cathar Commando".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(3),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
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

fn eaten_alive() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, sacrifice a creature or pay {3}{B}.\nExile target creature or planeswalker.";
    CardDefinition {
        key: "mid.eaten_alive",
        scryfall_id: "e7975a3c-570a-4bff-a60d-d274f758b93f",
        printed: PrintedCharacteristics {
            name: "Eaten Alive".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::Sacrifice(Box::new(SacrificeCost {
                predicate: CardPredicate::creature(),
                another: false,
                or_mana: Some(ManaCost::parse("{3}{B}")),
            }))],
            spell_effects: vec![Effect::exile(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                        ..Default::default()
                    },
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
