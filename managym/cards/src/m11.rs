//! Magic 2011.
use managym_state::{
    ability::{Ability, Effect, LinkedExile, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, ActivationZone, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics,
    },
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        CardDefinition {
            key: "m11.ajanis_pridemate",
            scryfall_id: "f70d1452-6b61-4c63-841f-4256ac498e9f",
            printed: PrintedCharacteristics {
                name: "Ajani's Pridemate".into(),
                mana_cost: Some(ManaCost::parse("{1}{W}")),
                types: CardTypes::new([CardType::Creature]),
                subtypes: vec!["Cat".into(), "Soldier".into()],
                power: Some(2),
                toughness: Some(2),
                abilities: vec![Ability::when_you_gain_life(Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                ))],
                text_box: "Whenever you gain life, put a +1/+1 counter on this creature.".into(),
                ..Default::default()
            },
            ..Default::default()
        },
        reassembling_skeleton(),
        hoarding_dragon(),
    ]
}

fn reassembling_skeleton() -> CardDefinition {
    let rules_text = "{1}{B}: Return this card from your graveyard to the battlefield tapped.";
    CardDefinition {
        key: "m11.reassembling_skeleton",
        scryfall_id: "a2ec6120-2de0-4166-be85-2d773fa7ebbb",
        printed: PrintedCharacteristics {
            name: "Reassembling Skeleton".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Skeleton".into(), "Warrior".into()],
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                zone: ActivationZone::Graveyard,
                mana_cost: ManaCost::parse("{1}{B}"),
                effects: vec![Effect::return_to_battlefield(ObjectSelector::This, true)],
                ..Default::default()
            }],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hoarding_dragon() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, you may search your library for an artifact card, exile it, then shuffle.\nWhen this creature dies, you may put the exiled card into its owner's hand.";
    CardDefinition {
        key: "m11.hoarding_dragon",
        scryfall_id: "1f8b6932-e62d-4d38-bd0e-9ab8d4a56762",
        printed: PrintedCharacteristics {
            name: "Hoarding Dragon".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![
                Ability::enters(vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::search_library(
                            CardPredicate {
                                card_type: Some(CardType::Artifact),
                                ..Default::default()
                            },
                            1.into(),
                            vec![Effect::exile_linked(
                                ObjectSelector::Selected,
                                LinkedExile::WithSource,
                            )],
                        )],
                        vec![],
                    ],
                }]),
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::Modal {
                        modes: vec![
                            vec![Effect::return_to_hand(ObjectSelector::ExiledWithSource)],
                            vec![],
                        ],
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
