//! Invasion.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![opt(), captain_sisay()]
}

fn captain_sisay() -> CardDefinition {
    let text_box = "{T}: Search your library for a legendary card, reveal that card, put it into your hand, then shuffle.";
    CardDefinition {
        key: "inv.captain_sisay",
        scryfall_id: "d24d441c-f37f-44fe-8a93-f5c89df807e4",
        printed: PrintedCharacteristics {
            name: "Captain Sisay".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::search_library(
                    CardPredicate {
                        supertype: Some("Legendary".into()),
                        ..Default::default()
                    },
                    EffectValue::Fixed(1),
                    vec![Effect::return_to_hand(ObjectSelector::Selected)],
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn opt() -> CardDefinition {
    let text_box = "Scry 1. (Look at the top card of your library. You may put that card on the bottom.)\nDraw a card.";
    CardDefinition {
        key: "inv.opt",
        scryfall_id: "958262ec-8e52-40cf-a9fd-a60e42643e15",
        printed: PrintedCharacteristics {
            name: "Opt".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::Scry { count: 1 },
                Effect::DrawCards { count: 1.into() },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
