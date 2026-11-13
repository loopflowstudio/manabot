//! Visions.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![man_o_war(), prosperity()]
}

fn man_o_war() -> CardDefinition {
    let text_box = "When this creature enters, return target creature to its owner's hand.";

    CardDefinition {
        key: "vis.man_o_war",
        scryfall_id: "4dbf9bf9-75cd-4b25-a3a1-43b7e029700b",
        printed: PrintedCharacteristics {
            name: "Man-o'-War".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Jellyfish".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::return_to_hand(ObjectSelector::Target(
                    TargetSpec::Creature,
                ))],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn prosperity() -> CardDefinition {
    let text_box = "Each player draws X cards.";
    CardDefinition {
        key: "vis.prosperity",
        scryfall_id: "3fa5e806-3cf2-4241-b45d-a05d2b715efd",
        printed: PrintedCharacteristics {
            name: "Prosperity".into(),
            mana_cost: Some(ManaCost::parse("{X}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::DrawCards {
                    count: EffectValue::X,
                },
                Effect::ForPlayer {
                    player: PlayerSelector::EachOpponent,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::X,
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
