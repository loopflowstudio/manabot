use crate::state::{
    ability::{Ability, Effect, EffectValue, TriggerCondition, TriggerSubject},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::{Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![prophetic_prism()]
}

fn prophetic_prism() -> CardDefinition {
    let text_box = "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color.";

    CardDefinition {
        key: "roe.prophetic_prism",
        scryfall_id: "cfb90d44-8cb1-4b83-b2f2-92c19d6304fb",
        name: "Prophetic Prism".into(),
        mana_cost: Some(ManaCost::parse("{2}")),
        types: CardTypes::new([CardType::Artifact]),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{1}"),
            effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::AddMana {
                        mana: Mana::parse("W"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("U"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("B"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("R"),
                        until_end_of_combat: false,
                    }],
                    vec![Effect::AddMana {
                        mana: Mana::parse("G"),
                        until_end_of_combat: false,
                    }],
                ],
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
