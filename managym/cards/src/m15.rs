//! Magic 2015.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![meteorite()]
}

fn meteorite() -> CardDefinition {
    let text_box = "When this artifact enters, it deals 2 damage to any target.\n{T}: Add one mana of any color.";
    CardDefinition {
        key: "m15.meteorite",
        scryfall_id: "d7a39c1a-7615-4ac8-8984-b8459d201cb2",
        name: "Meteorite".into(),
        mana_cost: Some(ManaCost::parse("{5}")),
        types: CardTypes::new([CardType::Artifact]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::deal_damage(
                2,
                ObjectSelector::Target(TargetSpec::AnyTarget),
            )],
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![Effect::any_color_mana()],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
