//! Ikoria: Lair of Behemoths.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, SacrificeCost},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bastion_of_remembrance(), weaponize_the_monsters()]
}

fn bastion_of_remembrance() -> CardDefinition {
    let text_box = "When this enchantment enters, create a 1/1 white Human Soldier creature token.\nWhenever a creature you control dies, each opponent loses 1 life and you gain 1 life.";

    CardDefinition {
        key: "iko.bastion_of_remembrance",
        scryfall_id: "2dd354dd-939e-4b1a-8ed6-fe89a7fd64bf",
        name: "Bastion of Remembrance".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::create_tokens("Human Soldier", 1)],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::AnyYouControl(CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::DrainLife {
                    amount: EffectValue::Fixed(1),
                }],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn weaponize_the_monsters() -> CardDefinition {
    let text_box = "{2}, Sacrifice a creature: This enchantment deals 2 damage to any target.";

    CardDefinition {
        key: "iko.weaponize_the_monsters",
        scryfall_id: "68bba622-a0ab-4c0e-88b1-9120690ea5a0",
        name: "Weaponize the Monsters".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            sacrifice: Some(SacrificeCost {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                another: false,
                or_mana: None,
            }),
            mana_cost: ManaCost::parse("{2}"),
            effects: vec![Effect::deal_damage(
                2,
                ObjectSelector::Target(TargetSpec::CreatureOrPlayer),
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
