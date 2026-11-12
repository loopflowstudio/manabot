//! Commander 2019.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, Scope, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dockside_extortionist()]
}

fn dockside_extortionist() -> CardDefinition {
    let text_box = "When this creature enters, create X Treasure tokens, where X is the number of artifacts and enchantments your opponents control. (Treasure tokens are artifacts with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    let artifacts_and_enchantments = CardPredicate {
        card_types_any: vec![CardType::Artifact, CardType::Enchantment],
        ..Default::default()
    };
    CardDefinition {
        key: "c19.dockside_extortionist",
        scryfall_id: "571bc9eb-8d13-4008-86b5-2e348a326d58",
        name: "Dockside Extortionist".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Goblin".into(), "Pirate".into()],
        power: Some(1),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::create_tokens(
                "Treasure",
                EffectValue::Objects(Box::new(ObjectSelector::each(
                    Scope::OpponentsControl,
                    artifacts_and_enchantments,
                ))),
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
