//! New Capenna Commander.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![extravagant_replication()]
}

fn extravagant_replication() -> CardDefinition {
    let target = TargetSpec::PermanentYouControl {
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "ncc.extravagant_replication",
        scryfall_id: "6a6f55d7-d689-43eb-a59a-b8be88269ee6",
        printed: PrintedCharacteristics {
            name: "Extravagant Replication".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::another(target.clone()),
                    effects: vec![Effect::create_token_copies(ObjectSelector::Target(target), 1)],
                }],
            }],
            text_box: "At the beginning of your upkeep, create a token that's a copy of another target nonland permanent you control.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
