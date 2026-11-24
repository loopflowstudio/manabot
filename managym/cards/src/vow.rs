//! Innistrad: Crimson Vow.
use managym_state::{
    ability::{
        Ability, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![undying_malice()]
}

fn undying_malice() -> CardDefinition {
    let text_box = "Until end of turn, target creature gains \"When this creature dies, return it to the battlefield tapped under its owner's control with a +1/+1 counter on it.\"";
    CardDefinition {
        key: "vow.undying_malice",
        scryfall_id: "8eb38041-043a-4b18-9d9a-f1283684e8f1",
        printed: PrintedCharacteristics {
            name: "Undying Malice".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::GrantTriggered(Box::new(Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::return_to_owner(
                        ObjectSelector::This,
                        true,
                        Some(CounterKind::PlusOnePlusOne.into()),
                    )],
                })),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
