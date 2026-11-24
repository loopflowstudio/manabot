//! March of the Machine.

use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![wary_thespian(), elspeths_smite()]
}

fn wary_thespian() -> CardDefinition {
    let text_box = "When this creature enters or dies, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "mom.wary_thespian",
        scryfall_id: "675b29bf-0b64-410f-9a92-c88e5615c27f",
        printed: PrintedCharacteristics {
            name: "Wary Thespian".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Druid".into()],
            power: Some(3),
            toughness: Some(1),
            abilities: vec![
                Ability::when_enters(TriggerSubject::This, vec![Effect::Surveil { count: 1 }]),
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::Surveil { count: 1 }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elspeths_smite() -> CardDefinition {
    let text_box = "Elspeth's Smite deals 3 damage to target attacking or blocking creature. If that creature would die this turn, exile it instead.";
    let target = || {
        ObjectSelector::Target(TargetSpec::Permanent {
            predicate: CardPredicate {
                in_combat: true,
                ..CardPredicate::creature()
            },
        })
    };
    CardDefinition {
        key: "mom.elspeths_smite",
        scryfall_id: "f03a480f-de67-4611-9db7-c0c3d020f597",
        printed: PrintedCharacteristics {
            name: "Elspeth's Smite".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::deal_damage(3, target()),
                Effect::until_end_of_turn(target(), Modification::ExileIfWouldDie),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
