//! Magic 2010.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, ObjectSelector, PaymentCost, TriggerCondition},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        StaticAbility, StaticPtBuff, StaticScope,
    },
    mana::{Color, Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![elvish_archdruid(), mold_adder()]
}

fn elvish_archdruid() -> CardDefinition {
    let text_box =
        "Other Elf creatures you control get +1/+1.\n{T}: Add {G} for each Elf you control.";
    let elf = CardPredicate {
        subtype: Some("Elf".into()),
        ..Default::default()
    };
    CardDefinition {
        key: "m10.elvish_archdruid",
        scryfall_id: "544bc214-e32d-4370-a15f-62812a0420be",
        printed: PrintedCharacteristics {
            name: "Elvish Archdruid".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Druid".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::OtherYouControl(CardPredicate {
                    card_type: Some(CardType::Creature),
                    ..elf.clone()
                }),
                condition: None,
                power: 1,
                toughness: 1,
                per_creature_type_max: None,
                per_color_among: None,
                per_graveyard_matching: None,
            })],
            // A mana ability whose amount is read as it is activated (CR
            // 605.1a, 605.3a): characteristics finalize it to concrete mana.
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("G"),
                    amount: EffectValue::BattlefieldMatching { predicate: elf },
                    until_end_of_combat: false,
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mold_adder() -> CardDefinition {
    let text_box = "Whenever an opponent casts a blue or black spell, you may put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "m10.mold_adder",
        scryfall_id: "a216a729-6283-4c2b-90fe-ec8f3b9c570f",
        printed: PrintedCharacteristics {
            name: "Mold Adder".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Fungus".into(), "Snake".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::Opponent,
                    CardPredicate {
                        colors_any: vec![Color::Blue, Color::Black],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                    if_declined: vec![],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
