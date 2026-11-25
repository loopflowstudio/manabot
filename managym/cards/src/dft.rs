//! Aetherdrift cards expressed in the shared execution and replacement rules.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, Per, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, ReplacementEffect,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![alacrian_jaguar(), caradora_heart_of_alacria()]
}

fn caradora_heart_of_alacria() -> CardDefinition {
    let text_box = "When Caradora enters, you may search your library for a Mount or Vehicle card, reveal it, put it into your hand, then shuffle.\nIf one or more +1/+1 counters would be put on a creature or Vehicle you control, that many plus one +1/+1 counters are put on it instead.";
    CardDefinition {
        key: "dft.caradora_heart_of_alacria",
        scryfall_id: "1256d22d-a2a9-41fb-b669-0661ba230bc7",
        printed: PrintedCharacteristics {
            name: "Caradora, Heart of Alacria".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Knight".into()],
            power: Some(4),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![
                            Effect::SelectCards {
                                from: ObjectSelector::Zone(ZoneType::Library),
                                predicate: CardPredicate {
                                    subtypes_any: vec!["Mount".into(), "Vehicle".into()],
                                    ..Default::default()
                                },
                                min: 0,
                                max: EffectValue::Fixed(1),
                                effects: vec![Effect::return_to_hand(ObjectSelector::Selected)],
                                if_declined: vec![],
                            },
                            Effect::ShuffleLibrary,
                        ],
                        vec![],
                    ],
                }],
            }],
            replacement_effects: vec![ReplacementEffect::ModifyCounters {
                effects_only: false,
                kind: Some(CounterKind::PlusOnePlusOne),
                predicate: CardPredicate {
                    any_of: vec![CardPredicate::creature(), CardPredicate::subtype("Vehicle")],
                    ..Default::default()
                },
                multiplier: 1,
                additional: 1,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn alacrian_jaguar() -> CardDefinition {
    let text_box = "Vigilance\nWhenever this creature attacks while saddled, it gets +2/+2 until end of turn.\nSaddle 1 (Tap any number of other creatures you control with total power 1 or more: This Mount becomes saddled until end of turn. Saddle only as a sorcery.)";
    CardDefinition {
        key: "dft.alacrian_jaguar",
        scryfall_id: "2bd40bca-aa54-4e52-8d48-b3709a11e633",
        printed: PrintedCharacteristics {
            name: "Alacrian Jaguar".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Mount".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::SourceSaddled,
                    condition: Box::new(TriggerCondition::attacks(
                        TriggerSubject::This,
                        Per::Object,
                    )),
                },
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(2, 2),
                )],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                saddle: Some(1),
                sorcery_timing: true,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::Saddled,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
