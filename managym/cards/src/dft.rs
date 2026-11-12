//! Aetherdrift cards expressed in the shared execution and replacement rules.
use managym_state::{
    ability::{Ability, Effect, EffectValue, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, ReplacementEffect},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![caradora_heart_of_alacria()]
}

fn caradora_heart_of_alacria() -> CardDefinition {
    let text_box = "When Caradora enters, you may search your library for a Mount or Vehicle card, reveal it, put it into your hand, then shuffle.\nIf one or more +1/+1 counters would be put on a creature or Vehicle you control, that many plus one +1/+1 counters are put on it instead.";
    CardDefinition {
        key: "dft.caradora_heart_of_alacria",
        scryfall_id: "1256d22d-a2a9-41fb-b669-0661ba230bc7",
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
    }
}
