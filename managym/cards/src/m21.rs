//! Core Set 2021.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![shipwreck_dowser(), heartfire_immolator()]
}

fn shipwreck_dowser() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nWhen this creature enters, return target instant or sorcery card from your graveyard to your hand.";
    CardDefinition {
        key: "m21.shipwreck_dowser",
        scryfall_id: "59d38ef7-5017-4ea3-b97f-a8fe12d03e98",
        printed: PrintedCharacteristics {
            name: "Shipwreck Dowser".into(),
            subtypes: vec!["Merfolk".into(), "Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![
                Ability::prowess(),
                Ability::when_enters(
                    TriggerSubject::This,
                    vec![Effect::return_to_hand(ObjectSelector::Target(
                        TargetSpec::GraveyardCardYouOwn {
                            predicate: CardPredicate::instant_or_sorcery(),
                        },
                    ))],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn heartfire_immolator() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\n{R}, Sacrifice this creature: It deals damage equal to its power to target creature or planeswalker.";

    CardDefinition {
        key: "m21.heartfire_immolator",
        scryfall_id: "869fb9f1-0d59-4874-aa52-ac665c3cc0e8",
        printed: PrintedCharacteristics {
            name: "Heartfire Immolator".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                ),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(1, 1),
                )],
            }],
            // "Its power": the sacrificed appearance's last known power
            // (CR 608.2h), read through the exact ability source.
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{R}"),
                sacrifice_source: true,
                effects: vec![Effect::deal_damage(
                    EffectValue::SourcePower,
                    ObjectSelector::Target(TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                            ..Default::default()
                        },
                    }),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
