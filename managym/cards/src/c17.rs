//! Commander 2017.
use managym_state::{
    ability::{
        Ability, Actor, Aggregate, Effect, EffectValue, ObjectProperty, ObjectSelector,
        TriggerCondition,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::{EffectDuration, Modification},
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![teferis_protection(), ramos_dragon_engine()]
}

fn teferis_protection() -> CardDefinition {
    let text_box = "Until your next turn, your life total can't change and you gain protection from everything. All permanents you control phase out. (While they're phased out, they're treated as though they don't exist. They phase in before you untap during your untap step.)\nExile Teferi's Protection.";
    let until_your_next_turn = |modification| Effect::ApplyContinuous {
        affected: ObjectSelector::You,
        modification,
        duration: EffectDuration::UntilYourNextTurn,
    };
    CardDefinition {
        key: "c17.teferis_protection",
        scryfall_id: "77f130c7-0138-4a1a-9f67-62d2c302dc48",
        printed: PrintedCharacteristics {
            name: "Teferi's Protection".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                until_your_next_turn(Modification::LifeTotalCantChange),
                until_your_next_turn(Modification::ProtectionFromEverything),
                // Phasing in ends the record, not a duration (CR 702.26a).
                Effect::ApplyContinuous {
                    affected: ObjectSelector::each_you_control(CardPredicate::default()),
                    modification: Modification::PhasedOut,
                    duration: EffectDuration::Indefinite,
                },
                Effect::exile(ObjectSelector::This),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ramos_dragon_engine() -> CardDefinition {
    let text_box = "Flying\nWhenever you cast a spell, put a +1/+1 counter on Ramos for each of that spell's colors.\nRemove five +1/+1 counters from Ramos: Add {W}{W}{U}{U}{B}{B}{R}{R}{G}{G}. Activate only once each turn.";
    CardDefinition {
        key: "c17.ramos_dragon_engine",
        scryfall_id: "2e747ef1-a1ad-4859-a70c-3f935f017310",
        printed: PrintedCharacteristics {
            name: "Ramos, Dragon Engine".into(),
            mana_cost: Some(ManaCost::parse("{6}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
                intervening_if: None,
                // The spell's colors as it is or last existed on the stack
                // (CR 608.2h), so a countered spell still counts.
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    EffectValue::Property {
                        of: Box::new(ObjectSelector::Triggering),
                        property: ObjectProperty::Colors,
                        aggregate: Aggregate::Sum,
                    },
                )],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                remove_counters: Some((CounterKind::PlusOnePlusOne, 5)),
                once_each_turn: true,
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("WWUUBBRRGG"),
                    amount: EffectValue::Fixed(1),
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
