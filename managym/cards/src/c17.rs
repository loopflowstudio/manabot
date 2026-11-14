//! Commander 2017.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![teferis_protection()]
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
