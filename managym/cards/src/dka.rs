//! Dark Ascension.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, Per, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    let grow = |condition| Ability::Triggered {
        condition,
        intervening_if: None,
        effects: vec![Effect::put_counters(
            ObjectSelector::This,
            CounterKind::PlusOnePlusOne,
            1,
        )],
    };
    let text_box = "Indestructible (Damage and effects that say \"destroy\" don't destroy this creature.)\nWhenever this creature attacks, put a +1/+1 counter on it.\nWhenever a creature dealt damage by this creature this turn dies, put a +1/+1 counter on this creature.";
    vec![CardDefinition {
        key: "dka.predator_ooze",
        scryfall_id: "73c71457-f7c9-4ab4-b89d-e235e3f15e16",
        printed: PrintedCharacteristics {
            name: "Predator Ooze".into(),
            mana_cost: Some(ManaCost::parse("{G}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Ooze".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Indestructible]),
            abilities: vec![
                grow(TriggerCondition::attacks(TriggerSubject::This, Per::Object)),
                grow(TriggerCondition::Dies {
                    subject: TriggerSubject::CreatureDamagedByThis,
                }),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }]
}
