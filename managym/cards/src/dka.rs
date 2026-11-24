//! Dark Ascension.
use managym_state::ability::{
    Ability, Effect, EffectValue, ObjectSelector, Per, TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
};
use managym_state::mana::ManaCost;
use managym_state::permanent::CounterKind;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![predator_ooze(), drogskol_reaver()]
}

fn predator_ooze() -> CardDefinition {
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
    CardDefinition {
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
    }
}

fn drogskol_reaver() -> CardDefinition {
    CardDefinition {
        key: "dka.drogskol_reaver",
        scryfall_id: "af2d9e0b-6433-40a2-9847-9fa4e3c008c4",
        printed: PrintedCharacteristics {
            name: "Drogskol Reaver".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into()],
            keywords: Keywords::of(&[Keyword::Flying, Keyword::DoubleStrike, Keyword::Lifelink]),
            power: Some(3),
            toughness: Some(5),
            abilities: vec![Ability::when_you_gain_life(Effect::DrawCards { count: EffectValue::Fixed(1) })],
            text_box: "Flying\nDouble strike (This creature deals both first-strike and regular combat damage.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhenever you gain life, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
