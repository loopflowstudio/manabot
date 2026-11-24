//! Oath of the Gatewatch.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![make_a_stand(), tajuru_pathwarden()]
}

fn tajuru_pathwarden() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "ogw.tajuru_pathwarden",
        scryfall_id: "b073e75b-b432-41a5-a71e-d169fecf774f",
        printed: PrintedCharacteristics {
            name: "Tajuru Pathwarden".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into(), "Ally".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance, Keyword::Trample]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_a_stand() -> CardDefinition {
    let text_box = "Creatures you control get +1/+0 and gain indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy them.)";
    CardDefinition {
        key: "ogw.make_a_stand",
        scryfall_id: "30cace63-91ca-493c-b67f-740fbbf06370",
        printed: PrintedCharacteristics {
            name: "Make a Stand".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                    Modification::pump(1, 0),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
