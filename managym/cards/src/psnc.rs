//! Streets of New Capenna promos.
use managym_state::{
    card::{
        CardDefinition, CardType, CardTypes, EntryScope, EntryValue, Keyword, Keywords,
        ManaAbility, PrintedCharacteristics, ReplacementEffect,
    },
    mana::{Mana, ManaCost, ManaRestriction},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![giada_font_of_hope()]
}

fn giada_font_of_hope() -> CardDefinition {
    let text_box = "Flying, vigilance\nEach other Angel you control enters with an additional +1/+1 counter on it for each Angel you already control.\n{T}: Add {W}. Spend this mana only to cast an Angel spell.";
    let angel = CardPredicate::subtype("Angel");
    CardDefinition {
        key: "psnc.giada_font_of_hope",
        scryfall_id: "8ced9f9b-2d20-41ed-90ff-5b37959a0b3b",
        printed: PrintedCharacteristics {
            name: "Giada, Font of Hope".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Angel".into()],
            power: Some(2),
            toughness: Some(2),
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                on: EntryScope::OtherYouControl(Box::new(angel.clone())),
                kind: CounterKind::PlusOnePlusOne,
                count: EntryValue::ControlledAlready(Box::new(angel)),
                condition: None,
            }],
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::restricted_to(ManaRestriction::spell(["Angel"]), [1, 0, 0, 0, 0, 0]),
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
