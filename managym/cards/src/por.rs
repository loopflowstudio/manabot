//! Portal.
use managym_state::{
    ability::{Effect, EffectValue, LibraryEnd, ObjectSelector, PlayerSelector},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![cruel_tutor(), raging_goblin(), wind_drake()]
}

fn raging_goblin() -> CardDefinition {
    let text_box =
        "Haste (This creature can attack and {T} as soon as it comes under your control.)";

    CardDefinition {
        key: "por.raging_goblin",
        scryfall_id: "fed57a17-7847-4e60-bc40-4452880f12a3",
        name: "Raging Goblin".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Goblin".into(), "Berserker".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords::of(&[Keyword::Haste]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wind_drake() -> CardDefinition {
    let text_box = "Flying";

    CardDefinition {
        key: "por.wind_drake",
        scryfall_id: "5486d2dc-9a5d-4f58-a5ec-d94de54b852f",
        name: "Wind Drake".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Drake".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords::of(&[Keyword::Flying]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn cruel_tutor() -> CardDefinition {
    let text_box =
        "Search your library for a card, then shuffle and put that card on top. You lose 2 life.";
    CardDefinition {
        key: "por.cruel_tutor",
        scryfall_id: "6c05bfb5-dd36-44c5-a60d-43f7f8c68a6b",
        name: "Cruel Tutor".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            // "then shuffle and put that card on top": the shuffle precedes
            // the placement, so the found card ends on top.
            Effect::SelectCards {
                from: ObjectSelector::Zone(ZoneType::Library),
                predicate: CardPredicate::default(),
                min: 0,
                max: EffectValue::Fixed(1),
                effects: vec![
                    Effect::ShuffleLibrary,
                    Effect::put_in_library(ObjectSelector::Selected, LibraryEnd::Top),
                ],
                if_declined: vec![Effect::ShuffleLibrary],
            },
            Effect::LoseLife {
                player: PlayerSelector::You,
                amount: EffectValue::Fixed(2),
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
