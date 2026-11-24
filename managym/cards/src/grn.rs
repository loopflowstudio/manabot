//! Guilds of Ravnica.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![swiftblade_vindicator(), healer_s_hawk()]
}

fn healer_s_hawk() -> CardDefinition {
    let text_box =
        "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)";

    CardDefinition {
        key: "grn.healer_s_hawk",
        scryfall_id: "3313bd5c-b657-47a3-822a-dd0d9165492a",
        printed: PrintedCharacteristics {
            name: "Healer's Hawk".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn swiftblade_vindicator() -> CardDefinition {
    let text_box = "Double strike (This creature deals both first-strike and regular combat damage.)\nVigilance (Attacking doesn't cause this creature to tap.)\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "grn.swiftblade_vindicator",
        scryfall_id: "285c4d9e-0f22-49a8-b68c-150fd0d4b617",
        printed: PrintedCharacteristics {
            name: "Swiftblade Vindicator".into(),
            mana_cost: Some(ManaCost::parse("{R}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Vigilance, Keyword::Trample, Keyword::DoubleStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
