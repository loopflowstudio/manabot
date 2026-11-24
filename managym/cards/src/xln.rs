//! Ixalan.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bishops_soldier()]
}

fn bishops_soldier() -> CardDefinition {
    let text_box =
        "Lifelink (Damage dealt by this creature also causes you to gain that much life.)";
    CardDefinition {
        key: "xln.bishops_soldier",
        scryfall_id: "d954677c-2de6-440a-90d0-bab2e0c8b4af",
        printed: PrintedCharacteristics {
            name: "Bishop's Soldier".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
