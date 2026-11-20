//! Antiquities.
use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![feldons_cane()]
}

fn feldons_cane() -> CardDefinition {
    let text_box = "{T}, Exile this artifact: Shuffle your graveyard into your library.";
    CardDefinition {
        key: "atq.feldons_cane",
        scryfall_id: "bb6af436-bcfd-4d47-a1aa-e84b587a725a",
        printed: PrintedCharacteristics {
            name: "Feldon's Cane".into(),
            mana_cost: Some(ManaCost::parse("{1}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                exile_source: true,
                effects: vec![Effect::ShuffleGraveyardIntoLibrary],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
