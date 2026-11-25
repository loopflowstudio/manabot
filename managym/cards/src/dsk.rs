//! Duskmourn: House of Horror.
use managym_state::{
    ability::Effect,
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![CardDefinition {
        key: "dsk.manifest_dread",
        scryfall_id: "a649265b-6c32-49e7-b6cb-6086c40d26e8",
        printed: PrintedCharacteristics {
            name: "Manifest Dread".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::ManifestDread],
            text_box: "Manifest dread. (Look at the top two cards of your library. Put one onto the battlefield face down as a 2/2 creature and the other into your graveyard. Turn it face up any time for its mana cost if it's a creature card.)".into(),
            ..Default::default()
        },
        ..Default::default()
    }]
}
