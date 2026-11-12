//! Odyssey.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![entomb()]
}

fn entomb() -> CardDefinition {
    let text_box =
        "Search your library for a card, put that card into your graveyard, then shuffle.";
    CardDefinition {
        key: "ody.entomb",
        scryfall_id: "f60a2091-fb97-4f04-911b-fce9b6351044",
        name: "Entomb".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::search_library(
            CardPredicate::default(),
            EffectValue::Fixed(1),
            vec![Effect::move_to(
                ObjectSelector::Selected,
                ZoneType::Graveyard,
            )],
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}
