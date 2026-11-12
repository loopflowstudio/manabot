//! Planeshift.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![eladamri_s_call()]
}

fn eladamri_s_call() -> CardDefinition {
    let text_box = "Search your library for a creature card, reveal that card, put it into your hand, then shuffle.";
    CardDefinition {
        key: "pls.eladamri_s_call",
        scryfall_id: "dcb79f39-5ef3-4ad6-9a43-04beb27d8480",
        name: "Eladamri's Call".into(),
        mana_cost: Some(ManaCost::parse("{G}{W}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::search_library(
            CardPredicate::creature(),
            EffectValue::Fixed(1),
            vec![Effect::return_to_hand(ObjectSelector::Selected)],
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}
