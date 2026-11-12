//! Salvat 2005.
use managym_state::{
    ability::{DistinctSearchEnd, Effect},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![three_dreams()]
}

fn three_dreams() -> CardDefinition {
    let text_box = "Search your library for up to three Aura cards with different names, reveal them, put them into your hand, then shuffle.";
    CardDefinition {
        key: "psal.three_dreams",
        scryfall_id: "ddd4bbfb-80c9-4bdd-86c6-96fba00d3d12",
        name: "Three Dreams".into(),
        mana_cost: Some(ManaCost::parse("{4}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::SearchDistinctNames {
            predicate: CardPredicate::subtype("Aura"),
            count: 3,
            end: DistinctSearchEnd::Hand,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
