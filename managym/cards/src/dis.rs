//! Dissension.

use managym_state::{
    ability::{Effect, ObjectSelector, TargetRequirement, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![macabre_waltz()]
}

fn macabre_waltz() -> CardDefinition {
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate::creature(),
    };
    let text_box = "Return up to two target creature cards from your graveyard to your hand, then discard a card.";
    CardDefinition {
        key: "dis.macabre_waltz",
        scryfall_id: "d9cd7bc3-73ba-4364-84b2-9954648cd8a9",
        printed: PrintedCharacteristics {
            name: "Macabre Waltz".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::return_to_hand(ObjectSelector::Targets(spec.clone())),
                Effect::discard(),
            ],
            targeting: vec![TargetRequirement::up_to(2, spec)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
