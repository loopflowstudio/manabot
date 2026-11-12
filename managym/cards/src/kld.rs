//! Kaladesh.
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dramatic_reversal()]
}

fn dramatic_reversal() -> CardDefinition {
    let text_box = "Untap all nonland permanents you control.";
    CardDefinition {
        key: "kld.dramatic_reversal",
        scryfall_id: "dcb59045-2743-48ae-8063-727e551b1c41",
        name: "Dramatic Reversal".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::untap(ObjectSelector::each_you_control(
            CardPredicate {
                not_card_types: vec![CardType::Land],
                ..Default::default()
            },
        ))],
        text_box: text_box.into(),
        ..Default::default()
    }
}
