//! Aether Revolt.
use managym_state::effect::Modification;
use managym_state::{
    ability::{Effect, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![heroic_intervention()]
}

fn heroic_intervention() -> CardDefinition {
    let text_box = "Permanents you control gain hexproof and indestructible until end of turn.";
    CardDefinition {
        key: "aer.heroic_intervention",
        scryfall_id: "8f5a620c-fde7-4b72-bf8a-efc4f14560c5",
        name: "Heroic Intervention".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::until_end_of_turn(
            ObjectSelector::each_you_control(CardPredicate::default()),
            Modification::AddKeywords(Keywords::of(&[Keyword::Hexproof, Keyword::Indestructible])),
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}
