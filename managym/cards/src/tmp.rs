//! Tempest.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![overrun()]
}

fn overrun() -> CardDefinition {
    let text_box = "Creatures you control get +3/+3 and gain trample until end of turn. (Each of those creatures can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "tmp.overrun",
        scryfall_id: "0ad7a961-d3a1-471a-8472-8407d1057de0",
        printed: PrintedCharacteristics {
            name: "Overrun".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                    Modification::pump(3, 3),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
