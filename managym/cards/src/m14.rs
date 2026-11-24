//! Magic 2014.
use managym_state::{
    ability::{Actor, Effect, ObjectSelector},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![rise_of_the_dark_realms()]
}

fn rise_of_the_dark_realms() -> CardDefinition {
    let text_box =
        "Put all creature cards from all graveyards onto the battlefield under your control.";

    CardDefinition {
        key: "m14.rise_of_the_dark_realms",
        scryfall_id: "073f81e8-8c0c-4430-bd3e-95ed3625340f",
        printed: PrintedCharacteristics {
            name: "Rise of the Dark Realms".into(),
            mana_cost: Some(ManaCost::parse("{7}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            // One move of every card: the creatures enter together and see
            // each other enter (CR 603.6a).
            spell_effects: vec![Effect::return_to_battlefield(
                ObjectSelector::ZoneMatching {
                    zone: ZoneType::Graveyard,
                    owner: Actor::AnyPlayer,
                    predicate: CardPredicate::creature(),
                },
                false,
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
