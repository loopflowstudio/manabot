//! Champions of Kamigawa.
use managym_state::{
    ability::Actor,
    card::{
        CardDefinition, CardType, CardTypes, PrintedCharacteristics, TriggeredManaAbility,
        TriggeredManaOutput,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![heartbeat_of_spring()]
}

fn heartbeat_of_spring() -> CardDefinition {
    let text_box = "Whenever a player taps a land for mana, that player adds one mana of any type that land produced.";
    CardDefinition {
        key: "chk.heartbeat_of_spring",
        scryfall_id: "6be3ba05-dce0-463b-bfe6-d51df63cc1a8",
        printed: PrintedCharacteristics {
            name: "Heartbeat of Spring".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            triggered_mana_abilities: vec![TriggeredManaAbility {
                who: Actor::AnyPlayer,
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
                output: TriggeredManaOutput::OneProducedType,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
