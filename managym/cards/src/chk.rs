//! Champions of Kamigawa.
use managym_state::{
    ability::{Actor, Effect},
    card::{
        CardDefinition, CardType, CardTypes, PrintedCharacteristics, TriggeredManaAbility,
        TriggeredManaOutput,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![heartbeat_of_spring(), time_stop()]
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

fn time_stop() -> CardDefinition {
    let text_box = "End the turn. (Exile all spells and abilities, including this spell. The player whose turn it is discards down to their maximum hand size. Damage heals and \"this turn\" and \"until end of turn\" effects end.)";
    CardDefinition {
        key: "chk.time_stop",
        scryfall_id: "f968c5e9-12a8-4542-90b4-84e0238fa375",
        printed: PrintedCharacteristics {
            name: "Time Stop".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::EndTheTurn],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
