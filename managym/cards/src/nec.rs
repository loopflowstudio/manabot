//! Neon Dynasty Commander.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![release_to_memory()]
}

fn release_to_memory() -> CardDefinition {
    let text_box = "Exile target opponent's graveyard. For each creature card exiled this way, create a 1/1 colorless Spirit creature token.";
    CardDefinition {
        key: "nec.release_to_memory",
        scryfall_id: "b45c4bbf-2c61-4d49-96c8-34a0fb0d6d02",
        printed: PrintedCharacteristics {
            name: "Release to Memory".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::exile(ObjectSelector::TargetPlayersZone {
                    spec: Box::new(TargetSpec::Opponent),
                    zone: ZoneType::Graveyard,
                }),
                Effect::create_tokens(
                    "token.colorless_spirit",
                    EffectValue::Objects {
                        of: Box::new(ObjectSelector::Moved),
                        predicate: Some(CardPredicate::creature()),
                    },
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
