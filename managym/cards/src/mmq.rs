//! Mercadian Masques.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bribery()]
}

fn bribery() -> CardDefinition {
    let text_box = "Search target opponent's library for a creature card and put that card onto the battlefield under your control. Then that player shuffles.";
    let they_shuffle = || Effect::ForPlayer {
        player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
        effects: vec![Effect::ShuffleLibrary],
    };
    CardDefinition {
        key: "mmq.bribery",
        scryfall_id: "dfc0ea8a-62f6-49e8-8eec-9748870bc596",
        printed: PrintedCharacteristics {
            name: "Bribery".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::SelectCards {
                from: ObjectSelector::TargetPlayersZone {
                    spec: Box::new(TargetSpec::Opponent),
                    zone: ZoneType::Library,
                },
                predicate: CardPredicate::creature(),
                min: 0,
                max: EffectValue::Fixed(1),
                effects: vec![
                    Effect::Move {
                        library_end: None,
                        what: ObjectSelector::Selected,
                        to: ZoneType::Battlefield,
                        tapped: false,
                        transformed: false,
                        under: Some(PlayerSelector::You),
                        with_counter: None,
                        linked: None,
                    },
                    they_shuffle(),
                ],
                if_declined: vec![they_shuffle()],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
