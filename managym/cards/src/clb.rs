//! Commander Legends: Battle for Baldur's Gate.
use std::sync::Arc;

use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword,
        PrintedCharacteristics,
    },
    mana::{Mana, ManaCost, SpendEffect},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![carnelian_orb_of_dragonkind()]
}

fn carnelian_orb_of_dragonkind() -> CardDefinition {
    let text_box = "{T}: Add {R}. If that mana is spent on a Dragon creature spell, it gains haste until end of turn.";
    let effect = SpendEffect::SpellGainsKeyword {
        spell: Arc::new(CardPredicate {
            card_type: Some(CardType::Creature),
            subtype: Some("Dragon".into()),
            ..Default::default()
        }),
        keyword: Keyword::Haste,
    };
    CardDefinition {
        key: "clb.carnelian_orb_of_dragonkind",
        scryfall_id: "e7e41166-bdaa-4aed-986a-7be1d043240c",
        printed: PrintedCharacteristics {
            name: "Carnelian Orb of Dragonkind".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::of_class(
                    effect.into(),
                    [0, 0, 0, 1, 0, 0],
                ))],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
