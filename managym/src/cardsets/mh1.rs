//! Modern Horizons.
use crate::state::ability::AffectedSpec;
use crate::state::effect::Modification;
use crate::state::{
    ability::{Effect, EffectValue, TargetSpec},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fists_of_flame(), prismatic_vista()]
}

fn fists_of_flame() -> CardDefinition {
    let text_box = "Draw a card. Until end of turn, target creature gains trample and gets +1/+0 for each card you've drawn this turn. (A creature with trample can deal excess combat damage to the player or planeswalker it's attacking.)";

    CardDefinition {
        key: "mh1.fists_of_flame",
        scryfall_id: "4fa3c0d8-0a3e-43b0-b81e-a3d22ce6a4b2",
        name: "Fists of Flame".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::ModifyPt {
                    power: EffectValue::CardsDrawnMinus { subtract: 0 },
                    toughness: EffectValue::Fixed(0),
                },
            ),
            Effect::until_end_of_turn(
                AffectedSpec::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords {
                    trample: true,
                    ..Default::default()
                }),
            ),
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn prismatic_vista() -> CardDefinition {
    let text_box = "{T}, Pay 1 life, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield, then shuffle.";

    CardDefinition {
        key: "mh1.prismatic_vista",
        scryfall_id: "e37da81e-be12-45a2-9128-376f1ad7b3e8",
        name: "Prismatic Vista".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            life_cost: 1,
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::SearchBasicLand {
                subtype: None,
                count: EffectValue::Fixed(1),
                tapped: false,
                untap_if_lands: None,
                bottom_source: false,
                to_hand: false,
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
