//! Commander 2015.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, PlayerSelector},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dread_summons()]
}

fn dread_summons() -> CardDefinition {
    let text_box = "Each player mills X cards. For each creature card put into a graveyard this way, you create a tapped 2/2 black Zombie creature token. (To mill a card, a player puts the top card of their library into their graveyard.)";
    CardDefinition {
        key: "c15.dread_summons",
        scryfall_id: "b2c20cb1-3e3d-4fea-b617-bd6d796c8d10",
        printed: PrintedCharacteristics {
            name: "Dread Summons".into(),
            mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::MillCards {
                    count: EffectValue::X,
                    target: None,
                    player: Some(PlayerSelector::EachPlayer),
                },
                // A milled card a replacement exiled was not put into a
                // graveyard this way.
                Effect::CreateToken {
                    token: "token.zombie_2_2_b".into(),
                    count: EffectValue::Objects {
                        of: Box::new(ObjectSelector::CardsThisWay),
                        predicate: Some(CardPredicate::creature()),
                        in_zone: Some(ZoneType::Graveyard),
                    },
                    tapped_and_attacking: false,
                    tapped: true,
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
