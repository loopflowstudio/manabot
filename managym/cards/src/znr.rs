//! Zendikar Rising.
use managym_state::{
    ability::{Aggregate, Effect, ObjectProperty, ObjectSelector, PlayerSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![feed_the_swarm()]
}

fn feed_the_swarm() -> CardDefinition {
    let text_box = "Destroy target creature or enchantment an opponent controls. You lose life equal to that permanent's mana value.";
    let target = || {
        ObjectSelector::Target(TargetSpec::PermanentOpponentControls {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Creature, CardType::Enchantment],
                ..Default::default()
            },
        })
    };
    CardDefinition {
        key: "znr.feed_the_swarm",
        scryfall_id: "f6b2eba7-862a-4efd-9f65-065fb2070855",
        printed: PrintedCharacteristics {
            name: "Feed the Swarm".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::destroy(target()),
                // The permanent as it is, or as it last existed once destroyed.
                Effect::LoseLife {
                    player: PlayerSelector::You,
                    amount: target().property(ObjectProperty::ManaValue, Aggregate::Sum),
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
