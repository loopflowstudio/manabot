//! Commander Legends.
use managym_state::{
    ability::{Ability, Effect, EffectValue, ObjectSelector, Scope, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![volcanic_torrent()]
}

fn volcanic_torrent() -> CardDefinition {
    let text_box = "Cascade (When you cast this spell, exile cards from the top of your library until you exile a nonland card that costs less. You may cast it without paying its mana cost. Put the exiled cards on the bottom in a random order.)\nVolcanic Torrent deals X damage to each creature and planeswalker your opponents control, where X is the number of spells you've cast this turn.";
    CardDefinition {
        key: "cmr.volcanic_torrent",
        scryfall_id: "eb586da6-670d-4c50-9d9b-f320f1c288d7",
        name: "Volcanic Torrent".into(),
        mana_cost: Some(ManaCost::parse("{4}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::YouCastThis,
            effects: vec![Effect::Cascade],
        }],
        spell_effects: vec![Effect::deal_damage(
            EffectValue::SpellsCastThisTurn,
            ObjectSelector::each(
                Scope::OpponentsControl,
                CardPredicate {
                    card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                    ..Default::default()
                },
            ),
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}
