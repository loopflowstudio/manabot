//! Odyssey.
use managym_state::{
    ability::{Ability, Effect, EffectValue, ObjectSelector, PlayerSelector, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![entomb(), standstill()]
}

fn entomb() -> CardDefinition {
    let text_box =
        "Search your library for a card, put that card into your graveyard, then shuffle.";
    CardDefinition {
        key: "ody.entomb",
        scryfall_id: "f60a2091-fb97-4f04-911b-fce9b6351044",
        name: "Entomb".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::search_library(
            CardPredicate::default(),
            EffectValue::Fixed(1),
            vec![Effect::move_to(
                ObjectSelector::Selected,
                ZoneType::Graveyard,
            )],
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn standstill() -> CardDefinition {
    let text_box = "When a player casts a spell, sacrifice this enchantment. If you do, each of that player's opponents draws three cards.";
    CardDefinition {
        key: "ody.standstill",
        scryfall_id: "3ede3f6f-e642-4fe4-aa37-0f01cdf4d149",
        name: "Standstill".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::PlayerCastsSpell {
                predicate: CardPredicate::default(),
            },
            effects: vec![
                Effect::sacrifice_this(),
                Effect::if_you_do(vec![Effect::ForPlayer {
                    player: PlayerSelector::TriggeringPlayersOpponents,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(3),
                    }],
                }]),
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
