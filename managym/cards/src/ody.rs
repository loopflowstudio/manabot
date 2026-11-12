//! Odyssey.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PlayerSelector, TriggerCondition,
    },
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, ManaAbility},
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![entomb(), standstill(), tarnished_citadel()]
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
            condition: TriggerCondition::casts(Actor::AnyPlayer, CardPredicate::default()),
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

fn tarnished_citadel() -> CardDefinition {
    let text_box =
        "{T}: Add {C}.\n{T}: Add one mana of any color. This land deals 3 damage to you.";
    CardDefinition {
        key: "ody.tarnished_citadel",
        scryfall_id: "30375d24-ccfe-47a2-babd-1bda0a6298fe",
        name: "Tarnished Citadel".into(),
        types: CardTypes::new([CardType::Land]),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("C"),
        }],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            effects: vec![
                Effect::any_color_mana(),
                Effect::DealDamageToPlayer {
                    player: PlayerSelector::You,
                    amount: EffectValue::Fixed(3),
                },
            ],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
