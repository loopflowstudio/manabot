//! Future Sight.
use managym_state::effect::Modification;
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PlayerSelector, TriggerCondition,
        TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![rites_of_flourishing(), scout_s_warning()]
}

fn scout_s_warning() -> CardDefinition {
    let text_box =
        "The next creature card you play this turn can be played as though it had flash.\nDraw a card.";
    CardDefinition {
        key: "fut.scout_s_warning",
        scryfall_id: "e6639e3d-b0c4-4c08-83d5-f3fc58fac9b2",
        name: "Scout's Warning".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::until_end_of_turn(ObjectSelector::You, Modification::NextCreatureCastFlash),
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn rites_of_flourishing() -> CardDefinition {
    let text_box = "At the beginning of each player's draw step, that player draws an additional card.\nEach player may play an additional land on each of their turns.";
    CardDefinition {
        key: "fut.rites_of_flourishing",
        scryfall_id: "811458c7-dcdc-43ef-8c3e-a90e21ce315e",
        name: "Rites of Flourishing".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::BeginningOf {
                step: TurnStep::Draw,
                whose: Actor::AnyPlayer,
            },
            effects: vec![Effect::ForPlayer {
                player: PlayerSelector::TriggeringPlayer,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
        }],
        statics: vec![StaticAbility::AdditionalLandPlayEachPlayer],
        text_box: text_box.into(),
        ..Default::default()
    }
}
