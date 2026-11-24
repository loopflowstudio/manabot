//! Future Sight.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PlayerSelector, TriggerCondition,
        TurnStep,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        thornweald_archer(),
        rites_of_flourishing(),
        scout_s_warning(),
    ]
}

fn scout_s_warning() -> CardDefinition {
    let text_box =
        "The next creature card you play this turn can be played as though it had flash.\nDraw a card.";
    CardDefinition {
        key: "fut.scout_s_warning",
        scryfall_id: "e6639e3d-b0c4-4c08-83d5-f3fc58fac9b2",
        printed: PrintedCharacteristics {
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
        },
        ..Default::default()
    }
}

fn rites_of_flourishing() -> CardDefinition {
    let text_box = "At the beginning of each player's draw step, that player draws an additional card.\nEach player may play an additional land on each of their turns.";
    CardDefinition {
        key: "fut.rites_of_flourishing",
        scryfall_id: "811458c7-dcdc-43ef-8c3e-a90e21ce315e",
        printed: PrintedCharacteristics {
            name: "Rites of Flourishing".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::beginning_of(TurnStep::Draw, Actor::AnyPlayer),
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
        },
        ..Default::default()
    }
}

fn thornweald_archer() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)\nDeathtouch (Any amount of damage this deals to a creature is enough to destroy it.)";
    CardDefinition {
        key: "fut.thornweald_archer",
        scryfall_id: "783a50a0-9394-4d97-856f-21eb126a0018",
        printed: PrintedCharacteristics {
            name: "Thornweald Archer".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Archer".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Reach, Keyword::Deathtouch]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
