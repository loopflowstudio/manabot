//! Magic 2014.
use managym_state::ability::{
    Ability, Actor, Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec,
    TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
};
use managym_state::mana::{Color, ManaCost};
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![mindsparker(), primeval_bounty()]
}

fn mindsparker() -> CardDefinition {
    let text_box = "First strike (This creature deals combat damage before creatures without first strike.)\nWhenever an opponent casts a white or blue instant or sorcery spell, this creature deals 2 damage to that player.";
    CardDefinition {
        key: "m14.mindsparker",
        scryfall_id: "a94295dc-d078-4f3f-9856-bd0a1899a9ca",
        printed: PrintedCharacteristics {
            name: "Mindsparker".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elemental".into()],
            power: Some(3),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(
                    Actor::Opponent,
                    CardPredicate {
                        colors_any: vec![Color::White, Color::Blue],
                        card_types_any: vec![CardType::Instant, CardType::Sorcery],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::TriggeringPlayer,
                    amount: EffectValue::Fixed(2),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn primeval_bounty() -> CardDefinition {
    let text_box = "Whenever you cast a creature spell, create a 3/3 green Beast creature token.\nWhenever you cast a noncreature spell, put three +1/+1 counters on target creature you control.\nLandfall — Whenever a land you control enters, you gain 3 life.";
    CardDefinition {
        key: "m14.primeval_bounty",
        scryfall_id: "e750d55d-d5e8-4abe-99cf-f6b8ba86cf16",
        printed: PrintedCharacteristics {
            name: "Primeval Bounty".into(),
            mana_cost: Some(ManaCost::parse("{5}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::casts(Actor::You, CardPredicate::creature()),
                    intervening_if: None,
                    effects: vec![Effect::create_tokens("token.beast_3_3_g", 1)],
                },
                Ability::Triggered {
                    condition: TriggerCondition::casts(Actor::You, CardPredicate::noncreature()),
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        3,
                    )],
                },
                Ability::when_enters(
                    TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                    vec![Effect::gain_life(3)],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
