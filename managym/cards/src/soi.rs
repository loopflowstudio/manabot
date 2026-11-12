//! Shadows over Innistrad.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
        TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![fevered_visions(), tireless_tracker()]
}

fn tireless_tracker() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, investigate. (Create a Clue token. It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a Clue, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "soi.tireless_tracker",
        scryfall_id: "ee8e9928-d9b2-4570-adb8-44b34115decd",
        name: "Tireless Tracker".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Scout".into()],
        power: Some(3),
        toughness: Some(2),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::create_tokens("Clue", 1)],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Sacrificed {
                    subject: TriggerSubject::you_control(CardPredicate {
                        subtype: Some("Clue".into()),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fevered_visions() -> CardDefinition {
    let text_box = "At the beginning of each player's end step, that player draws a card. If the player is your opponent and has four or more cards in hand, this enchantment deals 2 damage to that player.";
    let that_player = |effects| Effect::ForPlayer {
        player: PlayerSelector::TriggeringPlayer,
        effects,
    };
    CardDefinition {
        key: "soi.fevered_visions",
        scryfall_id: "3ee3fc2b-c2e4-421d-ac1d-6a111bb44f44",
        name: "Fevered Visions".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
            effects: vec![
                that_player(vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }]),
                // It is the step's active player, so an opponent only off your turn.
                Effect::If {
                    condition: Condition::YourTurn,
                    then: vec![],
                    otherwise: vec![that_player(vec![Effect::If {
                        condition: Condition::CardsInHandAtLeast { count: 4 },
                        then: vec![Effect::DealDamageToPlayer {
                            player: PlayerSelector::You,
                            amount: EffectValue::Fixed(2),
                        }],
                        otherwise: vec![],
                    }])],
                },
            ],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
