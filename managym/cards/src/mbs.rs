//! Mirrodin Besieged.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, PlayerSelector, Scope,
        TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![black_sun_s_zenith(), massacre_wurm(), slagstorm()]
}

fn slagstorm() -> CardDefinition {
    let text_box = "Choose one —\n• Slagstorm deals 3 damage to each creature.\n• Slagstorm deals 3 damage to each player.";
    CardDefinition {
        key: "mbs.slagstorm",
        scryfall_id: "9e318b03-2aad-462b-a2a9-8b6bdf0e93d6",
        printed: PrintedCharacteristics {
            name: "Slagstorm".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::deal_damage(
                        3,
                        ObjectSelector::each(Scope::Battlefield, CardPredicate::creature()),
                    )],
                    vec![Effect::DealDamageToPlayer {
                        player: PlayerSelector::EachPlayer,
                        amount: 3.into(),
                    }],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn massacre_wurm() -> CardDefinition {
    let text_box = "When this creature enters, creatures your opponents control get -2/-2 until end of turn.\nWhenever a creature an opponent controls dies, that player loses 2 life.";
    CardDefinition {
        key: "mbs.massacre_wurm",
        scryfall_id: "cdd32ec2-02a8-41fc-bf45-c9585bb2b3ee",
        printed: PrintedCharacteristics {
            name: "Massacre Wurm".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Phyrexian".into(), "Wurm".into()],
            power: Some(6),
            toughness: Some(5),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::OpponentsControl, CardPredicate::creature()),
                        Modification::pump(-2, -2),
                    )],
                },
                // "That player" is the dying creature's controller as it
                // last existed, not its owner.
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::matching(
                            Actor::Opponent,
                            CardPredicate::creature(),
                            false,
                        ),
                    },
                    intervening_if: None,
                    effects: vec![Effect::LoseLife {
                        player: PlayerSelector::TriggeringPlayer,
                        amount: EffectValue::Fixed(2),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn black_sun_s_zenith() -> CardDefinition {
    let text_box = "Put X -1/-1 counters on each creature. Shuffle Black Sun's Zenith into its owner's library.";
    CardDefinition {
        key: "mbs.black_sun_s_zenith",
        scryfall_id: "03bdcf52-50b8-42c0-9665-931d83f5f314",
        printed: PrintedCharacteristics {
            name: "Black Sun's Zenith".into(),
            mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::put_counters(
                    ObjectSelector::each(Scope::Battlefield, CardPredicate::creature()),
                    CounterKind::MinusOneMinusOne,
                    EffectValue::X,
                ),
                Effect::move_to(ObjectSelector::This, ZoneType::Library),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
