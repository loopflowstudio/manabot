//! Gatecrash.
use super::rtr::guildgate;
use managym_state::ability::{
    Ability, Actor, AttackSize, Effect, EffectValue, ObjectSelector, Per, PlayerSelector, Scope,
    TargetSpec, TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, EntryScope, EntryValue, Keyword, Keywords, LayerCount,
    LayerValue, PrintedCharacteristics, ReplacementEffect, StaticAbility,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        prime_speaker_zegana(),
        aurelia_the_warleader(),
        boros_guildgate(),
        consuming_aberration(),
        dimir_guildgate(),
        gruul_guildgate(),
        orzhov_guildgate(),
        simic_guildgate(),
        boros_charm(),
        aetherize(),
        angelic_edict(),
    ]
}

fn boros_guildgate() -> CardDefinition {
    guildgate(
        "gtc.boros_guildgate",
        "a0b447a8-524b-4bda-b975-7e194fd741fb",
        "Boros Guildgate",
        ["R", "W"],
    )
}

fn dimir_guildgate() -> CardDefinition {
    guildgate(
        "gtc.dimir_guildgate",
        "4951bf75-1a88-4c85-b4e9-063d84f1dabf",
        "Dimir Guildgate",
        ["U", "B"],
    )
}

fn gruul_guildgate() -> CardDefinition {
    guildgate(
        "gtc.gruul_guildgate",
        "99c54269-8798-4023-836f-640103e08ce0",
        "Gruul Guildgate",
        ["R", "G"],
    )
}

fn orzhov_guildgate() -> CardDefinition {
    guildgate(
        "gtc.orzhov_guildgate",
        "000d609c-deb7-4bd7-9c1d-e20fb3ed4f5f",
        "Orzhov Guildgate",
        ["W", "B"],
    )
}

fn simic_guildgate() -> CardDefinition {
    guildgate(
        "gtc.simic_guildgate",
        "1ce3f6f2-c52c-4fb8-afa0-b1ea723bb4c6",
        "Simic Guildgate",
        ["G", "U"],
    )
}

fn aurelia_the_warleader() -> CardDefinition {
    let text_box = "Flying, vigilance, haste\nWhenever Aurelia attacks for the first time each turn, untap all creatures you control. After this phase, there is an additional combat phase.";
    CardDefinition {
        key: "gtc.aurelia_the_warleader",
        scryfall_id: "4ec18e35-05e4-4bfc-b32b-c3e71c95a71d",
        printed: PrintedCharacteristics {
            name: "Aurelia, the Warleader".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Angel".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance, Keyword::Haste]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                    per: Per::Object,
                    attackers: AttackSize::Any,
                    nth: Some(1),
                },
                intervening_if: None,
                effects: vec![
                    Effect::Untap {
                        what: ObjectSelector::each_you_control(CardPredicate::creature()),
                    },
                    Effect::AdditionalCombatPhase { attackers: None },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn consuming_aberration() -> CardDefinition {
    let text_box = "Consuming Aberration's power and toughness are each equal to the number of cards in your opponents' graveyards.\nWhenever you cast a spell, each opponent reveals cards from the top of their library until they reveal a land card, then puts those cards into their graveyard.";
    CardDefinition {
        key: "gtc.consuming_aberration",
        scryfall_id: "6354de66-f7f8-4e33-98d0-52624d3d7828",
        printed: PrintedCharacteristics {
            name: "Consuming Aberration".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Horror".into()],
            statics: vec![StaticAbility::cda(vec![Modification::SetBasePt {
                power: Some(LayerValue::count(LayerCount::Graveyard {
                    who: Actor::Opponent,
                    predicate: CardPredicate::default(),
                })),
                toughness: Some(LayerValue::count(LayerCount::Graveyard {
                    who: Actor::Opponent,
                    predicate: CardPredicate::default(),
                })),
            }])],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
                intervening_if: None,
                effects: vec![Effect::MillUntil {
                    player: PlayerSelector::EachOpponent,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    },
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn angelic_edict() -> CardDefinition {
    CardDefinition {
        key: "gtc.angelic_edict",
        scryfall_id: "e24b62d3-c200-4330-a255-92d77f01ba44",
        printed: PrintedCharacteristics {
            name: "Angelic Edict".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::exile(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Creature, CardType::Enchantment],
                        ..Default::default()
                    },
                },
            ))],
            text_box: "Exile target creature or enchantment.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn aetherize() -> CardDefinition {
    CardDefinition {
        key: "gtc.aetherize",
        scryfall_id: "33303859-c6e0-4ebd-bb5f-44be7f5d7459",
        printed: PrintedCharacteristics {
            name: "Aetherize".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::return_to_hand(ObjectSelector::each(
                Scope::Battlefield,
                CardPredicate {
                    attacking: true,
                    ..CardPredicate::creature()
                },
            ))],
            text_box: "Return all attacking creatures to their owner's hand.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn boros_charm() -> CardDefinition {
    let text_box = "Choose one —\n• Boros Charm deals 4 damage to target player or planeswalker.\n• Permanents you control gain indestructible until end of turn.\n• Target creature gains double strike until end of turn.";
    CardDefinition {
        key: "gtc.boros_charm",
        scryfall_id: "d4ddf9cc-40a7-4b4f-bb51-b08171453c9a",
        printed: PrintedCharacteristics {
            name: "Boros Charm".into(),
            mana_cost: Some(ManaCost::parse("{R}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::deal_damage(
                        4,
                        ObjectSelector::Target(TargetSpec::PlayerOrPlaneswalker),
                    )],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each_you_control(CardPredicate::permanent()),
                        Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                    )],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::Creature),
                        Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn prime_speaker_zegana() -> CardDefinition {
    let text_box = "Prime Speaker Zegana enters with X +1/+1 counters on it, where X is the greatest power among other creatures you control.\nWhen Prime Speaker Zegana enters, draw cards equal to its power.";
    CardDefinition {
        key: "gtc.prime_speaker_zegana",
        scryfall_id: "f30dfb8e-f540-45ab-a4e8-63425099646a",
        printed: PrintedCharacteristics {
            name: "Prime Speaker Zegana".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Merfolk".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(1),
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                on: EntryScope::This,
                kind: CounterKind::PlusOnePlusOne,
                count: EntryValue::GreatestPower(Box::new(CardPredicate::creature())),
                condition: None,
            }],
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::DrawCards {
                    count: EffectValue::SourcePower,
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
