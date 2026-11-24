//! Edge of Eternities cards. Station is an activated ability whose cost taps
//! another creature and whose effect reads that creature's power from the
//! activation's receipt; a Spacecraft's "N+" lines are static abilities
//! conditioned on its charge counters.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, Per, PlayerSelector,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticKeywordGrant, StaticPtBuff, StaticScope,
        StaticTypeGrant,
    },
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        lumen_class_frigate(),
        virulent_silencer(),
        the_dominion_bracelet(),
    ]
}

fn lumen_class_frigate() -> CardDefinition {
    let text_box = "Station (Tap another creature you control: Put charge counters equal to its power on this Spacecraft. Station only as a sorcery. It's an artifact creature at 12+.)\n2+ | Other creatures you control get +1/+1.\n12+ | Flying, lifelink";
    let charged = |count| {
        Some(Condition::SourceCountersAtLeast {
            kind: CounterKind::Charge,
            count,
        })
    };
    CardDefinition {
        key: "eoe.lumen_class_frigate",
        scryfall_id: "0cc59b5a-65fa-47cc-8ac7-b7c3f533a782",
        printed: PrintedCharacteristics {
            name: "Lumen-Class Frigate".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Spacecraft".into()],
            power: Some(3),
            toughness: Some(5),
            statics: vec![
                StaticAbility::PtBuff(StaticPtBuff {
                    scope: StaticScope::OtherYouControl(CardPredicate::creature()),
                    condition: charged(2),
                    power: 1,
                    toughness: 1,
                    per_creature_type_max: None,
                    per_color_among: None,
                    per_graveyard_matching: None,
                }),
                StaticAbility::TypeGrant(StaticTypeGrant {
                    scope: StaticScope::This,
                    types: vec![CardType::Creature],
                    subtypes: Vec::new(),
                    condition: charged(12),
                }),
                StaticAbility::KeywordGrant(StaticKeywordGrant {
                    scope: StaticScope::This,
                    condition: charged(12),
                    keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
                }),
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                station: true,
                sorcery_timing: true,
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::Charge,
                    EffectValue::PowerTappedToPay,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn virulent_silencer() -> CardDefinition {
    CardDefinition {
        key: "eoe.virulent_silencer",
        scryfall_id: "4120fcfc-3547-4774-a15d-b9cccac04e76",
        printed: PrintedCharacteristics {
            name: "Virulent Silencer".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Robot".into(), "Assassin".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::you_control(CardPredicate {
                        token: Some(false),
                        card_types_any: vec![CardType::Artifact],
                        ..CardPredicate::creature()
                    }),
                    to: Some(Actor::Opponent),
                    per: Per::Object,
                },
                effects: vec![Effect::GivePlayerCounters {
                    player: PlayerSelector::EachOpponent,
                    kind: CounterKind::Poison,
                    count: 2,
                }],
            }],
            text_box: "Whenever a nontoken artifact creature you control deals combat damage to a player, that player gets two poison counters. (A player with ten or more poison counters loses the game.)".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn the_dominion_bracelet() -> CardDefinition {
    use managym_state::ability::{PlayerControlDuration, TargetSpec};
    let text_box = r#"Equipped creature gets +1/+1 and has "{15}, Exile The Dominion Bracelet: You control target opponent during their next turn. This ability costs {X} less to activate, where X is this creature's power. Activate only as a sorcery." (You see all cards that player could see and make all decisions for them.)
Equip {1}"#;
    CardDefinition {
        key: "eoe.the_dominion_bracelet",
        scryfall_id: "f5360880-2849-45d6-b1aa-08c7e01083af",
        printed: PrintedCharacteristics {
            name: "The Dominion Bracelet".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Equipment".into()],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{1}"))],
            statics: vec![
                StaticAbility::PtBuff(StaticPtBuff {
                    scope: StaticScope::Attached,
                    condition: None,
                    power: 1,
                    toughness: 1,
                    per_creature_type_max: None,
                    per_color_among: None,
                    per_graveyard_matching: None,
                }),
                StaticAbility::AttachedHas(ActivatedAbilityDefinition {
                    sorcery_timing: true,
                    mana_cost: ManaCost::parse("{15}"),
                    reduce_by_source_power: true,
                    exile_grantor: true,
                    effects: vec![Effect::ControlPlayer {
                        what: ObjectSelector::Target(TargetSpec::Opponent),
                        duration: PlayerControlDuration::NextTurn,
                    }],
                    ..Default::default()
                }),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
