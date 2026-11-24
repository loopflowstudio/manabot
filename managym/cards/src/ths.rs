use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, Per, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        ReplacementEffect,
    },
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        burnished_hart(),
        lightning_strike(),
        ordeal_of_nylea(),
        temple_of_abandon(),
        temple_of_deceit(),
        temple_of_mystery(),
        temple_of_silence(),
        temple_of_triumph(),
    ]
}

/// A Temple of the ten-card cycle: enters tapped, scries 1 on entering and
/// taps for either of two colors. The other five are filed in `bng.rs` and
/// `jou.rs`.
pub(super) fn temple(
    key: &'static str,
    scryfall_id: &'static str,
    name: &str,
    colors: [&str; 2],
) -> CardDefinition {
    let text_box = format!(
        "This land enters tapped.\nWhen this land enters, scry 1. (Look at the top card of your library. You may put that card on the bottom.)\n{{T}}: Add {{{}}} or {{{}}}.",
        colors[0], colors[1]
    );
    CardDefinition {
        key,
        scryfall_id,
        printed: PrintedCharacteristics {
            name: name.into(),
            types: CardTypes::new([CardType::Land]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Scry { count: 1 }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(colors.map(Mana::parse))],
                ..Default::default()
            }],
            replacement_effects: vec![ReplacementEffect::EntersTapped],
            text_box,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lightning_strike() -> CardDefinition {
    let text_box = "Lightning Strike deals 3 damage to any target.";

    CardDefinition {
        key: "ths.lightning_strike",
        scryfall_id: "bbb03f2e-2b92-4aa1-afae-301ed5d151d3",
        printed: PrintedCharacteristics {
            name: "Lightning Strike".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::deal_damage(
                3,
                ObjectSelector::Target(TargetSpec::AnyTarget),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ordeal_of_nylea() -> CardDefinition {
    let text_box = "Enchant creature\nWhenever enchanted creature attacks, put a +1/+1 counter on it. Then if it has three or more +1/+1 counters on it, sacrifice this Aura.\nWhen you sacrifice this Aura, search your library for up to two basic land cards, put them onto the battlefield tapped, then shuffle.";

    CardDefinition {
        key: "ths.ordeal_of_nylea",
        scryfall_id: "e5c48950-c246-47ad-94e1-bf42a62c2fe7",
        printed: PrintedCharacteristics {
            name: "Ordeal of Nylea".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::AttachSourceTo {
                what: ObjectSelector::Target(TargetSpec::Creature),
            }],
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::attacks(
                        TriggerSubject::EnchantedCreature,
                        Per::Object,
                    ),
                    effects: vec![
                        Effect::put_counters(
                            ObjectSelector::Attached,
                            CounterKind::PlusOnePlusOne,
                            1,
                        ),
                        Effect::If {
                            condition: Condition::AttachedCountersAtLeast {
                                kind: CounterKind::PlusOnePlusOne,
                                count: 3,
                            },
                            then: vec![Effect::sacrifice_this()],
                            otherwise: Vec::new(),
                        },
                    ],
                },
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::Sacrificed {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::search_library(
                        CardPredicate::basic_land(),
                        EffectValue::Fixed(2),
                        vec![Effect::return_to_battlefield(
                            ObjectSelector::Selected,
                            true,
                        )],
                    )],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn temple_of_abandon() -> CardDefinition {
    temple(
        "ths.temple_of_abandon",
        "46febc5d-1625-4e48-bb3f-31ee06fc13dd",
        "Temple of Abandon",
        ["R", "G"],
    )
}

fn temple_of_deceit() -> CardDefinition {
    temple(
        "ths.temple_of_deceit",
        "686559d7-8ac1-496b-a5a6-1467bf8fc7c5",
        "Temple of Deceit",
        ["U", "B"],
    )
}

fn temple_of_mystery() -> CardDefinition {
    temple(
        "ths.temple_of_mystery",
        "2f66945b-3e64-498a-9478-5f96a61d4ec7",
        "Temple of Mystery",
        ["G", "U"],
    )
}

fn temple_of_silence() -> CardDefinition {
    temple(
        "ths.temple_of_silence",
        "0f14b6b3-5f40-4328-a3be-28fe32dd7cb1",
        "Temple of Silence",
        ["W", "B"],
    )
}

fn temple_of_triumph() -> CardDefinition {
    temple(
        "ths.temple_of_triumph",
        "4f53049c-2491-4d20-aa19-00eb5c55b438",
        "Temple of Triumph",
        ["R", "W"],
    )
}

fn burnished_hart() -> CardDefinition {
    let text_box = "{3}, Sacrifice this creature: Search your library for up to two basic land cards, put them onto the battlefield tapped, then shuffle.";
    CardDefinition {
        key: "ths.burnished_hart",
        scryfall_id: "772cbcba-9efa-4894-9b57-e73fd296333d",
        printed: PrintedCharacteristics {
            name: "Burnished Hart".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Elk".into()],
            power: Some(2),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}"),
                sacrifice_source: true,
                effects: vec![Effect::search_library(
                    CardPredicate::basic_land(),
                    EffectValue::Fixed(2),
                    vec![Effect::return_to_battlefield(
                        ObjectSelector::Selected,
                        true,
                    )],
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
