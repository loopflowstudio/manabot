//! Zendikar.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, Per,
        PlayerSelector, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        CardDefinition, CardType, CardTypes, ManaAbility, ReplacementEffect, StaticAbility,
        StaticPtBuff, StaticScope,
    },
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        beastmaster_ascension(),
        bloodchief_ascension(),
        lotus_cobra(),
        valakut_the_molten_pinnacle(),
    ]
}

fn lotus_cobra() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, add one mana of any color.";

    CardDefinition {
        key: "zen.lotus_cobra",
        scryfall_id: "19adde22-e5eb-4815-beb6-c520b3274cc9",
        name: "Lotus Cobra".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Snake".into()],
        power: Some(2),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
            },
            effects: vec![Effect::mana_choice(
                ["W", "U", "B", "R", "G"].map(Mana::parse),
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn beastmaster_ascension() -> CardDefinition {
    let text_box = "Whenever a creature you control attacks, you may put a quest counter on this enchantment.\nAs long as this enchantment has seven or more quest counters on it, creatures you control get +5/+5.";
    CardDefinition {
        key: "zen.beastmaster_ascension",
        scryfall_id: "02d7ea78-d539-4c34-8bb3-941353e3da46",
        name: "Beastmaster Ascension".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::attacks(
                TriggerSubject::you_control(CardPredicate::creature()),
                Per::Object,
            ),
            effects: vec![Effect::OptionalPayment {
                cost: PaymentCost::Mana(ManaCost::default()),
                if_paid: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::Quest,
                    1,
                )],
                if_declined: vec![],
            }],
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::YouControl(CardPredicate::creature()),
            condition: Some(Condition::SourceCountersAtLeast {
                kind: CounterKind::Quest,
                count: 7,
            }),
            power: 5,
            toughness: 5,
            per_creature_type_max: None,
            per_color_among: None,
            per_graveyard_matching: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn bloodchief_ascension() -> CardDefinition {
    let text_box = "At the beginning of each end step, if an opponent lost 2 or more life this turn, you may put a quest counter on this enchantment. (Damage causes loss of life.)\nWhenever a card is put into an opponent's graveyard from anywhere, if this enchantment has three or more quest counters on it, you may have that player lose 2 life. If you do, you gain 2 life.";
    // "You may" with nothing to pay is an optional payment of no mana.
    let may = |if_paid: Vec<Effect>| Effect::OptionalPayment {
        cost: PaymentCost::Mana(ManaCost::default()),
        if_paid,
        if_declined: vec![],
    };
    CardDefinition {
        key: "zen.bloodchief_ascension",
        scryfall_id: "aa213dbb-c52a-4084-92aa-d0b5d97a97d9",
        name: "Bloodchief Ascension".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![
            Ability::Triggered {
                intervening_if: Some(Condition::OpponentLostLifeThisTurnAtLeast { amount: 2 }),
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                effects: vec![may(vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::Quest,
                    1,
                )])],
            },
            Ability::Triggered {
                intervening_if: Some(Condition::SourceCountersAtLeast {
                    kind: CounterKind::Quest,
                    count: 3,
                }),
                condition: TriggerCondition::PutIntoGraveyard {
                    whose: Actor::Opponent,
                    predicate: CardPredicate::default(),
                },
                effects: vec![may(vec![
                    Effect::LoseLife {
                        player: PlayerSelector::TriggeringPlayer,
                        amount: EffectValue::Fixed(2),
                    },
                    Effect::gain_life(2),
                ])],
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn valakut_the_molten_pinnacle() -> CardDefinition {
    let text_box = "This land enters tapped.\nWhenever a Mountain you control enters, if you control at least five other Mountains, you may have this land deal 3 damage to any target.\n{T}: Add {R}.";
    CardDefinition {
        key: "zen.valakut_the_molten_pinnacle",
        scryfall_id: "37bce60d-2cb0-4772-9f5c-122a7ed426a0",
        name: "Valakut, the Molten Pinnacle".into(),
        types: CardTypes::new([CardType::Land]),
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("R"),
        }],
        abilities: vec![Ability::Triggered {
            // The entering Mountain and at least five others.
            intervening_if: Some(Condition::ControlsAtLeast {
                count: 6,
                predicate: CardPredicate::subtype("Mountain"),
            }),
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::another_you_control(CardPredicate::subtype("Mountain")),
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement::one(TargetSpec::AnyTarget),
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![Effect::deal_damage(
                        3,
                        ObjectSelector::Target(TargetSpec::AnyTarget),
                    )],
                    if_declined: vec![],
                }],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
