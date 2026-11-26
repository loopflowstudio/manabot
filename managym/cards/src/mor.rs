//! Morningtide.
use managym_state::{
    ability::{Ability, Actor, Effect, ObjectSelector, PaymentCost, TargetSpec, TriggerCondition},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        ballyrush_banneret(),
        taurean_mauler(),
        kindled_fury(),
        negate(),
    ]
}

fn negate() -> CardDefinition {
    CardDefinition {
        key: "mor.negate",
        scryfall_id: "5a501252-e722-4ebf-bcf7-f53a42745fa7",
        printed: PrintedCharacteristics {
            name: "Negate".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::CounterSpell {
                what: ObjectSelector::Target(TargetSpec::SpellMatching {
                    predicate: CardPredicate::noncreature(),
                }),
                exile: false,
            }],
            text_box: "Counter target noncreature spell.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kindled_fury() -> CardDefinition {
    let text_box = "Target creature gets +1/+0 and gains first strike until end of turn. (It deals combat damage before creatures without first strike.)";
    CardDefinition {
        key: "mor.kindled_fury",
        scryfall_id: "993956c9-30d8-41ee-84c2-c06d0512aea4",
        printed: PrintedCharacteristics {
            name: "Kindled Fury".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                1,
                0,
                Keyword::FirstStrike,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ballyrush_banneret() -> CardDefinition {
    use managym_state::{
        ability::Condition,
        card::{SpellCostReduction, StaticAbility},
    };
    CardDefinition {
        key: "mor.ballyrush_banneret",
        scryfall_id: "a029814e-d84d-43e5-b483-e918871b3333",
        printed: PrintedCharacteristics {
            name: "Ballyrush Banneret".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Kithkin".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(1),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                predicate: CardPredicate {
                    subtypes_any: vec!["Kithkin".into(), "Soldier".into()],
                    ..Default::default()
                },
                condition: Condition::Always,
                generic: 1.into(),
                first_on_own_turn: false,
            })],
            text_box: "Kithkin spells and Soldier spells you cast cost {1} less to cast.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn taurean_mauler() -> CardDefinition {
    let text_box = "Changeling (This card is every creature type.)\nWhenever an opponent casts a spell, you may put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "mor.taurean_mauler",
        scryfall_id: "d50b5df1-b658-4df0-900e-79c44599b93e",
        printed: PrintedCharacteristics {
            name: "Taurean Mauler".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Shapeshifter".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Changeling]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::Opponent, CardPredicate::default()),
                intervening_if: None,
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                    if_declined: vec![],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
