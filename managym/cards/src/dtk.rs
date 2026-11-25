//! Dragons of Tarkir.
use managym_state::ability::{
    Ability, Actor, Aggregate, CompareOp, Condition, Effect, EffectValue, ObjectProperty,
    ObjectSelector, PlayerSelector, Scope, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
    SpellCostReduction, StaticAbility,
};
use managym_state::effect::Modification;
use managym_state::mana::{Color, ManaCost};
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        inspiring_call(),
        surrak_the_hunt_caller(),
        clone_legion(),
        dragonlords_servant(),
        rending_volley(),
        seismic_rupture(),
        impact_tremors(),
    ]
}

fn clone_legion() -> CardDefinition {
    let text_box =
        "For each creature target player controls, create a token that's a copy of that creature.";
    CardDefinition {
        key: "dtk.clone_legion",
        scryfall_id: "45c6a201-1d4d-4ae5-a84e-098e094eaf61",
        printed: PrintedCharacteristics {
            name: "Clone Legion".into(),
            mana_cost: Some(ManaCost::parse("{7}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::CreateTokenCopy {
                of: ObjectSelector::each(Scope::TargetPlayerControls, CardPredicate::creature()),
                exceptions: vec![],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rending_volley() -> CardDefinition {
    let text_box = "This spell can't be countered.\nRending Volley deals 4 damage to target white or blue creature.";
    let white_or_blue_creature = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_type: Some(CardType::Creature),
            colors_any: vec![Color::White, Color::Blue],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "dtk.rending_volley",
        scryfall_id: "8234090e-9df1-4915-90ef-8a4bc6212655",
        printed: PrintedCharacteristics {
            name: "Rending Volley".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::CantBeCountered],
            spell_effects: vec![Effect::deal_damage(
                4,
                ObjectSelector::Target(white_or_blue_creature),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dragonlords_servant() -> CardDefinition {
    let text_box = "Dragon spells you cast cost {1} less to cast.";
    CardDefinition {
        key: "dtk.dragonlords_servant",
        scryfall_id: "0ffcdd54-b6be-4d42-82c0-ae927037e859",
        printed: PrintedCharacteristics {
            name: "Dragonlord's Servant".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Shaman".into()],
            power: Some(1),
            toughness: Some(3),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                first_on_own_turn: false,
                predicate: CardPredicate::subtype("Dragon"),
                condition: Condition::Always,
                generic: EffectValue::Fixed(1),
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn surrak_the_hunt_caller() -> CardDefinition {
    let text_box = "Formidable — At the beginning of combat on your turn, if creatures you control have total power 8 or greater, target creature you control gains haste until end of turn. (It can attack and {T} no matter when it came under your control.)";
    CardDefinition {
        key: "dtk.surrak_the_hunt_caller",
        scryfall_id: "b374446d-44bc-4ac5-9829-8c49f0cca173",
        printed: PrintedCharacteristics {
            name: "Surrak, the Hunt Caller".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Warrior".into()],
            power: Some(5),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: Some(Condition::compare(
                    ObjectSelector::each_you_control(CardPredicate::creature())
                        .property(ObjectProperty::Power, Aggregate::Sum),
                    CompareOp::Ge,
                    8,
                )),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::CreatureYouControl),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn inspiring_call() -> CardDefinition {
    let text_box = "Draw a card for each creature you control with a +1/+1 counter on it. Those creatures gain indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy them.)";
    let countered = || {
        ObjectSelector::each_you_control(CardPredicate {
            card_type: Some(CardType::Creature),
            with_counter: true,
            ..Default::default()
        })
    };
    CardDefinition {
        key: "dtk.inspiring_call",
        scryfall_id: "c565b991-7021-4b81-b9c0-f7231daae360",
        printed: PrintedCharacteristics {
            name: "Inspiring Call".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::DrawCards {
                    count: countered().count(),
                },
                Effect::until_end_of_turn(
                    countered(),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn impact_tremors() -> CardDefinition {
    CardDefinition {
        key: "dtk.impact_tremors",
        scryfall_id: "56fb4035-197b-4d28-9bf7-bb62c304067e",
        printed: PrintedCharacteristics {
            name: "Impact Tremors".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::matching(Actor::You, CardPredicate::creature(), false),
                vec![Effect::DealDamageToPlayer { player: PlayerSelector::EachOpponent, amount: 1.into() }],
            )],
            text_box: "Whenever a creature you control enters, this enchantment deals 1 damage to each opponent.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn seismic_rupture() -> CardDefinition {
    let text_box = "Seismic Rupture deals 2 damage to each creature without flying.";
    CardDefinition {
        key: "dtk.seismic_rupture",
        scryfall_id: "9b952e4e-c1ed-4455-90d5-46b56478e6b0",
        printed: PrintedCharacteristics {
            name: "Seismic Rupture".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::deal_damage(
                2,
                ObjectSelector::each(
                    Scope::Battlefield,
                    CardPredicate {
                        not_flying: true,
                        ..CardPredicate::creature()
                    },
                ),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
