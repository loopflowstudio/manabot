//! Worldwake.
use managym_state::ability::{
    Ability, Actor, Condition, Effect, EffectValue, PaymentCost, Per, PlayerSelector,
    TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
    PrintedCharacteristics, StaticAbility, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        dragonmaster_outcast(),
        kalastria_highborn(),
        pulse_tracker(),
        basilisk_collar(),
    ]
}

fn kalastria_highborn() -> CardDefinition {
    let text_box = "Whenever this creature or another Vampire you control dies, you may pay {B}. If you do, target player loses 2 life and you gain 2 life.";
    let player = || PlayerSelector::Target(Box::new(TargetSpec::Player));
    CardDefinition {
        key: "wwk.kalastria_highborn",
        scryfall_id: "f1efd1dd-903c-47a0-b746-5571a3ea1755",
        printed: PrintedCharacteristics {
            name: "Kalastria Highborn".into(),
            mana_cost: Some(ManaCost::parse("{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Shaman".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate::subtype("Vampire")),
                },
                intervening_if: None,
                // The target is chosen as the ability is put on the stack
                // (CR 603.3d); the payment is offered as it resolves.
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Player),
                    effects: vec![Effect::OptionalPayment {
                        cost: PaymentCost::Mana(ManaCost::parse("{B}")),
                        if_paid: vec![
                            Effect::LoseLife {
                                player: player(),
                                amount: EffectValue::Fixed(2),
                            },
                            Effect::gain_life(2),
                        ],
                        if_declined: vec![],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dragonmaster_outcast() -> CardDefinition {
    let text_box = "At the beginning of your upkeep, if you control six or more lands, create a 5/5 red Dragon creature token with flying.";
    CardDefinition {
        key: "wwk.dragonmaster_outcast",
        scryfall_id: "c2297a4e-3c19-4748-9150-efbd2513066a",
        printed: PrintedCharacteristics {
            name: "Dragonmaster Outcast".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Shaman".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                intervening_if: Some(Condition::ControlsAtLeast {
                    count: 6,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    },
                    distinct_names: false,
                }),
                effects: vec![Effect::create_tokens("token.dragon_5_5_r_flying", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn basilisk_collar() -> CardDefinition {
    let text_box = "Equipped creature has deathtouch and lifelink. (Any amount of damage it deals to a creature is enough to destroy it. Damage dealt by this creature also causes you to gain that much life.)\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "wwk.basilisk_collar",
        scryfall_id: "55cdba1b-7a80-435f-9cff-b9365f62e311",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::Deathtouch,
                    Keyword::Lifelink,
                ]))],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            name: "Basilisk Collar".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            mana_cost: Some(ManaCost::parse("{1}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pulse_tracker() -> CardDefinition {
    let text_box = "Whenever this creature attacks, each opponent loses 1 life.";
    CardDefinition {
        key: "wwk.pulse_tracker",
        scryfall_id: "4604a63c-ebe0-420f-968e-3ffc7641ce22",
        printed: PrintedCharacteristics {
            name: "Pulse Tracker".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Rogue".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: 1.into(),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
