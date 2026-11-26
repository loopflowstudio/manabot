//! Fate Reforged.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost,
        PlayerSelector, TargetRequirement, TargetSpec, TriggerCondition, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, Restriction, StaticAbility, StaticRestriction, StaticScope,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![flamewake_phoenix(), valorous_stance(), humble_defector()]
}

fn flamewake_phoenix() -> CardDefinition {
    let text_box = "Flying, haste\nThis creature attacks each combat if able.\nFerocious — At the beginning of combat on your turn, if you control a creature with power 4 or greater, you may pay {R}. If you do, return this card from your graveyard to the battlefield.";
    CardDefinition {
        key: "frf.flamewake_phoenix",
        scryfall_id: "fefd5848-9fe1-4129-a5d7-e51606bf76ef",
        printed: PrintedCharacteristics {
            name: "Flamewake Phoenix".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Phoenix".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Haste]),
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::This,
                unless: None,
                kind: Restriction::AttacksEachCombat,
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::InGraveyard {
                    condition: Box::new(TriggerCondition::beginning_of(
                        TurnStep::Combat,
                        Actor::You,
                    )),
                },
                intervening_if: Some(Condition::ControlsAtLeast {
                    count: 1,
                    distinct_names: false,
                    predicate: CardPredicate {
                        min_power: Some(4),
                        ..CardPredicate::creature()
                    },
                }),
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::parse("{R}")),
                    if_paid: vec![Effect::return_to_battlefield(ObjectSelector::This, false)],
                    if_declined: vec![],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn humble_defector() -> CardDefinition {
    let text_box = "{T}: Draw two cards. Target opponent gains control of this creature. Activate only during your turn.";
    CardDefinition {
        key: "frf.humble_defector",
        scryfall_id: "4bb0c13f-6cea-48b4-a5c8-2c0cfcfea7fd",
        printed: PrintedCharacteristics {
            name: "Humble Defector".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                own_turn_only: true,
                effects: vec![
                    Effect::DrawCards {
                        count: EffectValue::Fixed(2),
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(TargetSpec::Opponent),
                        effects: vec![Effect::GainControl {
                            player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                            what: ObjectSelector::This,
                            duration: EffectDuration::Indefinite,
                        }],
                    },
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn valorous_stance() -> CardDefinition {
    let text_box = "Choose one —\n• Target creature gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)\n• Destroy target creature with toughness 4 or greater.";
    CardDefinition {
        key: "frf.valorous_stance",
        scryfall_id: "65998e94-15a0-41f1-8288-730b957f81df",
        printed: PrintedCharacteristics {
            name: "Valorous Stance".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::Creature),
                        Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                    )],
                    vec![Effect::destroy(ObjectSelector::Target(
                        TargetSpec::Permanent {
                            predicate: CardPredicate {
                                min_toughness: Some(4),
                                ..CardPredicate::creature()
                            },
                        },
                    ))],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
