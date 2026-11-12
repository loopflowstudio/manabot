//! Kaladesh.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![dramatic_reversal(), noxious_gearhulk()]
}

fn dramatic_reversal() -> CardDefinition {
    let text_box = "Untap all nonland permanents you control.";
    CardDefinition {
        key: "kld.dramatic_reversal",
        scryfall_id: "dcb59045-2743-48ae-8063-727e551b1c41",
        name: "Dramatic Reversal".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::untap(ObjectSelector::each_you_control(
            CardPredicate {
                not_card_types: vec![CardType::Land],
                ..Default::default()
            },
        ))],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn noxious_gearhulk() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)\nWhen this creature enters, you may destroy another target creature. If a creature is destroyed this way, you gain life equal to its toughness.";
    CardDefinition {
        key: "kld.noxious_gearhulk",
        scryfall_id: "9f86e5fe-8723-4494-b4cc-b7ac3a047bd1",
        name: "Noxious Gearhulk".into(),
        mana_cost: Some(ManaCost::parse("{4}{B}{B}")),
        types: CardTypes::new([CardType::Artifact, CardType::Creature]),
        subtypes: vec!["Construct".into()],
        power: Some(5),
        toughness: Some(4),
        keywords: Keywords::of(&[Keyword::Menace]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement {
                    another: true,
                    ..TargetRequirement::one(TargetSpec::Creature)
                },
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![
                        Effect::destroy(ObjectSelector::Target(TargetSpec::Creature)),
                        // Its last known toughness, only if it was destroyed.
                        Effect::If {
                            condition: Condition::Moved,
                            then: vec![Effect::gain_life(EffectValue::TargetToughness)],
                            otherwise: vec![],
                        },
                    ],
                    if_declined: vec![],
                }],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
