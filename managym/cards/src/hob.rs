//! The Hobbit cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, TargetRequirement, TargetSpec,
        TriggerCondition,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, LayerValue, StaticAbility,
        StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![beorn_the_fierce()]
}

fn beorn_the_fierce() -> CardDefinition {
    let text_box = "Trample\nOther Bears you control get +2/+2.\nAt the beginning of combat on your turn, put a trample counter on up to one target creature you control. It becomes a Bear in addition to its other types. Then if you control three or more Bears, draw two cards.";
    CardDefinition {
        key: "hob.beorn_the_fierce",
        scryfall_id: "367d5f8b-77ee-47f7-bc71-972d62c280a9",
        name: "Beorn the Fierce".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Bear".into(), "Shapeshifter".into(), "Warrior".into()],
        power: Some(6),
        toughness: Some(6),
        keywords: Keywords::of(&[Keyword::Trample]),
        statics: vec![StaticAbility::Continuous(StaticContinuous {
            scope: StaticScope::OtherYouControl(CardPredicate::subtype("Bear")),
            condition: None,
            characteristic_defining: false,
            modifications: vec![Modification::ModifyPt {
                power: LayerValue::Fixed(2),
                toughness: LayerValue::Fixed(2),
            }],
        })],
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourCombat,
            intervening_if: None,
            effects: vec![Effect::TargetClause {
                requirement: TargetRequirement::up_to(1, TargetSpec::CreatureYouControl),
                effects: vec![
                    Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::Trample,
                        1,
                    ),
                    Effect::indefinitely(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        Modification::AddSubtypes(vec!["Bear".into()]),
                    ),
                    Effect::If {
                        condition: Condition::ControlsAtLeast {
                            count: 3,
                            predicate: CardPredicate::subtype("Bear"),
                        },
                        then: vec![Effect::DrawCards {
                            count: EffectValue::Fixed(2),
                        }],
                        otherwise: vec![],
                    },
                ],
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
