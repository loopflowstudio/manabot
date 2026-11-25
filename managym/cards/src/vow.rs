//! Innistrad: Crimson Vow.
use managym_state::{
    ability::{
        Ability, Actor, Aggregate, Effect, ObjectProperty, ObjectSelector, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![halana_and_alena_partners(), undying_malice()]
}

fn halana_and_alena_partners() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)\nFirst strike (This creature deals combat damage before creatures without first strike.)\nAt the beginning of combat on your turn, put X +1/+1 counters on another target creature you control, where X is Halana and Alena's power. That creature gains haste until end of turn.";
    CardDefinition {
        key: "vow.halana_and_alena_partners",
        scryfall_id: "608fa232-f5fe-4c58-9efe-fb780f454b19",
        printed: PrintedCharacteristics {
            name: "Halana and Alena, Partners".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Ranger".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Reach, Keyword::FirstStrike]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::another(TargetSpec::CreatureYouControl),
                    effects: vec![
                        Effect::put_counters(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            CounterKind::PlusOnePlusOne,
                            ObjectSelector::This.property(ObjectProperty::Power, Aggregate::Sum),
                        ),
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                        ),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn undying_malice() -> CardDefinition {
    let text_box = "Until end of turn, target creature gains \"When this creature dies, return it to the battlefield tapped under its owner's control with a +1/+1 counter on it.\"";
    CardDefinition {
        key: "vow.undying_malice",
        scryfall_id: "8eb38041-043a-4b18-9d9a-f1283684e8f1",
        printed: PrintedCharacteristics {
            name: "Undying Malice".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::GrantTriggered(Box::new(Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::return_to_owner(
                        ObjectSelector::This,
                        true,
                        Some(CounterKind::PlusOnePlusOne.into()),
                    )],
                })),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
