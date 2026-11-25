//! Innistrad: Crimson Vow.
use managym_state::{
    ability::{
        Ability, Actor, Aggregate, Effect, ObjectProperty, ObjectSelector, TargetRequirement,
        TargetSpec, TriggerCondition, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![halana_and_alena_partners()]
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
