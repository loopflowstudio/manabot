//! Theros Beyond Death.
use managym_state::ability::{
    Ability, Actor, CompareOp, Condition, Effect, ObjectSelector, PlayerSelector, TargetSpec,
    TriggerCondition, TriggerSubject, TurnStep,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
    PrintedCharacteristics,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;
use managym_state::zone::ZoneType;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        nessian_hornbeetle(),
        terror_of_mount_velus(),
        soul_guide_lantern(),
    ]
}

fn nessian_hornbeetle() -> CardDefinition {
    let text_box = "At the beginning of combat on your turn, if you control another creature with power 4 or greater, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "thb.nessian_hornbeetle",
        scryfall_id: "8200fcda-e30c-460f-9964-47e657b7c758",
        printed: PrintedCharacteristics {
            name: "Nessian Hornbeetle".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Insect".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: Some(Condition::compare(
                    ObjectSelector::each_other_you_control(CardPredicate {
                        min_power: Some(4),
                        ..CardPredicate::creature()
                    })
                    .count(),
                    CompareOp::Ge,
                    1,
                )),
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn soul_guide_lantern() -> CardDefinition {
    let text_box = "When this artifact enters, exile target card from a graveyard.\n{T}, Sacrifice this artifact: Exile each opponent's graveyard.\n{1}, {T}, Sacrifice this artifact: Draw a card.";
    CardDefinition {
        key: "thb.soul_guide_lantern",
        scryfall_id: "7c850b94-75c9-4457-8b5e-1193352d6fcb",
        printed: PrintedCharacteristics {
            name: "Soul-Guide Lantern".into(),
            types: CardTypes::new([CardType::Artifact]),
            mana_cost: Some(ManaCost::parse("{1}")),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::exile(ObjectSelector::Target(
                    TargetSpec::GraveyardCard {
                        predicate: CardPredicate::default(),
                    },
                ))],
            )],
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    sacrifice_source: true,
                    effects: vec![Effect::ForPlayer {
                        player: PlayerSelector::EachOpponent,
                        effects: vec![Effect::exile(ObjectSelector::Zone(ZoneType::Graveyard))],
                    }],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    tap_source: true,
                    sacrifice_source: true,
                    mana_cost: ManaCost::parse("{1}"),
                    effects: vec![Effect::DrawCards { count: 1.into() }],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn terror_of_mount_velus() -> CardDefinition {
    let text_box = "Flying\nDouble strike (This creature deals both first-strike and regular combat damage.)\nWhen this creature enters, creatures you control gain double strike until end of turn.";
    CardDefinition {
        key: "thb.terror_of_mount_velus",
        scryfall_id: "332dc6c3-7802-4bde-aa4e-0feab70c216f",
        printed: PrintedCharacteristics {
            name: "Terror of Mount Velus".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::DoubleStrike]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
