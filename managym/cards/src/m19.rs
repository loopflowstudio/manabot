//! Core Set 2019.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, ObjectSelector, TargetSpec, TriggerCondition,
        TriggerSubject, TurnStep,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![leonin_vanguard(), volley_veteran()]
}

fn leonin_vanguard() -> CardDefinition {
    let text_box = "At the beginning of combat on your turn, if you control three or more creatures, this creature gets +1/+1 until end of turn and you gain 1 life.";
    CardDefinition {
        key: "m19.leonin_vanguard",
        scryfall_id: "724738ad-6a9b-4ef6-b637-558645cd8151",
        printed: PrintedCharacteristics {
            name: "Leonin Vanguard".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Soldier".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
                intervening_if: Some(Condition::ControlsAtLeast {
                    count: 3,
                    predicate: CardPredicate::creature(),
                    distinct_names: false,
                }),
                effects: vec![
                    Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::ModifyPt {
                            power: 1.into(),
                            toughness: 1.into(),
                        },
                    ),
                    Effect::gain_life(1),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn volley_veteran() -> CardDefinition {
    let text_box = "When this creature enters, it deals damage to target creature an opponent controls equal to the number of Goblins you control.";
    CardDefinition {
        key: "m19.volley_veteran",
        scryfall_id: "164960fc-6e80-4a53-90d7-5a18c0a28083",
        printed: PrintedCharacteristics {
            name: "Volley Veteran".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(4),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::deal_damage(
                    ObjectSelector::each_you_control(CardPredicate::subtype("Goblin")).count(),
                    ObjectSelector::Target(TargetSpec::CreatureOpponentControls),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
