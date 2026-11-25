//! War of the Spark.
use managym_state::{
    ability::{
        Ability, Effect, Except, ObjectSelector, PlayerSelector, Scope, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![liliana_dreadhorde_general(), mayhem_devil()]
}

fn liliana_dreadhorde_general() -> CardDefinition {
    let text_box = "Whenever a creature you control dies, draw a card.\n+1: Create a 2/2 black Zombie creature token.\n−4: Each player sacrifices two creatures of their choice.\n−9: Each opponent chooses a permanent they control of each permanent type and sacrifices the rest.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    let keep_one_of_each: Vec<Effect> = CardType::PERMANENT_TYPES
        .into_iter()
        .map(|card_type| Effect::remember_chosen(CardPredicate::card_type(card_type)))
        .chain([Effect::Sacrifice {
            what: ObjectSelector::Each {
                of: Scope::YouControl,
                predicate: CardPredicate::permanent(),
                except: Some(Except::Remembered),
                max_mana_value: None,
                exact_mana_value: None,
            },
            optional: false,
        }])
        .collect();
    CardDefinition {
        key: "war.liliana_dreadhorde_general",
        scryfall_id: "d75ebba8-34ca-47a0-bf13-8318ad73b343",
        printed: PrintedCharacteristics {
            name: "Liliana, Dreadhorde General".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}{B}")),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Liliana".into()],
            loyalty: Some(6),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards { count: 1.into() }],
            }],
            activated_abilities: vec![
                loyalty(1, vec![Effect::create_tokens("token.zombie_2_2_b", 1)]),
                loyalty(-4, vec![Effect::EachPlayerSacrificesCreatures { count: 2 }]),
                loyalty(
                    -9,
                    vec![Effect::ForPlayer {
                        player: PlayerSelector::EachOpponent,
                        effects: keep_one_of_each,
                    }],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mayhem_devil() -> CardDefinition {
    let text_box =
        "Whenever a player sacrifices a permanent, this creature deals 1 damage to any target.";

    CardDefinition {
        key: "war.mayhem_devil",
        scryfall_id: "17416926-168b-49b3-9231-acbb8f8a1d13",
        printed: PrintedCharacteristics {
            name: "Mayhem Devil".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Devil".into()],
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Sacrificed {
                    subject: TriggerSubject::any(CardPredicate::default()),
                },
                effects: vec![Effect::deal_damage(
                    1,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
