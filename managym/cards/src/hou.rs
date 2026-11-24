//! Hour of Devastation.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, PlayerSelector, TargetSpec, TriggerCondition,
    },
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![crash_through(), firebrand_archer(), abrade()]
}

fn abrade() -> CardDefinition {
    let text_box =
        "Choose one —\n• Abrade deals 3 damage to target creature.\n• Destroy target artifact.";
    CardDefinition {
        key: "hou.abrade",
        scryfall_id: "84319dfb-eaf7-4b98-8c4f-30f5e779591b",
        printed: PrintedCharacteristics {
            name: "Abrade".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::deal_damage(
                        3,
                        ObjectSelector::Target(TargetSpec::Creature),
                    )],
                    vec![Effect::destroy(ObjectSelector::Target(
                        TargetSpec::Permanent {
                            predicate: CardPredicate {
                                card_types_any: vec![CardType::Artifact],
                                ..Default::default()
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

fn firebrand_archer() -> CardDefinition {
    let text_box =
        "Whenever you cast a noncreature spell, this creature deals 1 damage to each opponent.";
    CardDefinition {
        key: "hou.firebrand_archer",
        scryfall_id: "6ddc6b73-298b-4afa-990a-63706e77dd9f",
        printed: PrintedCharacteristics {
            name: "Firebrand Archer".into(),
            subtypes: vec!["Human".into(), "Archer".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, CardPredicate::noncreature()),
                intervening_if: None,
                effects: vec![Effect::DealDamageToPlayer {
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

fn crash_through() -> CardDefinition {
    let text_box = "Creatures you control gain trample until end of turn. (Each of those creatures can deal excess combat damage to the player or planeswalker it's attacking.)\nDraw a card.";
    CardDefinition {
        key: "hou.crash_through",
        scryfall_id: "4bdaba76-b98d-4699-9a5f-e59285b09552",
        printed: PrintedCharacteristics {
            name: "Crash Through".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                ),
                Effect::DrawCards { count: 1.into() },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
