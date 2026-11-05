//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Condition, Effect, ObjectSelector, PlayWindow, TriggerCondition, TriggerSubject,
    },
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![kellan_planar_trailblazer()]
}

fn kellan_planar_trailblazer() -> CardDefinition {
    let text_box = "{1}{R}: If Kellan is a Scout, it becomes a Human Faerie Detective and gains \"Whenever Kellan deals combat damage to a player, exile the top card of your library. You may play that card this turn.\"\n{2}{R}: If Kellan is a Detective, it becomes a 3/2 Human Faerie Rogue and gains double strike.";
    CardDefinition {
        key: "fdn.kellan_planar_trailblazer",
        scryfall_id: "f46a9329-7b91-441d-8653-50c1152c9120",
        name: "Kellan, Planar Trailblazer".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Human".into(), "Faerie".into(), "Scout".into()],
        power: Some(2),
        toughness: Some(1),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{R}"),
                effects: vec![Effect::If {
                    condition: Condition::SourceMatches(CardPredicate::subtype("Scout")),
                    then: vec![
                        Effect::indefinitely(
                            ObjectSelector::This,
                            Modification::SetCreatureTypes(vec![
                                "Human".into(),
                                "Faerie".into(),
                                "Detective".into(),
                            ]),
                        ),
                        Effect::indefinitely(
                            ObjectSelector::This,
                            Modification::GrantTriggered(Box::new(Ability::Triggered {
                                condition: TriggerCondition::CreatureDealsCombatDamageToPlayer {
                                    subject: TriggerSubject::This,
                                },
                                intervening_if: None,
                                effects: vec![Effect::ExileTopCardsToPlay {
                                    count: 1,
                                    until: PlayWindow::ThisTurn,
                                }],
                            })),
                        ),
                    ],
                    otherwise: vec![],
                }],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{R}"),
                effects: vec![Effect::If {
                    condition: Condition::SourceMatches(CardPredicate::subtype("Detective")),
                    then: vec![
                        Effect::indefinitely(
                            ObjectSelector::This,
                            Modification::SetCreatureTypes(vec![
                                "Human".into(),
                                "Faerie".into(),
                                "Rogue".into(),
                            ]),
                        ),
                        Effect::indefinitely(
                            ObjectSelector::This,
                            Modification::base_pt(Some(3), Some(2)),
                        ),
                        Effect::indefinitely(
                            ObjectSelector::This,
                            Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
                        ),
                    ],
                    otherwise: vec![],
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
