//! Jumpstart 2022.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        deadly_plot(),
        lita_mechanical_engineer(),
        mild_mannered_librarian(),
    ]
}

fn lita_mechanical_engineer() -> CardDefinition {
    let text_box = "Vigilance\nAt the beginning of your end step, untap each other artifact creature you control.\n{3}{W}, {T}: Create a 5/5 colorless Vehicle artifact token named Zeppelin with flying and crew 3. (It has \"Tap any number of creatures you control with total power 3 or more: This token becomes an artifact creature until end of turn.\")";
    CardDefinition {
        key: "j22.lita_mechanical_engineer",
        scryfall_id: "f3a5a044-d474-43f4-95a9-e67808908e6c",
        printed: PrintedCharacteristics {
            name: "Lita, Mechanical Engineer".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Artificer".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                effects: vec![Effect::untap(ObjectSelector::each_other_you_control(
                    CardPredicate {
                        card_type: Some(CardType::Creature),
                        card_types_any: vec![CardType::Artifact],
                        ..Default::default()
                    },
                ))],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{W}"),
                tap_source: true,
                effects: vec![Effect::create_tokens("token.zeppelin", 1)],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deadly_plot() -> CardDefinition {
    let text_box = "Choose one —\n• Destroy target creature or planeswalker.\n• Return target Zombie creature card from your graveyard to the battlefield tapped.";
    CardDefinition {
        key: "j22.deadly_plot",
        scryfall_id: "9c78da23-e7ec-4a3d-9f79-09fb86993b26",
        printed: PrintedCharacteristics {
            name: "Deadly Plot".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::destroy(ObjectSelector::Target(
                        TargetSpec::Permanent {
                            predicate: CardPredicate {
                                card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                                ..Default::default()
                            },
                        },
                    ))],
                    vec![Effect::return_to_battlefield(
                        ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                            predicate: CardPredicate {
                                subtype: Some("Zombie".into()),
                                ..CardPredicate::creature()
                            },
                        }),
                        true,
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mild_mannered_librarian() -> CardDefinition {
    let text_box = "{3}{G}: This creature becomes a Werewolf. Put two +1/+1 counters on it and you draw a card. Activate only once.";
    CardDefinition {
        key: "j22.mild_mannered_librarian",
        scryfall_id: "3eee6f29-ef06-47f6-99af-fb0ff88f09d0",
        printed: PrintedCharacteristics {
            name: "Mild-Mannered Librarian".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into()],
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{G}"),
                once_only: true,
                effects: vec![
                    // "Becomes a Werewolf" replaces its creature types (CR 205.1b).
                    Effect::indefinitely(
                        ObjectSelector::This,
                        Modification::SetCreatureTypes(vec!["Werewolf".into()]),
                    ),
                    Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 2),
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
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
