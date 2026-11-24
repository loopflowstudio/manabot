//! The Hobbit cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, TargetRequirement,
        TargetSpec, TriggerCondition, TurnStep,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, LayerValue, PrintedCharacteristics,
        StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        beorn_the_fierce(),
        smaug_the_great_calamity(),
        bilbo_thief_in_the_night(),
    ]
}

fn beorn_the_fierce() -> CardDefinition {
    let text_box = "Trample\nOther Bears you control get +2/+2.\nAt the beginning of combat on your turn, put a trample counter on up to one target creature you control. It becomes a Bear in addition to its other types. Then if you control three or more Bears, draw two cards.";
    CardDefinition {
        key: "hob.beorn_the_fierce",
        scryfall_id: "367d5f8b-77ee-47f7-bc71-972d62c280a9",
        printed: PrintedCharacteristics {
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
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::You),
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
                                distinct_names: false,
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
        },
        ..Default::default()
    }
}

fn smaug_the_great_calamity() -> CardDefinition {
    CardDefinition {
        key: "hob.smaug_the_great_calamity",
        scryfall_id: "419ca9e5-8413-4378-a4ef-eda5a1024218",
        printed: PrintedCharacteristics {
            name: "Smaug, the Great Calamity".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: "Flying".into(),
            inset: Some(managym_state::card::InsetSpell::Adventure(
                PrintedCharacteristics {
                    name: "Spew Flame".into(),
                    mana_cost: Some(ManaCost::parse("{4}{R}")),
                    types: CardTypes::new([CardType::Sorcery]),
                    subtypes: vec!["Adventure".into()],
                    spell_effects: vec![Effect::deal_damage(
                        5,
                        ObjectSelector::Target(TargetSpec::Creature),
                    )],
                    text_box: "Spew Flame deals 5 damage to target creature.".into(),
                    ..Default::default()
                }
                .into(),
            )),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bilbo_thief_in_the_night() -> CardDefinition {
    use managym_state::ability::{Per, TriggerSubject};
    CardDefinition {
        key: "hob.bilbo_thief_in_the_night",
        scryfall_id: "484c7f83-8339-4ae1-8350-68ce1f7d05a3",
        printed: PrintedCharacteristics {
            name: "Bilbo, Thief in the Night".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Halfling".into(), "Rogue".into()],
            power: Some(2), toughness: Some(2),
            statics: vec![StaticAbility::SpellsFromOutsideHandCostLess(1)],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::CastFromGraveyard {
                    predicate: CardPredicate {card_types_any: vec![CardType::Artifact, CardType::Instant, CardType::Sorcery], ..Default::default()},
                    exile_instants_and_sorceries: true
                }]
            }],
            text_box: "Spells you cast from anywhere other than your hand cost {1} less to cast.\nWhenever Bilbo attacks, you may cast an artifact, instant, or sorcery spell from your graveyard. If an instant or sorcery spell cast this way would be put into your graveyard, exile it instead.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
