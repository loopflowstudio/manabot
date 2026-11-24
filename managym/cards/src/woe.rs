//! Wilds of Eldraine.
use managym_state::ability::{
    Ability, CompareOp, Condition, Effect, ObjectSelector, Per, PlayerSelector, TargetSpec,
    TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
    Keywords, PrintedCharacteristics, SpellCostReduction, StaticAbility,
};
use managym_state::effect::Modification;
use managym_state::mana::{Mana, ManaCost};
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        ruby_daring_tracker(),
        archons_glory(),
        mocking_sprite(),
        scarecrow_guide(),
        stroke_of_midnight(),
        quick_study(),
    ]
}

fn stroke_of_midnight() -> CardDefinition {
    let text_box = "Destroy target nonland permanent. Its controller creates a 1/1 white Human creature token.";
    let nonland = || {
        ObjectSelector::Target(TargetSpec::Permanent {
            predicate: CardPredicate {
                not_card_types: vec![CardType::Land],
                ..Default::default()
            },
        })
    };
    CardDefinition {
        key: "woe.stroke_of_midnight",
        scryfall_id: "289ba7ec-e30e-436f-b8d4-c88b65ecd137",
        printed: PrintedCharacteristics {
            name: "Stroke of Midnight".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::destroy(nonland()),
                // Its controller as it last existed, destroyed or not.
                Effect::ForPlayer {
                    player: PlayerSelector::TargetController,
                    effects: vec![Effect::create_tokens("token.human", 1)],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn scarecrow_guide() -> CardDefinition {
    let text_box = "Reach\n{1}: Add one mana of any color. Activate only once each turn.";
    CardDefinition {
        key: "woe.scarecrow_guide",
        scryfall_id: "3dc08461-bec6-4a18-b782-da4c92abb789",
        printed: PrintedCharacteristics {
            name: "Scarecrow Guide".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Scarecrow".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Reach]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                once_each_turn: true,
                mana_cost: ManaCost::parse("1"),
                effects: vec![Effect::any_color_mana()],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn archons_glory() -> CardDefinition {
    let text_box = "Bargain (You may sacrifice an artifact, enchantment, or token as you cast this spell.)\nTarget creature gets +2/+2 until end of turn. If this spell was bargained, that creature also gains flying and lifelink until end of turn.";
    let creature = || ObjectSelector::Target(TargetSpec::Creature);
    CardDefinition {
        key: "woe.archons_glory",
        scryfall_id: "e71768e7-4ef7-4fb2-838b-eb3a7f662d38",
        printed: PrintedCharacteristics {
            name: "Archon's Glory".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Bargain],
            spell_effects: vec![
                Effect::until_end_of_turn(creature(), Modification::pump(2, 2)),
                Effect::If {
                    condition: Condition::Kicked,
                    then: vec![Effect::until_end_of_turn(
                        creature(),
                        Modification::AddKeywords(Keywords::of(&[
                            Keyword::Flying,
                            Keyword::Lifelink,
                        ])),
                    )],
                    otherwise: vec![],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mocking_sprite() -> CardDefinition {
    let text_box = "Flying\nInstant and sorcery spells you cast cost {1} less to cast.";
    CardDefinition {
        key: "woe.mocking_sprite",
        scryfall_id: "e595014d-4ff4-4561-b7f2-a9bd56300b01",
        printed: PrintedCharacteristics {
            name: "Mocking Sprite".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Faerie".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                first_on_own_turn: false,
                predicate: CardPredicate::instant_or_sorcery(),
                condition: Condition::Always,
                generic: 1,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ruby_daring_tracker() -> CardDefinition {
    let text_box = "Haste (This creature can attack and {T} as soon as it comes under your control.)\nWhenever Ruby attacks while you control a creature with power 4 or greater, Ruby gets +2/+2 until end of turn.\n{T}: Add {R} or {G}.";
    CardDefinition {
        key: "woe.ruby_daring_tracker",
        scryfall_id: "ffb5786b-6825-4ebf-a1e1-80011340adbb",
        printed: PrintedCharacteristics {
            name: "Ruby, Daring Tracker".into(),
            mana_cost: Some(ManaCost::parse("{R}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Scout".into()],
            power: Some(1),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Haste]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::compare(
                        ObjectSelector::each_you_control(CardPredicate {
                            min_power: Some(4),
                            ..CardPredicate::creature()
                        })
                        .count(),
                        CompareOp::Ge,
                        1,
                    ),
                    condition: Box::new(TriggerCondition::attacks(
                        TriggerSubject::This,
                        Per::Object,
                    )),
                },
                intervening_if: None,
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(2, 2),
                )],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(["R", "G"].map(Mana::parse))],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn quick_study() -> CardDefinition {
    let text_box = "Draw two cards.";
    CardDefinition {
        key: "woe.quick_study",
        scryfall_id: "b78e2bca-bc93-464a-8911-8361abff2ac6",
        printed: PrintedCharacteristics {
            name: "Quick Study".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::DrawCards { count: 2.into() }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
