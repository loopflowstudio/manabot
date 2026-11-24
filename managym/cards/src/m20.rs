//! Core Set 2020.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
        SpellTargetFilter, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
        TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::EffectDuration,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        agent_of_treachery(),
        brought_back(),
        spectral_sailor(),
        brineborn_cutthroat(),
        gnarlback_rhino(),
    ]
}

fn spectral_sailor() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.";

    CardDefinition {
        key: "m20.spectral_sailor",
        scryfall_id: "67483891-36d1-46f2-8b4f-b8b7bd54bdcc",
        printed: PrintedCharacteristics {
            name: "Spectral Sailor".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into(), "Pirate".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Flash]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{U}"),
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn agent_of_treachery() -> CardDefinition {
    let text_box = "When this creature enters, gain control of target permanent.\nAt the beginning of your end step, if you control three or more permanents you don't own, draw three cards.";
    CardDefinition {
        key: "m20.agent_of_treachery",
        scryfall_id: "cc6686e6-4535-49be-b0b3-e76464656cd2",
        printed: PrintedCharacteristics {
            name: "Agent of Treachery".into(),
            mana_cost: Some(ManaCost::parse("{5}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::GainControl {
                        player: PlayerSelector::You,
                        what: ObjectSelector::Target(TargetSpec::Permanent {
                            predicate: CardPredicate::default(),
                        }),
                        duration: EffectDuration::Indefinite,
                    }],
                },
                Ability::Triggered {
                    intervening_if: Some(Condition::ControlsUnownedAtLeast { count: 3 }),
                    condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(3),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn brought_back() -> CardDefinition {
    let text_box = "Choose up to two target permanent cards in your graveyard that were put there from the battlefield this turn. Return them to the battlefield tapped.";
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate {
            card_types_any: vec![
                CardType::Artifact,
                CardType::Battle,
                CardType::Creature,
                CardType::Enchantment,
                CardType::Land,
                CardType::Planeswalker,
            ],
            from_battlefield_this_turn: true,
            ..Default::default()
        },
    };
    CardDefinition {
        key: "m20.brought_back",
        scryfall_id: "aecdc315-95e5-450c-bedc-998eea46cf8c",
        printed: PrintedCharacteristics {
            name: "Brought Back".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::return_to_battlefield(
                ObjectSelector::Target(spec.clone()),
                true,
            )],
            targeting: vec![TargetRequirement::up_to(2, spec)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn brineborn_cutthroat() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhenever you cast a spell during an opponent's turn, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "m20.brineborn_cutthroat",
        scryfall_id: "0857765f-afd7-418a-a93b-c0bd1b1f037e",
        printed: PrintedCharacteristics {
            name: "Brineborn Cutthroat".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Merfolk".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::not(Condition::YourTurn),
                    condition: Box::new(TriggerCondition::casts(
                        Actor::You,
                        CardPredicate::default(),
                    )),
                },
                intervening_if: None,
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

fn gnarlback_rhino() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nWhenever you cast a spell that targets this creature, draw a card.";
    CardDefinition {
        key: "m20.gnarlback_rhino",
        scryfall_id: "68a69558-aca0-413d-9762-2fa115b44abd",
        printed: PrintedCharacteristics {
            name: "Gnarlback Rhino".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rhino".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Casts {
                    who: Actor::You,
                    predicate: CardPredicate::default(),
                    nth: None,
                    targets: SpellTargetFilter::TargetsThis,
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
