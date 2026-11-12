//! Core Set 2020.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords},
    effect::EffectDuration,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![agent_of_treachery(), brought_back(), spectral_sailor()]
}

fn spectral_sailor() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.";

    CardDefinition {
        key: "m20.spectral_sailor",
        scryfall_id: "67483891-36d1-46f2-8b4f-b8b7bd54bdcc",
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
    }
}

fn agent_of_treachery() -> CardDefinition {
    let text_box = "When this creature enters, gain control of target permanent.\nAt the beginning of your end step, if you control three or more permanents you don't own, draw three cards.";
    CardDefinition {
        key: "m20.agent_of_treachery",
        scryfall_id: "cc6686e6-4535-49be-b0b3-e76464656cd2",
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
    }
}
