//! Bloomburrow.
use managym_state::effect::Modification;
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector, TargetSpec,
        TriggerCondition, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes,
        PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![diresight(), feed_the_cycle(), lunar_convocation(), savor()]
}

fn lunar_convocation() -> CardDefinition {
    let end_step = |condition, effect| Ability::Triggered {
        condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
        intervening_if: Some(condition),
        effects: vec![effect],
    };
    CardDefinition {
        key: "blb.lunar_convocation",
        scryfall_id: "a9ee50d4-c878-457b-964d-29c039ce9852",
        printed: PrintedCharacteristics {
            name: "Lunar Convocation".into(),
            mana_cost: Some(ManaCost::parse("{W}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![
                end_step(Condition::GainedLifeThisTurn, Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::Fixed(1),
                }),
                end_step(Condition::AllOf(vec![Condition::GainedLifeThisTurn, Condition::LostLifeThisTurn]), Effect::create_tokens("token.bat", 1)),
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{B}"),
                life_cost: 2,
                effects: vec![Effect::DrawCards { count: EffectValue::Fixed(1) }],
                ..Default::default()
            }],
            text_box: "At the beginning of your end step, if you gained life this turn, each opponent loses 1 life.\nAt the beginning of your end step, if you gained and lost life this turn, create a 1/1 black Bat creature token with flying.\n{1}{B}, Pay 2 life: Draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn savor() -> CardDefinition {
    let text_box = "Target creature gets -2/-2 until end of turn. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";

    CardDefinition {
        key: "blb.savor",
        scryfall_id: "1397f689-dca1-4d35-864b-92c5606afb9a",
        printed: PrintedCharacteristics {
            name: "Savor".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(-2, -2),
                ),
                Effect::create_tokens("token.food", 1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn diresight() -> CardDefinition {
    let text_box = "Surveil 2, then draw two cards. You lose 2 life. (To surveil 2, look at the top two cards of your library, then put any number of them into your graveyard and the rest on top of your library in any order.)";
    CardDefinition {
        key: "blb.diresight",
        scryfall_id: "fada29c0-5293-40a4-b36d-d073ee99e650",
        printed: PrintedCharacteristics {
            name: "Diresight".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::Surveil { count: 2 },
                Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                },
                Effect::LoseLife {
                    player: PlayerSelector::You,
                    amount: EffectValue::Fixed(2),
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn feed_the_cycle() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, forage or pay {B}. (To forage, exile three cards from your graveyard or sacrifice a Food.)\nDestroy target creature or planeswalker.";
    CardDefinition {
        key: "blb.feed_the_cycle",
        scryfall_id: "7e017ff8-2936-4a1b-bece-00004cfbad06",
        printed: PrintedCharacteristics {
            name: "Feed the Cycle".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::ForageOr(ManaCost::parse("{B}"))],
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                        ..Default::default()
                    },
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
