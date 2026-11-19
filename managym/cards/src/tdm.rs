//! Tarkir: Dragonstorm.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, PlayerSelector, TriggerCondition},
    card::{
        AdditionalCost, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![teval_arbiter_of_virtue()]
}

fn teval_arbiter_of_virtue() -> CardDefinition {
    let text_box = "Flying, lifelink\nSpells you cast have delve. (Each card you exile from your graveyard while casting those spells pays for {1}.)\nWhenever you cast a spell, you lose life equal to its mana value.";
    CardDefinition {
        key: "tdm.teval_arbiter_of_virtue",
        scryfall_id: "27a93f5b-7b32-49f0-a179-b897828fe49a",
        printed: PrintedCharacteristics {
            name: "Teval, Arbiter of Virtue".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Spirit".into(), "Dragon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
            statics: vec![StaticAbility::SpellsHave {
                predicate: CardPredicate::default(),
                cost: AdditionalCost::Delve,
            }],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::You,
                    amount: EffectValue::TriggeringManaValue,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
