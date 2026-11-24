//! Magic 2014.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![primeval_bounty()]
}

fn primeval_bounty() -> CardDefinition {
    let text_box = "Whenever you cast a creature spell, create a 3/3 green Beast creature token.\nWhenever you cast a noncreature spell, put three +1/+1 counters on target creature you control.\nLandfall — Whenever a land you control enters, you gain 3 life.";
    CardDefinition {
        key: "m14.primeval_bounty",
        scryfall_id: "e750d55d-d5e8-4abe-99cf-f6b8ba86cf16",
        printed: PrintedCharacteristics {
            name: "Primeval Bounty".into(),
            mana_cost: Some(ManaCost::parse("{5}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::casts(Actor::You, CardPredicate::creature()),
                    intervening_if: None,
                    effects: vec![Effect::create_tokens("token.beast_3_3_g", 1)],
                },
                Ability::Triggered {
                    condition: TriggerCondition::casts(Actor::You, CardPredicate::noncreature()),
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        3,
                    )],
                },
                Ability::when_enters(
                    TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                    vec![Effect::gain_life(3)],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
