//! Magic 2014.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![primeval_bounty(), rise_of_the_dark_realms()]
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

fn rise_of_the_dark_realms() -> CardDefinition {
    let text_box =
        "Put all creature cards from all graveyards onto the battlefield under your control.";

    CardDefinition {
        key: "m14.rise_of_the_dark_realms",
        scryfall_id: "073f81e8-8c0c-4430-bd3e-95ed3625340f",
        printed: PrintedCharacteristics {
            name: "Rise of the Dark Realms".into(),
            mana_cost: Some(ManaCost::parse("{7}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            // One move of every card: the creatures enter together and see
            // each other enter (CR 603.6a).
            spell_effects: vec![Effect::return_to_battlefield(
                ObjectSelector::ZoneMatching {
                    zone: ZoneType::Graveyard,
                    owner: Actor::AnyPlayer,
                    predicate: CardPredicate::creature(),
                },
                false,
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
