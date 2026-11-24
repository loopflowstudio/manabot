//! Innistrad.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, Per, Scope, TriggerCondition, TriggerSubject,
    },
    card::{
        BlockRestriction, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, ReplacementEffect, StaticAbility,
    },
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        stromkirk_noble(),
        vampire_interloper(),
        blasphemous_act(),
        diregraf_ghoul(),
        typhoid_rats(),
    ]
}

fn diregraf_ghoul() -> CardDefinition {
    CardDefinition {
        key: "isd.diregraf_ghoul",
        scryfall_id: "4ed5790a-3354-49c2-89b6-3fc0de8dcc7c",
        printed: PrintedCharacteristics {
            name: "Diregraf Ghoul".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into()],
            power: Some(2),
            toughness: Some(2),
            replacement_effects: vec![ReplacementEffect::EntersTapped],
            text_box: "This creature enters tapped.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn typhoid_rats() -> CardDefinition {
    let text_box =
        "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)";

    CardDefinition {
        key: "isd.typhoid_rats",
        scryfall_id: "4490ce65-c73a-4809-abd1-ccc3175bd2a4",
        printed: PrintedCharacteristics {
            name: "Typhoid Rats".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rat".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn blasphemous_act() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast for each creature on the battlefield.\nBlasphemous Act deals 13 damage to each creature.";
    CardDefinition {
        key: "isd.blasphemous_act",
        scryfall_id: "509ce648-fb76-486d-8b39-183e368b7cb7",
        printed: PrintedCharacteristics {
            name: "Blasphemous Act".into(),
            mana_cost: Some(ManaCost::parse("{8}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            statics: vec![StaticAbility::CostsLessPerPermanentOnBattlefield(
                CardPredicate::creature(),
            )],
            spell_effects: vec![Effect::deal_damage(
                13,
                ObjectSelector::each(Scope::Battlefield, CardPredicate::creature()),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn stromkirk_noble() -> CardDefinition {
    let text_box = "This creature can't be blocked by Humans.\nWhenever this creature deals combat damage to a player, put a +1/+1 counter on it.";
    CardDefinition {
        key: "isd.stromkirk_noble",
        scryfall_id: "9c16cf74-f9e0-4d80-9a29-b91dec0b6b38",
        printed: PrintedCharacteristics {
            name: "Stromkirk Noble".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Noble".into()],
            power: Some(1),
            toughness: Some(1),
            statics: vec![StaticAbility::BlockRestriction(
                BlockRestriction::CantBeBlockedBy(CardPredicate::subtype("Human")),
            )],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Some(Actor::AnyPlayer),
                    per: Per::Object,
                },
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

fn vampire_interloper() -> CardDefinition {
    let text_box = "Flying\nThis creature can't block.";
    CardDefinition {
        key: "isd.vampire_interloper",
        scryfall_id: "48105c2e-ee36-4117-b56b-3440298da995",
        printed: PrintedCharacteristics {
            name: "Vampire Interloper".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Scout".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            // "Can't block [any] creatures": the unconditional prohibition.
            statics: vec![StaticAbility::BlockRestriction(
                BlockRestriction::CantBlock(CardPredicate::default()),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
