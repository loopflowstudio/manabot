//! Champions of Kamigawa.
use managym_state::{
    ability::{Actor, Condition, Effect, PlayerSelector},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, EntryScope, Keyword,
        Keywords, PrintedCharacteristics, ReplacementEffect, StaticAbility, StaticKeywordGrant,
        StaticScope, TriggeredManaAbility, TriggeredManaOutput,
    },
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![myojin_of_nights_reach(), heartbeat_of_spring(), time_stop()]
}

fn heartbeat_of_spring() -> CardDefinition {
    let text_box = "Whenever a player taps a land for mana, that player adds one mana of any type that land produced.";
    CardDefinition {
        key: "chk.heartbeat_of_spring",
        scryfall_id: "6be3ba05-dce0-463b-bfe6-d51df63cc1a8",
        printed: PrintedCharacteristics {
            name: "Heartbeat of Spring".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            triggered_mana_abilities: vec![TriggeredManaAbility {
                who: Actor::AnyPlayer,
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
                output: TriggeredManaOutput::OneProducedType,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn time_stop() -> CardDefinition {
    let text_box = "End the turn. (Exile all spells and abilities, including this spell. The player whose turn it is discards down to their maximum hand size. Damage heals and \"this turn\" and \"until end of turn\" effects end.)";
    CardDefinition {
        key: "chk.time_stop",
        scryfall_id: "f968c5e9-12a8-4542-90b4-84e0238fa375",
        printed: PrintedCharacteristics {
            name: "Time Stop".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::EndTheTurn],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn myojin_of_nights_reach() -> CardDefinition {
    let text_box = "Myojin of Night's Reach enters with a divinity counter on it if you cast it from your hand.\nMyojin of Night's Reach has indestructible as long as it has a divinity counter on it.\nRemove a divinity counter from Myojin of Night's Reach: Each opponent discards their hand.";
    CardDefinition {
        key: "chk.myojin_of_nights_reach",
        scryfall_id: "13a295b0-535e-4c2d-879d-62603d1f2f1b",
        printed: PrintedCharacteristics {
            name: "Myojin of Night's Reach".into(),
            mana_cost: Some(ManaCost::parse("{5}{B}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Spirit".into()],
            power: Some(5),
            toughness: Some(2),
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                on: EntryScope::This,
                kind: CounterKind::Divinity,
                count: 1.into(),
                condition: Some(Condition::EntryCastFrom(Some(ZoneType::Hand))),
            }],
            statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::This,
                keywords: Keywords::of(&[Keyword::Indestructible]),
                condition: Some(Condition::SourceCountersAtLeast {
                    kind: CounterKind::Divinity,
                    count: 1,
                }),
            })],
            activated_abilities: vec![ActivatedAbilityDefinition {
                remove_counters: Some((CounterKind::Divinity, 1)),
                effects: vec![Effect::DiscardHand {
                    who: PlayerSelector::EachOpponent,
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
