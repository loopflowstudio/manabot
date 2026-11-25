//! Magic: The Gathering-Commander.
use managym_state::{
    ability::{Condition, Effect, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![scavenging_ooze()]
}

fn scavenging_ooze() -> CardDefinition {
    let text_box = "{G}: Exile target card from a graveyard. If it was a creature card, put a +1/+1 counter on this creature and you gain 1 life.";
    CardDefinition {
        key: "cmd.scavenging_ooze",
        scryfall_id: "371ceb58-f498-4616-a7f0-eb118fe2e4ff",
        printed: PrintedCharacteristics {
            name: "Scavenging Ooze".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Ooze".into()],
            power: Some(2),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{G}"),
                effects: vec![
                    Effect::exile(ObjectSelector::Target(TargetSpec::GraveyardCard {
                        predicate: CardPredicate::default(),
                    })),
                    // The exiled card the target became (CR 400.7j).
                    Effect::If {
                        condition: Condition::TargetCardExiled(CardPredicate::creature()),
                        then: vec![
                            Effect::put_counters(
                                ObjectSelector::This,
                                CounterKind::PlusOnePlusOne,
                                1,
                            ),
                            Effect::gain_life(1),
                        ],
                        otherwise: vec![],
                    },
                ],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
