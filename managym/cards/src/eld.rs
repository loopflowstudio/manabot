//! Throne of Eldraine.
use managym_state::{
    ability::{Condition, Effect, EffectValue, ObjectSelector, TargetSpec},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![epic_downfall(), fabled_passage()]
}

fn epic_downfall() -> CardDefinition {
    let text_box = "Exile target creature with mana value 3 or greater.";

    CardDefinition {
        key: "eld.epic_downfall",
        scryfall_id: "63da83fe-fa59-40cb-a42e-e1b14b650bc8",
        name: "Epic Downfall".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::exile(ObjectSelector::Target(
            TargetSpec::Permanent {
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    min_mana_value: Some(3),
                    ..Default::default()
                },
            },
        ))],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fabled_passage() -> CardDefinition {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.";

    CardDefinition {
        key: "eld.fabled_passage",
        scryfall_id: "b841bfa8-7c17-4df2-8466-780ab9a4a53a",
        name: "Fabled Passage".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::SelectCards {
                from: ObjectSelector::Zone(ZoneType::Library),
                predicate: CardPredicate::basic_land(),
                min: 0,
                max: EffectValue::Fixed(1),
                effects: vec![
                    Effect::return_to_battlefield(ObjectSelector::Selected, true),
                    Effect::ShuffleLibrary,
                    Effect::If {
                        condition: Condition::ControlsAtLeast {
                            count: 4,
                            predicate: CardPredicate {
                                card_type: Some(CardType::Land),
                                ..Default::default()
                            },
                        },
                        then: vec![Effect::Untap {
                            what: ObjectSelector::Moved,
                        }],
                        otherwise: vec![],
                    },
                ],
                if_declined: vec![Effect::ShuffleLibrary],
            }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
