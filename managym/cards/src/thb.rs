//! Theros Beyond Death.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, PlayerSelector, TargetSpec, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![soul_guide_lantern()]
}

fn soul_guide_lantern() -> CardDefinition {
    let text_box = "When this artifact enters, exile target card from a graveyard.\n{T}, Sacrifice this artifact: Exile each opponent's graveyard.\n{1}, {T}, Sacrifice this artifact: Draw a card.";
    CardDefinition {
        key: "thb.soul_guide_lantern",
        scryfall_id: "7c850b94-75c9-4457-8b5e-1193352d6fcb",
        printed: PrintedCharacteristics {
            name: "Soul-Guide Lantern".into(),
            types: CardTypes::new([CardType::Artifact]),
            mana_cost: Some(ManaCost::parse("{1}")),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::exile(ObjectSelector::Target(
                    TargetSpec::GraveyardCard {
                        predicate: CardPredicate::default(),
                    },
                ))],
            )],
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    sacrifice_source: true,
                    effects: vec![Effect::ForPlayer {
                        player: PlayerSelector::EachOpponent,
                        effects: vec![Effect::exile(ObjectSelector::Zone(ZoneType::Graveyard))],
                    }],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    tap_source: true,
                    sacrifice_source: true,
                    mana_cost: ManaCost::parse("{1}"),
                    effects: vec![Effect::DrawCards { count: 1.into() }],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
