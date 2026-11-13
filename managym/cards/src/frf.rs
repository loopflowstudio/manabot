//! Fate Reforged.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, PlayerSelector, TargetRequirement, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    effect::EffectDuration,
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![humble_defector()]
}

fn humble_defector() -> CardDefinition {
    let text_box = "{T}: Draw two cards. Target opponent gains control of this creature. Activate only during your turn.";
    CardDefinition {
        key: "frf.humble_defector",
        scryfall_id: "4bb0c13f-6cea-48b4-a5c8-2c0cfcfea7fd",
        printed: PrintedCharacteristics {
            name: "Humble Defector".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                own_turn_only: true,
                effects: vec![
                    Effect::DrawCards {
                        count: EffectValue::Fixed(2),
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(TargetSpec::Opponent),
                        effects: vec![Effect::GainControl {
                            player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                            what: ObjectSelector::This,
                            duration: EffectDuration::Indefinite,
                        }],
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
