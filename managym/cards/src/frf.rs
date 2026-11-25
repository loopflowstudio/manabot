//! Fate Reforged.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, PlayerSelector, TargetRequirement, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![valorous_stance(), humble_defector()]
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

fn valorous_stance() -> CardDefinition {
    let text_box = "Choose one —\n• Target creature gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)\n• Destroy target creature with toughness 4 or greater.";
    CardDefinition {
        key: "frf.valorous_stance",
        scryfall_id: "65998e94-15a0-41f1-8288-730b957f81df",
        printed: PrintedCharacteristics {
            name: "Valorous Stance".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::Creature),
                        Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                    )],
                    vec![Effect::destroy(ObjectSelector::Target(
                        TargetSpec::Permanent {
                            predicate: CardPredicate {
                                min_toughness: Some(4),
                                ..CardPredicate::creature()
                            },
                        },
                    ))],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
