//! The Brothers' War.
use managym_state::{
    ability::{
        Ability, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue,
        ObjectSelector, Per, PlayerSelector, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, ActivationZone, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, StaticAbility,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![cityscape_leveler(), giant_cindermaw()]
}

/// Unearth `cost` (CR 702.84a): "Return this card from your graveyard to the
/// battlefield. It gains haste. Exile it at the beginning of the next end step
/// or if it would leave the battlefield. Unearth only as a sorcery." The
/// returned object is `Moved`; its haste and exile replacement are records on
/// that exact incarnation, and the delayed trigger remembers it.
fn unearth(cost: &str) -> ActivatedAbilityDefinition {
    let returned = |modification| Effect::ApplyContinuous {
        affected: ObjectSelector::Moved,
        modification,
        duration: EffectDuration::Indefinite,
    };
    ActivatedAbilityDefinition {
        zone: ActivationZone::Graveyard,
        sorcery_timing: true,
        mana_cost: ManaCost::parse(cost),
        effects: vec![
            Effect::move_to(ObjectSelector::This, ZoneType::Battlefield),
            returned(Modification::AddKeywords(Keywords::of(&[Keyword::Haste]))),
            returned(Modification::ExileIfWouldLeave),
            Effect::CreateDelayedTrigger {
                remember: Some(ObjectSelector::Moved),
                event: DelayedEvent::EndStep { your_turn: false },
                frequency: DelayedFrequency::Once,
                expiry: DelayedExpiry::Never,
                effects: vec![Effect::exile(ObjectSelector::Remembered)],
            },
        ],
        ..Default::default()
    }
}

fn cityscape_leveler() -> CardDefinition {
    let text_box = "Trample\nWhen you cast this spell and whenever this creature attacks, destroy up to one target nonland permanent. Its controller creates a tapped Powerstone token.\nUnearth {8}";
    let nonland = TargetSpec::Permanent {
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land],
            ..Default::default()
        },
    };
    // The token's creator is the target's controller as it last existed, so
    // an indestructible or already-destroyed permanent's controller still
    // gets one.
    let level = || {
        vec![Effect::TargetClause {
            requirement: TargetRequirement::up_to(1, nonland.clone()),
            effects: vec![
                Effect::destroy(ObjectSelector::Target(nonland.clone())),
                Effect::ForPlayer {
                    player: PlayerSelector::TargetController,
                    effects: vec![Effect::CreateToken {
                        token_name: "Powerstone".into(),
                        count: EffectValue::Fixed(1),
                        tapped_and_attacking: false,
                        tapped: true,
                    }],
                },
            ],
        }]
    };
    CardDefinition {
        key: "bro.cityscape_leveler",
        scryfall_id: "67a87278-4c82-4056-8354-253d86b0ef3d",
        name: "Cityscape Leveler".into(),
        mana_cost: Some(ManaCost::parse("{8}")),
        types: CardTypes::new([CardType::Artifact, CardType::Creature]),
        subtypes: vec!["Construct".into()],
        power: Some(8),
        toughness: Some(8),
        keywords: Keywords::of(&[Keyword::Trample]),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::YouCastThis,
                effects: level(),
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: level(),
            },
        ],
        activated_abilities: vec![unearth("{8}")],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn giant_cindermaw() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nPlayers can't gain life.";

    CardDefinition {
        key: "bro.giant_cindermaw",
        scryfall_id: "1349465f-d29f-4d4b-a653-f4388574c336",
        name: "Giant Cindermaw".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dinosaur".into(), "Beast".into()],
        power: Some(4),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Trample]),
        statics: vec![StaticAbility::PlayersCantGainLife],
        text_box: text_box.into(),
        ..Default::default()
    }
}
