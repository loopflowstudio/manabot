//! Kaldheim.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerCondition},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        SacrificeCost, StaticAbility,
    },
    effect::{EffectDuration, Modification},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![koma_cosmos_serpent()]
}

fn koma_cosmos_serpent() -> CardDefinition {
    let text_box = "This spell can't be countered.\nAt the beginning of each upkeep, create a 3/3 blue Serpent creature token named Koma's Coil.\nSacrifice another Serpent: Choose one —\n• Tap target permanent. Its activated abilities can't be activated this turn.\n• Koma gains indestructible until end of turn.";
    let target = TargetSpec::Permanent {
        predicate: CardPredicate::default(),
    };
    // "Choose one" of an activated ability is chosen as it is activated
    // (CR 700.2a), so each mode is its own ability with the shared cost.
    let mode = |effects: Vec<Effect>| ActivatedAbilityDefinition {
        sacrifice: Some(SacrificeCost {
            predicate: CardPredicate::subtype("Serpent"),
            another: true,
            or_mana: None,
        }),
        effects,
        ..Default::default()
    };
    CardDefinition {
        key: "khm.koma_cosmos_serpent",
        scryfall_id: "2de25ea4-284a-4c16-b823-048ff00c6a03",
        name: "Koma, Cosmos Serpent".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}{G}{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        supertypes: vec!["Legendary".into()],
        subtypes: vec!["Serpent".into()],
        power: Some(6),
        toughness: Some(6),
        statics: vec![StaticAbility::CantBeCountered],
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::BeginningOfEachUpkeep,
            effects: vec![Effect::create_tokens("Koma's Coil", 1)],
        }],
        activated_abilities: vec![
            mode(vec![
                Effect::Tap {
                    what: ObjectSelector::Target(target.clone()),
                },
                Effect::ApplyContinuous {
                    affected: ObjectSelector::Target(target),
                    modification: Modification::CantActivateAbilities,
                    duration: EffectDuration::EndOfTurn,
                },
            ]),
            mode(vec![Effect::ApplyContinuous {
                affected: ObjectSelector::This,
                modification: Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
                duration: EffectDuration::EndOfTurn,
            }]),
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
