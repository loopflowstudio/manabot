//! Command-zone abilities granted by planeswalker ultimates (CR 114).
use managym_state::{
    ability::{Ability, Actor, Effect, ObjectSelector, TargetSpec, TriggerCondition},
    card::{
        EmblemDefinition, Keyword, Keywords, PrintedCharacteristics, StaticAbility,
        StaticContinuous, StaticScope,
    },
    effect::Modification,
    predicate::CardPredicate,
};
use std::sync::OnceLock;

pub fn emblem_catalog() -> &'static [EmblemDefinition] {
    static EMBLEMS: OnceLock<Vec<EmblemDefinition>> = OnceLock::new();
    EMBLEMS.get_or_init(|| {
        vec![
            anthem(
                "emblem.ajani_resolute",
                "Emblem — Ajani Resolute",
                "Creatures you control get +2/+2.",
                &[],
            ),
            cast_trigger(
                "emblem.chandra_torch_of_defiance",
                "Emblem — Chandra, Torch of Defiance",
                "Whenever you cast a spell, this emblem deals 5 damage to any target.",
                Actor::You,
                Effect::deal_damage(5, ObjectSelector::Target(TargetSpec::AnyTarget)),
            ),
            cast_trigger(
                "emblem.kaito_cunning_infiltrator",
                "Emblem — Kaito, Cunning Infiltrator",
                "Whenever a player casts a spell, you create a 2/1 blue Ninja creature token.",
                Actor::AnyPlayer,
                Effect::create_tokens("token.ninja_2_1_u", 1),
            ),
            anthem(
                "emblem.vivien_reid",
                "Emblem — Vivien Reid",
                "Creatures you control get +2/+2 and have vigilance, trample, and indestructible.",
                &[
                    Keyword::Vigilance,
                    Keyword::Trample,
                    Keyword::Indestructible,
                ],
            ),
        ]
    })
}

fn anthem(
    key: &'static str,
    label: &'static str,
    text: &str,
    keywords: &[Keyword],
) -> EmblemDefinition {
    let mut modifications = vec![Modification::ModifyPt {
        power: 2.into(),
        toughness: 2.into(),
    }];
    if !keywords.is_empty() {
        modifications.push(Modification::AddKeywords(Keywords::of(keywords)));
    }
    EmblemDefinition {
        key,
        label,
        printed: PrintedCharacteristics {
            text_box: text.into(),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::YouControl(CardPredicate::creature()),
                condition: None,
                characteristic_defining: false,
                modifications,
            })],
            ..Default::default()
        },
    }
}

fn cast_trigger(
    key: &'static str,
    label: &'static str,
    text: &str,
    who: Actor,
    effect: Effect,
) -> EmblemDefinition {
    EmblemDefinition {
        key,
        label,
        printed: PrintedCharacteristics {
            text_box: text.into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(who, CardPredicate::default()),
                intervening_if: None,
                effects: vec![effect],
            }],
            ..Default::default()
        },
    }
}
