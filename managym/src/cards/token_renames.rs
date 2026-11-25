//! Historical learning names no longer emitted by the engine. A missing
//! successor means the old meaning was removed or split across several tokens.
//! Stored programs can still read the learned vector of a retired name.

use super::tokens::TokenKind::{self, Field, Structure, Text, Variant};

pub const TOKENS: &[(TokenKind, &str, Option<&str>)] = &[
    (Variant, "AttackSize::WithAnother", None),
    (
        Variant,
        "ReplacementEffect::PreventDamageToController",
        None,
    ),
    (
        Structure,
        "/ReplacementEffect::PreventDamageToController",
        None,
    ),
    (Variant, "ReplacementEffect::DoubleDamageToController", None),
    (
        Structure,
        "/ReplacementEffect::DoubleDamageToController",
        None,
    ),
    (Structure, "/Effect::SearchLandsOpponentSplits", None),
    (Structure, "/ManaUse", None),
    (Structure, "/TriggerCondition::AttacksAlone", None),
    (
        Structure,
        "/TriggerCondition::CreatureDealsCombatDamageToPlayer",
        None,
    ),
    (
        Structure,
        "/TriggerCondition::OneOrMoreDealCombatDamageToPlayer",
        None,
    ),
    (
        Structure,
        "/TriggerCondition::OpponentDrawsNthCardThisTurn",
        None,
    ),
    (Structure, "/TriggerCondition::PlayerCastsSpell", None),
    (
        Structure,
        "/TriggerCondition::YouCastNthSpellThisTurn",
        None,
    ),
    (Structure, "/TriggerCondition::YouCastSpell", None),
    (Structure, "/TriggerCondition::YouDrawNthCardThisTurn", None),
    (
        Structure,
        "/TriggerCondition::YouPutCountersOnCreature",
        None,
    ),
    (Structure, "/TriggerSubject::Another", None),
    (Structure, "/TriggerSubject::AnotherYouControl", None),
    (Structure, "/TriggerSubject::AnyYouControl", None),
    (Structure, "ManaUse", None),
    (Variant, "Effect::WinWithDistinctGates", None),
    (Variant, "Effect::SearchLandsOpponentSplits", None),
    (Variant, "TriggerCondition::AttacksAlone", None),
    (Variant, "TriggerCondition::AttacksWithAnother", None),
    (Variant, "TriggerCondition::BeginningOfEachUpkeep", None),
    (Variant, "TriggerCondition::BeginningOfYourCombat", None),
    (Variant, "TriggerCondition::BeginningOfYourEndStep", None),
    (
        Variant,
        "TriggerCondition::BeginningOfYourFirstMainPhase",
        None,
    ),
    (Variant, "TriggerCondition::BeginningOfYourUpkeep", None),
    (
        Variant,
        "TriggerCondition::CreatureDealsCombatDamageToPlayer",
        None,
    ),
    (
        Variant,
        "TriggerCondition::OneOrMoreDealCombatDamageToPlayer",
        None,
    ),
    (
        Variant,
        "TriggerCondition::OpponentDrawsNthCardThisTurn",
        None,
    ),
    (Variant, "TriggerCondition::OpponentSearchesLibrary", None),
    (Variant, "TriggerCondition::PlayerCastsSpell", None),
    (Variant, "TriggerCondition::YouBend", None),
    (Variant, "TriggerCondition::YouCastNthSpellThisTurn", None),
    (Variant, "TriggerCondition::YouCastSpell", None),
    (Variant, "TriggerCondition::YouDrawNthCardThisTurn", None),
    (Variant, "TriggerCondition::YouPutCountersOnCreature", None),
    (Variant, "TriggerCondition::YouScryOrSurveil", None),
    (Variant, "TriggerSubject::Another", None),
    (Variant, "TriggerSubject::AnotherYouControl", None),
    (Variant, "TriggerSubject::AnyYouControl", None),
    // Auras and Equipment share one attachment subject.
    (
        Variant,
        "TriggerSubject::EnchantedCreature",
        Some("TriggerSubject::Attached"),
    ),
    (Field, "abilities", None),
    (Field, "controller_only", None),
    (Field, "opponent_picks", Some("picks")),
    (Field, "restricted", None),
    (Field, "spells", None),
    (
        Variant,
        "Effect::EachPlayerSacrificesCreatures",
        Some("Effect::EachPlayerSacrifices"),
    ),
    (Text, "creatures each player sacrifices", None),
    (Text, "kept creature", None),
    (Variant, "Effect::FightLastTarget", Some("Effect::Fight")),
    (
        Structure,
        "/Effect::EachPlayerSacrificesCreatures",
        Some("/Effect::EachPlayerSacrifices"),
    ),
    (Text, "attached at activation", None),
    (Text, "land choices remaining", None),
    (Text, "selected lands", None),
    // Split into `contribution` facts naming how and toward what each paid.
    (Text, "waterbend contributions", None),
];

/// The search now describes arbitrary distinct names and a structured ending;
/// a payment's contributions are `contribution` facts.
pub const RECORDS: &[(&str, &str, Option<&str>)] = &[
    (
        "token",
        "execution/6/kept creature",
        Some("execution/6/kept permanent"),
    ),
    (
        "token",
        "execution/6/creatures each player sacrifices",
        Some("execution/6/permanents each player sacrifices"),
    ),
    (
        "token",
        "execution/1//ReplacementEffect::DoubleDamageToController",
        None,
    ),
    (
        "token",
        "execution/3/ReplacementEffect::DoubleDamageToController",
        None,
    ),
    (
        "token",
        "execution/1//ReplacementEffect::PreventDamageToController",
        None,
    ),
    (
        "token",
        "execution/3/ReplacementEffect::PreventDamageToController",
        None,
    ),
    ("token", "execution/6/attached at activation", None),
    ("token", "execution/6/land choices remaining", None),
    ("token", "execution/6/selected lands", None),
    ("token", "execution/6/waterbend contributions", None),
];

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::cards::{
        content_pack::default_content_pack,
        tokens::{catalog_tokens, name_value},
    };

    #[test]
    fn each_historical_token_has_one_disposition_and_renames_resolve() {
        let catalog = catalog_tokens(default_content_pack()).unwrap();
        let current: HashSet<_> = catalog
            .token_kind
            .into_iter()
            .zip(catalog.token_value)
            .collect();
        let mut classified = HashSet::new();
        for &(kind, old, successor) in TOKENS {
            assert!(classified.insert((kind as u16, name_value(old))), "{old}");
            if let Some(new) = successor {
                assert!(
                    current.contains(&(kind as u16, name_value(new))),
                    "{old} -> {new}"
                );
            }
        }
    }
}
