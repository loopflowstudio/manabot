//! Ixalan.
use managym_state::ability::{
    Ability, Condition, Effect, ObjectSelector, Scope, TargetSpec, TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, EntryChoiceKind, Keyword,
    Keywords, PrintedCharacteristics, ReplacementEffect, Restriction, StaticAbility,
    StaticRestriction, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        sorcerous_spyglass(),
        storm_fleet_spy(),
        pirate_s_cutlass(),
        dive_down(),
        rivers_rebuke(),
        bishops_soldier(),
    ]
}

fn storm_fleet_spy() -> CardDefinition {
    let text_box = "Raid — When this creature enters, if you attacked this turn, draw a card.";
    CardDefinition {
        key: "xln.storm_fleet_spy",
        scryfall_id: "f7c33ef4-60bb-4f95-92a5-7abedaac6767",
        printed: PrintedCharacteristics {
            name: "Storm Fleet Spy".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::DrawCards { count: 1.into() }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bishops_soldier() -> CardDefinition {
    let text_box =
        "Lifelink (Damage dealt by this creature also causes you to gain that much life.)";
    CardDefinition {
        key: "xln.bishops_soldier",
        scryfall_id: "d954677c-2de6-440a-90d0-bab2e0c8b4af",
        printed: PrintedCharacteristics {
            name: "Bishop's Soldier".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Soldier".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rivers_rebuke() -> CardDefinition {
    CardDefinition {
        key: "xln.rivers_rebuke",
        scryfall_id: "fda8ef30-bbfa-4857-9750-0dd0def8b13f",
        printed: PrintedCharacteristics {
            name: "River's Rebuke".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::return_to_hand(ObjectSelector::each(
                Scope::TargetPlayerControls,
                CardPredicate {
                    not_card_types: vec![CardType::Land],
                    ..Default::default()
                },
            ))],
            text_box: "Return all nonland permanents target player controls to their owner's hand."
                .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dive_down() -> CardDefinition {
    let text_box = "Target creature you control gets +0/+3 and gains hexproof until end of turn. (It can't be the target of spells or abilities your opponents control.)";
    CardDefinition {
        key: "xln.dive_down",
        scryfall_id: "b33e493e-1aef-43b3-9716-52158b002430",
        printed: PrintedCharacteristics {
            name: "Dive Down".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::Target(TargetSpec::CreatureYouControl),
                0,
                3,
                Keyword::Hexproof,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pirate_s_cutlass() -> CardDefinition {
    let text_box = "When this Equipment enters, attach it to target Pirate you control.\nEquipped creature gets +2/+1.\nEquip {2} ({2}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "xln.pirate_s_cutlass",
        scryfall_id: "b3e7e871-19cf-486d-bacb-1499fe066974",
        printed: PrintedCharacteristics {
            name: "Pirate's Cutlass".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::PermanentYouControl {
                        predicate: CardPredicate::subtype("Pirate"),
                    }),
                }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{2}"))],
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::pump(2, 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sorcerous_spyglass() -> CardDefinition {
    let text_box = "As this artifact enters, look at an opponent's hand, then choose any card name.\nActivated abilities of sources with the chosen name can't be activated unless they're mana abilities.";
    CardDefinition {
        key: "xln.sorcerous_spyglass",
        scryfall_id: "85506a24-8d60-475c-9f43-65994caca7d4",
        printed: PrintedCharacteristics {
            name: "Sorcerous Spyglass".into(),
            types: CardTypes::new([CardType::Artifact]),
            mana_cost: Some(ManaCost::parse("{2}")),
            replacement_effects: vec![ReplacementEffect::AsEntersChoose {
                kind: EntryChoiceKind::CardName,
                look_at_opponent_hand: true,
            }],
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::All(CardPredicate {
                    chosen_name: true,
                    ..Default::default()
                }),
                kind: Restriction::CantActivateNonmanaAbilities,
                unless: None,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
