//! Hand-registered native cards that have no typed set definition yet.
//!
//! Every other card is authored once in `cardsets/` and lowered by
//! `semantic::compile`. These remain native because the typed card language
//! lacks an operation they need (target-player draw, damage to each creature,
//! ward, defender, conditional combat triggers, a modal test fixture) or because
//! no one has re-authored them yet. Prefer moving a card into its set file over
//! extending this list.
//!
//! [`legacy_slots`] fixes the registration order of the legacy prefix so the
//! default pack keeps the `CardDefId`s that older evidence records; typed
//! definitions fill their historical slots by name.

use crate::state::{
    ability::{
        Ability, Effect, EffectValue, StaticCondition, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, ManaAbility,
    },
    mana::{Color, Mana, ManaCost},
};

/// One position in the default pack's registration order.
#[allow(clippy::large_enum_variant)] // Built once per default pack.
pub(super) enum Slot {
    Native(CardDefinition),
    /// A typed definition, by printed name, registered at this position.
    Typed(&'static str),
}

use Slot::{Native, Typed};

/// The legacy prefix of the default pack, in its historical order.
pub(super) fn legacy_slots() -> Vec<Slot> {
    let mut slots = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .into_iter()
        .map(Typed)
        .collect::<Vec<_>>();

    slots.push(Native(CardDefinition {
        name: "Llanowar Elves".to_string(),
        mana_cost: Some(ManaCost::parse("G")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elf".to_string(), "Druid".to_string()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::single(Color::Green),
        }],
        text_box: "{T}: Add {G}.".to_string(),
        power: Some(1),
        toughness: Some(1),
        ..Default::default()
    }));

    slots.push(Native(creature(
        "Gray Ogre",
        "2R",
        &["Ogre"],
        2,
        2,
        Keywords::default(),
        "",
    )));
    slots.push(Typed("Lightning Bolt"));

    // DEVIATION (documented in wave/rules/01-two-deck-slice.md): oracle
    // is "Target player draws three cards." — the engine has no
    // target-player draw, so the caster always draws (the self-target
    // case; the opponent-draw line is not available).
    slots.push(Native(CardDefinition {
        name: "Ancestral Recall".to_string(),
        mana_cost: Some(ManaCost::parse("U")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::DrawCards { count: 3 }],
        text_box: "Target player draws three cards.".to_string(),
        ..Default::default()
    }));

    slots.push(Native(CardDefinition {
        name: "Counterspell".to_string(),
        mana_cost: Some(ManaCost::parse("UU")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::CounterSpell {
            target: TargetSpec::Spell,
        }],
        text_box: "Counter target spell.".to_string(),
        ..Default::default()
    }));

    slots.push(Native(creature(
        "Wind Drake",
        "2U",
        &["Drake"],
        2,
        2,
        Keywords {
            flying: true,
            ..Default::default()
        },
        "Flying",
    )));
    slots.push(Native(creature(
        "Giant Spider",
        "3G",
        &["Spider"],
        2,
        4,
        Keywords {
            reach: true,
            ..Default::default()
        },
        "Reach (This creature can block creatures with flying.)",
    )));
    slots.push(Native(creature(
        "Raging Goblin",
        "R",
        &["Goblin", "Berserker"],
        1,
        1,
        Keywords {
            haste: true,
            ..Default::default()
        },
        "Haste (This creature can attack and {T} as soon as it comes under your control.)",
    )));
    slots.push(Native(creature(
        "Serra Angel",
        "3WW",
        &["Angel"],
        4,
        4,
        Keywords {
            flying: true,
            vigilance: true,
            ..Default::default()
        },
        "Flying\nVigilance (Attacking doesn't cause this creature to tap.)",
    )));
    slots.push(Native(creature(
        "Typhoid Rats",
        "B",
        &["Rat"],
        1,
        1,
        Keywords {
            deathtouch: true,
            ..Default::default()
        },
        "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)",
    )));
    slots.push(Native(creature(
        "War Mammoth",
        "3G",
        &["Elephant"],
        3,
        3,
        Keywords {
            trample: true,
            ..Default::default()
        },
        "Trample",
    )));
    slots.push(Native(creature(
        "Wall of Stone",
        "1RR",
        &["Wall"],
        0,
        8,
        Keywords {
            defender: true,
            ..Default::default()
        },
        "Defender (This creature can't attack.)",
    )));
    slots.push(Native(creature(
        "Boggart Brute",
        "2R",
        &["Goblin", "Warrior"],
        3,
        2,
        Keywords {
            menace: true,
            ..Default::default()
        },
        "Menace (This creature can't be blocked except by two or more creatures.)",
    )));
    slots.push(Native(creature(
        "Youthful Knight",
        "1W",
        &["Human", "Knight"],
        2,
        1,
        Keywords {
            first_strike: true,
            ..Default::default()
        },
        "First strike",
    )));
    slots.push(Native(creature(
        "Fencing Ace",
        "1W",
        &["Human", "Soldier"],
        1,
        1,
        Keywords {
            double_strike: true,
            ..Default::default()
        },
        "Double strike (This creature deals both first-strike and regular combat damage.)",
    )));
    slots.push(Native(creature(
        "Healer's Hawk",
        "W",
        &["Bird"],
        1,
        1,
        Keywords {
            flying: true,
            lifelink: true,
            ..Default::default()
        },
        "Flying\nLifelink (Damage dealt by this creature also causes you to gain that much life.)",
    )));
    slots.push(Native(creature(
        "Craw Wurm",
        "4GG",
        &["Wurm"],
        6,
        4,
        Keywords::default(),
        "",
    )));
    slots.push(Native(CardDefinition {
        name: "Shivan Dragon".to_string(),
        mana_cost: Some(ManaCost::parse("4RR")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dragon".to_string()],
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: false,
            mana_cost: ManaCost::parse("R"),
            sacrifice_source: false,
            waterbend: false,
            effect: Effect::ModifyUntilEot {
                power_delta: 1,
                toughness_delta: 0,
            },
            ..Default::default()
        }],
        text_box: "Flying\n{R}: This creature gets +1/+0 until end of turn.".to_string(),
        power: Some(5),
        toughness: Some(5),
        ..Default::default()
    }));

    slots.push(Native(CardDefinition {
        name: "Pyroclasm".to_string(),
        mana_cost: Some(ManaCost::parse("1R")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::MassDamage { amount: 2 }],
        text_box: "Pyroclasm deals 2 damage to each creature.".to_string(),
        ..Default::default()
    }));

    slots.push(Native(CardDefinition {
        name: "Man-o'-War".to_string(),
        mana_cost: Some(ManaCost::parse("2U")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Jellyfish".to_string()],
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::ReturnToHand {
                target: TargetSpec::Creature,
            }],
        }],
        text_box: "When this creature enters, return target creature to its owner's hand."
            .to_string(),
        power: Some(2),
        toughness: Some(2),
        ..Default::default()
    }));

    slots.extend(["Pop Quiz", "Igneous Inspiration", "Divide by Zero"].map(Typed));

    slots.push(Native(CardDefinition {
            name: "Waterfall Aerialist".to_string(),
            mana_cost: Some(ManaCost::parse("3U")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Djinn".into(), "Wizard".into()],
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            ward: Some(ManaCost::parse("2")),
            text_box: "Flying\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)".to_string(),
            power: Some(3),
            toughness: Some(1),
            ..Default::default()
        }));

    slots
}

/// Native cards registered after every typed definition.
pub(super) fn trailing_slots() -> Vec<Slot> {
    let mut slots = Vec::new();

    slots.push(Native(CardDefinition {
        name: "Crossroads of Destiny".to_string(),
        mana_cost: Some(ManaCost::parse("1")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::Modal {
            modes: vec![
                vec![Effect::GainLife { amount: 3 }],
                vec![Effect::DrawCards { count: 1 }],
            ],
        }],
        text_box: "Choose one —\n• You gain 3 life.\n• Draw a card.".to_string(),
        ..Default::default()
    }));

    let drawn_twice = StaticCondition::CardsDrawnAtLeast { count: 2 };
    slots.push(Native(CardDefinition {
            name: "Proft's Eidetic Memory".into(),
            mana_cost: Some(ManaCost::parse("1U")),
            types: CardTypes::new([CardType::Enchantment]),
            supertypes: vec!["Legendary".into()],
            no_maximum_hand_size: true,
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield { subject: TriggerSubject::This },
                    effects: vec![Effect::DrawCards { count: 1 }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::ActiveIf {
                        active_if: drawn_twice.clone(),
                        condition: Box::new(TriggerCondition::BeginningOfYourCombat),
                    },
                    effects: vec![Effect::IfCondition {
                        condition: drawn_twice,
                        effects: vec![Effect::PutCountersValue {
                            count: EffectValue::CardsDrawnMinus { subtract: 1 },
                            target: TargetSpec::CreatureYouControl,
                        }],
                    }],
                },
            ],
            text_box: "When Proft's Eidetic Memory enters, draw a card.\nYou have no maximum hand size.\nAt the beginning of combat on your turn, if you've drawn more than one card this turn, put X +1/+1 counters on target creature you control, where X is the number of cards you've drawn this turn minus one.".into(),
            ..Default::default()
        }));

    slots
}

#[allow(clippy::too_many_arguments)]
fn creature(
    name: &str,
    mana_cost: &str,
    subtypes: &[&str],
    power: i32,
    toughness: i32,
    keywords: Keywords,
    text_box: &str,
) -> CardDefinition {
    CardDefinition {
        name: name.to_string(),
        mana_cost: Some(ManaCost::parse(mana_cost)),
        types: CardTypes::new([CardType::Creature]),
        subtypes: subtypes
            .iter()
            .map(|subtype| (*subtype).to_string())
            .collect(),
        keywords,
        text_box: text_box.to_string(),
        power: Some(power),
        toughness: Some(toughness),
        ..Default::default()
    }
}
