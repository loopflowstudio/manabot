//! Limited Edition Alpha.
use super::native_creature;
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Op, PrintedPower, Program, ProgramKind, Selector, Target,
};
use crate::state::{
    ability::{Effect, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType as NativeCardType, CardTypes,
        Keywords, ManaAbility,
    },
    mana::{Color, Mana, ManaCost},
};

pub(super) fn cards() -> Vec<Card> {
    vec![
        plains(),
        island(),
        swamp(),
        mountain(),
        forest(),
        hill_giant(),
        lightning_bolt(),
        savannah_lions(),
    ]
}

/// Native definitions, until the typed card language has the operations they need.
pub(super) fn native_cards() -> Vec<CardDefinition> {
    vec![
        llanowar_elves(),
        gray_ogre(),
        ancestral_recall(),
        counterspell(),
        giant_spider(),
        serra_angel(),
        war_mammoth(),
        wall_of_stone(),
        craw_wurm(),
        shivan_dragon(),
    ]
}

fn hill_giant() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("3R"),
            subtypes: Some(vec!["Giant"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "lea.hill_giant",
        programs: vec![],
        registry_name: "Hill Giant",
        text_box,
    }
}

fn lightning_bolt() -> Card {
    let text_box = "Lightning Bolt deals 3 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "lea.lightning_bolt",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DealDamage {
                amount: 3,
                target: "victim",
            }],
        )
        .cost(Cost::mana("R"))
        .targets(vec![Target::one("victim", Selector::CreatureOrPlayer)])],
        registry_name: "Lightning Bolt",
        text_box,
    }
}

fn savannah_lions() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("W"),
            subtypes: Some(vec!["Cat"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "lea.savannah_lions",
        programs: vec![],
        registry_name: "Savannah Lions",
        text_box,
    }
}

fn plains() -> Card {
    let text_box = "({T}: Add {W}.)";

    basic_land("lea.plains", "Plains", text_box, "W")
}

fn swamp() -> Card {
    let text_box = "({T}: Add {B}.)";

    basic_land("lea.swamp", "Swamp", text_box, "B")
}

fn forest() -> Card {
    let text_box = "({T}: Add {G}.)";

    basic_land("lea.forest", "Forest", text_box, "G")
}

fn island() -> Card {
    let text_box = "({T}: Add {U}.)";

    basic_land("lea.island", "Island", text_box, "U")
}

fn mountain() -> Card {
    let text_box = "({T}: Add {R}.)";

    basic_land("lea.mountain", "Mountain", text_box, "R")
}

fn basic_land(
    key: &'static str,
    registry_name: &'static str,
    text_box: &'static str,
    mana: &'static str,
) -> Card {
    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![registry_name]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec!["basic"]),
            ..Default::default()
        },
        key,
        programs: vec![Program::new(
            "p0",
            ProgramKind::Mana,
            vec![Op::AddMana { mana, until: None }],
        )],
        registry_name,
        text_box,
    }
}

fn llanowar_elves() -> CardDefinition {
    CardDefinition {
        name: "Llanowar Elves".to_string(),
        mana_cost: Some(ManaCost::parse("G")),
        types: CardTypes::new([NativeCardType::Creature]),
        subtypes: vec!["Elf".to_string(), "Druid".to_string()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::single(Color::Green),
        }],
        text_box: "{T}: Add {G}.".to_string(),
        power: Some(1),
        toughness: Some(1),
        ..Default::default()
    }
}

fn gray_ogre() -> CardDefinition {
    native_creature("Gray Ogre", "2R", &["Ogre"], 2, 2, Keywords::default(), "")
}

fn ancestral_recall() -> CardDefinition {
    // DEVIATION (documented in wave/rules/01-two-deck-slice.md): oracle
    // is "Target player draws three cards." — the engine has no
    // target-player draw, so the caster always draws (the self-target
    // case; the opponent-draw line is not available).
    CardDefinition {
        name: "Ancestral Recall".to_string(),
        mana_cost: Some(ManaCost::parse("U")),
        types: CardTypes::new([NativeCardType::Instant]),
        spell_effects: vec![Effect::DrawCards { count: 3 }],
        text_box: "Target player draws three cards.".to_string(),
        ..Default::default()
    }
}

fn counterspell() -> CardDefinition {
    CardDefinition {
        name: "Counterspell".to_string(),
        mana_cost: Some(ManaCost::parse("UU")),
        types: CardTypes::new([NativeCardType::Instant]),
        spell_effects: vec![Effect::CounterSpell {
            target: TargetSpec::Spell,
        }],
        text_box: "Counter target spell.".to_string(),
        ..Default::default()
    }
}

fn giant_spider() -> CardDefinition {
    native_creature(
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
    )
}

fn serra_angel() -> CardDefinition {
    native_creature(
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
    )
}

fn war_mammoth() -> CardDefinition {
    native_creature(
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
    )
}

fn wall_of_stone() -> CardDefinition {
    native_creature(
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
    )
}

fn craw_wurm() -> CardDefinition {
    native_creature("Craw Wurm", "4GG", &["Wurm"], 6, 4, Keywords::default(), "")
}

fn shivan_dragon() -> CardDefinition {
    CardDefinition {
        name: "Shivan Dragon".to_string(),
        mana_cost: Some(ManaCost::parse("4RR")),
        types: CardTypes::new([NativeCardType::Creature]),
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
    }
}
