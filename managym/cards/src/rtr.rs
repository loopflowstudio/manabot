//! Return to Ravnica.
use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, ReplacementEffect,
    },
    mana::{Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        azorius_guildgate(),
        fencing_ace(),
        golgari_guildgate(),
        izzet_guildgate(),
        rakdos_guildgate(),
        selesnya_guildgate(),
    ]
}

/// A Gate of the ten-card cycle: enters tapped and taps for either of its
/// guild's two colors. The Gatecrash five are filed in `gtc.rs`.
pub(super) fn guildgate(
    key: &'static str,
    scryfall_id: &'static str,
    name: &str,
    colors: [&str; 2],
) -> CardDefinition {
    let text_box = format!(
        "This land enters tapped.\n{{T}}: Add {{{}}} or {{{}}}.",
        colors[0], colors[1]
    );
    CardDefinition {
        key,
        scryfall_id,
        printed: PrintedCharacteristics {
            name: name.into(),
            types: CardTypes::new([CardType::Land]),
            subtypes: vec!["Gate".into()],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(colors.map(Mana::parse))],
                ..Default::default()
            }],
            replacement_effects: vec![ReplacementEffect::EntersTapped],
            text_box,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fencing_ace() -> CardDefinition {
    let text_box =
        "Double strike (This creature deals both first-strike and regular combat damage.)";

    CardDefinition {
        key: "rtr.fencing_ace",
        scryfall_id: "a42d3066-f4ec-4d28-83ab-e48141206c72",
        printed: PrintedCharacteristics {
            name: "Fencing Ace".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::DoubleStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn azorius_guildgate() -> CardDefinition {
    guildgate(
        "rtr.azorius_guildgate",
        "984e37df-0734-493a-a958-f519a0c98580",
        "Azorius Guildgate",
        ["W", "U"],
    )
}

fn golgari_guildgate() -> CardDefinition {
    guildgate(
        "rtr.golgari_guildgate",
        "8fe2fd1a-f7d3-48b4-bad8-be5ee45d6121",
        "Golgari Guildgate",
        ["B", "G"],
    )
}

fn izzet_guildgate() -> CardDefinition {
    guildgate(
        "rtr.izzet_guildgate",
        "6951d84f-2d3c-4203-8d31-e08f4bc707f0",
        "Izzet Guildgate",
        ["U", "R"],
    )
}

fn rakdos_guildgate() -> CardDefinition {
    guildgate(
        "rtr.rakdos_guildgate",
        "207048f5-268b-4cdb-b4e7-c8282cac1b28",
        "Rakdos Guildgate",
        ["B", "R"],
    )
}

fn selesnya_guildgate() -> CardDefinition {
    guildgate(
        "rtr.selesnya_guildgate",
        "ff61d4e4-3c8c-48f7-a994-ec2317bbd9a0",
        "Selesnya Guildgate",
        ["G", "W"],
    )
}
