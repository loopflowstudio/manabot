//! Khans of Tarkir.
use managym_state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, ReplacementEffect,
    },
    mana::{Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        bloodfell_caves(),
        blossoming_sands(),
        dismal_backwater(),
        jungle_hollow(),
        monastery_swiftspear(),
        rugged_highlands(),
        scoured_barrens(),
        swiftwater_cliffs(),
        thornwood_falls(),
        tranquil_cove(),
        wind_scarred_crag(),
    ]
}

/// A land of the ten-card cycle: enters tapped, gains 1 life on entering and
/// taps for either of two colors.
fn gain_land(
    key: &'static str,
    scryfall_id: &'static str,
    name: &str,
    colors: [&str; 2],
) -> CardDefinition {
    let text_box = format!(
        "This land enters tapped.\nWhen this land enters, you gain 1 life.\n{{T}}: Add {{{}}} or {{{}}}.",
        colors[0], colors[1]
    );
    CardDefinition {
        key,
        scryfall_id,
        printed: PrintedCharacteristics {
            name: name.into(),
            types: CardTypes::new([CardType::Land]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::gain_life(1)],
            }],
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

fn bloodfell_caves() -> CardDefinition {
    gain_land(
        "ktk.bloodfell_caves",
        "15a7b30a-c59f-4a87-9e8a-b29daea27422",
        "Bloodfell Caves",
        ["B", "R"],
    )
}

fn blossoming_sands() -> CardDefinition {
    gain_land(
        "ktk.blossoming_sands",
        "a32a1c0b-f6ea-475a-aa01-3618ea7d8647",
        "Blossoming Sands",
        ["G", "W"],
    )
}

fn dismal_backwater() -> CardDefinition {
    gain_land(
        "ktk.dismal_backwater",
        "63742780-47ee-4a66-993a-69e06c14967d",
        "Dismal Backwater",
        ["U", "B"],
    )
}

fn jungle_hollow() -> CardDefinition {
    gain_land(
        "ktk.jungle_hollow",
        "fea27aa7-7fcf-4198-b03a-5034a03ba81f",
        "Jungle Hollow",
        ["B", "G"],
    )
}

fn monastery_swiftspear() -> CardDefinition {
    let text_box = "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    CardDefinition {
        key: "ktk.monastery_swiftspear",
        scryfall_id: "b81c6c8b-a9cf-4866-89ba-7f8ad077b836",
        printed: PrintedCharacteristics {
            name: "Monastery Swiftspear".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Monk".into()],
            power: Some(1),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Haste]),
            abilities: vec![Ability::prowess()],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rugged_highlands() -> CardDefinition {
    gain_land(
        "ktk.rugged_highlands",
        "501ce6cb-0324-4cca-bc79-903cefe1ac1f",
        "Rugged Highlands",
        ["R", "G"],
    )
}

fn scoured_barrens() -> CardDefinition {
    gain_land(
        "ktk.scoured_barrens",
        "0824a960-dd89-45c5-90f0-3ec9eb47d9ce",
        "Scoured Barrens",
        ["W", "B"],
    )
}

fn swiftwater_cliffs() -> CardDefinition {
    gain_land(
        "ktk.swiftwater_cliffs",
        "e782d005-a563-4738-978a-73a3465de78f",
        "Swiftwater Cliffs",
        ["U", "R"],
    )
}

fn thornwood_falls() -> CardDefinition {
    gain_land(
        "ktk.thornwood_falls",
        "9e57abd9-e864-4047-a3c8-618952071858",
        "Thornwood Falls",
        ["G", "U"],
    )
}

fn tranquil_cove() -> CardDefinition {
    gain_land(
        "ktk.tranquil_cove",
        "0f840bd2-c4f5-4ac4-918c-91b4feeb8783",
        "Tranquil Cove",
        ["W", "U"],
    )
}

fn wind_scarred_crag() -> CardDefinition {
    gain_land(
        "ktk.wind_scarred_crag",
        "3b296781-78ac-411f-88fc-2d924ad22986",
        "Wind-Scarred Crag",
        ["R", "W"],
    )
}
