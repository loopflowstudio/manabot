//! Return to Ravnica.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, PlayerSelector, TargetSpec, TriggerCondition,
        TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        ManaAbility, PrintedCharacteristics, ReplacementEffect,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        desecration_demon(),
        rogues_passage(),
        guttersnipe(),
        azorius_guildgate(),
        fencing_ace(),
        golgari_guildgate(),
        izzet_guildgate(),
        rakdos_guildgate(),
        selesnya_guildgate(),
    ]
}

fn desecration_demon() -> CardDefinition {
    let text_box = "Flying\nAt the beginning of each combat, any opponent may sacrifice a creature of their choice. If a player does, tap this creature and put a +1/+1 counter on it.";
    CardDefinition {
        key: "rtr.desecration_demon",
        scryfall_id: "8242fade-754c-4404-b3fb-f3cccf84b3b6",
        printed: PrintedCharacteristics {
            name: "Desecration Demon".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Demon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::Combat, Actor::AnyPlayer),
                intervening_if: None,
                effects: vec![
                    Effect::ForPlayer {
                        player: PlayerSelector::EachOpponent,
                        effects: vec![Effect::may_sacrifice(CardPredicate::creature(), false)],
                    },
                    Effect::if_you_do(vec![
                        Effect::Tap {
                            what: ObjectSelector::This,
                        },
                        Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 1),
                    ]),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
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

fn guttersnipe() -> CardDefinition {
    let text_box = "Whenever you cast an instant or sorcery spell, this creature deals 2 damage to each opponent.";
    CardDefinition {
        key: "rtr.guttersnipe",
        scryfall_id: "9d8590ea-512c-4e09-97cc-7f07d0706f2b",
        printed: PrintedCharacteristics {
            name: "Guttersnipe".into(),
            subtypes: vec!["Goblin".into(), "Shaman".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, CardPredicate::instant_or_sorcery()),
                intervening_if: None,
                effects: vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::EachOpponent,
                    amount: 2.into(),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rogues_passage() -> CardDefinition {
    CardDefinition {
        key: "rtr.rogues_passage",
        scryfall_id: "f416e36a-15cb-43c8-b27f-82f65a95ddef",
        printed: PrintedCharacteristics {
            name: "Rogue's Passage".into(),
            types: CardTypes::new([CardType::Land]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("C"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{4}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::CantBeBlocked,
                )],
                ..Default::default()
            }],
            text_box: "{T}: Add {C}.\n{4}, {T}: Target creature can't be blocked this turn.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
