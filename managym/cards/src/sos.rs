//! Secrets of Strixhaven.
use super::enters_prepared;
use managym_state::{
    ability::{
        Ability, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue,
        ObjectSelector, PaymentCost, PlayerSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        adventurous_eater(),
        vastlands_scavenger(),
        pigment_wrangler(),
        jadzi_steward_of_fate(),
        elite_interceptor(),
        blazing_firesinger(),
        campus_composer(),
        cheerful_osteomancer(),
        goblin_glasswright(),
        honorbound_page(),
        infirmary_healer(),
        landscape_painter(),
        maelstrom_artisan(),
        quill_blade_laureate(),
        skycoach_conductor(),
        spellbook_seeker(),
        strife_scholar(),
        studious_first_year(),
    ]
}

fn adventurous_eater() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.adventurous_eater",
        scryfall_id: "d40cc7da-c731-418e-8547-7033d1939450",
        printed: PrintedCharacteristics {
            name: "Adventurous Eater".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            text_box: text.into(),
            subtypes: vec!["Human".into(), "Warlock".into()],
            power: Some(3),
            toughness: Some(2),
            ..enters_prepared(have_a_bite())
        },
        ..Default::default()
    }
}

fn blazing_firesinger() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.blazing_firesinger",
        scryfall_id: "3ba971e7-0b7a-4750-896f-7cf063e66b2a",
        printed: PrintedCharacteristics {
            name: "Blazing Firesinger".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            text_box: text.into(),
            subtypes: vec!["Dwarf".into(), "Bard".into()],
            power: Some(2),
            toughness: Some(3),
            ..enters_prepared(seething_song())
        },
        ..Default::default()
    }
}

fn campus_composer() -> CardDefinition {
    let text = "Ward {2}\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.campus_composer",
        scryfall_id: "fac8ac39-ecb4-4142-bf37-131c65660a9b",
        printed: PrintedCharacteristics {
            name: "Campus Composer".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            text_box: text.into(),
            subtypes: vec!["Merfolk".into(), "Bard".into()],
            power: Some(3),
            toughness: Some(4),
            statics: vec![StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse(
                "{2}",
            )))],
            ..enters_prepared(aqueous_aria())
        },
        ..Default::default()
    }
}

fn cheerful_osteomancer() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.cheerful_osteomancer",
        scryfall_id: "3c34660c-25e3-4ff5-9b2b-5554ded2bcc3",
        printed: PrintedCharacteristics {
            name: "Cheerful Osteomancer".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            text_box: text.into(),
            subtypes: vec!["Orc".into(), "Warlock".into()],
            power: Some(4),
            toughness: Some(2),
            ..enters_prepared(raise_dead())
        },
        ..Default::default()
    }
}

fn goblin_glasswright() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.goblin_glasswright",
        scryfall_id: "c85c5f06-dd31-4e2c-97be-2f64d65069ea",
        printed: PrintedCharacteristics {
            name: "Goblin Glasswright".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            text_box: text.into(),
            subtypes: vec!["Goblin".into(), "Sorcerer".into()],
            power: Some(2),
            toughness: Some(2),
            ..enters_prepared(craft_with_pride())
        },
        ..Default::default()
    }
}

fn honorbound_page() -> CardDefinition {
    let text = "First strike\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.honorbound_page",
        scryfall_id: "79a70863-860f-4a7b-9cb2-d3546b689d44",
        printed: PrintedCharacteristics {
            name: "Honorbound Page".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            text_box: text.into(),
            subtypes: vec!["Cat".into(), "Cleric".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            ..enters_prepared(forum_s_favor())
        },
        ..Default::default()
    }
}

fn infirmary_healer() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.infirmary_healer",
        scryfall_id: "911442e3-3003-4683-a766-e791e9553667",
        printed: PrintedCharacteristics {
            name: "Infirmary Healer".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            text_box: text.into(),
            subtypes: vec!["Cat".into(), "Cleric".into()],
            power: Some(2),
            toughness: Some(3),
            ..enters_prepared(stream_of_life())
        },
        ..Default::default()
    }
}

fn landscape_painter() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.landscape_painter",
        scryfall_id: "c0bd30c4-3cdf-4eda-8be5-0fb5e5ddddbf",
        printed: PrintedCharacteristics {
            name: "Landscape Painter".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            text_box: text.into(),
            subtypes: vec!["Merfolk".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            ..enters_prepared(vibrant_idea())
        },
        ..Default::default()
    }
}

fn maelstrom_artisan() -> CardDefinition {
    let text = "Haste\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.maelstrom_artisan",
        scryfall_id: "5c88391d-271f-4021-a5d9-158ebc1e6357",
        printed: PrintedCharacteristics {
            name: "Maelstrom Artisan".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
            text_box: text.into(),
            subtypes: vec!["Minotaur".into(), "Sorcerer".into()],
            power: Some(3),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Haste]),
            ..enters_prepared(rocket_volley())
        },
        ..Default::default()
    }
}

fn quill_blade_laureate() -> CardDefinition {
    let text = "Double strike\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.quill_blade_laureate",
        scryfall_id: "62a47835-5719-48c4-a740-a0c5f00dce11",
        printed: PrintedCharacteristics {
            name: "Quill-Blade Laureate".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            text_box: text.into(),
            subtypes: vec!["Human".into(), "Cleric".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::DoubleStrike]),
            ..enters_prepared(twofold_intent())
        },
        ..Default::default()
    }
}

fn skycoach_conductor() -> CardDefinition {
    let text = "Flash\nFlying, vigilance\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.skycoach_conductor",
        scryfall_id: "4ecbca71-9a1d-44c5-b709-d6f565941d5e",
        printed: PrintedCharacteristics {
            name: "Skycoach Conductor".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            text_box: text.into(),
            subtypes: vec!["Bird".into(), "Pilot".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance, Keyword::Flash]),
            ..enters_prepared(all_aboard())
        },
        ..Default::default()
    }
}

fn spellbook_seeker() -> CardDefinition {
    let text = "Flying\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.spellbook_seeker",
        scryfall_id: "cc44eaa4-59a4-419e-b1d1-d92f354ff588",
        printed: PrintedCharacteristics {
            name: "Spellbook Seeker".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            text_box: text.into(),
            subtypes: vec!["Bird".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            ..enters_prepared(careful_study())
        },
        ..Default::default()
    }
}

fn strife_scholar() -> CardDefinition {
    let text = "Ward—Pay 2 life.\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.strife_scholar",
        scryfall_id: "8de79312-2046-425e-9919-49afe19be81b",
        printed: PrintedCharacteristics {
            name: "Strife Scholar".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            text_box: text.into(),
            subtypes: vec!["Orc".into(), "Sorcerer".into()],
            power: Some(3),
            toughness: Some(2),
            statics: vec![StaticAbility::Ward(PaymentCost::Life(2))],
            ..enters_prepared(awaken_the_ages())
        },
        ..Default::default()
    }
}

fn studious_first_year() -> CardDefinition {
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    CardDefinition {
        key: "sos.studious_first_year",
        scryfall_id: "24f888dd-785c-4089-a89c-03f9080130ed",
        printed: PrintedCharacteristics {
            name: "Studious First-Year".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            text_box: text.into(),
            subtypes: vec!["Bear".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(1),
            ..enters_prepared(rampant_growth())
        },
        ..Default::default()
    }
}

fn have_a_bite() -> PrintedCharacteristics {
    let text = "Put a +1/+1 counter on target creature. You gain 1 life.";
    PrintedCharacteristics {
        name: "Have a Bite".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![
            Effect::put_counters(
                ObjectSelector::Target(TargetSpec::Creature),
                CounterKind::PlusOnePlusOne,
                1,
            ),
            Effect::gain_life(1),
        ],
        ..Default::default()
    }
}

fn seething_song() -> PrintedCharacteristics {
    let text = "Add {R}{R}{R}{R}{R}.";
    PrintedCharacteristics {
        name: "Seething Song".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Instant]),
        text_box: text.into(),
        spell_effects: vec![Effect::add_mana(Mana::parse("RRRRR"))],
        ..Default::default()
    }
}

fn aqueous_aria() -> PrintedCharacteristics {
    let text = "Create a 3/3 blue and red Elemental creature token with flying.";
    PrintedCharacteristics {
        name: "Aqueous Aria".into(),
        mana_cost: Some(ManaCost::parse("{4}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::create_tokens("token.blue_red_elemental", 1)],
        ..Default::default()
    }
}

fn raise_dead() -> PrintedCharacteristics {
    let text = "Return target creature card from your graveyard to your hand.";
    PrintedCharacteristics {
        name: "Raise Dead".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::return_to_hand(ObjectSelector::Target(
            TargetSpec::GraveyardCardYouOwn {
                predicate: CardPredicate::creature(),
            },
        ))],
        ..Default::default()
    }
}

fn craft_with_pride() -> PrintedCharacteristics {
    let text = "Create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    PrintedCharacteristics {
        name: "Craft with Pride".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::create_tokens("token.treasure", 1)],
        ..Default::default()
    }
}

fn forum_s_favor() -> PrintedCharacteristics {
    let text = "Target creature gets +1/+0 and gains flying until end of turn.";
    PrintedCharacteristics {
        name: "Forum's Favor".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::pump(1, 0),
            ),
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
            ),
        ],
        ..Default::default()
    }
}

fn stream_of_life() -> PrintedCharacteristics {
    let text = "Target player gains X life.";
    PrintedCharacteristics {
        name: "Stream of Life".into(),
        mana_cost: Some(ManaCost::parse("{X}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::ForPlayer {
            player: PlayerSelector::Target(Box::new(TargetSpec::Player)),
            effects: vec![Effect::gain_life(EffectValue::X)],
        }],
        ..Default::default()
    }
}

fn vibrant_idea() -> PrintedCharacteristics {
    let text = "Draw two cards.";
    PrintedCharacteristics {
        name: "Vibrant Idea".into(),
        mana_cost: Some(ManaCost::parse("{4}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::DrawCards { count: 2.into() }],
        ..Default::default()
    }
}

fn rocket_volley() -> PrintedCharacteristics {
    let text = "Destroy target nonbasic land.";
    PrintedCharacteristics {
        name: "Rocket Volley".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::destroy(ObjectSelector::Target(
            TargetSpec::Permanent {
                predicate: CardPredicate {
                    card_type: Some(CardType::Land),
                    nonbasic: true,
                    ..Default::default()
                },
            },
        ))],
        ..Default::default()
    }
}

fn twofold_intent() -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: "Twofold Intent".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: "Target creature gets +1/+0 and gains double strike until end of turn.".into(),
        spell_effects: vec![
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::pump(1, 0),
            ),
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
            ),
        ],
        ..Default::default()
    }
}

fn all_aboard() -> PrintedCharacteristics {
    let text = "Exile target non-Pilot creature you control, then return that card to the battlefield under its owner's control.";
    PrintedCharacteristics {
        name: "All Aboard".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Instant]),
        text_box: text.into(),
        spell_effects: vec![
            Effect::move_to(
                ObjectSelector::Target(TargetSpec::PermanentYouControl {
                    predicate: CardPredicate {
                        not_subtypes: vec!["Pilot".into()],
                        ..CardPredicate::creature()
                    },
                }),
                ZoneType::Exile,
            ),
            Effect::Move {
                what: ObjectSelector::Moved,
                to: ZoneType::Battlefield,
                under: Some(PlayerSelector::Owner),
                tapped: false,
                transformed: false,
                with_counter: None,
                linked: None,
                library_end: None,
            },
        ],
        ..Default::default()
    }
}

fn careful_study() -> PrintedCharacteristics {
    let text = "Draw two cards, then discard two cards.";
    PrintedCharacteristics {
        name: "Careful Study".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![
            Effect::DrawCards { count: 2.into() },
            Effect::Discard {
                what: ObjectSelector::You,
                count: 2,
            },
        ],
        ..Default::default()
    }
}

fn awaken_the_ages() -> PrintedCharacteristics {
    let text = "Create two 2/2 red and white Spirit creature tokens.";
    PrintedCharacteristics {
        name: "Awaken the Ages".into(),
        mana_cost: Some(ManaCost::parse("{5}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::create_tokens("token.red_white_spirit", 2)],
        ..Default::default()
    }
}

fn rampant_growth() -> PrintedCharacteristics {
    let text = "Search your library for a basic land card, put that card onto the battlefield tapped, then shuffle.";
    PrintedCharacteristics {
        name: "Rampant Growth".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        text_box: text.into(),
        spell_effects: vec![Effect::search_library(
            CardPredicate::basic_land(),
            1.into(),
            vec![Effect::return_to_battlefield(
                ObjectSelector::Selected,
                true,
            )],
        )],
        ..Default::default()
    }
}

fn elite_interceptor() -> CardDefinition {
    use managym_state::ability::TargetRequirement;
    let text = "This creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    let creature = ObjectSelector::Target(TargetSpec::Creature);
    let rejoinder = PrintedCharacteristics {
        name: "Rejoinder".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Sorcery]),
        targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
        spell_effects: vec![
            Effect::Modal {
                modes: vec![
                    vec![Effect::Tap {
                        what: creature.clone(),
                    }],
                    vec![Effect::Untap { what: creature }],
                    vec![],
                ],
            },
            Effect::DrawCards { count: 1.into() },
        ],
        text_box: "You may tap or untap target creature.\nDraw a card.".into(),
        ..Default::default()
    };
    CardDefinition {
        key: "sos.elite_interceptor",
        scryfall_id: "2970683e-e69c-42cb-a067-34abd56fb42b",
        printed: PrintedCharacteristics {
            name: "Elite Interceptor".into(),
            mana_cost: Some(ManaCost::parse("{W}")),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(2),
            text_box: text.into(),
            ..enters_prepared(rejoinder)
        },
        ..Default::default()
    }
}

fn jadzi_steward_of_fate() -> CardDefinition {
    let text = "Jadzi enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)\nWhen Jadzi enters, draw two cards, then discard two cards.";
    let spell_text = "Create X 0/0 green and blue Fractal creature tokens, then put X +1/+1 counters on each Fractal you control.";
    CardDefinition {
        key: "sos.jadzi_steward_of_fate",
        scryfall_id: "a95b6baf-01e6-49c3-9a26-394b127d53c3",
        printed: PrintedCharacteristics {
            name: "Jadzi, Steward of Fate".into(),
            supertypes: vec!["Legendary".into()],
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![
                    Effect::DrawCards { count: 2.into() },
                    Effect::Discard {
                        what: ObjectSelector::You,
                        count: 2,
                    },
                ],
            }],
            text_box: text.into(),
            ..enters_prepared(PrintedCharacteristics {
                name: "Oracle's Gift".into(),
                mana_cost: Some(ManaCost::parse("{X}{X}{U}")),
                types: CardTypes::new([CardType::Sorcery]),
                text_box: spell_text.into(),
                spell_effects: vec![
                    Effect::create_tokens("token.fractal", EffectValue::X),
                    Effect::put_counters(
                        ObjectSelector::each_you_control(CardPredicate::subtype("Fractal")),
                        CounterKind::PlusOnePlusOne,
                        EffectValue::X,
                    ),
                ],
                ..Default::default()
            })
        },
        ..Default::default()
    }
}

fn pigment_wrangler() -> CardDefinition {
    let text = "Flying\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    let spell_text = "When you next cast an instant or sorcery spell this turn, copy that spell. You may choose new targets for the copy.";
    CardDefinition {
        key: "sos.pigment_wrangler",
        scryfall_id: "c2faf4cf-c4b6-4721-ac06-0e045dd9704a",
        printed: PrintedCharacteristics {
            name: "Pigment Wrangler".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            subtypes: vec!["Orc".into(), "Sorcerer".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),

            text_box: text.into(),
            ..enters_prepared(PrintedCharacteristics {
                name: "Striking Palette".into(),
                mana_cost: Some(ManaCost::parse("{R}")),
                types: CardTypes::new([CardType::Sorcery]),
                text_box: spell_text.into(),
                spell_effects: vec![Effect::CreateDelayedTrigger {
                    remember: None,
                    event: DelayedEvent::YouCastSpell(Box::new(CardPredicate {
                        card_types_any: vec![CardType::Instant, CardType::Sorcery],
                        ..Default::default()
                    })),
                    frequency: DelayedFrequency::Once,
                    expiry: DelayedExpiry::ThisTurn,
                    effects: vec![Effect::CopyTriggeringObject],
                }],
                ..Default::default()
            })
        },
        ..Default::default()
    }
}

fn vastlands_scavenger() -> CardDefinition {
    let text = "Deathtouch\nThis creature enters prepared. (While it's prepared, you may cast a copy of its spell. Doing so unprepares it.)";
    let spell_text =
        "Mill seven cards. Then put a creature card from among them onto the battlefield.";
    CardDefinition {
        key: "sos.vastlands_scavenger",
        scryfall_id: "476b6a4d-cc05-4e98-8a45-a5c6582ec514",
        printed: PrintedCharacteristics {
            name: "Vastlands Scavenger".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}{G}")),
            subtypes: vec!["Bear".into(), "Druid".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Deathtouch]),

            text_box: text.into(),
            ..enters_prepared(PrintedCharacteristics {
                name: "Bind to Life".into(),
                mana_cost: Some(ManaCost::parse("{4}{G}")),
                types: CardTypes::new([CardType::Instant]),
                text_box: spell_text.into(),
                spell_effects: vec![
                    Effect::MillCards {
                        count: 7.into(),
                        target: None,
                    },
                    Effect::SelectCards {
                        from: ObjectSelector::CardsThisWay,
                        predicate: CardPredicate::creature(),
                        min: 0,
                        max: 1.into(),
                        effects: vec![Effect::return_to_battlefield(
                            ObjectSelector::Selected,
                            false,
                        )],
                        if_declined: vec![],
                    },
                ],
                ..Default::default()
            })
        },
        ..Default::default()
    }
}
