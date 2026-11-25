//! Every token definition (`token.*`). Tokens are not cards (CR 108.2,
//! 111.1): each is a [`TokenDefinition`] whose [`PrintedCharacteristics`] are
//! the same type a card prints, so copy effects treat tokens and cards alike. Effects
//! create a token by its key, and each carries the Scryfall id of its first
//! token printing, or `None` when it has none.
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, ObjectSelector, Per, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, BlockRestriction, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, TokenDefinition,
    },
    effect::Modification,
    mana::{Color, Colors, Mana, ManaCost, ManaRestriction},
    predicate::CardPredicate,
};

pub(super) fn tokens() -> Vec<TokenDefinition> {
    vec![
        ally(),
        ballistic_boulder(),
        bat(),
        bear(),
        cat(),
        lifelink_cat(),
        clue(),
        dragon(),
        drake(),
        elf_warrior(),
        fish(),
        elf(),
        food(),
        goblin(),
        human(),
        human_soldier(),
        komas_coil(),
        marit_lage(),
        monk(),
        plain_spirit(),
        flying_spirit(),
        powerstone(),
        rabbit(),
        black_rat(),
        scion_of_the_deep(),
        soldier(),
        white_soldier(),
        spirit(),
        treasure(),
        knight_3_3_w(),
        dragon_4_4_r_flying(),
        dragon_5_5_r_flying(),
        soldier_1_1_w(),
        elf_warrior_1_1_g(),
        insect_1_1_bg_flying(),
        faerie_1_1_u_flying(),
        zombie_2_2_b(),
        dog_1_1_w(),
        beast_3_3_g(),
        beast_4_4_g(),
        villain(),
        zeppelin(),
        zombie(),
        cadet(),
        fractal(),
        heartwood(),
        blue_red_elemental(),
        inkling(),
        red_white_spirit(),
        rat(),
        jace(),
        token(
            "token.ninja_2_1_u",
            "aeec04b1-475c-4e55-b72f-327ea5258146",
            creature(
                face("Ninja", [CardType::Creature], &["Ninja"], ""),
                2,
                1,
                [Color::Blue],
            ),
        ),
        token(
            "token.ajanis_pridemate",
            "b0819e8e-fb7e-43c7-a7cf-d768f43193ac",
            PrintedCharacteristics {
                abilities: vec![Ability::when_you_gain_life(Effect::put_counters(
                    ObjectSelector::This,
                    managym_state::permanent::CounterKind::PlusOnePlusOne,
                    1,
                ))],
                ..creature(
                    face(
                        "Ajani's Pridemate",
                        [CardType::Creature],
                        &["Cat", "Soldier"],
                        "Whenever you gain life, put a +1/+1 counter on this token.",
                    ),
                    2,
                    2,
                    [Color::White],
                )
            },
        ),
        token(
            "token.illusion_1_1_u",
            "7bda0b63-fd0f-495d-9eb0-20057571c0ee",
            creature(
                face("Illusion", [CardType::Creature], &["Illusion"], ""),
                1,
                1,
                [Color::Blue],
            ),
        ),
    ]
}

/// A token's printed name, type line and text; callers add the rest.
fn face(
    name: &str,
    types: impl IntoIterator<Item = CardType>,
    subtypes: &[&str],
    text_box: &str,
) -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: name.into(),
        types: CardTypes::new(types),
        subtypes: subtypes.iter().map(|&subtype| subtype.into()).collect(),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn token(
    key: &'static str,
    scryfall_id: &'static str,
    printed: PrintedCharacteristics,
) -> TokenDefinition {
    TokenDefinition {
        key,
        scryfall_id: Some(scryfall_id),
        printed,
    }
}

/// A creature token with printed power, toughness and colors.
fn creature(
    definition: PrintedCharacteristics,
    power: i32,
    toughness: i32,
    colors: impl IntoIterator<Item = Color>,
) -> PrintedCharacteristics {
    PrintedCharacteristics {
        power: Some(power),
        toughness: Some(toughness),
        color_override: Some(Colors::from_iter(colors)),
        ..definition
    }
}

fn jace() -> TokenDefinition {
    let text_box = "−1: Surveil 1. (Look at the top card of your library. You may put that card into your graveyard.)\n−3: Draw a card.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    token(
        "token.jace",
        "8d1e424f-6407-4f8c-bed2-eaf148237c81",
        PrintedCharacteristics {
            loyalty: Some(0),
            color_override: Some(Colors::from_iter([Color::Blue])),
            activated_abilities: vec![
                loyalty(-1, vec![Effect::Surveil { count: 1 }]),
                loyalty(-3, vec![Effect::DrawCards { count: 1.into() }]),
            ],
            ..face("Jace", [CardType::Planeswalker], &["Jace"], text_box)
        },
    )
}

fn goblin() -> TokenDefinition {
    token(
        "token.goblin_1_1_r",
        "fbd728ed-d5dc-44b3-9159-6c86cab3be0c",
        creature(
            face("Goblin", [CardType::Creature], &["Goblin"], ""),
            1,
            1,
            [Color::Red],
        ),
    )
}

fn cat() -> TokenDefinition {
    token(
        "token.cat_1_1_w",
        "5f458f39-27b6-4121-bda9-1a0d1b42f5fb",
        creature(
            face("Cat", [CardType::Creature], &["Cat"], ""),
            1,
            1,
            [Color::White],
        ),
    )
}

fn lifelink_cat() -> TokenDefinition {
    let base = PrintedCharacteristics {
        keywords: Keywords::of(&[Keyword::Lifelink]),
        ..face("Cat", [CardType::Creature], &["Cat"], "Lifelink")
    };
    token(
        "token.cat_1_1_w_lifelink",
        "af0086cd-56b5-450e-818c-e54f45d1a104",
        creature(base, 1, 1, [Color::White]),
    )
}

fn ally() -> TokenDefinition {
    let base = face("Ally", [CardType::Creature], &["Ally"], "");
    token(
        "token.ally",
        "f842c52a-04b6-4625-ba58-45b043ada340",
        creature(base, 1, 1, [Color::White]),
    )
}

fn ballistic_boulder() -> TokenDefinition {
    let base = face(
        "Ballistic Boulder",
        [CardType::Creature, CardType::Artifact],
        &["Construct"],
        "Flying",
    );
    token(
        "token.ballistic_boulder",
        "b9d70342-9d3c-44cb-9026-db5e5941b9fc",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(base, 2, 1, [])
        },
    )
}

fn bat() -> TokenDefinition {
    let base = face("Bat", [CardType::Creature], &["Bat"], "Flying");
    token(
        "token.bat",
        "89d7ed94-564b-415c-b590-d89c972384a1",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(base, 1, 1, [Color::Black])
        },
    )
}

fn bear() -> TokenDefinition {
    let base = face("Bear", [CardType::Creature], &["Bear"], "");
    token(
        "token.bear",
        "ca3dae7d-3880-4c0a-acfb-8fd227cf9fab",
        creature(base, 4, 4, [Color::Green]),
    )
}

/// Redcap Gutter-Dweller's Rats (first token printing, WOE).
fn rat() -> TokenDefinition {
    let base = face(
        "Rat",
        [CardType::Creature],
        &["Rat"],
        "This token can't block.",
    );
    token(
        "token.rat",
        "1e0205f2-25c1-403b-b408-56e3f2d63b4d",
        PrintedCharacteristics {
            statics: vec![StaticAbility::BlockRestriction(
                BlockRestriction::CantBlock(CardPredicate::default()),
            )],
            ..creature(base, 1, 1, [Color::Black])
        },
    )
}

fn clue() -> TokenDefinition {
    token(
        "token.clue",
        "f2c859e1-181e-44d1-afbd-bbd6e52cf42a",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
                ..Default::default()
            }],
            ..face(
                "Clue",
                [CardType::Artifact],
                &["Clue"],
                "{2}, Sacrifice this token: Draw a card.",
            )
        },
    )
}

fn drake() -> TokenDefinition {
    let base = face("Drake", [CardType::Creature], &["Drake"], "Flying");
    token(
        "token.drake",
        "f4a73034-e20f-4e7e-ac15-3460b1e9c69b",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(base, 2, 2, [Color::Blue])
        },
    )
}

fn dragon() -> TokenDefinition {
    let text_box = "Flying, firebending 4 (Whenever this token attacks, add {R}{R}{R}{R} to your mana pool. This mana lasts until end of combat.)";
    let base = face("Dragon", [CardType::Creature], &["Dragon"], text_box);
    token(
        "token.dragon",
        "cddc0746-6d3e-441f-866f-28587bf54801",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RRRR"),
                    amount: EffectValue::Fixed(1),
                    until_end_of_combat: true,
                }],
            }],
            ..creature(base, 4, 4, [Color::Red])
        },
    )
}

fn elf_warrior() -> TokenDefinition {
    let base = face("Elf Warrior", [CardType::Creature], &["Elf", "Warrior"], "");
    token(
        "token.elf_warrior",
        "49180301-f72f-48ca-8be1-df295e89cdd7",
        creature(base, 1, 1, [Color::Green, Color::White]),
    )
}

fn food() -> TokenDefinition {
    token(
        "token.food",
        "bf36408d-ed85-497f-8e68-d3a922c388a0",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{2}"),
                sacrifice_source: true,
                effects: vec![Effect::gain_life(3)],
                ..Default::default()
            }],
            ..face(
                "Food",
                [CardType::Artifact],
                &["Food"],
                "{2}, {T}, Sacrifice this token: You gain 3 life.",
            )
        },
    )
}

fn human() -> TokenDefinition {
    let base = face("Human", [CardType::Creature], &["Human"], "");
    token(
        "token.human",
        "7d13a93a-a43d-4cf5-8300-8341f3b7f1b1",
        creature(base, 1, 1, [Color::White]),
    )
}

fn human_soldier() -> TokenDefinition {
    let base = face(
        "Human Soldier",
        [CardType::Creature],
        &["Human", "Soldier"],
        "",
    );
    token(
        "token.human_soldier",
        "d9cbf36e-4044-4f08-9bae-f0dcb2455716",
        creature(base, 1, 1, [Color::White]),
    )
}

fn komas_coil() -> TokenDefinition {
    let base = face("Koma's Coil", [CardType::Creature], &["Serpent"], "");
    token(
        "token.komas_coil",
        "54a1c6a9-3531-4432-9157-e4400dbc89fd",
        creature(base, 3, 3, [Color::Blue]),
    )
}

fn marit_lage() -> TokenDefinition {
    let base = face(
        "Marit Lage",
        [CardType::Creature],
        &["Avatar"],
        "Flying, indestructible",
    );
    token(
        "token.marit_lage",
        "8f5c3863-c876-48a8-9bc8-be20cd61074c",
        PrintedCharacteristics {
            supertypes: vec!["Legendary".into()],
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Indestructible]),
            ..creature(base, 20, 20, [Color::Black])
        },
    )
}

fn monk() -> TokenDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";
    let base = face("Monk", [CardType::Creature], &["Monk"], text_box);
    token(
        "token.monk",
        "388180bd-cc96-4b18-9ddd-18a3f7e06282",
        PrintedCharacteristics {
            abilities: vec![Ability::prowess()],
            ..creature(base, 1, 1, [Color::Red])
        },
    )
}

fn rabbit() -> TokenDefinition {
    let base = face("Rabbit", [CardType::Creature], &["Rabbit"], "");
    token(
        "token.rabbit",
        "22ca55e0-d269-4178-bc90-920a12066e4f",
        creature(base, 1, 1, [Color::White]),
    )
}

fn soldier() -> TokenDefinition {
    let text_box = "Firebending 1 (Whenever this token attacks, add {R}. This mana lasts until end of combat.)";
    let base = face("Soldier", [CardType::Creature], &["Soldier"], text_box);
    token(
        "token.soldier",
        "2de43b03-9ac5-4292-ab29-2dc6210ef3d9",
        PrintedCharacteristics {
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("R"),
                    amount: EffectValue::Fixed(1),
                    until_end_of_combat: true,
                }],
            }],
            ..creature(base, 2, 2, [Color::Red])
        },
    )
}

fn spirit() -> TokenDefinition {
    let text_box = "This token can't block or be blocked by non-Spirit creatures.";
    let base = face("Spirit", [CardType::Creature], &["Spirit"], text_box);
    let non_spirit = || CardPredicate {
        not_subtypes: vec!["Spirit".into()],
        ..Default::default()
    };
    token(
        "token.spirit",
        "f59eba51-458a-40e0-b754-999f91d5d839",
        PrintedCharacteristics {
            statics: vec![
                StaticAbility::BlockRestriction(BlockRestriction::CantBlock(non_spirit())),
                StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(non_spirit())),
            ],
            ..creature(base, 1, 1, [])
        },
    )
}

/// Cityscape Leveler's token (first token printing, BRO).
fn powerstone() -> TokenDefinition {
    token(
        "token.powerstone",
        "d45fe4b6-aeaf-4f84-b660-c7b482ed8512",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::restricted_to(
                    ManaRestriction::ExceptNonartifactSpells,
                    [0, 0, 0, 0, 0, 1],
                ))],
                ..Default::default()
            }],
            ..face(
                "Powerstone",
                [CardType::Artifact],
                &["Powerstone"],
                "{T}: Add {C}. This mana can't be spent to cast a nonartifact spell.",
            )
        },
    )
}

fn treasure() -> TokenDefinition {
    token(
        "token.treasure",
        "720f3e68-84c0-462e-a0d1-90236ccc494a",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(Mana::parse),
                )],
                ..Default::default()
            }],
            ..face(
                "Treasure",
                [CardType::Artifact],
                &["Treasure"],
                "{T}, Sacrifice this token: Add one mana of any color.",
            )
        },
    )
}

/// Plain colorless Spirit, first printed in the 2004 player rewards tokens.
fn plain_spirit() -> TokenDefinition {
    let base = face("Spirit", [CardType::Creature], &["Spirit"], "");
    token(
        "token.plain_spirit",
        "46b60d95-b9bc-40f8-b986-bfa8e3eb74f3",
        creature(base, 1, 1, []),
    )
}

/// Lita's named Vehicle token has no printed token card.
fn zeppelin() -> TokenDefinition {
    TokenDefinition {
        key: "token.zeppelin",
        scryfall_id: None,
        printed: PrintedCharacteristics {
            power: Some(5),
            toughness: Some(5),
            color_override: Some(Colors::default()),
            keywords: Keywords::of(&[Keyword::Flying]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                crew: Some(3),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::BecomeCreature,
                )],
                ..Default::default()
            }],
            ..face(
                "Zeppelin",
                [CardType::Artifact],
                &["Vehicle"],
                "Flying\nCrew 3",
            )
        },
    }
}

fn knight_3_3_w() -> TokenDefinition {
    token(
        "token.knight_3_3_w",
        "0d287806-07bb-4709-9260-b9ef34fe2a13",
        creature(
            face("Knight", [CardType::Creature], &["Knight"], ""),
            3,
            3,
            [Color::White],
        ),
    )
}

fn dragon_4_4_r_flying() -> TokenDefinition {
    token(
        "token.dragon_4_4_r_flying",
        "cbb3e91d-9cae-49c8-ba92-087ad8171b9d",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(
                face("Dragon", [CardType::Creature], &["Dragon"], "Flying"),
                4,
                4,
                [Color::Red],
            )
        },
    )
}

fn dragon_5_5_r_flying() -> TokenDefinition {
    token(
        "token.dragon_5_5_r_flying",
        "56c0fa29-fbcb-41fe-94ab-a39b1154c99b",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(
                face("Dragon", [CardType::Creature], &["Dragon"], "Flying"),
                5,
                5,
                [Color::Red],
            )
        },
    )
}

fn soldier_1_1_w() -> TokenDefinition {
    token(
        "token.soldier_1_1_w",
        "b159b57d-bc52-4cef-ac7a-e364e40c3d03",
        creature(
            face("Soldier", [CardType::Creature], &["Soldier"], ""),
            1,
            1,
            [Color::White],
        ),
    )
}

fn elf_warrior_1_1_g() -> TokenDefinition {
    token(
        "token.elf_warrior_1_1_g",
        "27b171ac-b2ef-4a80-92d1-6d9e71f3e3ca",
        creature(
            face("Elf Warrior", [CardType::Creature], &["Elf", "Warrior"], ""),
            1,
            1,
            [Color::Green],
        ),
    )
}

fn insect_1_1_bg_flying() -> TokenDefinition {
    token(
        "token.insect_1_1_bg_flying",
        "377f1a20-b270-4b07-9892-7170cd0bee38",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(
                face("Insect", [CardType::Creature], &["Insect"], "Flying"),
                1,
                1,
                [Color::Black, Color::Green],
            )
        },
    )
}

fn faerie_1_1_u_flying() -> TokenDefinition {
    token(
        "token.faerie_1_1_u_flying",
        "bcd82cb0-ff4b-4f4d-b3d0-3ac53883b099",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(
                face("Faerie", [CardType::Creature], &["Faerie"], "Flying"),
                1,
                1,
                [Color::Blue],
            )
        },
    )
}

fn zombie_2_2_b() -> TokenDefinition {
    token(
        "token.zombie_2_2_b",
        "8e7b5995-ee8b-40d1-b157-8df96c02cc5b",
        creature(
            face("Zombie", [CardType::Creature], &["Zombie"], ""),
            2,
            2,
            [Color::Black],
        ),
    )
}

fn dog_1_1_w() -> TokenDefinition {
    token(
        "token.dog_1_1_w",
        "4f8107b3-8539-4b9c-8d0d-c512c940838f",
        creature(
            face("Dog", [CardType::Creature], &["Dog"], ""),
            1,
            1,
            [Color::White],
        ),
    )
}

fn beast_4_4_g() -> TokenDefinition {
    token(
        "token.beast_4_4_g",
        "ce45e037-5efb-4735-afee-12d7dc3127d1",
        creature(
            face("Beast", [CardType::Creature], &["Beast"], ""),
            4,
            4,
            [Color::Green],
        ),
    )
}

fn beast_3_3_g() -> TokenDefinition {
    token(
        "token.beast_3_3_g",
        "d02d5670-fff3-4011-8e45-e6843e4e8988",
        creature(
            face("Beast", [CardType::Creature], &["Beast"], ""),
            3,
            3,
            [Color::Green],
        ),
    )
}
fn villain() -> TokenDefinition {
    let base = face("Villain", [CardType::Creature], &["Villain"], "Menace");
    token(
        "token.villain",
        "cd76092f-0f5e-4890-996c-33f4f47c659c",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Menace]),
            ..creature(base, 2, 1, [Color::Black])
        },
    )
}

fn flying_spirit() -> TokenDefinition {
    let base = face("Spirit", [CardType::Creature], &["Spirit"], "Flying");
    token(
        "token.flying_spirit",
        "14ef4815-3dfe-47b3-ad81-1506925280d3",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(base, 1, 1, [Color::White])
        },
    )
}

fn fish() -> TokenDefinition {
    let base = face("Fish", [CardType::Creature], &["Fish"], "");
    token(
        "token.fish",
        "297fc694-08f6-4477-b898-60ec641a0757",
        creature(base, 1, 1, [Color::Blue]),
    )
}

fn black_rat() -> TokenDefinition {
    let base = face("Rat", [CardType::Creature], &["Rat"], "");
    token(
        "token.black_rat",
        "1a85fe9d-ef18-46c4-88b0-cf2e222e30e4",
        creature(base, 1, 1, [Color::Black]),
    )
}

fn scion_of_the_deep() -> TokenDefinition {
    let base = face("Scion of the Deep", [CardType::Creature], &["Octopus"], "");
    token(
        "token.scion_of_the_deep",
        "26c478f7-b426-4055-8d06-e5782226c826",
        PrintedCharacteristics {
            supertypes: vec!["Legendary".into()],
            ..creature(base, 8, 8, [Color::Blue])
        },
    )
}
fn inkling() -> TokenDefinition {
    token(
        "token.inkling",
        "bab52920-9d67-4cd4-9015-6e645ff9764f",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(
                face("Inkling", [CardType::Creature], &["Inkling"], "Flying"),
                1,
                1,
                [Color::White, Color::Black],
            )
        },
    )
}

fn zombie() -> TokenDefinition {
    let base = face("Zombie", [CardType::Creature], &["Zombie"], "");
    token(
        "token.zombie",
        "e29a360b-11fd-42ba-82ba-de8f7e136c20",
        creature(base, 2, 2, [Color::Black]),
    )
}

fn white_soldier() -> TokenDefinition {
    let base = face("Soldier", [CardType::Creature], &["Soldier"], "");
    token(
        "token.white_soldier",
        "7ddd5153-2d16-4a4e-b9bc-f20313d322da",
        creature(base, 2, 2, [Color::White]),
    )
}

fn elf() -> TokenDefinition {
    let base = face("Elf", [CardType::Creature], &["Elf"], "");
    token(
        "token.elf",
        "761c7c31-c6c5-44e2-a845-f590542b6eda",
        creature(base, 1, 1, [Color::Green]),
    )
}

fn cadet() -> TokenDefinition {
    token(
        "token.cadet",
        "8f4534d8-2783-484f-8ebf-a47b1cc4c6df",
        creature(
            face("Cadet", [CardType::Creature], &["Wizard", "Soldier"], ""),
            2,
            2,
            [],
        ),
    )
}

fn heartwood() -> TokenDefinition {
    token(
        "token.heartwood",
        "eb4bf635-04d7-4994-9f70-4ff573a767e7",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(["R", "G"].map(Mana::parse))],
                ..Default::default()
            }],
            color_override: Some(Colors::from_iter([Color::Red, Color::Green])),
            ..face(
                "Heartwood",
                [CardType::Artifact],
                &["Heartwood"],
                "{T}: Add {R} or {G}.",
            )
        },
    )
}

fn blue_red_elemental() -> TokenDefinition {
    token(
        "token.blue_red_elemental",
        "57b98846-85e3-47c7-a903-29953d0b0e8a",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(
                face("Elemental", [CardType::Creature], &["Elemental"], "Flying"),
                3,
                3,
                [Color::Blue, Color::Red],
            )
        },
    )
}

fn red_white_spirit() -> TokenDefinition {
    token(
        "token.red_white_spirit",
        "877f7ddb-ed70-41a0-b845-d9bf8ac65f9b",
        creature(
            face("Spirit", [CardType::Creature], &["Spirit"], ""),
            2,
            2,
            [Color::Red, Color::White],
        ),
    )
}

fn fractal() -> TokenDefinition {
    token(
        "token.fractal",
        "910f48ab-b04e-4874-b31d-a86a7bc5af14",
        creature(
            face("Fractal", [CardType::Creature], &["Fractal"], ""),
            0,
            0,
            [Color::Green, Color::Blue],
        ),
    )
}
