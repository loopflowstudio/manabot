//! Limited Edition Alpha.
use crate::state::ability::AffectedSpec;
use crate::state::effect::Modification;
use crate::state::{
    ability::{Effect, TargetRequirement, TargetSpec},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, ManaAbility,
    },
    mana::{Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        ancestral_recall(),
        counterspell(),
        craw_wurm(),
        forest(),
        giant_spider(),
        gray_ogre(),
        hill_giant(),
        island(),
        lightning_bolt(),
        llanowar_elves(),
        mountain(),
        plains(),
        savannah_lions(),
        serra_angel(),
        shivan_dragon(),
        swamp(),
        wall_of_stone(),
        war_mammoth(),
    ]
}

fn ancestral_recall() -> CardDefinition {
    let text_box = "Target player draws three cards.";

    CardDefinition {
        key: "lea.ancestral_recall",
        scryfall_id: "70e7ddf2-5604-41e7-bb9d-ddd03d3e9d0b",
        name: "Ancestral Recall".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::DrawCards { count: 3 }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn counterspell() -> CardDefinition {
    let text_box = "Counter target spell.";

    CardDefinition {
        key: "lea.counterspell",
        scryfall_id: "0df55e3f-14de-46ef-b6b1-616618724d9e",
        name: "Counterspell".into(),
        mana_cost: Some(ManaCost::parse("{U}{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::CounterSpell {
            target: TargetSpec::Spell,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn craw_wurm() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "lea.craw_wurm",
        scryfall_id: "bfed1a95-bd67-4e16-a781-81866028af2f",
        name: "Craw Wurm".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Wurm".into()],
        power: Some(6),
        toughness: Some(4),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn forest() -> CardDefinition {
    let text_box = "({T}: Add {G}.)";

    CardDefinition {
        key: "lea.forest",
        scryfall_id: "6f1c8cb0-38eb-408b-94e8-16db83999b3b",
        name: "Forest".into(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".into()],
        subtypes: vec!["Forest".into()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("G"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn giant_spider() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)";

    CardDefinition {
        key: "lea.giant_spider",
        scryfall_id: "77636b4c-faea-4bf5-b88c-dd5bb88dc930",
        name: "Giant Spider".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Spider".into()],
        power: Some(2),
        toughness: Some(4),
        keywords: Keywords {
            reach: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn gray_ogre() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "lea.gray_ogre",
        scryfall_id: "73ae5276-b607-4f23-a9d2-e8cc7b8e3693",
        name: "Gray Ogre".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Ogre".into()],
        power: Some(2),
        toughness: Some(2),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn hill_giant() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "lea.hill_giant",
        scryfall_id: "0ddb98e8-13fe-4786-83f7-b72c56db135a",
        name: "Hill Giant".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Giant".into()],
        power: Some(3),
        toughness: Some(3),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn island() -> CardDefinition {
    let text_box = "({T}: Add {U}.)";

    CardDefinition {
        key: "lea.island",
        scryfall_id: "90a57c0e-fa61-45ef-955d-d296403967d5",
        name: "Island".into(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".into()],
        subtypes: vec!["Island".into()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("U"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn lightning_bolt() -> CardDefinition {
    let text_box = "Lightning Bolt deals 3 damage to any target.";

    CardDefinition {
        key: "lea.lightning_bolt",
        scryfall_id: "d573ef03-4730-45aa-93dd-e45ac1dbaf4a",
        name: "Lightning Bolt".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::DealDamage {
            amount: 3,
            target: TargetSpec::CreatureOrPlayer,
        }],
        targeting: vec![TargetRequirement {
            shares_creature_type: false,
            spec: TargetSpec::CreatureOrPlayer,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn llanowar_elves() -> CardDefinition {
    let text_box = "{T}: Add {G}.";

    CardDefinition {
        key: "lea.llanowar_elves",
        scryfall_id: "d4f1cc9e-4f99-4c26-ac1b-8ef069fa8ceb",
        name: "Llanowar Elves".into(),
        mana_cost: Some(ManaCost::parse("{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elf".into(), "Druid".into()],
        power: Some(1),
        toughness: Some(1),
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("G"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn mountain() -> CardDefinition {
    let text_box = "({T}: Add {R}.)";

    CardDefinition {
        key: "lea.mountain",
        scryfall_id: "eace2c85-976c-425e-9800-5a6ccbd91b56",
        name: "Mountain".into(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".into()],
        subtypes: vec!["Mountain".into()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("R"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn plains() -> CardDefinition {
    let text_box = "({T}: Add {W}.)";

    CardDefinition {
        key: "lea.plains",
        scryfall_id: "b1623d57-4729-4796-b3f7-f1837a05c6ed",
        name: "Plains".into(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".into()],
        subtypes: vec!["Plains".into()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("W"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn savannah_lions() -> CardDefinition {
    let text_box = "";

    CardDefinition {
        key: "lea.savannah_lions",
        scryfall_id: "d05b92bd-797e-413f-a8b0-32e0937a1ee0",
        name: "Savannah Lions".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Cat".into()],
        power: Some(2),
        toughness: Some(1),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn serra_angel() -> CardDefinition {
    let text_box = "Flying\nVigilance (Attacking doesn't cause this creature to tap.)";

    CardDefinition {
        key: "lea.serra_angel",
        scryfall_id: "f8ac5006-91bd-4803-93da-f87cf196dd2f",
        name: "Serra Angel".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Angel".into()],
        power: Some(4),
        toughness: Some(4),
        keywords: Keywords {
            flying: true,
            vigilance: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn shivan_dragon() -> CardDefinition {
    let text_box = "Flying\n{R}: This creature gets +1/+0 until end of turn.";

    CardDefinition {
        key: "lea.shivan_dragon",
        scryfall_id: "fefbf149-f988-4f8b-9f53-56f5878116a6",
        name: "Shivan Dragon".into(),
        mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dragon".into()],
        power: Some(5),
        toughness: Some(5),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{R}"),
            effects: vec![Effect::until_end_of_turn(
                AffectedSpec::Source,
                Modification::pump(1, 0),
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn swamp() -> CardDefinition {
    let text_box = "({T}: Add {B}.)";

    CardDefinition {
        key: "lea.swamp",
        scryfall_id: "6176936d-72e2-4205-8871-4c5a4f1cb2d8",
        name: "Swamp".into(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".into()],
        subtypes: vec!["Swamp".into()],
        mana_abilities: vec![ManaAbility {
            mana: Mana::parse("B"),
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wall_of_stone() -> CardDefinition {
    let text_box = "Defender (This creature can't attack.)";

    CardDefinition {
        key: "lea.wall_of_stone",
        scryfall_id: "140e567c-6e4a-42b0-8084-d6c9695ae802",
        name: "Wall of Stone".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Wall".into()],
        power: Some(0),
        toughness: Some(8),
        keywords: Keywords {
            defender: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn war_mammoth() -> CardDefinition {
    let text_box = "Trample";

    CardDefinition {
        key: "lea.war_mammoth",
        scryfall_id: "c8d6081e-f686-4263-a0a2-21c0d9af5fdb",
        name: "War Mammoth".into(),
        mana_cost: Some(ManaCost::parse("{3}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elephant".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords {
            trample: true,
            ..Default::default()
        },
        text_box: text_box.into(),
        ..Default::default()
    }
}
