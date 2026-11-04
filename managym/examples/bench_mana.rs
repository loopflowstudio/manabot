//! Bounded payment/offer fixtures, not a stress benchmark. Ten samples per
//! position retain raw timings, exact remainders and opt-in search counters.
//! Run with `cargo run --release --features benchmark --example bench_mana`.
#[path = "../tests/support/mod.rs"]
mod support;

use managym::{
    agent::{
        action::Action,
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig, FACT_DIM},
    },
    flow::{mana::measurement, turn::StepKind},
    state::{
        game_object::PlayerId,
        mana::{Mana, ManaCost, ManaRestriction, SpendingContext},
        zone::ZoneType,
    },
};
use std::{hint::black_box, time::Instant};
use support::game::{deck, Scenario};

fn position(names: &[&str]) -> Scenario {
    let mut cards = deck(&[("Mountain", 24), ("Gray Ogre", 12), ("Prophetic Prism", 4)]);
    for name in names {
        cards.insert((*name).into(), 8);
    }
    let mut s = Scenario::new(cards.clone(), cards, 242);
    s.game_mut().skip_trivial = true;
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().scenario_clear_hand(PlayerId(0));
    for name in names {
        s.force_permanent_on_battlefield(0, name);
    }
    s.game_mut()
        .scenario_force_card_in_hand(PlayerId(0), "Gray Ogre")
        .unwrap();
    s.set_player_mana_pool(0, "0");
    s
}

/// Follow a constructed spell payment with native forced-decision skipping.
/// Only its printed cost/subtypes change to match the declared obligation;
/// the mana sources and all later Commands use ordinary engine behavior.
fn trace(name: &str, mut s: Scenario, payable: &str) {
    let spell = s.game().state.zones.zone_cards(ZoneType::Hand, PlayerId(0))[0];
    s.game_mut().state.cards[spell].mana_cost = Some(ManaCost::parse(payable));
    s.game_mut().state.cards[spell].subtypes = vec!["Citizen".into(), "Elf".into()];
    s.game_mut().scenario_refresh_priority().unwrap();
    let skipped = s.game().skip_trivial_count;
    let mut frames = Vec::new();
    for step in 0..32 {
        if s.game().state.zones.zone_of(spell) != Some(ZoneType::Hand) {
            break;
        }
        let actions = &s.action_space().actions;
        assert!(actions.len() > 1, "native forced decision escaped skipping");
        let chosen = if step == 0 {
            actions
                .iter()
                .find(|a| matches!(a, Action::CastSpell { card, .. } if *card == spell))
                .unwrap()
        } else {
            actions
                .iter()
                .find(|a| matches!(a, Action::PayCost { .. }))
                .unwrap_or(&actions[0])
        }
        .clone();
        let observation = Observation::for_player(s.game(), PlayerId(0));
        let config = ObservationEncoderConfig {
            policy_input_version: 9,
            ..Default::default()
        };
        let mut times = Vec::new();
        let mut facts = 0;
        for _ in 0..10 {
            let start = Instant::now();
            let encoded = black_box(encode(&observation, &config).unwrap());
            times.push(start.elapsed().as_secs_f64() * 1e6);
            facts = encoded
                .facts
                .chunks_exact(FACT_DIM)
                .filter(|r| r[11] == 1.0)
                .count();
        }
        frames.push(serde_json::json!({"kind":format!("{:?}",s.action_space().kind),"legal_actions":actions.len(),"facts":facts,"encode_microseconds":times}));
        support::play::command(&mut s, |a| *a == chosen);
    }
    assert!(matches!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Stack | ZoneType::Battlefield)
    ));
    println!(
        "{}",
        serde_json::json!({"case":name,"retained_decisions":frames.len(),"skipped_decisions":s.game().skip_trivial_count-skipped,"frames":frames,"remaining_mana":s.game().state.players[0].mana_pool})
    );
}

fn main() {
    let mut mixed = position(&[]);
    for (names, amount) in [
        (&["Citizen"][..], 2),
        (&["Elf"][..], 1),
        (&["Citizen", "Elf"][..], 1),
    ] {
        mixed.game_mut().state.players[0]
            .mana_pool
            .add(&Mana::restricted_to(
                ManaRestriction::spell(names.iter().copied()),
                Mana::parse(&"R".repeat(amount)).mana,
            ));
    }
    let mut nested = position(&["Prophetic Prism"]);
    nested
        .game_mut()
        .create_token("Treasure", PlayerId(0), false)
        .unwrap();
    let cases = [
        (
            "fixed",
            position(&["Mountain", "Mountain", "Mountain"]),
            "2R",
            "3R",
        ),
        ("restrictions", mixed, "2R", "4R"),
        (
            "shared_taps",
            position(&[
                "Supportive Parents",
                "Gray Ogre",
                "Gray Ogre",
                "Llanowar Elves",
            ]),
            "RR",
            "RRR",
        ),
        ("prism_treasure", nested, "R", "RR"),
        (
            "variable",
            position(&["White Lotus Tile", "Gray Ogre", "Gray Ogre", "Mountain"]),
            "RRR",
            "RRRR",
        ),
    ];
    for (name, mut s, payable, unpayable) in cases {
        if std::env::args().nth(1).as_deref() == Some("--trace") {
            trace(name, s, payable);
            continue;
        }
        for (label, text, expected) in [("payable", payable, true), ("unpayable", unpayable, false)]
        {
            let mut cost = ManaCost::parse(text);
            cost.purpose = SpendingContext::spell(["Citizen", "Elf"]);
            let mut times = Vec::new();
            let mut counters = Vec::new();
            let mut remainders = Vec::new();
            for _ in 0..10 {
                measurement::take();
                let start = Instant::now();
                let remainder = black_box(measurement::payment(s.game(), PlayerId(0), &cost));
                times.push(start.elapsed().as_secs_f64() * 1e6);
                counters.push(measurement::take());
                assert_eq!(remainder.is_some(), expected, "{name}/{label}");
                remainders.push(remainder);
            }
            println!(
                "{}",
                serde_json::json!({"case":name,"obligation":label,"cost":text,"microseconds":times,"counters":counters,"remainders":remainders})
            );
        }
        let mut times = Vec::new();
        let mut counters = Vec::new();
        for _ in 0..10 {
            measurement::take();
            let start = Instant::now();
            black_box(s.game_mut().scenario_refresh_priority().unwrap());
            times.push(start.elapsed().as_secs_f64() * 1e6);
            counters.push(measurement::take());
        }
        println!(
            "{}",
            serde_json::json!({"case":name,"operation":"offers","microseconds":times,"counters":counters})
        );
    }
}
