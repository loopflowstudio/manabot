//! The Academy curated matchup (UR Lessons versus GW Allies main decks) as
//! benchmark seats. Both engine benchmarks play exactly these lists so their
//! baselines describe one workload; change the lists only with the baseline.
use managym::state::player::PlayerConfig;
use std::collections::BTreeMap;

fn deck(cards: &[(&str, usize)]) -> BTreeMap<String, usize> {
    cards
        .iter()
        .map(|&(name, n)| (name.to_string(), n))
        .collect()
}

pub fn players() -> Vec<PlayerConfig> {
    let lessons = deck(&[
        ("Island", 9),
        ("Mountain", 8),
        ("Tiger-Seal", 2),
        ("Otter-Penguin", 2),
        ("Fire Nation Cadets", 2),
        ("First-Time Flyer", 2),
        ("Forecasting Fortune Teller", 1),
        ("Dragonfly Swarm", 1),
        ("Firebending Lesson", 4),
        ("Igneous Inspiration", 2),
        ("Pop Quiz", 2),
        ("Divide by Zero", 2),
        ("It'll Quench Ya!", 2),
        ("Accumulate Wisdom", 2),
    ]);
    let allies = deck(&[
        ("Plains", 9),
        ("Forest", 8),
        ("Water Tribe Rallier", 2),
        ("Invasion Reinforcements", 2),
        ("Compassionate Healer", 2),
        ("Earth Kingdom Jailer", 2),
        ("White Lotus Reinforcements", 2),
        ("Earth King's Lieutenant", 2),
        ("Kyoshi Warriors", 2),
        ("Badgermole Cub", 2),
        ("Suki, Kyoshi Warrior", 1),
        ("South Pole Voyager", 1),
        ("Allies at Last", 2),
        ("Yip Yip!", 1),
        ("Fancy Footwork", 2),
    ]);
    vec![
        PlayerConfig::new("lessons", lessons),
        PlayerConfig::new("allies", allies),
    ]
}

/// A bounded layer-rich position using the reference world's existing cards.
/// The old curated workload stays intact; these roots add overlapping Ally
/// anthems and type-dependent power to ordinary steps and complete search scores.
pub fn add_layer_board(env: &mut managym::agent::env::Env) {
    for name in [
        "White Lotus Reinforcements",
        "White Lotus Reinforcements",
        "Earth King's Lieutenant",
        "Kyoshi Warriors",
        "Suki, Kyoshi Warrior",
    ] {
        env.scenario_force_battlefield(1, name, true).unwrap();
    }
    for name in ["Tiger-Seal", "Otter-Penguin", "Fire Nation Cadets"] {
        env.scenario_force_battlefield(0, name, true).unwrap();
    }
    env.scenario_refresh().unwrap();
}
