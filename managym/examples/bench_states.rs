//! Constructed game workloads through the same measurement layers. Recipes use
//! catalog cards; checkpoints are reached by ordinary actions with trivial
//! skipping enabled. Setup and restoration are outside each timed operation.
//! JSON lines retain deterministic work beside the median of three timings.
//! `--allocations` instead counts one operation per row, without timing it;
//! setup and forks remain outside the counted region.
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        env::Env,
        observation::Observation,
        observation_encoder::{encode, encode_into, ObservationEncoderConfig},
        player_policy::PlayerPolicy,
        vector_env::VectorEnv,
    },
    benchmark_allocations, benchmark_layers, benchmark_payments,
    flow::event::GameEvent,
    state::{
        game_object::PlayerId,
        permanent::CounterKind,
        player::{GameSetup, ManaFloats, PlayerConfig},
        zone::ZoneType,
    },
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

const STATES: &[(&str, &str)] = &[
    ("duplicate-lands", "Forest + Airship Engine Room; interchangeable sources, unpayable Sokka"),
    ("herbalist", "Forest + Hermitic Herbalist; modal and Lesson-only output"),
    ("paid-filter", "Forest + Prophetic Prism; nested unpayable payment"),
    ("counted-tap", "Forest, Supportive Parents, Compassionate Healer; counted tap scans stop once a single use is funded. Residual rollout replanning spans changing tapped resources and saved callers; sharing answers across steps needs a separate cache/invalidation design"),
    ("waterbend-x", "Forest, Katara, Bender's Waterskin, Herbalist; one payment per X"),
    ("duplicate-creatures", "Gray Ogre on both sides; attack, block and damage checkpoints"),
    ("duplicate-tokens", "Ally tokens on both sides; observation and combat rows"),
    ("diverse-mana", "cycle of seven modal/fixed/filter/restricted sources; diversity saturates at seven. Avoids counted-tap scans on ordinary activations; residual nested-filter branches need an exact bound accounting for upfront spending purposes and saved outputs"),
    ("restricted-filter", "Tea Shop, Hideout, Herbalist + Prism; expensive nested filters; defaults 2,3,4. Repeats three restrictions. Residual branch pruning must keep restricted mana that can fund a filter but not the parent; subtracting fees from gross output is not a safe bound"),
    ("distinct-restrictions", "2/3/4 pairwise-distinct restrictions + paid Prism: Powerstone, Tea Shop, Hideout, Herbalist. Catalog maximum four; cannot construct arbitrary N or claim 2^(N+1) growth. Nested filters retain the same pruning limitation as restricted-filter"),
    ("diverse-statics", "White Lotus Reinforcements, Earth King's Lieutenant, Kyoshi Warriors, Badgermole Cub; diversity saturates at four"),
    ("attachments", "Gray Ogre + Kyoshi Battle Fan + Avatar Destiny, attached, with counters"),
    ("triggers", "many Airbender Ascensions; Ally enters, ordering checkpoint"),
    ("zones", "large hands/graveyards and a stack of Pillar Launch spells; target choices"),
    ("mixed", "Gray Ogre + cycle of statics and mana sources on both sides; avoids ordinary-activation tap scans. Residual many-small-plan rollout cost needs cross-step payment caching keyed by resources, pools, life and saved obligations; not a per-plan explosion"),
    ("choices", "Azula Always Lies modes and targets; Joo Dee chosen sacrifice among many creatures"),
    ("names", "The Rise of Sozin chapter II; catalog-wide bounded naming groups beside a large graveyard"),
    ("convoke-duplicates", "Guardian of Vitu-Ghazi + Appeal to Eirdu; N other Guardians and N Forests. Identical sources still yield one direct-tap offer per creature"),
    ("convoke-diverse", "Guardian of Vitu-Ghazi + Appeal to Eirdu; N creatures cycling eight color mixes (two are mana sources) and N lands cycling four kinds. The payment step grows with N squared and a search reply with N cubed: a step asks a number of payment plans that grows with N, each over the whole board, and search tries one tap per source. One payment model answers every tap offer of a decision"),
    ("delve-graveyard", "Teval, Sire of Seven Deaths, three lands and N graveyard cards cycling eight names; one exile offer per card, no payment search"),
    ("improvise-artifacts", "Ironheart, one Island and N artifacts cycling Treasure, Prism, Waterskin, Scarecrow Guide and Battle Fan; the payment step and search grow as in convoke-diverse, beside one plan per mana ability"),
    ("float-origins", "diverse-mana with mana_floats = all; the floated checkpoint has every mana ability used at priority, so the pool holds one origin per unit and a clone copies each unit's class"),
    ("crew-saddle", "Cultivator's Caravan, Alacrian Jaguar and Lumen-Class Frigate with N creatures cycling eight names; crew, saddle and station tap choices, one offer per creature"),
    ("bargain-forage", "Archon's Glory + Feed the Cycle; N artifacts and tokens cycling five kinds and N graveyard cards; bargain's sacrifice and forage's exile are single picks among N, never subsets"),
];
const LAYERS: &[&str] = &[
    "actions",
    "step:cast",
    "step:activate",
    "step:attack",
    "step:block",
    "step:damage",
    "step:choice",
    "characteristics",
    "triggers",
    "observe",
    "vector",
    "clone",
    "search",
];
const MANA: &[&str] = &[
    "Forest",
    "Island",
    "Airship Engine Room",
    "Hermitic Herbalist",
    "Prophetic Prism",
    "Jasmine Dragon Tea Shop",
    "White Lotus Hideout",
];
const CREATURES: &[&str] = &[
    "White Lotus Reinforcements",
    "Kyoshi Warriors",
    "Badgermole Cub",
    "Hermitic Herbalist",
    "Gray Ogre",
    "Earth King's Lieutenant",
    "Supportive Parents",
    "Tiger-Seal",
];
const ARTIFACTS: &[&str] = &[
    "token.treasure",
    "Prophetic Prism",
    "Bender's Waterskin",
    "Scarecrow Guide",
    "Kyoshi Battle Fan",
];
const GRAVEYARD: &[&str] = &[
    "Gray Ogre",
    "Forest",
    "Pillar Launch",
    "Badgermole Cub",
    "Island",
    "Kyoshi Warriors",
    "Prophetic Prism",
    "Tiger-Seal",
];
const STATICS: &[&str] = &[
    "White Lotus Reinforcements",
    "Earth King's Lieutenant",
    "Kyoshi Warriors",
    "Badgermole Cub",
];

#[derive(Clone, Debug, Default, Serialize)]
struct Seat {
    battlefield: Vec<String>,
    hand: Vec<String>,
    graveyard: Vec<String>,
    tokens: Vec<String>,
}
impl Seat {
    fn add(&mut self, name: &str, count: usize) {
        self.battlefield
            .extend(std::iter::repeat_n(name.to_string(), count));
    }
    /// `size` permanents cycling `names`; a `token.` name is a token key.
    fn cycle(&mut self, names: &[&str], size: usize) {
        for &name in names.iter().cycle().take(size) {
            if name.starts_with("token.") {
                self.tokens.push(name.into());
            } else {
                self.add(name, 1);
            }
        }
    }
    /// `size` graveyard cards cycling `GRAVEYARD`.
    fn fill_graveyard(&mut self, size: usize) {
        let cards = GRAVEYARD.iter().cycle().take(size);
        self.graveyard.extend(cards.map(|name| name.to_string()));
    }
}
#[derive(Clone, Debug, Serialize)]
struct Recipe {
    state: String,
    size: usize,
    seats: [Seat; 2],
}
fn recipe(name: &str, size: usize) -> Recipe {
    let mut seats = [Seat::default(), Seat::default()];
    match name {
        "duplicate-lands" | "herbalist" | "paid-filter" | "counted-tap" | "waterbend-x" => {
            let seat = &mut seats[0];
            seat.add("Forest", size);
            seat.hand.push("Sokka, Bold Boomeranger".into());
            match name {
                "duplicate-lands" => seat.add("Airship Engine Room", 1),
                "herbalist" => {
                    seat.add("Hermitic Herbalist", 1);
                    seat.hand.extend([
                        "Crashing Wave".into(),
                        "Waterbending Lesson".into(),
                        "Dragonfly Swarm".into(),
                    ]);
                }
                "paid-filter" => seat.add("Prophetic Prism", 1),
                "counted-tap" => {
                    seat.add("Supportive Parents", 1);
                    seat.add("Compassionate Healer", size);
                }
                _ => {
                    for card in [
                        "Katara, Water Tribe's Hope",
                        "Bender's Waterskin",
                        "Hermitic Herbalist",
                    ] {
                        seat.add(card, 1);
                    }
                }
            }
        }
        "convoke-duplicates" | "convoke-diverse" => {
            let seat = &mut seats[0];
            if name == "convoke-duplicates" {
                seat.add("Guardian of Vitu-Ghazi", size);
                seat.add("Forest", size);
            } else {
                seat.cycle(CREATURES, size);
                seat.cycle(&["Forest", "Plains", "Airship Engine Room", "Island"], size);
            }
            seat.hand
                .extend(["Guardian of Vitu-Ghazi".into(), "Appeal to Eirdu".into()]);
        }
        "delve-graveyard" => {
            let seat = &mut seats[0];
            for card in ["Teval, Arbiter of Virtue", "Swamp", "Forest", "Island"] {
                seat.add(card, 1);
            }
            seat.hand.push("Sire of Seven Deaths".into());
            seat.fill_graveyard(size);
        }
        "improvise-artifacts" => {
            let seat = &mut seats[0];
            seat.add("Island", 1);
            seat.cycle(ARTIFACTS, size);
            seat.hand.push("Ironheart, Clever Champion".into());
        }
        "crew-saddle" => {
            let seat = &mut seats[0];
            for card in [
                "Cultivator's Caravan",
                "Alacrian Jaguar",
                "Lumen-Class Frigate",
            ] {
                seat.add(card, 1);
            }
            seat.cycle(CREATURES, size);
        }
        "bargain-forage" => {
            seats[1].add("Gray Ogre", 1);
            let seat = &mut seats[0];
            for card in ["Gray Ogre", "Plains", "Swamp", "Swamp", "Swamp"] {
                seat.add(card, 1);
            }
            seat.cycle(
                &[
                    "token.food",
                    "token.treasure",
                    "Prophetic Prism",
                    "Kyoshi Battle Fan",
                    "token.ally",
                ],
                size,
            );
            seat.hand
                .extend(["Archon's Glory".into(), "Feed the Cycle".into()]);
            seat.fill_graveyard(size);
        }
        "diverse-mana" | "restricted-filter" | "distinct-restrictions" | "float-origins" => {
            let names = if matches!(name, "diverse-mana" | "float-origins") {
                MANA
            } else {
                &[
                    "Jasmine Dragon Tea Shop",
                    "White Lotus Hideout",
                    "Hermitic Herbalist",
                ]
            };
            let count = if name == "distinct-restrictions" {
                assert!(
                    (2..=4).contains(&size),
                    "distinct-restrictions supports sizes 2..4: the catalog has four restrictions"
                );
                seats[0].tokens.push("token.powerstone".into());
                size - 1
            } else {
                size
            };
            for i in 0..count {
                seats[0].add(names[i % names.len()], 1);
            }
            seats[0].add("Prophetic Prism", 1);
            seats[0].hand.extend([
                "Sokka, Bold Boomeranger".into(),
                "Dragonfly Swarm".into(),
                "Waterbending Lesson".into(),
            ]);
        }
        "duplicate-creatures"
        | "duplicate-tokens"
        | "diverse-statics"
        | "attachments"
        | "triggers"
        | "zones"
        | "mixed" => {
            for seat in &mut seats {
                match name {
                    "duplicate-tokens" => seat.tokens = vec!["token.ally".into(); size],
                    "diverse-statics" => {
                        for i in 0..size {
                            seat.add(STATICS[i % STATICS.len()], 1);
                        }
                    }
                    "triggers" => {
                        seat.add("Airbender Ascension", size);
                        seat.add("Gray Ogre", 1);
                    }
                    "mixed" => {
                        for i in 0..size {
                            seat.add(
                                if i % 2 == 0 {
                                    "Gray Ogre"
                                } else {
                                    STATICS[(i / 2) % STATICS.len()]
                                },
                                1,
                            );
                            seat.add(MANA[i % MANA.len()], 1);
                        }
                    }
                    "zones" => {
                        seat.add("Gray Ogre", size);
                        seat.add("Forest", size);
                        seat.hand
                            .extend(std::iter::repeat_n("Pillar Launch".into(), size));
                        seat.graveyard
                            .extend(std::iter::repeat_n("Gray Ogre".into(), size));
                    }
                    _ => seat.add("Gray Ogre", size),
                }
                if name == "attachments" {
                    seat.add("Kyoshi Battle Fan", size);
                    seat.add("Avatar Destiny", size);
                }
                seat.add("Forest", 2);
                seat.hand.push("Pillar Launch".into());
            }
        }
        "choices" | "names" => {
            for seat in &mut seats {
                seat.add("Gray Ogre", size);
                seat.add("Swamp", 4);
                seat.graveyard
                    .extend(std::iter::repeat_n("Gray Ogre".into(), size));
            }
            if name == "names" {
                seats[0].add("The Rise of Sozin", 1);
            } else {
                seats[0].add("Joo Dee, One of Many", 1);
                seats[0].hand.push("Azula Always Lies".into());
            }
        }
        _ => panic!("unknown state {name}; use --list"),
    }
    Recipe {
        state: name.into(),
        size,
        seats,
    }
}
fn build(recipe: &Recipe) -> Env {
    let players: Vec<_> = recipe
        .seats
        .iter()
        .enumerate()
        .map(|(i, seat)| {
            let mut deck = BTreeMap::from([("Plains".into(), 24)]);
            for name in seat
                .battlefield
                .iter()
                .chain(&seat.hand)
                .chain(&seat.graveyard)
            {
                *deck.entry(name.clone()).or_default() += 1;
            }
            PlayerConfig::new(format!("seat {i}"), deck)
        })
        .collect();
    let mut setup = GameSetup::from(players);
    if recipe.state == "float-origins" {
        setup.mana_floats[0] = ManaFloats::All;
    }
    let mut env = Env::new(7, true, false, false);
    env.reset(setup).unwrap();
    for (i, seat) in recipe.seats.iter().enumerate() {
        env.scenario_clear_hand(i).unwrap();
        for name in &seat.battlefield {
            env.scenario_force_battlefield(i, name, true).unwrap();
        }
        for name in &seat.hand {
            env.scenario_force_card_in_hand(i, name).unwrap();
        }
        for name in &seat.graveyard {
            env.scenario_force_card_in_hand(i, name).unwrap();
            let game = env.benchmark_game_mut();
            let card = *game
                .state
                .zones
                .zone_objects(ZoneType::Hand, PlayerId(i))
                .last()
                .unwrap();
            let state = &mut *game.state;
            state
                .zones
                .move_object(&state.objects, card, PlayerId(i), ZoneType::Graveyard);
        }
        for token in &seat.tokens {
            let game = env.benchmark_game_mut();
            let id = game.create_token(token, PlayerId(i), false).unwrap();
            game.state.permanents[id].as_mut().unwrap().summoning_sick = false;
        }
    }
    if recipe.state == "attachments" {
        let game = env.benchmark_game_mut();
        for i in 0..2 {
            let creatures: Vec<_> = game
                .state
                .permanents
                .iter()
                .flatten()
                .filter(|p| {
                    let card = &game.state[p.object];
                    card.owner == PlayerId(i) && card.name == "Gray Ogre"
                })
                .map(|p| p.object)
                .collect();
            let targets: Vec<_> = creatures
                .iter()
                .map(|card| game.state.live(*card).unwrap())
                .collect();
            let mut index = 0;
            let state = &mut *game.state;
            let roles: Vec<_> = state
                .permanents
                .iter()
                .map(|p| {
                    p.as_ref().map(|p| {
                        let object = &state[p.object];
                        (
                            object.owner,
                            object.name == "Gray Ogre",
                            object.name == "Forest",
                        )
                    })
                })
                .collect();
            for (p, role) in state.permanents.iter_mut().zip(roles) {
                let (Some(p), Some((owner, ogre, forest))) = (p.as_mut(), role) else {
                    continue;
                };
                if owner == PlayerId(i) {
                    if ogre {
                        p.change_counters(CounterKind::PlusOnePlusOne, 1);
                    } else if !forest {
                        p.attached_to = Some(targets[index % targets.len()]);
                        index += 1;
                    }
                }
            }
        }
    }
    if recipe.state == "names" {
        let game = env.benchmark_game_mut();
        let id = game
            .state
            .permanents
            .iter()
            .enumerate()
            .find_map(|(i, p)| {
                p.as_ref()
                    .filter(|p| game.state[p.object].name == "The Rise of Sozin")
                    .map(|_| i)
            })
            .unwrap();
        game.state.permanents[managym::state::game_object::PermanentId(id)]
            .as_mut()
            .unwrap()
            .change_counters(CounterKind::Lore, 1);
    }
    env.scenario_refresh().unwrap();
    env
}
fn actions(env: &mut Env) -> Vec<Action> {
    env.benchmark_game_mut()
        .current_action_space
        .as_ref()
        .unwrap()
        .actions
        .clone()
}
fn first_action(env: &mut Env, kind: &str) -> Option<usize> {
    nth_action(env, kind, 0)
}
fn nth_action(env: &mut Env, kind: &str, nth: usize) -> Option<usize> {
    let offered = actions(env);
    let matching = offered.iter().enumerate().filter(|(_, a)| match kind {
        "cast" => matches!(a, Action::CastSpell { .. }),
        "activate" => matches!(
            a,
            Action::ActivateAbility { .. } | Action::ActivateCardAbility { .. }
        ),
        "attack" => matches!(a, Action::DeclareAttacker { attack: true, .. }),
        "block" => matches!(
            a,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        ),
        "damage" => matches!(a, Action::PassPriority { .. }),
        "choice" => !matches!(
            a,
            Action::PassPriority { .. }
                | Action::CastSpell { .. }
                | Action::PlayLand { .. }
                | Action::ActivateAbility { .. }
                | Action::ActivateCardAbility { .. }
                | Action::DeclareAttacker { .. }
                | Action::DeclareBlocker { .. }
        ),
        _ => false,
    });
    matching.map(|(index, _)| index).nth(nth)
}
/// The decision a cost choice is made at: each step of `path` takes the nth
/// offered action of a kind. Fixture preparation only.
fn follow(root: &Env, path: &[(&str, usize)]) -> Env {
    let mut env = root.fork().unwrap();
    for &(kind, nth) in path {
        let index = nth_action(&mut env, kind, nth)
            .unwrap_or_else(|| panic!("no {kind} action {nth} on the way to a cost choice"));
        env.step(index as i64).unwrap();
    }
    env
}
/// Tap every mana source at priority, taking the first offer of each choice
/// an ability asks, so the pool holds every unit the board can float.
fn float_everything(root: &Env) -> Env {
    let mut env = root.fork().unwrap();
    for _ in 0..256 {
        let index = if env.action_space_kind() == Some(ActionSpaceKind::Priority) {
            let offered = actions(&mut env);
            let game = env.benchmark_game_mut();
            let source = offered.iter().position(|a| match a {
                Action::TapForMana { .. } => true,
                Action::ActivateAbility {
                    permanent,
                    ability_index,
                    ..
                } => game.characteristics(*permanent).activated_abilities[*ability_index]
                    .mana_choices()
                    .is_some(),
                _ => false,
            });
            match source {
                Some(index) => index,
                None => return env,
            }
        } else {
            0
        };
        env.step(index as i64).unwrap();
    }
    panic!("mana sources were still offered after 256 decisions")
}
/// Reach combat by passing priority and declaring every attacker. Bound the
/// preparation trace, not the rules search; error if it never reaches combat.
fn combat(root: &Env, block: bool) -> Option<Env> {
    let mut env = root.fork().unwrap();
    for _ in 0..256 {
        if env.is_game_over() {
            return None;
        }
        let kind = env.action_space_kind().unwrap();
        if kind
            == if block {
                ActionSpaceKind::DeclareBlocker
            } else {
                ActionSpaceKind::DeclareAttacker
            }
        {
            return Some(env);
        }
        let options = actions(&mut env);
        let index = if kind == ActionSpaceKind::DeclareAttacker {
            options
                .iter()
                .position(|a| matches!(a, Action::DeclareAttacker { attack: true, .. }))
        } else {
            options
                .iter()
                .position(|a| matches!(a, Action::PassPriority { .. }))
        };
        env.step(index? as i64).unwrap();
        if env.is_game_over() || env.benchmark_game_mut().state.turn.turn_number > 1 {
            return None;
        }
    }
    panic!("combat preparation exceeded 256 decisions")
}
/// Finish declarations and stop one pass before combat damage. Both hands
/// retain an instant, so ordinary trivial skipping leaves this priority open.
fn damage(root: &Env) -> Option<Env> {
    use managym::flow::turn::StepKind;
    let mut env = combat(root, true)?;
    for _ in 0..256 {
        if env.action_space_kind() != Some(ActionSpaceKind::DeclareBlocker) {
            break;
        }
        let index = first_action(&mut env, "block").unwrap_or(0);
        env.step(index as i64).unwrap();
        if env.is_game_over() {
            return None;
        }
    }
    if env.action_space_kind() != Some(ActionSpaceKind::Priority) {
        return None;
    }
    if env.benchmark_game_mut().state.priority.consecutive_passes == 0 {
        let index = first_action(&mut env, "damage").unwrap();
        env.step(index as i64).unwrap();
        if env.is_game_over() {
            return None;
        }
    }
    (env.benchmark_game_mut().state.turn.current_step_kind() == StepKind::DeclareBlockers)
        .then_some(env)
}

/// Resolve by passing priority and taking the first offered continuation.
/// This is fixture preparation only, never an automation policy for play.
fn reach(root: &Env, label: &str) -> Env {
    use managym::flow::decision::Decision;
    let mut env = root.fork().unwrap();
    for _ in 0..256 {
        let game = env.benchmark_game_mut();
        let decision = game.state.suspended_decision.as_ref().map(|p| &p.decision);
        let reached = match label {
            "name" => matches!(decision, Some(Decision::NameCard { .. })),
            "object" => matches!(decision, Some(Decision::ChooseObject { .. })),
            "order" => env.action_space_kind() == Some(ActionSpaceKind::OrderTriggers),
            _ => false,
        };
        if reached {
            return env;
        }
        let options = actions(&mut env);
        let index = options
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .or_else(|| {
                options
                    .iter()
                    .position(|a| matches!(a, Action::DeclareAttacker { attack: false, .. }))
            })
            .unwrap_or(0);
        env.step(index as i64).unwrap();
        assert!(!env.is_game_over(), "game ended before {label}");
    }
    panic!("fixture never reached {label} in 256 decisions")
}

fn checkpoints(root: &Env, recipe: &Recipe) -> Vec<(String, Env)> {
    let mut result = vec![("priority".into(), root.fork().unwrap())];
    if recipe.state == "names" {
        return vec![("name".into(), reach(root, "name"))];
    }
    if recipe.state == "triggers" {
        let mut entered = root.fork().unwrap();
        entered
            .benchmark_game_mut()
            .create_token("token.ally", PlayerId(0), false)
            .unwrap();
        result.push(("order".into(), reach(&entered, "order")));
    }
    let cost_choices: &[(&str, &[(&str, usize)])] = match recipe.state.as_str() {
        "crew-saddle" => &[
            ("crew", &[("activate", 0)]),
            ("saddle", &[("activate", 1)]),
            ("station", &[("activate", 2)]),
        ],
        "bargain-forage" => &[
            ("bargain", &[("cast", 0), ("choice", 0), ("choice", 0)]),
            ("forage", &[("cast", 1), ("choice", 0)]),
        ],
        _ => &[],
    };
    for (label, path) in cost_choices {
        result.push(((*label).into(), follow(root, path)));
    }
    if recipe.state == "float-origins" {
        result.push(("floated".into(), float_everything(root)));
    }
    if recipe.state == "choices" {
        let mut activated = root.fork().unwrap();
        let index = first_action(&mut activated, "activate").unwrap();
        activated.step(index as i64).unwrap();
        result.push(("object".into(), reach(&activated, "object")));
    }
    if recipe.state == "zones" {
        let mut stack = root.fork().unwrap();
        for _ in 0..recipe.size {
            let index =
                first_action(&mut stack, "cast").expect("stack recipe has a payable instant");
            stack.step(index as i64).unwrap();
            while stack.action_space_kind() != Some(ActionSpaceKind::Priority) {
                stack.step(0).unwrap();
            }
        }
        result.push(("stack".into(), stack));
    }
    for (label, block) in [("attack", false), ("block", true)] {
        if let Some(env) = combat(root, block) {
            result.push((label.into(), env));
        }
    }
    if let Some(env) = damage(root) {
        result.push(("damage".into(), env));
    }
    let mut env = root.fork().unwrap();
    if let Some(index) = first_action(&mut env, "cast") {
        env.step(index as i64).unwrap();
        if !env.is_game_over() && env.action_space_kind() != Some(ActionSpaceKind::Priority) {
            result.push(("cast-choice".into(), env));
        }
    }
    result
}
fn config(obs: &Observation) -> ObservationEncoderConfig {
    ObservationEncoderConfig {
        max_cards_per_player: obs.agent_cards.len().max(obs.opponent_cards.len()).max(60),
        max_permanents_per_player: obs
            .agent_permanents
            .len()
            .max(obs.opponent_permanents.len())
            .max(40),
        max_actions: obs.action_space.actions.len().max(512),
        max_facts: 4096,
        max_events: obs.recent_events.len().max(32),
        turn_state: true,
        ..Default::default()
    }
}
fn surface(env: &mut Env) -> Value {
    json!({"actions": actions(env), "state": env.state_digest().unwrap()})
}
// Hold computed results until the timer stops; JSON formatting is not part
// of action enumeration, search, or observation construction.
#[derive(Serialize)]
#[serde(untagged)]
enum Output {
    Actions {
        actions: Vec<Action>,
    },
    Views {
        views: usize,
    },
    Triggers {
        pending: usize,
    },
    Observation {
        cards: usize,
        permanents: usize,
        actions: usize,
    },
    Steps {
        steps: usize,
    },
    Clone {},
    Search {
        simulations: u64,
        cap_hits: u64,
        scores: Vec<f64>,
    },
}

fn run(
    recipe: &Recipe,
    checkpoint: &str,
    root: &Env,
    layers: &[String],
    allocations: bool,
    digest: &mut blake3::Hasher,
) {
    let actor = root.current_agent_index().unwrap();
    let obs = root.observation_for_player(actor).unwrap();
    let config = config(&obs);
    for &layer in LAYERS {
        if !layers.is_empty()
            && !layers
                .iter()
                .any(|v| v == layer || (v == "step" && layer.starts_with("step:")))
        {
            continue;
        }
        if layer == "actions" && root.action_space_kind() != Some(ActionSpaceKind::Priority) {
            continue;
        }
        if layer == "step:damage" && checkpoint != "damage" {
            continue;
        }
        let mut probe = root.fork().unwrap();
        let index = layer
            .strip_prefix("step:")
            .and_then(|kind| first_action(&mut probe, kind));
        if layer.starts_with("step:") && index.is_none() {
            continue;
        }
        let mut times = Vec::new();
        let mut work = Value::Null;
        let mut clone_bytes = None;
        let samples = if allocations { 1 } else { 3 };
        let mut traffic = None;
        for sample in 0..samples {
            let mut env = root.fork().unwrap();
            let mut vector = VectorEnv::new(1, 7, true, [PlayerPolicy::Agent; 2]);
            let buffer = std::cell::RefCell::new(encode(&obs, &config).unwrap());
            if layer == "vector" {
                vector.reset_all(root.player_configs().to_vec()).unwrap();
                vector.benchmark_load_root(root).unwrap();
            }
            // Cold characteristics attribution: a conservative state write
            // invalidates the cloned caches, without changing any value.
            if layer == "characteristics" {
                let _ = &mut *env.benchmark_game_mut().state;
            }
            // Cold payment attribution: preparation already answered these
            // questions, and the fork carries their cached answers.
            if layer == "actions" {
                benchmark_payments::clear_cache(env.benchmark_game_mut());
            }
            benchmark_layers::reset();
            benchmark_payments::reset();
            if allocations {
                benchmark_allocations::start();
            }
            let start = Instant::now();
            let output = match layer {
                "actions" => Output::Actions {
                    actions: benchmark_layers::actions(env.benchmark_game_mut(), PlayerId(actor)),
                },
                "characteristics" => Output::Views {
                    views: benchmark_layers::characteristics(env.benchmark_game_mut()),
                },
                "triggers" => {
                    let game = env.benchmark_game_mut();
                    // Feed one controlled entry event to the ordinary indexed
                    // matcher. On the trigger recipe every Ascension sees it.
                    let source = game
                        .state
                        .permanents
                        .iter()
                        .flatten()
                        .find(|p| game.state[p.object].name == "Gray Ogre")
                        .or_else(|| game.state.permanents.iter().flatten().next())
                        .map(|p| p.object);
                    if let Some(card) = source {
                        benchmark_layers::triggers(
                            game,
                            GameEvent::CardMoved {
                                object_id: game.state.live(card).expect("battlefield appearance"),
                                departed: None,
                                from: Some(ZoneType::Hand),
                                to: ZoneType::Battlefield,
                                controller: PlayerId(actor),
                                face_down: false,
                            },
                        );
                    }
                    Output::Triggers {
                        pending: game.state.pending_triggers.len(),
                    }
                }
                "observe" => {
                    let observation = env.observation_for_player(actor).unwrap();
                    encode_into(&observation, &config, buffer.borrow_mut().buffers_mut()).unwrap();
                    black_box(&buffer);
                    Output::Observation {
                        cards: observation.agent_cards.len() + observation.opponent_cards.len(),
                        permanents: observation.agent_permanents.len()
                            + observation.opponent_permanents.len(),
                        actions: observation.action_space.actions.len(),
                    }
                }
                "clone" => {
                    black_box(env.fork().unwrap());
                    Output::Clone {}
                }
                "vector" => {
                    let result = vector
                        .step_isolated_into(&[0], &[true], false, |_, obs, _, _, _| {
                            encode_into(obs, &config, buffer.borrow_mut().buffers_mut())?;
                            Ok(())
                        })
                        .unwrap();
                    if result[0].contains_key("game_error") {
                        panic!("vector failed: {:?}", result[0]);
                    }
                    Output::Steps { steps: 1 }
                }
                "search" => {
                    let scores = env.flat_mc_scores(4, 1, 23, 4).unwrap();
                    Output::Search {
                        simulations: scores.simulations,
                        cap_hits: scores.cap_hits,
                        scores: scores.scores,
                    }
                }
                _ => {
                    env.step(index.unwrap() as i64).unwrap();
                    Output::Steps { steps: 1 }
                }
            };
            if allocations {
                traffic = Some(benchmark_allocations::stop());
            } else {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let output = serde_json::to_value(output).unwrap();
            let payment = benchmark_payments::snapshot();
            let counters = benchmark_layers::snapshot();
            work = json!({"plans":payment.plans,"calls":payment.calls,"states":payment.states,"max_plan_calls":payment.max_plan_calls,"cache_hits":payment.cache_hits,"cache_misses":payment.cache_misses,"witness_digest":payment.witness_digest,
                "boards":counters.boards,"source_plans":counters.source_plans,"comparisons":counters.comparisons,"trigger_sources":counters.trigger_sources,"triggers_found":counters.triggers_found});
            if sample == 0 {
                digest.update(
                    &serde_json::to_vec(&(
                        recipe.state.as_str(),
                        recipe.size,
                        checkpoint,
                        layer,
                        &output,
                        payment.witness_digest,
                    ))
                    .unwrap(),
                );
                if layer.starts_with("step:") {
                    digest.update(
                        &serde_json::to_vec(&(
                            env.is_game_over(),
                            env.winner_index(),
                            env.state_digest().unwrap(),
                        ))
                        .unwrap(),
                    );
                    if !env.is_game_over() {
                        digest.update(&serde_json::to_vec(&surface(&mut env)).unwrap());
                    }
                }
                // Clone bytes are allocation traffic, measured separately.
                if layer == "clone" {
                    benchmark_allocations::start();
                    black_box(root.fork().unwrap());
                    let (_, bytes) = benchmark_allocations::stop();
                    clone_bytes = Some(bytes);
                }
            }
            if sample == samples - 1 {
                work["output"] = match layer {
                    "search" => {
                        let mut summary = output.clone();
                        summary.as_object_mut().unwrap().remove("scores");
                        summary
                    }
                    "actions" => json!({"actions": output["actions"].as_array().unwrap().len()}),
                    _ => output,
                };
            }
        }
        if let Some(bytes) = clone_bytes {
            work["allocated_bytes"] = json!(bytes);
        }
        let mut row = json!({"state":recipe.state,"size":recipe.size,"checkpoint":checkpoint,"layer":layer,"work":work});
        if let Some((count, bytes)) = traffic {
            row["allocations"] = json!(count);
            row["allocated_bytes"] = json!(bytes);
        } else {
            times.sort_by(f64::total_cmp);
            row["ms"] = json!(times[1]);
        }
        println!("{row}");
    }
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut names = Vec::new();
    let mut sizes: Option<Vec<usize>> = None;
    let mut layers = Vec::new();
    let mut recipes = false;
    let mut allocations = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => {
                for (name, description) in STATES {
                    println!("{name}: {description}");
                }
                return;
            }
            "--recipes" => recipes = true,
            "--allocations" => allocations = true,
            "--sizes" | "--layers" => {
                let option = &args[i];
                i += 1;
                let value = args.get(i).expect("option requires comma-separated values");
                if option == "--sizes" {
                    let given: Vec<usize> = value
                        .split(',')
                        .map(|s| s.parse().expect("positive size"))
                        .collect();
                    assert!(
                        given.iter().all(|&s| s > 0 && s <= 64),
                        "sizes must be 1..64"
                    );
                    sizes = Some(given);
                } else {
                    layers = value.split(',').map(str::to_string).collect();
                    assert!(
                        layers
                            .iter()
                            .all(|s| s == "step" || LAYERS.contains(&s.as_str())),
                        "unknown layer in {value}; layers are step, {}",
                        LAYERS.join(", ")
                    );
                }
            }
            name => {
                assert!(
                    STATES.iter().any(|(n, _)| *n == name),
                    "unknown state {name}; --list names the states"
                );
                names.push(args[i].clone());
            }
        }
        i += 1;
    }
    let mut digest = blake3::Hasher::new();
    for &(name, _) in STATES {
        if !names.is_empty() && !names.iter().any(|n| n == name) {
            continue;
        }
        let sizes = match (name, &sizes) {
            (_, Some(given)) => given.clone(),
            ("restricted-filter" | "distinct-restrictions", None) => vec![2, 3, 4],
            (_, None) => vec![4, 8, 16],
        };
        for size in sizes {
            let recipe = recipe(name, size);
            let root = build(&recipe);
            for (checkpoint, mut env) in checkpoints(&root, &recipe) {
                if recipes {
                    let actor = env.current_agent_index().unwrap();
                    let observation = env.observation_for_player(actor).unwrap();
                    let capture = env.capture_decision(actor, 0).unwrap();
                    println!(
                        "{}",
                        json!({
                            "recipe": recipe,
                            "checkpoint": checkpoint,
                            "observation": observation.to_json(),
                            "continuation": serde_json::to_string(&observation.continuation).unwrap(),
                            "capture": serde_json::to_string(&capture).unwrap(),
                        })
                    );
                } else {
                    digest.update(&serde_json::to_vec(&surface(&mut env)).unwrap());
                    run(
                        &recipe,
                        &checkpoint,
                        &env,
                        &layers,
                        allocations,
                        &mut digest,
                    );
                }
            }
        }
    }
    if !recipes {
        println!("{}", json!({"digest":digest.finalize().to_hex().as_str()}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recipes_reach_large_boards_and_real_combat_choices() {
        for &(name, _) in STATES {
            let recipe = recipe(name, 4);
            let root = build(&recipe);
            let mut again = build(&recipe);
            let mut first = root.fork().unwrap();
            assert_eq!(surface(&mut first), surface(&mut again), "{name}");
            let points = checkpoints(&root, &recipe);
            if name == "convoke-duplicates" {
                let (_, payment) = points
                    .iter()
                    .find(|(label, _)| label == "cast-choice")
                    .expect("duplicate convoke reaches a payment choice");
                let mut env = payment.fork().unwrap();
                let taps = actions(&mut env)
                    .into_iter()
                    .filter(|action| matches!(action, Action::WaterbendTap { .. }))
                    .count();
                assert_eq!(taps, recipe.size);
            }
            if name == "duplicate-creatures" {
                assert!(points.iter().any(|(label, _)| label == "attack"));
                assert!(points.iter().any(|(label, _)| label == "block"));
                let (_, damage) = points
                    .iter()
                    .find(|(label, _)| label == "damage")
                    .expect("combat damage checkpoint");
                let mut env = damage.fork().unwrap();
                let before = env
                    .observation_for_player(env.current_agent_index().unwrap())
                    .unwrap();
                let index = first_action(&mut env, "damage").unwrap();
                let after = env.step(index as i64).unwrap().0;
                assert!(
                    after.game_over
                        || after.agent_permanents.len() + after.opponent_permanents.len()
                            < before.agent_permanents.len() + before.opponent_permanents.len()
                );
            }
        }
    }
}
