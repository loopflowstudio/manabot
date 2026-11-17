// Full Elemental admission: exact rosters, complete native coherence and bounded games.
use managym::{
    agent::{
        action::{ActionSpaceKind, ActionType},
        env::Env,
    },
    cards::setups::authored_setup,
    Game,
};

#[test]
fn retained_professor_cast_prefix_can_enter_and_finish_mana_payment() {
    // Native failure retained by MTG-175: one Forest, off-color lands and
    // Prophetic Prism can pay 2GG, provided Prism preserves the Forest.
    let pack = authored_setup("elemental_world").unwrap();
    let configs = vec![
        pack.player_config("arena-seat-0", "elemental_wb").unwrap(),
        pack.player_config("arena-seat-1", "elemental_ug").unwrap(),
    ];
    let mut env = managym::agent::env::Env::new(2165384, true, false, false);
    env.reset(configs).unwrap();
    let prefix = [
        2, 2, 0, 0, 4, 3, 3, 5, 2, 0, 2, 2, 2, 6, 2, 3, 1, 2, 1, 0, 4, 0, 0, 8, 0, 0, 1, 4, 6, 2,
        2, 5, 1, 1,
    ];
    for action in prefix {
        let choices = old_choices(&mut env);
        env.step(choices[action]).expect("retained original choice");
    }
    let choices = old_choices(&mut env);
    env.step(choices[4])
        .expect("offered Professor cast enters payment");
    assert!(env
        .random_playout(2165384, 30_000)
        .expect("payment and subsequent offered actions execute")
        .is_some());
    assert!(env.is_game_over());
}

#[test]
fn every_pair_and_mirror_completes_on_the_catalog() {
    let semantic = authored_setup("optional_payments").unwrap();
    for (left, first) in semantic.decks.iter().enumerate() {
        for (right, second) in semantic.decks.iter().enumerate() {
            let configs = vec![
                semantic.player_config("left", &first.key).unwrap(),
                semantic.player_config("right", &second.key).unwrap(),
            ];
            let seed = 16400 + (left * 10 + right) as u64;
            let mut game = Game::new(configs, seed, true);
            let winner = game
                .random_playout(30_000, None)
                .unwrap_or_else(|e| panic!("{} vs {} seed {seed}: {e:?}", first.key, second.key));
            assert!(
                game.is_game_over(),
                "cap: {} vs {} seed {seed}",
                first.key,
                second.key
            );
            assert!(
                winner.is_some(),
                "no winner: {} vs {} seed {seed}",
                first.key,
                second.key
            );
        }
    }
}

/// Automatic completion is explicit at the new fixed-source decisions; old
/// mana-ability choices retain their indices after removing the added taps.
fn old_choices(env: &mut Env) -> Vec<i64> {
    loop {
        let actor = env.current_agent_index().expect("acting seat");
        let obs = env.observation_for_player(actor).unwrap();
        let sources = obs.action_space.action_space_type == ActionSpaceKind::PayOrNot
            && obs
                .action_space
                .actions
                .iter()
                .any(|a| a.action_type == ActionType::TapForCost);
        let choices: Vec<usize> = obs
            .action_space
            .actions
            .iter()
            .enumerate()
            .filter(|(_, a)| !sources || a.action_type != ActionType::TapForCost)
            .map(|(i, _)| i)
            .collect();
        if sources
            && choices.len() == 1
            && obs.action_space.actions[choices[0]].action_type == ActionType::PayCost
        {
            env.step(choices[0] as i64)
                .expect("automatic source payment");
        } else {
            return choices.into_iter().map(|index| index as i64).collect();
        }
    }
}
