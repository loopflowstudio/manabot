// Full Elemental admission: exact rosters, complete native coherence and bounded games.
use managym::{
    cards::setups::authored_setup,
    state::player::{GameSetup, ManaFloats},
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
    let mut setup = GameSetup::from(configs);
    setup.mana_floats = [ManaFloats::Usable; 2];
    env.reset(setup).unwrap();
    // Includes the retained deliberate green Prism float and manual source decisions.
    let prefix = [
        2, 2, 0, 0, 4, 3, 3, 5, 2, 0, 2, 2, 2, 6, 2, 3, 3, 0, 1, 0, 1, 0, 4, 0, 0, 8, 0, 0, 0, 0,
        1, 4, 6, 2, 2, 5, 0, 0,
    ];
    for action in prefix {
        env.step(action).expect("retained Professor path");
    }
    env.step(4).expect("offered Professor cast enters payment");
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
