// possible_worlds_tests.rs
// Integration proof for the first viewer-relative PossibleWorldSpace slice on
// the selected UR-Lessons-vs-GW-Allies authored match.

use std::collections::HashSet;

use managym::{
    possible_worlds::{
        viewer_observation, CanonicalWorldQuery, ConditioningError, CountQuery, PossibleWorldSpace,
        WorldQuery,
    },
    semantic::SemanticPack,
    state::game_object::PlayerId,
    state::player::PlayerConfig,
    state::zone::ZoneType,
    Game,
};
use rand::Rng;

const VIEWER: PlayerId = PlayerId(1);
const OPPONENT: PlayerId = PlayerId(0);
const CARD: &str = "Firebending Lesson";

fn local_binom(n: u32, k: u32) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc * (n - i) as u128 / (i + 1) as u128;
    }
    acc
}

fn authored_match() -> Game {
    let semantic = SemanticPack::two_deck().expect("checked-in semantic IR parses");
    let configs = vec![
        PlayerConfig::new(
            "UR Lessons",
            semantic.decklist("ur_lessons").expect("UR deck compiles"),
        ),
        PlayerConfig::new(
            "GW Allies",
            semantic.decklist("gw_allies").expect("GW deck compiles"),
        ),
    ];
    Game::new(configs, 0, true)
}

/// Step the authored match deterministically until GW Allies (player 1) is
/// acting on a non-terminal decision with Firebending Lesson in UR Lessons'
/// unseen pool. Player 0 = UR Lessons (opponent), player 1 = GW Allies
/// (viewer). Seed 0 makes the reached state stable.
fn gwallies_decision_with_lesson() -> Game {
    let mut game = authored_match();
    for _ in 0..500 {
        if game.is_game_over() {
            break;
        }
        let acting = game.action_space().and_then(|space| space.player);
        if acting == Some(VIEWER) {
            let space = PossibleWorldSpace::for_viewer(&game, VIEWER);
            if space.pool().get(CARD).copied().unwrap_or(0) >= 1 {
                return game;
            }
        }
        let n = game.action_space().map_or(0, |space| space.actions.len());
        assert!(
            n > 0,
            "no legal action while stepping to GW Allies decision"
        );
        let idx = game.state.rng.gen_range(0..n);
        game.step(idx).expect("authored match step succeeds");
    }
    panic!("did not reach a GW Allies decision with Firebending Lesson in the opponent pool");
}

#[test]
fn exact_weights_total_to_choose_n_h() {
    let game = gwallies_decision_with_lesson();
    let space = PossibleWorldSpace::for_viewer(&game, VIEWER);
    let h = space.hand_size();
    let n: u32 = space.pool().values().copied().sum();
    assert!(n >= h, "opponent unseen pool must cover the hand size");
    assert_eq!(space.total_weight(), local_binom(n, h));
    let sum: u128 = space.worlds().iter().map(|w| w.weight).sum();
    assert_eq!(sum, space.total_weight());
    assert!(
        space.worlds().len() >= 2,
        "fixture must yield multiple worlds"
    );
}

#[test]
fn viewer_relative_slice_preserves_observation_and_hides_truth() {
    let game = gwallies_decision_with_lesson();
    let space = PossibleWorldSpace::for_viewer(&game, VIEWER);
    let h = space.hand_size();
    assert!(space.pool().get(CARD).copied().unwrap_or(0) >= 1);

    // Condition on Has("Firebending Lesson", 1): non-empty support.
    let query = WorldQuery::Has {
        card: CARD.to_string(),
        at_least: 1,
    };
    let receipt = space
        .condition(&query)
        .expect("Firebending Lesson support is non-empty");
    assert!(receipt.receipt.support_size > 0);
    assert!(receipt.receipt.total_weight > 0);
    assert_eq!(receipt.receipt.support_size, receipt.worlds.len());

    // The support and its complement (Lacks(card, 1)) partition total weight.
    let lacks = space
        .condition(&WorldQuery::Lacks {
            card: CARD.to_string(),
            fewer_than: 1,
        })
        .expect("complement support is non-empty");
    assert_eq!(
        receipt.receipt.total_weight + lacks.receipt.total_weight,
        space.total_weight(),
    );

    // Materialize one support world into an exact branch.
    let world = &receipt.worlds[0];
    let source_obs = viewer_observation(&game, VIEWER);
    let source_opponent_hand: Vec<usize> = game
        .state
        .zones
        .zone_cards(ZoneType::Hand, OPPONENT)
        .iter()
        .map(|c| c.0)
        .collect();
    let branch = space
        .materialize(&game, world, 0)
        .expect("world materializes into an exact branch");
    let branch_obs = viewer_observation(&branch, VIEWER);

    // Proof 3: the viewer Observation and viewer-visible legal decision are
    // preserved across materialization (byte-identical projection).
    assert_eq!(
        branch_obs, source_obs,
        "viewer projection must be byte-identical"
    );

    // The source Game is not mutated (materialize takes &Game).
    let after_opponent_hand: Vec<usize> = game
        .state
        .zones
        .zone_cards(ZoneType::Hand, OPPONENT)
        .iter()
        .map(|c| c.0)
        .collect();
    assert_eq!(
        after_opponent_hand, source_opponent_hand,
        "source must not be mutated by materialize"
    );

    // Proof 2: opponent hand card identities are absent from the viewer
    // projection; only the public hand-size count is present.
    let opponent_hand_ids: HashSet<i32> = branch
        .state
        .zones
        .zone_cards(ZoneType::Hand, OPPONENT)
        .iter()
        .map(|c| c.0 as i32)
        .collect();
    let projected_ids: HashSet<i32> = branch_obs
        .agent_cards
        .iter()
        .chain(branch_obs.opponent_cards.iter())
        .map(|c| c.id)
        .collect();
    for id in &opponent_hand_ids {
        assert!(
            !projected_ids.contains(id),
            "opponent hand card {id} leaked into viewer projection"
        );
    }
    assert_eq!(
        branch_obs.opponent.zone_counts[1], h as i32,
        "public opponent hand-size count must be present"
    );
}

#[test]
fn distinct_worlds_yield_identical_viewer_projections() {
    let game = gwallies_decision_with_lesson();
    let space = PossibleWorldSpace::for_viewer(&game, VIEWER);
    let all = space
        .condition(&WorldQuery::True)
        .expect("True has full support");
    assert!(all.worlds.len() >= 2);
    let w0 = all.worlds[0].clone();
    let w1 = all.worlds[1].clone();
    assert_ne!(
        w0.hand, w1.hand,
        "worlds must differ in opponent hand contents"
    );
    let b0 = space
        .materialize(&game, &w0, 0)
        .expect("first world materializes");
    let b1 = space
        .materialize(&game, &w1, 0)
        .expect("second world materializes");
    let o0 = viewer_observation(&b0, VIEWER);
    let o1 = viewer_observation(&b1, VIEWER);
    let osrc = viewer_observation(&game, VIEWER);
    assert_eq!(
        o0, o1,
        "distinct worlds project to identical viewer observations"
    );
    assert_eq!(
        o0, osrc,
        "each world preserves the source viewer observation"
    );
}

#[test]
fn impossible_query_reports_empty_support() {
    let game = gwallies_decision_with_lesson();
    let space = PossibleWorldSpace::for_viewer(&game, VIEWER);
    let impossible = WorldQuery::Has {
        card: CARD.to_string(),
        at_least: 99,
    };
    let receipt = space.support_receipt(&impossible);
    assert_eq!(receipt.support_size, 0);
    assert_eq!(receipt.total_weight, 0);
    assert!(matches!(
        space.condition(&impossible),
        Err(ConditioningError::EmptySupport { .. }),
    ));
}

#[test]
fn equivalent_queries_share_canonical_digest_and_support() {
    let game = gwallies_decision_with_lesson();
    let space = PossibleWorldSpace::for_viewer(&game, VIEWER);
    let true_worlds = space.condition(&WorldQuery::True).unwrap();
    let true_digest = CanonicalWorldQuery::True.digest();

    // Has(card, 0) ≡ True; Lacks(card, 99) ≡ True; Not(Q(card, 99)) ≡ True.
    let equivalent_to_true = [
        WorldQuery::Has {
            card: CARD.to_string(),
            at_least: 0,
        },
        WorldQuery::Lacks {
            card: CARD.to_string(),
            fewer_than: 99,
        },
        WorldQuery::Not(CountQuery::Exactly {
            card: CARD.to_string(),
            count: 99,
        }),
    ];
    for q in equivalent_to_true {
        assert_eq!(
            q.canonicalize(&space),
            CanonicalWorldQuery::True,
            "{q:?} -> True"
        );
        assert_eq!(q.canonicalize(&space).digest(), true_digest);
        let got = space.condition(&q).unwrap();
        assert_eq!(
            got.worlds, true_worlds.worlds,
            "{q:?} support must equal True support"
        );
        assert_eq!(
            got.receipt.canonical_digest,
            true_worlds.receipt.canonical_digest
        );
        assert_ne!(got.receipt.query_digest, true_worlds.receipt.query_digest);
    }

    // Q(card, 0) ≡ Lacks(card, 1).
    let q0 = WorldQuery::Q(CountQuery::Exactly {
        card: CARD.to_string(),
        count: 0,
    });
    let lacks1 = WorldQuery::Lacks {
        card: CARD.to_string(),
        fewer_than: 1,
    };
    assert_eq!(q0.canonicalize(&space), lacks1.canonicalize(&space));
    assert_eq!(
        q0.canonicalize(&space).digest(),
        lacks1.canonicalize(&space).digest()
    );
    assert_eq!(
        space.condition(&q0).unwrap().worlds,
        space.condition(&lacks1).unwrap().worlds,
    );

    // Not(Q(card, 0)) ≡ Has(card, 1).
    let not0 = WorldQuery::Not(CountQuery::Exactly {
        card: CARD.to_string(),
        count: 0,
    });
    let has1 = WorldQuery::Has {
        card: CARD.to_string(),
        at_least: 1,
    };
    assert_eq!(not0.canonicalize(&space), has1.canonicalize(&space));
    assert_eq!(
        not0.canonicalize(&space).digest(),
        has1.canonicalize(&space).digest()
    );
    assert_eq!(
        space.condition(&not0).unwrap().worlds,
        space.condition(&has1).unwrap().worlds,
    );
}
