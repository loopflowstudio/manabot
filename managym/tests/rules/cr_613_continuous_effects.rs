//! CR 611 / 613: continuous effects from resolved spells and abilities are
//! records on `GameState::effects`. Layer 6 applies them by timestamp
//! (CR 613.7), each duration ends at its boundary (CR 514.2, CR 400.7), and
//! the records are journaled and hashed like the rest of the match state.

use managym::{
    flow::turn::StepKind,
    search_state::{BranchDriver, ClonePlusUndoDriver},
    state::{
        card::Keywords,
        effect::{Affected, EffectDuration, Modification},
        game_object::{PermanentId, PlayerId},
        zone::ZoneType,
    },
};

use crate::support::game::*;
use crate::support::inspect::card_of;

/// Player 0's Gray Ogre (no printed keywords) in player 0's main phase.
fn ogre_scenario() -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Mountain", 40)]),
        613_007,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    (s, ogre)
}

fn flying() -> Modification {
    Modification::AddKeywords(Keywords {
        flying: true,
        ..Default::default()
    })
}

fn indefinitely(s: &mut Scenario, id: PermanentId, modification: Modification) {
    let affected = s.game().affected_permanent(id).unwrap();
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [affected],
        modification,
        EffectDuration::Indefinite,
    );
}

/// CR 613.7: a loss of all abilities removes grants with earlier timestamps
/// only; a later grant applies.
#[test]
fn cr_613_7_a_grant_later_than_ability_loss_applies_and_an_earlier_one_does_not() {
    let (mut later_grant, ogre) = ogre_scenario();
    until_end_of_turn(&mut later_grant, ogre, Modification::LoseAllAbilities);
    until_end_of_turn(&mut later_grant, ogre, flying());
    assert!(later_grant.game().has_lost_abilities(ogre));
    assert!(later_grant.game().effective_keywords(ogre).flying);

    let (mut earlier_grant, ogre) = ogre_scenario();
    until_end_of_turn(&mut earlier_grant, ogre, flying());
    until_end_of_turn(&mut earlier_grant, ogre, Modification::LoseAllAbilities);
    assert!(earlier_grant.game().has_lost_abilities(ogre));
    assert!(!earlier_grant.game().effective_keywords(ogre).flying);
}

/// CR 514.2: "until end of turn" records end in the cleanup step; the
/// permanent's characteristics return to its printed ones.
#[test]
fn cr_514_2_an_end_of_turn_record_ends_at_cleanup() {
    let (mut s, ogre) = ogre_scenario();
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::ModifyPt {
            power: 2,
            toughness: 1,
        },
    );
    indefinitely(&mut s, ogre, flying());
    assert_eq!(s.game().effective_pt(ogre), (4, 3));
    assert_eq!(s.game().effects_on(ogre).count(), 2);

    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.game().effective_pt(ogre), (2, 2));
    let remaining: Vec<_> = s
        .game()
        .effects_on(ogre)
        .map(|effect| (effect.modification.clone(), effect.duration))
        .collect();
    assert_eq!(
        remaining,
        [(flying(), EffectDuration::Indefinite)],
        "an indefinite record outlives cleanup"
    );
}

/// CR 400.7: an indefinite record on an object ends when it changes zones; the
/// card returns as a new object with no records.
#[test]
fn cr_400_7_an_indefinite_record_ends_when_its_object_changes_zones() {
    let (mut s, ogre) = ogre_scenario();
    indefinitely(&mut s, ogre, flying());
    let old = s.game().affected_permanent(ogre).unwrap();
    assert!(s.game().effective_keywords(ogre).flying);

    let card = card_of(&s, ogre);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let returned = s.game().state.card_to_permanent[card].expect("returned");
    assert_ne!(s.game().affected_permanent(returned), Some(old));
    assert!(s.game().state.effects.is_empty());
    assert_eq!(s.game().effects_on(returned).count(), 0);
    assert!(!s.game().effective_keywords(returned).flying);
}

/// Records are journaled: rolling back to a mark removes a record applied
/// after it and restores records a zone change ended after it.
#[test]
fn rollback_restores_continuous_effect_records_and_the_state_hash() {
    let (s, ogre) = ogre_scenario();
    let driver = ClonePlusUndoDriver::default();
    let mut branch = driver.fork_exact(s.game());
    let effects = branch.state.effects.clone();
    let hash = branch.state.deterministic_hash();

    let mark = driver.mark(&mut branch);
    let affected = branch.affected_permanent(ogre).unwrap();
    branch.apply_continuous(
        None,
        PlayerId(0),
        [affected],
        flying(),
        EffectDuration::EndOfTurn,
    );
    assert_ne!(branch.state.effects, effects);
    assert_ne!(branch.state.deterministic_hash(), hash);
    driver.rollback(&mut branch, mark);
    assert_eq!(branch.state.effects, effects);
    assert_eq!(branch.state.deterministic_hash(), hash);
    assert!(!branch.effective_keywords(ogre).flying);

    branch.apply_continuous(
        None,
        PlayerId(0),
        [affected],
        flying(),
        EffectDuration::Indefinite,
    );
    let effects = branch.state.effects.clone();
    let hash = branch.state.deterministic_hash();
    let mark = driver.mark(&mut branch);
    let card = branch.state.permanents[ogre].as_ref().unwrap().card;
    branch.move_card(card, ZoneType::Hand);
    assert!(branch.state.effects.is_empty());
    driver.rollback(&mut branch, mark);
    assert_eq!(branch.state.effects, effects);
    assert_eq!(branch.state.deterministic_hash(), hash);
    assert!(branch.effective_keywords(ogre).flying);
}

/// Equal games with equal records hash equally; the records are part of the
/// hash.
#[test]
fn equal_records_give_equal_deterministic_hashes() {
    let build = |records: bool| {
        let (mut s, ogre) = ogre_scenario();
        if records {
            until_end_of_turn(&mut s, ogre, Modification::LoseAllAbilities);
            until_end_of_turn(&mut s, ogre, flying());
            player_effect(
                &mut s,
                1,
                Modification::NoMaximumHandSize,
                EffectDuration::Indefinite,
            );
        }
        s
    };
    let (first, second) = (build(true), build(true));
    assert_eq!(first.game().state.effects, second.game().state.effects);
    assert_eq!(
        first.game().state.deterministic_hash(),
        second.game().state.deterministic_hash()
    );
    assert_ne!(
        first.game().state.deterministic_hash(),
        build(false).game().state.deterministic_hash()
    );
    assert!(first
        .game()
        .state
        .effects
        .iter()
        .any(|effect| effect.affected == Affected::Player(PlayerId(1))));
}
