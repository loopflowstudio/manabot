use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionType},
    flow::turn::StepKind,
};

use super::helpers::*;

fn set_not_summoning_sick(
    s: &mut Scenario,
    permanent_id: managym::state::game_object::PermanentId,
) {
    s.game_mut().state.permanents[permanent_id]
        .as_mut()
        .expect("permanent should exist")
        .summoning_sick = false;
}

/// CR 702.9 — Creatures with flying can't be blocked except by flying/reach creatures.
#[test]
fn cr_702_flying_cant_be_blocked_by_ground() {
    let mut s = Scenario::new(wind_drake_deck(), forest_elves_deck(), 70201);
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, drake);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(
        !s.action_space().actions.iter().any(|action| {
            matches!(
                action,
                Action::DeclareBlocker {
                    attacker: Some(_),
                    ..
                }
            )
        }),
        "ground creature should not have a legal block action against flyer"
    );
}

/// CR 702.9 / negative — Non-flyers can be blocked by ground creatures.
#[test]
fn cr_702_flying_negative_nonflyer_is_blockable() {
    let mut s = Scenario::new(ogre_deck(), forest_elves_deck(), 70202);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// CR 702.17 — Reach allows a creature to block creatures with flying.
#[test]
fn cr_702_reach_blocks_flying() {
    let mut s = Scenario::new(wind_drake_deck(), giant_spider_deck(), 70203);
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    let _spider = s.force_permanent_on_battlefield(1, "Giant Spider");
    set_not_summoning_sick(&mut s, drake);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// Negative: reach does not grant evasion while attacking.
#[test]
fn cr_702_reach_negative_no_attacking_evasion() {
    let mut s = Scenario::new(giant_spider_deck(), forest_elves_deck(), 70204);
    let spider = s.force_permanent_on_battlefield(0, "Giant Spider");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, spider);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// CR 702.10 — Haste allows attacking on the turn the creature enters.
#[test]
fn cr_702_haste_can_attack_turn_it_enters() {
    let mut s = Scenario::new(raging_goblin_deck(), mountain_deck(), 70205);
    s.play_land_and_cast_creature(0, "Mountain", "Raging Goblin");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::DeclareAttacker { attack: true, .. })));
}

/// Negative: non-haste creatures remain summoning sick.
#[test]
fn cr_702_haste_negative_nonhaste_cant_attack_immediately() {
    let mut s = Scenario::new(forest_elves_deck(), mountain_deck(), 70206);
    s.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::DeclareAttacker { attack: true, .. })));
}

/// CR 702.20 — Vigilance attackers don't tap.
#[test]
fn cr_702_vigilance_attacker_stays_untapped() {
    let mut s = Scenario::new(serra_angel_deck(), mountain_deck(), 70207);
    let serra = s.force_permanent_on_battlefield(0, "Serra Angel");
    set_not_summoning_sick(&mut s, serra);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    assert!(
        !s.game().state.permanents[serra]
            .as_ref()
            .expect("serra should exist")
            .tapped
    );
}

/// Negative: non-vigilance attackers tap.
#[test]
fn cr_702_vigilance_negative_attacker_taps_without_vigilance() {
    let mut s = Scenario::new(ogre_deck(), mountain_deck(), 70208);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    set_not_summoning_sick(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    assert!(
        s.game().state.permanents[ogre]
            .as_ref()
            .expect("ogre should exist")
            .tapped
    );
}

/// CR 702.3 — Defender creatures can't attack.
#[test]
fn cr_702_defender_cant_attack() {
    let mut s = Scenario::new(
        BTreeMap::from([
            ("Mountain".to_string(), 20),
            ("Wall of Stone".to_string(), 10),
            ("Raging Goblin".to_string(), 10),
        ]),
        raging_goblin_deck(),
        70209,
    );
    let wall = s.force_permanent_on_battlefield(0, "Wall of Stone");
    let goblin = s.force_permanent_on_battlefield(0, "Raging Goblin");
    set_not_summoning_sick(&mut s, wall);
    set_not_summoning_sick(&mut s, goblin);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert!(!s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareAttacker {
                permanent,
                attack: true,
                ..
            } if *permanent == wall
        )
    }));
    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareAttacker {
                permanent,
                attack: true,
                ..
            } if *permanent == goblin
        )
    }));
}

/// Defender can still block.
#[test]
fn cr_702_defender_can_block() {
    let mut s = Scenario::new(raging_goblin_deck(), wall_of_stone_deck(), 70210);
    let goblin = s.force_permanent_on_battlefield(0, "Raging Goblin");
    let _wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    set_not_summoning_sick(&mut s, goblin);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// CR 702.19 — Trample assigns excess damage to defending player.
#[test]
fn cr_702_trample_excess_hits_player() {
    let mut s = Scenario::new(war_mammoth_deck(), forest_elves_deck(), 70211);
    let mammoth = s.force_permanent_on_battlefield(0, "War Mammoth");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, mammoth);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), life_before - 2);
}

/// Negative: without trample, excess damage isn't dealt to defending player.
#[test]
fn cr_702_trample_negative_no_excess_without_trample() {
    let mut s = Scenario::new(ogre_deck(), forest_elves_deck(), 70212);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), life_before);
}

/// Interaction: trample + deathtouch only needs 1 lethal damage assignment.
#[test]
fn cr_702_trample_with_deathtouch_assigns_one_lethal() {
    let mut s = Scenario::new(war_mammoth_deck(), ogre_deck(), 70213);
    let mammoth = s.force_permanent_on_battlefield(0, "War Mammoth");
    let _ogre = s.force_permanent_on_battlefield(1, "Grey Ogre");
    set_not_summoning_sick(&mut s, mammoth);
    let mammoth_card = s.game().state.permanents[mammoth]
        .as_ref()
        .expect("mammoth should exist")
        .card;
    s.game_mut().state.cards[mammoth_card].keywords.deathtouch = true;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), life_before - 2);
}

/// CR 702.2 — Any nonzero deathtouch damage is lethal to a creature.
#[test]
fn cr_702_deathtouch_one_damage_is_lethal() {
    let mut s = Scenario::new(craw_wurm_deck(), typhoid_rats_deck(), 70214);
    let wurm = s.force_permanent_on_battlefield(0, "Craw Wurm");
    let _rats = s.force_permanent_on_battlefield(1, "Typhoid Rats");
    set_not_summoning_sick(&mut s, wurm);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(
        s.battlefield_permanents_named(0, "Craw Wurm").is_empty(),
        "deathtouch damage should destroy the wurm"
    );
}

/// Negative: without deathtouch, 1 damage doesn't kill large creatures.
#[test]
fn cr_702_deathtouch_negative_normal_one_damage_not_lethal() {
    let mut s = Scenario::new(craw_wurm_deck(), forest_elves_deck(), 70215);
    let wurm = s.force_permanent_on_battlefield(0, "Craw Wurm");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, wurm);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(!s.battlefield_permanents_named(0, "Craw Wurm").is_empty());
}

/// CR 702.15 — Lifelink causes its controller to gain life with damage dealt.
#[test]
fn cr_702_lifelink_gains_life_on_damage() {
    let mut s = Scenario::new(hawk_deck(), mountain_deck(), 70216);
    let hawk = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    set_not_summoning_sick(&mut s, hawk);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(0), 21);
    assert_eq!(s.life(1), 19);
}

/// Negative: non-lifelink damage doesn't gain life.
#[test]
fn cr_702_lifelink_negative_no_life_gain_without_lifelink() {
    let mut s = Scenario::new(wind_drake_deck(), mountain_deck(), 70217);
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    set_not_summoning_sick(&mut s, drake);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(0), 20);
    assert_eq!(s.life(1), 18);
}

/// CR 702.111 + 509.1b — Menace attacker can't be blocked by exactly one creature.
#[test]
fn cr_702_menace_single_block_is_removed() {
    let mut s = Scenario::new(boggart_brute_deck(), forest_elves_deck(), 70218);
    let brute = s.force_permanent_on_battlefield(0, "Boggart Brute");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, brute);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let combat = s.game().state.combat.as_ref().expect("combat should exist");
    let attacker = *combat.attackers.first().expect("attacker should exist");
    let blockers = combat
        .attacker_to_blockers
        .get(&attacker)
        .expect("blockers should exist");
    assert!(blockers.is_empty());
}

/// Menace is satisfied by two or more blockers.
#[test]
fn cr_702_menace_two_blockers_is_legal() {
    let mut s = Scenario::new(boggart_brute_deck(), forest_elves_deck(), 70219);
    let brute = s.force_permanent_on_battlefield(0, "Boggart Brute");
    let _elf_one = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    let _elf_two = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, brute);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.declare_block();

    let combat = s.game().state.combat.as_ref().expect("combat should exist");
    let attacker = *combat.attackers.first().expect("attacker should exist");
    let blockers = combat
        .attacker_to_blockers
        .get(&attacker)
        .expect("blockers should exist");
    assert_eq!(blockers.len(), 2);
}

/// CR 702.7 — First strike damage is dealt before normal combat damage.
#[test]
fn cr_702_first_strike_kills_before_normal_damage() {
    let mut s = Scenario::new(youthful_knight_deck(), ogre_deck(), 70220);
    let knight = s.force_permanent_on_battlefield(0, "Youthful Knight");
    let _ogre = s.force_permanent_on_battlefield(1, "Grey Ogre");
    set_not_summoning_sick(&mut s, knight);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(!s
        .battlefield_permanents_named(0, "Youthful Knight")
        .is_empty());
    assert!(s.battlefield_permanents_named(1, "Grey Ogre").is_empty());
}

/// Negative: without first strike, creatures deal combat damage simultaneously.
#[test]
fn cr_702_first_strike_negative_normal_combat_is_simultaneous() {
    let mut s = Scenario::new(ogre_deck(), ogre_deck(), 70221);
    let attacker = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let _blocker = s.force_permanent_on_battlefield(1, "Grey Ogre");
    set_not_summoning_sick(&mut s, attacker);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(s.battlefield_permanents_named(0, "Grey Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Grey Ogre").is_empty());
}

/// CR 702.4 — Double strike deals damage in both combat damage steps.
#[test]
fn cr_702_double_strike_deals_damage_twice() {
    let mut s = Scenario::new(fencing_ace_deck(), mountain_deck(), 70222);
    let ace = s.force_permanent_on_battlefield(0, "Fencing Ace");
    set_not_summoning_sick(&mut s, ace);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before - 2);
}

/// Interaction: double strike + trample can deal player damage in second sub-step.
#[test]
fn cr_702_double_strike_with_trample_interaction() {
    let mut s = Scenario::new(fencing_ace_deck(), forest_elves_deck(), 70223);
    let ace = s.force_permanent_on_battlefield(0, "Fencing Ace");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    set_not_summoning_sick(&mut s, ace);
    let ace_card = s.game().state.permanents[ace]
        .as_ref()
        .expect("ace should exist")
        .card;
    s.game_mut().state.cards[ace_card].keywords.trample = true;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before - 1);
}

/// Regression: multi-blocker assignment uses total attacker power, not full power per blocker.
#[test]
fn cr_702_regression_multi_blocker_damage_is_split_not_duplicated() {
    let mut s = Scenario::new(ogre_deck(), ogre_deck(), 70224);
    let ogre = s.force_permanent_on_battlefield(0, "Grey Ogre");
    let _ogre_one = s.force_permanent_on_battlefield(1, "Grey Ogre");
    let _ogre_two = s.force_permanent_on_battlefield(1, "Grey Ogre");
    set_not_summoning_sick(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    let surviving_ogres = s.battlefield_permanents_named(1, "Grey Ogre").len();
    assert_eq!(surviving_ogres, 1);
}

/// Sanity: deathtouch-enabled creature should still be castable/attackable via normal priority.
#[test]
fn cr_702_deathtouch_card_still_uses_normal_actions() {
    let mut s = Scenario::new(typhoid_rats_deck(), mountain_deck(), 70225);
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Swamp");
    s.force_card_in_hand(0, "Typhoid Rats");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
}
