//! Engine mechanics the TLA cards introduced, witnessed with vanilla cards,
//! injected abilities and fixture-only cards rather than one printed card:
//! trigger events, token lifetime, block restrictions, counters on lands,
//! observation tags, modal choice, cleanup discard and a full-deck replay.
use super::{
    helpers::{forest_elves_deck, ogre_only_deck, raging_goblin_deck, serra_angel_deck, Scenario},
    support::{
        card_of, cast_and_resolve, cast_only, clear_hand, deck, find_owned_card_off_battlefield,
        force_lands, hand_of, permanent_state, put_permanent, resolve, resolve_all,
        ur_lessons_setup,
    },
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    decision::Command,
    flow::turn::StepKind,
    semantic::SemanticPack,
    state::{
        ability::{Ability, Effect, TriggerCondition, TriggerSubject},
        card::CardType,
        game_object::{CardId, PermanentId, PlayerId},
        predicate::CardPredicate,
        zone::ZoneType,
    },
    Game,
};

/// Move a specific card from the library into the hand (avoids colliding
/// with same-named cards already on the battlefield or in hand).
fn force_from_library(s: &mut Scenario, player: usize, name: &str) -> CardId {
    let game = s.game();
    let card_id = game
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(player))
        .iter()
        .copied()
        .find(|card_id| game.state.cards[card_id].name == name)
        .unwrap_or_else(|| panic!("no {name} in player {player}'s library"));
    s.game_mut()
        .state
        .zones
        .move_card(card_id, PlayerId(player), ZoneType::Hand);
    card_id
}

// ---------------------------------------------------------------------------
// Attack triggers, tapped-and-attacking tokens, vigilance & haste.
// ---------------------------------------------------------------------------

fn inject_ability(s: &mut Scenario, card_id: CardId, ability: Ability) {
    s.game_mut().state.cards[card_id].abilities.push(ability);
}

fn battlefield_card(s: &Scenario, permanent_id: PermanentId) -> CardId {
    permanent_state(s, permanent_id).card
}

// ---------------------------------------------------------------------------
// Crossroads of Destiny — MODAL machinery proof.
// ---------------------------------------------------------------------------

fn crossroads_setup(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Crossroads of Destiny", 16)]),
        deck(&[("Plains", 40)]),
        seed,
    );
    force_lands(&mut s, 0, "Island", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Crossroads of Destiny");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);
    s
}

/// A creature token that dies goes to the graveyard (firing zone-change
/// events) and then ceases to exist via state-based actions.
#[test]
fn creature_token_ceases_to_exist_after_dying() {
    let mut s = Scenario::new(deck(&[("Plains", 40)]), deck(&[("Plains", 40)]), 37);
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    let token = s.battlefield_permanents_named(0, "Ally")[0];
    let token_card = permanent_state(&s, token).card;

    s.game_mut().state.permanents[token]
        .as_mut()
        .unwrap()
        .damage = 1;
    s.pass_priority();

    assert!(s.battlefield_permanents_named(0, "Ally").is_empty());
    assert_eq!(s.game().state.zones.zone_of(token_card), None);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 0);
}

/// "Whenever this creature attacks" fires after the declare-attackers batch;
/// a token created tapped-and-attacking joins combat and deals damage.
#[test]
fn attack_trigger_creates_tapped_and_attacking_token() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 43);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CreateToken {
                token_name: "Ally".to_string(),
                count: 1,
                tapped_and_attacking: true,
                attach_source: false,
            }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    resolve_all(&mut s);

    let tokens = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(tokens.len(), 1, "attack trigger should create one token");
    assert!(permanent_state(&s, tokens[0]).tapped);
    assert!(permanent_state(&s, tokens[0]).attacking);

    // Unblocked: Gray Ogre (2) + token (1) = 3 damage.
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    s.assert_life(1, 17);
}

/// "Whenever one or more creatures you control attack" fires once per
/// combat, not once per attacker.
#[test]
fn one_or_more_attack_trigger_fires_once() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 47);
    let ogre_a = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_b = s.force_permanent_on_battlefield(0, "Gray Ogre");
    for ogre in [ogre_a, ogre_b] {
        s.game_mut().state.permanents[ogre]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
    }
    let ogre_a_card = battlefield_card(&s, ogre_a);
    inject_ability(
        &mut s,
        ogre_a_card,
        Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::AnyYouControl(CardPredicate::creature()),
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.declare_attack();
    resolve_all(&mut s);

    s.assert_life(0, 21);
}

/// A creature with haste can attack the turn it enters and its attack
/// trigger fires.
#[test]
fn haste_attacker_fires_attack_trigger_on_entry_turn() {
    let mut s = Scenario::new(raging_goblin_deck(), deck(&[("Plains", 40)]), 53);
    force_lands(&mut s, 0, "Mountain", 1);
    clear_hand(&mut s, 0);
    // Inject on the card before it enters; abilities are per-card-instance.
    let goblin_card = force_from_library(&mut s, 0, "Raging Goblin");
    inject_ability(
        &mut s,
        goblin_card,
        Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    resolve_all(&mut s);

    s.assert_life(0, 21);
}

/// Attacking taps a non-vigilance creature and fires "becomes tapped".
#[test]
fn becomes_tapped_fires_when_attacker_taps() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 59);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    resolve_all(&mut s);

    assert!(permanent_state(&s, ogre).tapped);
    s.assert_life(0, 21);
}

/// A vigilance attacker does not tap, so "becomes tapped" must NOT fire.
#[test]
fn vigilance_attacker_does_not_become_tapped() {
    let mut s = Scenario::new(serra_angel_deck(), deck(&[("Plains", 40)]), 61);
    let angel = s.force_permanent_on_battlefield(0, "Serra Angel");
    s.game_mut().state.permanents[angel]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let angel_card = battlefield_card(&s, angel);
    inject_ability(
        &mut s,
        angel_card,
        Ability::Triggered {
            condition: TriggerCondition::BecomesTapped {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::EndOfCombat);

    assert!(
        !permanent_state(&s, angel).tapped,
        "vigilance attacker stays untapped"
    );
    s.assert_life(0, 20);
}

/// Tapping a creature for mana fires "tapped for mana" (and it does not
/// fire for combat taps).
#[test]
fn tapped_for_mana_trigger() {
    let mut s = Scenario::new(forest_elves_deck(), deck(&[("Plains", 40)]), 67);
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    s.game_mut().state.permanents[elves]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let elves_card = battlefield_card(&s, elves);
    inject_ability(
        &mut s,
        elves_card,
        Ability::Triggered {
            condition: TriggerCondition::TappedForMana {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    // Cast another Llanowar Elves paying {G} with the battlefield Elves.
    clear_hand(&mut s, 0);
    force_from_library(&mut s, 0, "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    resolve_all(&mut s);

    assert!(permanent_state(&s, elves).tapped);
    s.assert_life(0, 21);
}

/// A combat tap is not "tapped for mana".
#[test]
fn attack_tap_is_not_tapped_for_mana() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 71);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::TappedForMana {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::EndOfCombat);

    assert!(permanent_state(&s, ogre).tapped);
    s.assert_life(0, 20);
}

// ---------------------------------------------------------------------------
// Death triggers.
// ---------------------------------------------------------------------------

/// "When this creature dies" fires on battlefield → graveyard.
#[test]
fn dies_trigger_fires_on_death() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 73);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.game_mut().state.permanents[ogre].as_mut().unwrap().damage = 2;
    s.pass_priority();
    resolve_all(&mut s);

    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
    s.assert_life(0, 21);
}

/// A creature going battlefield → hand did not die.
#[test]
fn bounce_is_not_death() {
    let mut s = Scenario::new(ogre_only_deck(), deck(&[("Plains", 40)]), 79);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = battlefield_card(&s, ogre);
    inject_ability(
        &mut s,
        ogre_card,
        Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::GainLife { amount: 1 }],
        },
    );

    s.game_mut().move_card(ogre_card, ZoneType::Hand);
    s.pass_priority();
    resolve_all(&mut s);

    s.assert_life(0, 20);
}

// ---------------------------------------------------------------------------
// Conditional unblockability.
// ---------------------------------------------------------------------------

/// "Can't be blocked by creatures with power 2 or less" — the predicate is
/// checked against effective power, so +1/+1 counters can re-enable a block.
#[test]
fn block_restriction_respects_effective_power() {
    let mut s = Scenario::new(ogre_only_deck(), ogre_only_deck(), 83);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let attacker_card = battlefield_card(&s, attacker);
    s.game_mut().state.cards[attacker_card].block_restriction = Some(CardPredicate {
        card_type: Some(CardType::Creature),
        max_power: Some(2),
        ..Default::default()
    });

    let weak = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let strong = s.force_permanent_on_battlefield(1, "Gray Ogre");
    // The "strong" ogre has a +1/+1 counter: effective power 3 > 2.
    s.game_mut().state.permanents[strong]
        .as_mut()
        .unwrap()
        .plus1_counters = 1;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    // Two blocker prompts follow; the countered ogre may block, the plain
    // one may not.
    for _ in 0..2 {
        s.advance_until(
            |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
            "expected blocker prompt".to_string(),
        );
        let actions = s.action_space().actions.clone();
        let blocker = actions
            .iter()
            .find_map(|action| match action {
                Action::DeclareBlocker { blocker, .. } => Some(*blocker),
                _ => None,
            })
            .expect("blocker action should reference a blocker");
        let can_block = actions.iter().any(|action| {
            matches!(
                action,
                Action::DeclareBlocker {
                    attacker: Some(_),
                    ..
                }
            )
        });
        if blocker == weak {
            assert!(!can_block, "power-2 creature must not be able to block");
        } else {
            assert_eq!(blocker, strong);
            assert!(can_block, "power-3 creature may block");
        }
        s.decline_block();
    }
}

/// A permanent flagged "can't be blocked this turn" cannot be blocked.
#[test]
fn cant_be_blocked_this_turn_blocks_all_blocks() {
    let mut s = Scenario::new(ogre_only_deck(), ogre_only_deck(), 89);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .cant_be_blocked_this_turn = true;
    s.force_permanent_on_battlefield(1, "Gray Ogre");

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "expected blocker prompt".to_string(),
    );
    let can_block = s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    });
    assert!(!can_block);
}

// ---------------------------------------------------------------------------
// Counters on lands, upkeep trigger scoping, observation encoding.
// ---------------------------------------------------------------------------

/// +1/+1 counters sit on lands without killing them (earthbend prereq —
/// Stage 3 animates lands; the counter state must not assume creature-ness).
#[test]
fn counters_on_lands_are_inert_and_persistent() {
    let mut s = Scenario::new(deck(&[("Forest", 40)]), deck(&[("Forest", 40)]), 97);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.game_mut().state.permanents[forest]
        .as_mut()
        .unwrap()
        .plus1_counters = 1;

    s.pass_priority();
    s.pass_priority();

    let forest_perm = permanent_state(&s, forest);
    assert_eq!(forest_perm.plus1_counters, 1);
    assert!(s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Battlefield, PlayerId(0))
        .contains(&forest_perm.card));
}

/// Observation encoding: counter counts, token flag, Ally/Lesson type tags,
/// and per-player graveyard Lesson counts are all agent-visible.
#[test]
fn observation_encodes_counters_tokens_and_type_tags() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        deck(&[("Plains", 40)]),
        101,
    );
    // A token with counters on the battlefield.
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_all(&mut s);
    let token = s.battlefield_permanents_named(0, "Ally")[0];
    s.game_mut().state.permanents[token]
        .as_mut()
        .unwrap()
        .plus1_counters = 3;

    // A Lesson in the graveyard (subtype injected — no Lesson spell is
    // registered until Stage 2 machinery lands).
    let bolt = find_owned_card_off_battlefield(&s, 0, "Lightning Bolt");
    s.game_mut().state.cards[bolt]
        .subtypes
        .push("Lesson".to_string());
    s.game_mut().move_card(bolt, ZoneType::Graveyard);
    s.pass_priority();

    let events = s.game_mut().take_observation_events();
    let obs = Observation::new(s.game(), &events);
    let (me, them) = if obs.agent.player_index == 0 {
        (&obs.agent, &obs.opponent)
    } else {
        (&obs.opponent, &obs.agent)
    };
    assert_eq!(me.graveyard_lessons, 1);
    assert_eq!(them.graveyard_lessons, 0);

    let all_cards = obs.agent_cards.iter().chain(obs.opponent_cards.iter());
    let token_card = all_cards
        .clone()
        .find(|card| card.name == "Ally")
        .expect("token card should be observable");
    assert!(token_card.is_token);
    assert!(token_card.is_ally);
    assert!(!token_card.is_lesson);

    let lesson_card = all_cards
        .clone()
        .find(|card| card.name == "Lightning Bolt")
        .expect("graveyard lesson should be observable");
    assert!(lesson_card.is_lesson);
    assert!(!lesson_card.is_token);

    let token_perm = obs
        .agent_permanents
        .iter()
        .chain(obs.opponent_permanents.iter())
        .find(|perm| perm.plus1_counters == 3)
        .expect("counter count should be observable");
    assert!(!token_perm.cant_be_blocked_this_turn);
}

#[test]
fn modal_mode_one_gains_life() {
    let mut s = crossroads_setup(111);
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::Modal);
    assert_eq!(space.actions.len(), 2);
    assert!(space
        .actions
        .iter()
        .all(|action| action.action_type() == ActionType::ChooseMode));
    s.step_action(0);
    s.pass_priority();
    s.pass_priority();
    s.assert_life(0, 23);
}

#[test]
fn modal_mode_two_draws_a_card() {
    let mut s = crossroads_setup(112);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    s.step_action(1);
    let hand_before = hand_of(&s, 0).len();
    s.pass_priority();
    s.pass_priority();
    assert_eq!(hand_of(&s, 0).len(), hand_before + 1);
}

#[test]
fn cleanup_sba_grants_priority_and_then_repeats_mandatory_discard() {
    let mut s = ur_lessons_setup();
    let land = put_permanent(&mut s, 0, "Island");
    let card = card_of(&s, land);
    // A 0/0 animated land survives only while its until-end-of-turn buff lasts.
    let permanent = s.game_mut().state.permanents[land].as_mut().unwrap();
    permanent.animated = true;
    permanent.temp_toughness = 1;
    s.game_mut().draw_cards(PlayerId(0), 4);
    s.advance_to_active_step(0, StepKind::End);
    resolve(&mut s);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.turn.current_step_kind(), StepKind::Cleanup);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);

    // CR 514.3a: cards drawn during exceptional cleanup priority must be
    // discarded in a new cleanup step before the next turn starts.
    let turn = s.game().state.turn.turn_number;
    s.game_mut().draw_cards(PlayerId(0), 2);
    resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(s.game().state.turn.turn_number, turn);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
    assert!(s.game().state.turn.turn_number > turn);
}

#[test]
fn revised_forty_card_deck_completes_and_replays_every_command_exactly() {
    let semantic = SemanticPack::current("two_deck").unwrap();
    let mut ur = semantic
        .player_config("UR candidate", "ur_lessons")
        .unwrap();
    for (name, n) in [
        ("It'll Quench Ya!", 2),
        ("First-Time Flyer", 2),
        ("Pop Quiz", 1),
        ("Igneous Inspiration", 1),
    ] {
        *ur.decklist.get_mut(name).unwrap() -= n;
    }
    ur.decklist.retain(|_, n| *n > 0);
    for (name, n) in [
        ("Combustion Technique", 2),
        ("Proft's Eidetic Memory", 1),
        ("Gran-Gran", 1),
        ("Accumulate Wisdom", 1),
    ] {
        *ur.decklist.entry(name.into()).or_default() += n;
    }
    assert_eq!(ur.decklist.values().sum::<usize>(), 40);
    let gw = semantic.player_config("GW", "gw_allies").unwrap();
    for seed in 0..4 {
        for reverse in [false, true] {
            let configs = if reverse {
                vec![gw.clone(), ur.clone()]
            } else {
                vec![ur.clone(), gw.clone()]
            };
            let mut game = Game::new(configs.clone(), seed, true);
            let mut replay = Game::new(configs, seed, true);
            for step in 0..20_000_u64 {
                if game.is_game_over() {
                    break;
                }
                let frame = game.semantic_decision_frame().unwrap();
                let i = ((step
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed + 1)) as usize)
                    % frame.offers.len();
                let command = Command {
                    command_id: format!("etu88-{step}"),
                    expected_revision: frame.revision,
                    offer_id: frame.offers[i].id.0,
                    answers: vec![],
                    object_preconditions: vec![],
                };
                game.execute_semantic_command(&command).unwrap();
                replay.execute_semantic_command(&command).unwrap();
                assert_eq!(
                    game.state.deterministic_hash(),
                    replay.state.deterministic_hash()
                );
                assert_eq!(
                    serde_json::to_value(&game.state.events).unwrap(),
                    serde_json::to_value(&replay.state.events).unwrap()
                );
                assert_eq!(
                    Observation::new(&game, &[]).to_json(),
                    Observation::new(&replay, &[]).to_json()
                );
            }
            assert!(game.is_game_over(), "seed {seed}, reverse {reverse}");
        }
    }
}
