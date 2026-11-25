//! Source choices and automatic completion share the pending payment.

use managym::state::player::ManaFloats;
use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionType},
    state::{
        game_object::{PermanentId, PlayerId},
        mana::{Color, Mana},
        payment::PaymentReceipt,
        player::{ManaPayment, PlayerConfig},
        stack_object::StackObject,
    },
    Game,
};

fn game(lands: &[&str]) -> Game {
    game_with(lands, &[("Mountain", 20), ("Forest", 12), ("Gray Ogre", 8)])
}

fn game_with(lands: &[&str], deck: &[(&str, usize)]) -> Game {
    position(lands, deck, "Gray Ogre")
}

fn position(lands: &[&str], deck: &[(&str, usize)], spell: &str) -> Game {
    let deck: BTreeMap<String, usize> = deck
        .iter()
        .copied()
        .map(|(name, count)| (name.to_string(), count))
        .collect();
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", deck.clone()),
            PlayerConfig::new("p1", deck),
        ],
        347,
        true,
    );
    for _ in 0..2 {
        game.step(0).unwrap(); // Both seats keep.
    }
    let player = PlayerId(0);
    game.scenario_clear_hand(player);
    game.scenario_force_card_in_hand(player, spell).unwrap();
    for land in lands {
        game.scenario_force_battlefield(player, land, true).unwrap();
    }
    game.scenario_refresh_priority().unwrap();
    game
}

fn cast(game: &mut Game) {
    let space = game.action_space().expect("priority").clone();
    let index = space
        .actions
        .iter()
        .position(|action| action.action_type() == ActionType::PriorityCastSpell)
        .expect("the spell is castable");
    game.step(index).unwrap();
}

fn play_land(game: &mut Game, land: &str) {
    game.scenario_force_card_in_hand(PlayerId(0), land).unwrap();
    game.scenario_refresh_priority().unwrap();
    let space = game.action_space().expect("priority").clone();
    let index = space
        .actions
        .iter()
        .position(|action| {
            matches!(action, Action::PlayLand { object, .. } if game.state[*object].name == land)
        })
        .unwrap_or_else(|| panic!("{land} is playable from hand"));
    game.step(index).unwrap();
}

fn name(game: &Game, permanent: PermanentId) -> String {
    let object = game.state.permanents[permanent].as_ref().unwrap().object;
    game.state[object].name.clone()
}

/// The tap offers of the pending payment, as (action index, land name).
fn taps(game: &Game) -> Vec<(usize, String)> {
    game.action_space()
        .expect("payment decision")
        .actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| match action {
            Action::TapForMana { permanent, .. } => Some((index, name(game, *permanent))),
            _ => None,
        })
        .collect()
}

/// Untapped lands; the Ogre may already have resolved.
fn untapped(game: &Game) -> Vec<String> {
    let mut names: Vec<String> = game
        .state
        .permanents
        .iter()
        .flatten()
        .filter(|permanent| !permanent.tapped)
        .map(|permanent| game.state[permanent.object].name.clone())
        .filter(|land| land != "Gray Ogre")
        .collect();
    names.sort();
    names
}

#[test]
fn per_source_taps_choose_what_stays_untapped_and_pay_cost_finishes() {
    // {2}{R} from three Mountains and a Forest: every land can pay a pip and
    // one will be left, so all four are offered beside PayCost.
    let lands = ["Mountain", "Mountain", "Mountain", "Forest"];
    let mut by_hand = game(&lands);
    cast(&mut by_hand);
    let offers = taps(&by_hand);
    assert_eq!(offers.len(), 4);
    let space = by_hand.action_space().unwrap();
    assert_eq!(space.player, Some(PlayerId(0)));
    assert!(space
        .actions
        .iter()
        .any(|action| matches!(action, Action::PayCost { .. })));

    // Choosing the Forest leaves only interchangeable Mountains to pay.
    let forest = offers.iter().find(|(_, land)| land == "Forest").unwrap().0;
    by_hand.step(forest).unwrap();
    assert!(by_hand.pending_choice.is_none());
    assert_eq!(untapped(&by_hand), ["Mountain"]);

    // With two Mountains and two Forests, tapping a Forest leaves {1}{R} and
    // tapping a Mountain leaves {1}, keeping the Mountain-vs-Forest choice
    // until the final tap finishes the payment.
    let mut multi_step = game(&["Mountain", "Mountain", "Forest", "Forest"]);
    cast(&mut multi_step);
    let forest = taps(&multi_step)
        .into_iter()
        .find(|(_, land)| land == "Forest")
        .unwrap()
        .0;
    multi_step.step(forest).unwrap();
    let offers = taps(&multi_step);
    assert_eq!(offers.len(), 3);
    let mountain = offers
        .iter()
        .find(|(_, land)| land == "Mountain")
        .unwrap()
        .0;
    multi_step.step(mountain).unwrap();
    let offers = taps(&multi_step);
    assert_eq!(offers.len(), 2);
    let mountain = offers
        .iter()
        .find(|(_, land)| land == "Mountain")
        .unwrap()
        .0;
    multi_step.step(mountain).unwrap();
    assert!(multi_step.pending_choice.is_none());
    assert_eq!(untapped(&multi_step), ["Forest"]);

    // PayCost at once is the automatic payer: Mountains first, in order.
    let mut automatic = game(&lands);
    cast(&mut automatic);
    let pay = automatic
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::PayCost { .. }))
        .unwrap();
    automatic.step(pay).unwrap();
    assert_eq!(untapped(&automatic), ["Forest"]);
}

fn payment_position(spell: &str, sources: &[&str]) -> Game {
    let mut deck = vec![("Mountain", 20), (spell, 8)];
    for source in sources {
        if !deck.iter().any(|(name, _)| name == source) {
            deck.push((source, 8));
        }
    }
    position(sources, &deck, spell)
}

fn payment_labels(game: &Game) -> Vec<String> {
    game.semantic_decision_frame()
        .unwrap()
        .offers
        .into_iter()
        .map(|offer| offer.label)
        .collect()
}

#[test]
fn exact_dual_land_payments_cast_without_a_source_or_color_prompt() {
    for lands in [
        ["Gruul Guildgate", "Forest", "Forest"],
        ["Gruul Guildgate", "Mountain", "Forest"],
        ["Gruul Guildgate"; 3],
    ] {
        let mut game = payment_position("Gray Ogre", &lands);
        cast(&mut game);
        assert!(game.pending_choice.is_none(), "{lands:?}");
        assert!(untapped(&game).is_empty(), "{lands:?}");
        assert_eq!(spell_receipt(&game, "Gray Ogre").total(), 3);
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn a_spare_source_keeps_every_payment_offer_including_the_dual() {
    let mut game = payment_position(
        "Gray Ogre",
        &["Gruul Guildgate", "Mountain", "Mountain", "Forest"],
    );
    cast(&mut game);
    let labels = payment_labels(&game);
    assert_eq!(labels.len(), 5, "{labels:?}");
    assert_eq!(labels[0], "Tap Gruul Guildgate for {R}/{G}");
    assert_eq!(taps(&game).len(), 3);
    game.step(0).unwrap();
    // Red leaves two generic pips, green leaves a red pip and a generic.
    assert_eq!(payment_labels(&game), ["Add {R}", "Add {G}"]);
}

#[test]
fn required_dual_pays_with_one_of_two_interchangeable_plains() {
    let mut game = payment_position("Badgermole Cub", &["Plains", "Plains", "Kyoshi Village"]);
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
    assert_eq!(
        untapped(&game)
            .iter()
            .filter(|name| name.as_str() == "Plains")
            .count(),
        1
    );
    assert!(!untapped(&game).contains(&"Kyoshi Village".to_string()));
}

#[test]
fn interchangeable_single_unit_sources_pay_and_leave_the_spare_untapped() {
    for (spell, land, count, owed) in [
        ("Gray Ogre", "Mountain", 4, 3),
        ("Leonin Skyhunter", "Plains", 3, 2),
        ("Firebrand Archer", "Gruul Guildgate", 3, 2),
    ] {
        let mut game = payment_position(spell, &vec![land; count]);
        cast(&mut game);
        assert!(game.pending_choice.is_none(), "{spell}");
        assert_eq!(untapped(&game).iter().filter(|s| *s == land).count(), 1);
        assert_eq!(spell_receipt(&game, spell).total(), owed);
        assert_eq!(game.state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn choosing_village_keeps_two_plains_and_skips_the_color_choice() {
    let mut game = payment_position(
        "Youthful Knight",
        &["Kyoshi Village", "Plains", "Plains", "Plains"],
    );
    cast(&mut game);
    assert_eq!(
        payment_labels(&game),
        [
            "Tap Kyoshi Village for {G}/{W}",
            "Tap Plains for {W}",
            "Tap Plains for {W}",
            "Tap Plains for {W}",
            "PayCost",
        ]
    );
    assert!(game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .any(|action| matches!(action, Action::PayCost { .. })));
    game.step(0).unwrap();
    assert!(game.pending_choice.is_none());
    assert_eq!(untapped(&game).iter().filter(|s| *s == "Plains").count(), 2);
    assert!(!untapped(&game).iter().any(|s| s == "Kyoshi Village"));
    assert_eq!(spell_receipt(&game, "Youthful Knight").total(), 2);
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
}

#[test]
fn playing_a_plain_land_for_turn_stays_interchangeable_for_manual_payment() {
    // 4th Mountain played this turn still skips Gray Ogre {2}{R}.
    let mut mountains = payment_position("Gray Ogre", &["Mountain"; 3]);
    play_land(&mut mountains, "Mountain");
    assert!(mountains
        .state
        .permanents
        .iter()
        .flatten()
        .any(|p| p.summoning_sick && p.entered_this_turn(mountains.state.turn.turn_number)));
    cast(&mut mountains);
    assert!(mountains.pending_choice.is_none());
    assert_eq!(untapped(&mountains), ["Mountain"]);
    assert_eq!(spell_receipt(&mountains, "Gray Ogre").total(), 3);
    assert_eq!(mountains.state.players[0].mana_pool.total(), 0);

    // 2nd Plains played this turn beside Kyoshi Village still skips Badgermole Cub {1}{G}.
    let mut village_pair = payment_position("Badgermole Cub", &["Plains", "Kyoshi Village"]);
    play_land(&mut village_pair, "Plains");
    cast(&mut village_pair);
    assert!(village_pair.pending_choice.is_none());
    assert_eq!(
        untapped(&village_pair)
            .iter()
            .filter(|name| name.as_str() == "Plains")
            .count(),
        1
    );
    assert!(!untapped(&village_pair).contains(&"Kyoshi Village".to_string()));
    assert_eq!(village_pair.state.players[0].mana_pool.total(), 0);

    // 3rd Plains played this turn beside Kyoshi Village + 2 Plains: tapping Village
    // skips the color modal and finishes directly.
    let mut village_three =
        payment_position("Youthful Knight", &["Kyoshi Village", "Plains", "Plains"]);
    play_land(&mut village_three, "Plains");
    cast(&mut village_three);
    assert_eq!(
        payment_labels(&village_three),
        [
            "Tap Kyoshi Village for {G}/{W}",
            "Tap Plains for {W}",
            "Tap Plains for {W}",
            "Tap Plains for {W}",
            "PayCost",
        ]
    );
    village_three.step(0).unwrap();
    assert!(village_three.pending_choice.is_none());
    assert_eq!(
        untapped(&village_three)
            .iter()
            .filter(|name| name.as_str() == "Plains")
            .count(),
        2
    );
    assert!(!untapped(&village_three).contains(&"Kyoshi Village".to_string()));
    assert_eq!(spell_receipt(&village_three, "Youthful Knight").total(), 2);
    assert_eq!(village_three.state.players[0].mana_pool.total(), 0);
}

#[test]
fn self_animating_land_played_this_turn_stays_distinct() {
    let mut game = payment_position("Sol Ring", &["Soulstone Sanctuary"]);
    play_land(&mut game, "Soulstone Sanctuary");
    cast(&mut game);
    // One Soulstone Sanctuary can animate and attack this turn; the newly
    // played one is summoning-sick, so the choice of which to tap stays open.
    assert_eq!(taps(&game).len(), 2);
}

#[test]
fn a_targeted_plains_keeps_its_tap_offer_in_response() {
    use managym::state::game_object::Target;

    let player = PlayerId(0);
    let deck = [
        ("Plains", 24),
        ("Imprisoned in the Moon", 8),
        ("Teferi's Protection", 8),
    ];
    let mut game = position(&["Plains"; 4], &deck, "Imprisoned in the Moon");
    game.skip_trivial = false;
    game.scenario_force_card_in_hand(player, "Teferi's Protection")
        .unwrap();
    game.state.players[0].mana_pool = Mana::parse("2U").into();
    game.scenario_refresh_priority().unwrap();
    let aura = game.action_space().unwrap().actions.iter().position(|action|
        matches!(action, Action::CastSpell { object, .. } if game.state[*object].name == "Imprisoned in the Moon")
    ).unwrap();
    game.step(aura).unwrap();
    let target = Target::Permanent(PermanentId(0));
    let index = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::ChooseTarget { target: t, .. } if *t == target))
        .unwrap();
    game.step(index).unwrap();
    assert_eq!(
        game.state.stack_objects.last().unwrap().targets(),
        &[target]
    );
    game.skip_trivial = true;
    cast(&mut game);
    assert_eq!(taps(&game).len(), 4);
}

#[test]
fn enchanted_and_animated_plains_keep_the_source_choice() {
    use managym::state::effect::{Affected, ContinuousEffect, EffectDuration, Modification};
    let player = PlayerId(0);
    let deck = [
        ("Plains", 24),
        ("Imprisoned in the Moon", 8),
        ("Youthful Knight", 8),
    ];
    let base = position(&["Plains"; 3], &deck, "Youthful Knight");
    let plains = base.state.permanents[PermanentId(0)].as_ref().unwrap();
    let mut enchanted = base.clone();
    let aura = enchanted
        .scenario_force_battlefield(player, "Imprisoned in the Moon", true)
        .unwrap();
    enchanted.state.permanents[aura]
        .as_mut()
        .unwrap()
        .attached_to = Some(plains.object);
    enchanted.scenario_refresh_priority().unwrap();
    cast(&mut enchanted);
    assert_eq!(payment_labels(&enchanted).len(), 4);
    assert!(payment_labels(&enchanted).contains(&"Tap Plains for {C}".into()));

    let mut animated = base.clone();
    for modification in [
        Modification::BecomeCreature,
        Modification::SetBasePt {
            power: Some(2),
            toughness: Some(2),
        },
    ] {
        animated.state.effects.push(ContinuousEffect {
            timestamp: plains.timestamp,
            source: None,
            controller: player,
            affected: Affected::Object(plains.object),
            modification,
            duration: EffectDuration::EndOfTurn,
            source_object: None,
        });
    }
    animated.scenario_refresh_priority().unwrap();
    cast(&mut animated);
    assert_eq!(taps(&animated).len(), 3);
}

#[test]
fn a_dual_paying_generic_mana_skips_only_equivalent_colors() {
    let mut game = payment_position(
        "Gray Ogre",
        &["Kyoshi Village", "Plains", "Mountain", "Island"],
    );
    cast(&mut game);
    tap(&mut game, "Mountain");
    let labels = payment_labels(&game);
    let dual = labels
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    game.step(dual).unwrap();
    assert_eq!(owed(&game), "{1}");
    assert_eq!(taps(&game).len(), 2);
    assert_eq!(committed(&game).len(), 2);
}

#[test]
fn two_forced_completions_tapping_different_lands_keep_the_color_modal() {
    let mut game = payment_position(
        "Anthem of Champions",
        &["Kyoshi Village", "Forest", "Plains"],
    );
    cast(&mut game);
    let dual = payment_labels(&game)
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    game.step(dual).unwrap();
    assert_eq!(payment_labels(&game), ["Add {G}", "Add {W}"]);
    for (mode, kept) in [(0, "Forest"), (1, "Plains")] {
        let mut branch = game.clone();
        branch.step(mode).unwrap();
        assert!(branch.pending_choice.is_none());
        assert!(untapped(&branch).contains(&kept.to_string()));
        assert_eq!(branch.state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn mana_abilities_on_one_permanent_only_fund_one_tap() {
    for source in ["Jasmine Dragon Tea Shop", "Kyoshi Village"] {
        let mut game = payment_position("Compassionate Healer", &[source, "Plains"]);
        cast(&mut game);
        assert!(game.pending_choice.is_none());
        for permanent in game.state.permanents.iter().flatten() {
            if [source, "Plains"].contains(&game.state[permanent.object].name.as_str()) {
                assert!(permanent.tapped);
            }
        }
    }
    let mut game = payment_position("Badgermole Cub", &["Animal Attendant", "Plains"]);
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    // The spend effect is part of the receipt whichever color was chosen.
    assert_eq!(spell_receipt(&game, "Badgermole Cub").entry_counters(), 1);
}

#[test]
fn turning_a_manifest_face_up_skips_when_forced_and_offers_taps_on_real_choice() {
    let player = PlayerId(0);
    let turn_face_up = |sources: &[&str]| {
        let mut game = payment_position("Compassionate Healer", sources);
        let card = game
            .state
            .zones
            .zone_objects(managym::state::zone::ZoneType::Hand, player)
            .iter()
            .copied()
            .next()
            .unwrap();
        let face_down = game.manifest(card, player);
        game.scenario_refresh_priority().unwrap();
        let index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::TurnFaceUp { object, .. } if *object == face_down))
            .unwrap();
        game.step(index).unwrap();
        (game, face_down)
    };

    // Forced exact (dual + basic) and interchangeable basics skip without a prompt.
    for (sources, expected_untapped_lands) in [
        (&["Kyoshi Village", "Plains"][..], 0_usize),
        (&["Plains", "Plains", "Plains"][..], 1_usize),
    ] {
        let (game, face_down) = turn_face_up(sources);
        assert!(!game.state[face_down].is_face_down(), "{sources:?}");
        assert_eq!(game.state.players[0].mana_pool.total(), 0, "{sources:?}");
        let untapped_lands = untapped(&game)
            .into_iter()
            .filter(|name| name != "Compassionate Healer")
            .count();
        assert_eq!(untapped_lands, expected_untapped_lands, "{sources:?}");
    }

    // Real choice (dual + spare basics or mixed basics): offers per-source taps
    // beside PayCost; clicking Kyoshi Village beside 3 Plains skips the color
    // modal and finishes directly, leaving 2 Plains untapped.
    let (mut choice_dual, face_down) =
        turn_face_up(&["Kyoshi Village", "Plains", "Plains", "Plains"]);
    assert!(choice_dual.state[face_down].is_face_down());
    let labels = payment_labels(&choice_dual);
    assert!(
        labels.contains(&"Tap Kyoshi Village for {G}/{W}".into()),
        "{labels:?}"
    );
    assert_eq!(taps(&choice_dual).len(), 3);
    let dual = labels
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    choice_dual.step(dual).unwrap();
    assert!(!choice_dual.state[face_down].is_face_down());
    assert_eq!(choice_dual.state.players[0].mana_pool.total(), 0);
    let remaining: Vec<_> = untapped(&choice_dual)
        .into_iter()
        .filter(|name| name != "Compassionate Healer")
        .collect();
    assert_eq!(remaining, ["Plains", "Plains"]);

    // Mixed basics offer per-source taps and PayCost finishes.
    let (mut choice_basics, face_down) = turn_face_up(&["Plains", "Plains", "Forest"]);
    assert!(choice_basics.state[face_down].is_face_down());
    assert_eq!(taps(&choice_basics).len(), 3);
    pay_cost(&mut choice_basics);
    assert!(!choice_basics.state[face_down].is_face_down());
    assert_eq!(choice_basics.state.players[0].mana_pool.total(), 0);
    let remaining_count = untapped(&choice_basics)
        .into_iter()
        .filter(|name| name != "Compassionate Healer")
        .count();
    assert_eq!(remaining_count, 1);
}

#[test]
fn ward_offers_per_source_taps_on_real_choice_and_skips_when_forced() {
    let player = PlayerId(0);
    let deck = [
        ("Mountain", 20),
        ("Lightning Bolt", 8),
        ("Kyoshi Village", 4),
        ("Plains", 8),
        ("Forest", 8),
        ("Waterfall Aerialist", 4),
        ("Sol Ring", 4),
    ];
    let cast_at_ward = |sources: &[&str], with_treasure: bool| {
        let mut game = position(sources, &deck, "Lightning Bolt");
        let target = game
            .scenario_force_battlefield(PlayerId(1), "Waterfall Aerialist", true)
            .unwrap();
        let treasure =
            with_treasure.then(|| game.create_token("token.treasure", player, false).unwrap());
        game.state.players[0].mana_pool = Mana::parse("R").into();
        game.scenario_refresh_priority().unwrap();
        cast(&mut game);
        let index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| {
                matches!(
                    a,
                    Action::ChooseTarget {
                        target: managym::state::game_object::Target::Permanent(p),
                        ..
                    } if *p == target
                )
            })
            .unwrap();
        game.step(index).unwrap();
        if game.pending_choice.is_some() {
            pay_cost(&mut game);
        }
        (game, treasure)
    };

    // Exact dual + basic and interchangeable basics: PayCost or Decline only.
    for sources in [
        &["Kyoshi Village", "Plains"][..],
        &["Plains", "Plains", "Plains"][..],
    ] {
        let (mut forced, _) = cast_at_ward(sources, false);
        assert_eq!(
            forced.action_space().unwrap().actions,
            [Action::PayCost { player }, Action::Decline { player }],
            "{sources:?}"
        );
        let mut decline = forced.clone();
        decline.step(1).unwrap();
        assert_eq!(
            untapped(&decline)
                .into_iter()
                .filter(|name| name != "Waterfall Aerialist")
                .count(),
            sources.len(),
            "{sources:?}"
        );
        forced.step(0).unwrap();
        assert_eq!(
            untapped(&forced)
                .into_iter()
                .filter(|name| name != "Waterfall Aerialist")
                .count(),
            sources.len() - 2,
            "{sources:?}"
        );
    }

    // Real choice (dual or basic beside Sol Ring): both offer per-source taps
    // beside PayCost/Decline, and PayCost still taps Sol Ring while leaving the
    // land untapped.
    for first in ["Kyoshi Village", "Plains"] {
        let (mut game, _) = cast_at_ward(&[first, "Sol Ring"], false);
        let labels = payment_labels(&game);
        assert!(
            labels
                .iter()
                .any(|l| l.starts_with(&format!("Tap {first} for "))),
            "{first}: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l.starts_with("Tap Sol Ring for ")),
            "{first}: {labels:?}"
        );
        assert!(game
            .action_space()
            .unwrap()
            .actions
            .contains(&Action::PayCost { player }));
        assert!(game
            .action_space()
            .unwrap()
            .actions
            .contains(&Action::Decline { player }));
        pay_cost(&mut game);
        assert!(untapped(&game).contains(&first.to_string()), "{first}");
        assert!(
            !untapped(&game).contains(&"Sol Ring".to_string()),
            "{first}"
        );
    }

    // Real choice on Ward {2} with Kyoshi Village + Mountain + Forest: tapping
    // Kyoshi Village skips the redundant {G}/{W} modal and returns to the
    // remaining choice between Mountain and Forest.
    let (mut dual_choice, _) = cast_at_ward(&["Kyoshi Village", "Mountain", "Forest"], false);
    let dual = payment_labels(&dual_choice)
        .iter()
        .position(|l| l == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    dual_choice.step(dual).unwrap();
    assert_eq!(taps(&dual_choice).len(), 2);
    tap(&mut dual_choice, "Mountain");
    assert_eq!(
        dual_choice.action_space().unwrap().actions,
        [Action::PayCost { player }, Action::Decline { player }]
    );
    pay_cost(&mut dual_choice);
    assert!(untapped(&dual_choice).contains(&"Forest".into()));

    // A second Kyoshi Village tap completes Ward {2} beside the first one's
    // floating mana: either color is spent whole, so no {G}/{W} modal.
    let (mut second_dual, _) = cast_at_ward(&["Kyoshi Village", "Kyoshi Village", "Plains"], false);
    for _ in 0..2 {
        let dual = payment_labels(&second_dual)
            .iter()
            .position(|l| l == "Tap Kyoshi Village for {G}/{W}")
            .unwrap();
        second_dual.step(dual).unwrap();
    }
    assert_eq!(
        second_dual.action_space().unwrap().actions,
        [Action::PayCost { player }, Action::Decline { player }]
    );
    pay_cost(&mut second_dual);
    assert_eq!(
        untapped(&second_dual)
            .into_iter()
            .filter(|name| name != "Waterfall Aerialist")
            .collect::<Vec<_>>(),
        ["Plains"]
    );

    // Non-simple mana abilities (Treasure) remain offered alongside PayCost/Decline.
    let (with_treasure, treasure) = cast_at_ward(&["Plains", "Plains"], true);
    let treasure = treasure.unwrap();
    let actions = &with_treasure.action_space().unwrap().actions;
    assert!(
        actions.iter().any(
            |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == treasure)
        ),
        "{actions:?}"
    );
    assert!(actions.contains(&Action::PayCost { player }));
    assert!(actions.contains(&Action::Decline { player }));
}

#[test]
fn partial_multi_tap_resolution_payments_skip_redundant_color_modals() {
    let player = PlayerId(0);
    // TurnFaceUp {2}{W} with Plains + Kyoshi Village + Mountain + Forest:
    // tapping Plains ({W} in pool) then Kyoshi Village leaves {1} under either
    // {G} or {W}, so the {G}/{W} modal skips and Mountain vs Forest stays open.
    let mut turn_up = payment_position(
        "Cat Collector",
        &["Plains", "Kyoshi Village", "Mountain", "Forest"],
    );
    let card = turn_up
        .state
        .zones
        .zone_objects(managym::state::zone::ZoneType::Hand, player)
        .iter()
        .copied()
        .next()
        .unwrap();
    let face_down = turn_up.manifest(card, player);
    turn_up.scenario_refresh_priority().unwrap();
    let index = turn_up
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::TurnFaceUp { object, .. } if *object == face_down))
        .unwrap();
    turn_up.step(index).unwrap();
    tap(&mut turn_up, "Plains");
    let dual = payment_labels(&turn_up)
        .iter()
        .position(|l| l == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    turn_up.step(dual).unwrap();
    // Skipped the color modal; Mountain and Forest remain offered beside PayCost.
    let offers = taps(&turn_up);
    assert_eq!(offers.len(), 2, "{offers:?}");
    tap(&mut turn_up, "Mountain");
    assert!(!turn_up.state[face_down].is_face_down());
    assert_eq!(turn_up.state.players[0].mana_pool.total(), 0);
    assert_eq!(
        untapped(&turn_up)
            .into_iter()
            .filter(|name| name != "Cat Collector")
            .collect::<Vec<_>>(),
        ["Forest"]
    );

    // Multi-unit remaining source (Sol Ring) can overshoot into floating mana,
    // so tapping Kyoshi Village beside Plains + Sol Ring keeps the color modal.
    let mut with_sol_ring =
        payment_position("Cat Collector", &["Plains", "Kyoshi Village", "Sol Ring"]);
    let card = with_sol_ring
        .state
        .zones
        .zone_objects(managym::state::zone::ZoneType::Hand, player)
        .iter()
        .copied()
        .next()
        .unwrap();
    let face_down = with_sol_ring.manifest(card, player);
    with_sol_ring.scenario_refresh_priority().unwrap();
    let index = with_sol_ring
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::TurnFaceUp { object, .. } if *object == face_down))
        .unwrap();
    with_sol_ring.step(index).unwrap();
    tap(&mut with_sol_ring, "Plains");
    let dual = payment_labels(&with_sol_ring)
        .iter()
        .position(|l| l == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    with_sol_ring.step(dual).unwrap();
    assert_eq!(payment_labels(&with_sol_ring), ["Add {G}", "Add {W}"]);

    // Optional resolution payment {4} (Mystic Remora trigger) with Plains +
    // Kyoshi Village + Mountain + Forest + Island: tapping Plains then Kyoshi
    // Village skips the {G}/{W} modal and keeps the remaining land choice open.
    let deck = [
        ("Mountain", 12),
        ("Forest", 8),
        ("Island", 8),
        ("Plains", 8),
        ("Kyoshi Village", 4),
        ("Lightning Bolt", 4),
        ("Mystic Remora", 4),
    ];
    let mut remora = position(
        &["Plains", "Kyoshi Village", "Mountain", "Forest", "Island"],
        &deck,
        "Lightning Bolt",
    );
    remora
        .scenario_force_battlefield(PlayerId(1), "Mystic Remora", true)
        .unwrap();
    remora.state.players[0].mana_pool = Mana::parse("R").into();
    remora.scenario_refresh_priority().unwrap();
    cast(&mut remora);
    let target = remora
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::ChooseTarget {
                    target: managym::state::game_object::Target::Player(PlayerId(1)),
                    ..
                }
            )
        })
        .unwrap();
    remora.step(target).unwrap();
    // Pass priority twice to resolve Mystic Remora's trigger into the {4} payment.
    for _ in 0..4 {
        if remora
            .action_space()
            .unwrap()
            .actions
            .contains(&Action::Decline { player })
        {
            break;
        }
        let pass = remora
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .unwrap();
        remora.step(pass).unwrap();
    }
    tap(&mut remora, "Plains");
    let dual = payment_labels(&remora)
        .iter()
        .position(|l| l == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    remora.step(dual).unwrap();
    assert_eq!(taps(&remora).len(), 3);
    assert!(remora
        .action_space()
        .unwrap()
        .actions
        .contains(&Action::Decline { player }));
}

#[test]
fn granted_mana_does_not_double_count_a_forest_and_treasure_stays_explicit() {
    let deck = [
        ("Forest", 20),
        ("Great Divide Guide", 4),
        ("Firebrand Archer", 8),
    ];
    let mut game = position(&["Forest", "Forest"], &deck, "Firebrand Archer");
    game.scenario_force_battlefield(PlayerId(0), "Great Divide Guide", false)
        .unwrap();
    game.scenario_refresh_priority().unwrap();
    let mut with_treasure = game.clone();
    let treasure = with_treasure
        .create_token("token.treasure", PlayerId(0), false)
        .unwrap();
    with_treasure.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    cast(&mut with_treasure);
    assert!(with_treasure
        .action_space()
        .unwrap()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == treasure)));
}

#[test]
fn fourteen_lands_with_granted_color_choices_stay_under_the_policy_action_capacity() {
    let deck = [
        ("Forest", 24),
        ("Great Divide Guide", 4),
        ("Firebrand Archer", 8),
    ];
    let mut game = position(&["Forest"; 14], &deck, "Firebrand Archer");
    // One Forest has its own state, so choosing which lands to keep matters.
    game.state.permanents[PermanentId(0)]
        .as_mut()
        .unwrap()
        .change_counters(managym::state::permanent::CounterKind::PlusOnePlusOne, 1);
    game.scenario_force_battlefield(PlayerId(0), "Great Divide Guide", false)
        .unwrap();
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    let labels = payment_labels(&game);
    assert_eq!(
        labels
            .iter()
            .filter(|s| s.as_str() == "Tap Forest for any color")
            .count(),
        14
    );
    assert_eq!(taps(&game).len(), 14);
    assert_eq!(labels.len(), 29);
}

#[test]
fn paying_a_convoke_spell_entirely_in_mana_charges_its_generic_cost_once() {
    let mut game = payment_position("Guardian of Vitu-Ghazi", &["Kyoshi Village", "Forest"]);
    game.state.players[0].mana_pool = Mana::parse("6").into();
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    assert_eq!(spell_receipt(&game, "Guardian of Vitu-Ghazi").total(), 8);
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
}

#[test]
fn lesson_mana_uses_two_units_only_when_the_cost_needs_both() {
    let mut exact = payment_position("Energybending", &["Hermitic Herbalist"]);
    cast(&mut exact);
    assert!(exact.pending_choice.is_none());
    assert_eq!(exact.state.players[0].mana_pool.total(), 0);
    assert!(!untapped(&exact).contains(&"Hermitic Herbalist".into()));

    let mut spare = payment_position("Boomerang Basics", &["Hermitic Herbalist"]);
    cast(&mut spare);
    let labels = payment_labels(&spare);
    assert!(
        labels.contains(&"Tap Hermitic Herbalist for any color".into()),
        "{labels:?}"
    );
    assert!(
        labels.contains(&"Tap Hermitic Herbalist for 2 mana (Lesson spells only)".into()),
        "{labels:?}"
    );
}

#[test]
fn converge_keeps_a_color_choice_with_exact_sources() {
    let mut game = payment_position(
        "Uncle's Musings",
        &["Kyoshi Village", "Forest", "Forest", "Forest"],
    );
    cast(&mut game);
    let labels = payment_labels(&game);
    let dual = labels
        .iter()
        .position(|label| label == "Tap Kyoshi Village for {G}/{W}")
        .unwrap();
    game.step(dual).unwrap();
    assert_eq!(payment_labels(&game), ["Add {G}", "Add {W}"]);
}

#[test]
fn funded_and_all_sources_needed_payments_remain_trivial() {
    let mut exact = game(&["Mountain", "Mountain", "Mountain"]);
    cast(&mut exact);
    assert!(exact.pending_choice.is_none());
    assert!(untapped(&exact).is_empty());

    let lands = ["Mountain", "Mountain", "Mountain", "Forest"];
    let mut funded = game(&lands);
    funded.state.players[0].mana_pool = Mana::parse("2R").into();
    cast(&mut funded);
    assert!(funded.pending_choice.is_none());
    assert_eq!(
        untapped(&funded),
        ["Forest", "Mountain", "Mountain", "Mountain"]
    );
}

#[test]
fn priority_taps_append_each_source_and_float_mana_for_a_cast() {
    let lands = ["Mountain", "Forest", "Forest"];
    let mut game = game(&lands);
    let ordinary = game.action_space().unwrap().actions.clone();
    game.mana_floats = [ManaFloats::Off, ManaFloats::All];
    game.scenario_refresh_priority().unwrap();
    assert_eq!(game.action_space().unwrap().actions, ordinary);

    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    assert_eq!(
        &game.action_space().unwrap().actions[..ordinary.len()],
        ordinary
    );
    let offers = taps(&game);
    assert_eq!(
        offers,
        [
            (ordinary.len(), "Mountain".into()),
            (ordinary.len() + 1, "Forest".into()),
            (ordinary.len() + 2, "Forest".into())
        ]
    );
    game.step(offers[1].0).unwrap();
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
    assert_eq!(game.action_space().unwrap().player, Some(PlayerId(0)));
    assert_eq!(taps(&game).len(), 2);
    cast(&mut game);
    pay_cost(&mut game);
    assert!(game.pending_choice.is_none());
    assert!(untapped(&game).is_empty());
    assert_eq!(game.state.players[0].mana_pool, Mana::default());
}

#[test]
fn funded_cast_can_tap_a_surplus_forest_before_paying() {
    let mut game = game(&["Forest", "Mountain"]);
    game.state.players[0].mana_pool = Mana::parse("2R").into();
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(matches!(
        game.action_space().unwrap().actions[0],
        Action::PayCost { .. }
    ));
    let forest = taps(&game)
        .into_iter()
        .find(|(_, name)| name == "Forest")
        .unwrap()
        .0;
    game.step(forest).unwrap();
    // The Forest's {G} could pay {1} that floating colorless could pay
    // instead, so it waits with the floating mana for the spend choice.
    assert_eq!(*game.state.players[0].mana_pool.mana(), Mana::parse("2RG"));
    assert_eq!(owed(&game), "{2}{R}");
    assert!(committed(&game).is_empty());
    pay_cost(&mut game);
    // Generic spending can ask which floating unit to keep.
    while game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .all(|a| matches!(a, Action::ChooseMode { .. }))
    {
        game.step(0).unwrap();
    }
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.players[0].mana_pool.total(), 1);
    assert_eq!(untapped(&game), ["Mountain"]);
}

/// The mana the pending decision says is still owed.
fn owed(game: &Game) -> String {
    game.decision_context(PlayerId(0))
        .and_then(|context| context.mana_cost.0)
        .expect("a mana cost is being paid")
}

/// What the pending decision says is committed to the cost, in order.
fn committed(game: &Game) -> Vec<String> {
    game.decision_context(PlayerId(0))
        .map(|context| context.paid.into_iter().map(|paid| paid.phrase).collect())
        .unwrap_or_default()
}

fn tap(game: &mut Game, land: &str) {
    let index = taps(game)
        .into_iter()
        .find(|(_, name)| name == land)
        .unwrap_or_else(|| panic!("{land} is offered"))
        .0;
    game.step(index).unwrap();
}

/// CR 601.2g-h: mana a payer makes while paying a cost pays toward it as it
/// is made. The owed cost ticks down, the unit joins the receipt with its
/// source and the decision names it, as a convoke tap does.
#[test]
fn mana_made_while_paying_commits_to_the_cost() {
    let mut game = game(&["Mountain", "Mountain", "Forest", "Forest"]);
    cast(&mut game);
    assert_eq!(owed(&game), "{2}{R}");
    let labels: Vec<_> = game
        .semantic_decision_frame()
        .unwrap()
        .offers
        .iter()
        .map(|offer| offer.label.clone())
        .collect();
    assert!(
        labels.contains(&"Tap Forest for {G}".to_string()),
        "{labels:?}"
    );
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{1}{R}");
    assert_eq!(committed(&game), ["{G} from Forest"]);
    assert_eq!(game.state.players[0].mana_pool, Mana::default());
    // A red unit pays the red pip, which leaves the most that can pay the rest.
    tap(&mut game, "Mountain");
    assert_eq!(owed(&game), "{1}");
    assert_eq!(committed(&game), ["{G} from Forest", "{R} from Mountain"]);

    // A fork at the last tap finishes once each way.
    for last in ["Mountain", "Forest"] {
        let mut fork = game.clone();
        tap(&mut fork, last);
        assert!(fork.pending_choice.is_none());
        let receipt = spell_receipt(&fork, "Gray Ogre");
        let colors: Vec<u8> = receipt.mana().iter().map(|unit| unit.color).collect();
        let third = if last == "Forest" { GREEN } else { RED };
        assert_eq!(colors, [GREEN, RED, third]);
        assert_eq!(untapped(&fork).len(), 1);
        assert_eq!(fork.state.players[0].mana_pool, Mana::default());
        assert_eq!(owed(&game), "{1}");
    }
}

#[test]
fn required_sol_ring_can_leave_a_mountain_or_floating_mana() {
    let mut game = payment_position("Gray Ogre", &["Sol Ring", "Mountain", "Mountain"]);
    cast(&mut game);
    assert_eq!(taps(&game).len(), 3);
    game.skip_trivial = false; // Inspect priority before the floating mana empties.

    let mut exact = game.clone();
    pay_cost(&mut exact);
    assert_eq!(spell_receipt(&exact, "Gray Ogre").total(), 3);
    assert_eq!(untapped(&exact), ["Mountain"]);
    assert_eq!(exact.state.players[0].mana_pool.total(), 0);

    tap(&mut game, "Mountain");
    tap(&mut game, "Mountain");
    if game.pending_choice.is_some() {
        pay_cost(&mut game);
    }
    assert_eq!(spell_receipt(&game, "Gray Ogre").total(), 3);
    assert!(untapped(&game).is_empty());
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("C"));
}

/// A source that makes more than is owed pays what is owed; the rest floats.
#[test]
fn surplus_of_mana_made_while_paying_floats() {
    let deck = [
        ("Mountain", 20),
        ("Forest", 11),
        ("Sol Ring", 1),
        ("Gray Ogre", 8),
    ];
    let mut game = game_with(&["Mountain", "Forest", "Sol Ring", "Mountain"], &deck);
    // Floats keep priority a real choice after the cast, so the pool is seen.
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    tap(&mut game, "Mountain");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{1}");
    // The Sol Ring's {C} paid the last {1}: the Ogre is cast, the surplus
    // floats.
    tap(&mut game, "Sol Ring");
    assert!(game.pending_choice.is_none());
    assert_eq!(untapped(&game), ["Mountain"]);
    assert_eq!(
        spell_receipt(&game, "Gray Ogre").amounts(),
        [0, 0, 0, 1, 1, 1]
    );
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("C"));
}

/// Mana that can pay nothing still owed stays in the pool, and the cost
/// does not move.
#[test]
fn mana_made_while_paying_that_cannot_pay_floats() {
    // A spare Mountain keeps priority a real choice after the cast.
    let mut game = game(&["Mountain", "Mountain", "Forest", "Forest", "Forest"]);
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    tap(&mut game, "Forest");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{R}");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{R}");
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
    assert_eq!(committed(&game).len(), 2);
    tap(&mut game, "Mountain");
    assert!(game.pending_choice.is_none());
    assert_eq!(
        spell_receipt(&game, "Gray Ogre").amounts(),
        [0, 0, 0, 1, 2, 0]
    );
    assert_eq!(game.state.players[0].mana_pool, Mana::parse("G"));
}

/// A payment ends when no mana is owed, however it was paid: the last unit
/// here comes from an activated mana ability while another one is still
/// untapped beside two lands, floats off, and the Ogre is cast with no
/// "Pay the cost".
#[test]
fn a_payment_ends_when_its_last_unit_comes_from_a_mana_ability() {
    let deck = [
        ("Mountain", 20),
        ("Forest", 10),
        ("Meteorite", 2),
        ("Gray Ogre", 8),
    ];
    let lands = [
        "Mountain",
        "Mountain",
        "Forest",
        "Forest",
        "Meteorite",
        "Meteorite",
    ];
    let mut game = game_with(&lands, &deck);
    cast(&mut game);
    tap(&mut game, "Mountain");
    tap(&mut game, "Forest");
    assert_eq!(owed(&game), "{1}");
    let activate = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|action| matches!(action, Action::ActivateAbility { .. }))
        .expect("a Meteorite's mana ability is offered");
    game.step(activate).unwrap();
    while game.pending_choice.is_some() {
        // The Meteorite's color: any pays {1}.
        let space = game.action_space().unwrap();
        assert!(
            space
                .actions
                .iter()
                .all(|action| matches!(action, Action::ChooseMode { .. })),
            "only the Meteorite's color is chosen: {space:?}"
        );
        game.step(0).unwrap();
    }
    assert!(game.pending_choice.is_none());
    assert_eq!(untapped(&game), ["Forest", "Meteorite", "Mountain"]);
    assert_eq!(
        spell_receipt(&game, "Gray Ogre")
            .amounts()
            .iter()
            .sum::<u8>(),
        3
    );
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
}

fn pay_cost(game: &mut Game) {
    let index = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
        .unwrap();
    game.step(index).unwrap();
}

#[test]
fn surplus_sources_follow_the_existing_payment_actions() {
    let mut off = game(&["Mountain", "Mountain", "Forest"]);
    off.state.players[0].mana_pool = Mana::parse("2").into();
    let mut on = off.clone();
    on.mana_floats = [ManaFloats::All, ManaFloats::Off];
    let mut usable = off.clone();
    usable.mana_floats = [ManaFloats::Usable, ManaFloats::Off];
    cast(&mut off);
    cast(&mut on);
    cast(&mut usable);
    assert_eq!(
        on.action_space().unwrap().actions,
        usable.action_space().unwrap().actions
    );
    let ordinary = &off.action_space().unwrap().actions;
    assert_eq!(taps(&off).len(), 2);
    assert_eq!(
        &on.action_space().unwrap().actions[..ordinary.len()],
        ordinary
    );
    assert_eq!(taps(&on).last(), Some(&(ordinary.len(), "Forest".into())));
    assert_eq!(on.action_space().unwrap().actions.len(), ordinary.len() + 1);
}

#[test]
fn usable_priority_requires_a_spend_or_a_trigger_on_that_source() {
    use managym::state::ability::{Ability, Effect, TriggerCondition, TriggerSubject};
    let mut game = game(&["Mountain", "Forest", "Forest"]);
    game.mana_floats = [ManaFloats::Usable, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    assert_eq!(taps(&game).len(), 3); // Ogre costs {2}{R}.
    game.scenario_clear_hand(PlayerId(0));
    game.scenario_refresh_priority().unwrap();
    assert!(taps(&game).is_empty());

    let mountain = game
        .state
        .permanents
        .iter()
        .flatten()
        .map(|permanent| permanent.object)
        .find(|object| game.state[*object].name == "Mountain")
        .unwrap();
    game.state[mountain].abilities.push(Ability::Triggered {
        condition: TriggerCondition::TappedForMana {
            subject: TriggerSubject::This,
        },
        intervening_if: None,
        effects: vec![Effect::gain_life(1)],
    });
    game.scenario_refresh_priority().unwrap();
    let offers = taps(&game);
    assert_eq!(offers.len(), 1);
    assert_eq!(offers[0].1, "Mountain");
    game.step(offers[0].0).unwrap();
    assert_eq!(game.state.players[0].life, 21);
}

/// The receipt on a spell while it is on the stack,
/// then on the permanent it became.
fn spell_receipt(game: &Game, name: &str) -> PaymentReceipt {
    let cast = game
        .state
        .stack_objects
        .iter()
        .find_map(|object| match object {
            StackObject::Spell(spell) => Some(spell.spent.clone()),
            _ => None,
        });
    cast.unwrap_or_else(|| {
        game.state
            .permanents
            .iter()
            .flatten()
            .find(|permanent| game.state[permanent.object].name == name)
            .expect("the spell was cast")
            .spell_entry
            .as_deref()
            .unwrap()
            .payment
            .clone()
            .unwrap()
    })
}

/// The lands a receipt names, sorted, with the color each one's unit had.
fn paid_by(game: &Game, receipt: &PaymentReceipt) -> Vec<(String, u8)> {
    let mut paid: Vec<(String, u8)> = receipt
        .mana()
        .iter()
        .map(|unit| {
            let origin = unit.source.expect("a land produced this unit");
            // The origin is the land's own battlefield appearance.
            let permanent = game.state.permanent_of(origin.object).expect("the land");
            assert!(game.state.permanents[permanent].as_ref().unwrap().tapped);
            assert_eq!(game.state[origin.object].definition_id, origin.definition);
            (name(game, permanent), unit.color)
        })
        .collect();
    paid.sort();
    paid
}

const RED: u8 = Color::Red as u8;
const GREEN: u8 = Color::Green as u8;

/// CR 601.2h: the receipt names the object that produced each unit spent,
/// however the payment was made: taps chosen by hand, the automatic payer,
/// and mana tapped at priority before the cast.
#[test]
fn receipt_names_the_source_of_each_unit_under_manual_auto_and_floated_payment() {
    let lands = ["Mountain", "Mountain", "Mountain", "Forest"];
    let expected = [
        ("Forest".to_string(), GREEN),
        ("Mountain".to_string(), RED),
        ("Mountain".to_string(), RED),
    ];

    // Choosing the Forest determines the two remaining Mountain taps.
    let mut by_hand = game(&lands);
    cast(&mut by_hand);
    let forest = taps(&by_hand)
        .into_iter()
        .find(|(_, land)| land == "Forest")
        .unwrap()
        .0;
    by_hand.step(forest).unwrap();
    while by_hand.pending_choice.is_some() {
        let tap = taps(&by_hand)[0].0;
        by_hand.step(tap).unwrap();
    }
    assert_eq!(
        paid_by(&by_hand, &spell_receipt(&by_hand, "Gray Ogre")),
        expected
    );

    // Auto: the payer taps Mountains first; the receipt names exactly those.
    let mut automatic = game(&lands);
    automatic.mana_payment[0] = ManaPayment::Auto;
    cast(&mut automatic);
    assert!(automatic.pending_choice.is_none());
    assert_eq!(
        paid_by(&automatic, &spell_receipt(&automatic, "Gray Ogre")),
        [RED; 3].map(|color| ("Mountain".to_string(), color))
    );
    assert_eq!(untapped(&automatic), ["Forest"]);

    // Floated: the Forest is tapped at priority. Its unit keeps its origin
    // in the pool and the later receipt names it.
    let mut floated = game(&lands);
    floated.mana_floats = [ManaFloats::All, ManaFloats::Off];
    floated.scenario_refresh_priority().unwrap();
    let forest = taps(&floated)
        .into_iter()
        .find(|(_, land)| land == "Forest")
        .unwrap()
        .0;
    floated.step(forest).unwrap();
    let pool = &floated.state.players[0].mana_pool;
    let origins: Vec<_> = pool.origins(None, Color::Green).collect();
    assert_eq!(origins.len(), 1);
    let origin = origins[0].expect("the Forest");
    assert_eq!(floated.state[origin.object].name, "Forest");
    cast(&mut floated);
    // Floats stay offered during payment, so the payer stops by hand: two
    // Mountains, then PayCost.
    for _ in 0..2 {
        let tap = taps(&floated)[0].0;
        floated.step(tap).unwrap();
    }
    if floated.pending_choice.is_some() {
        pay_cost(&mut floated);
    }
    assert_eq!(
        paid_by(&floated, &spell_receipt(&floated, "Gray Ogre")),
        expected
    );
    assert_eq!(untapped(&floated), ["Mountain"]);
    assert_eq!(floated.state.players[0].mana_pool, Mana::default());

    // Whole object ids, so a later appearance of a land is never mistaken
    // for the one that paid.
    let json = serde_json::to_value(spell_receipt(&floated, "Gray Ogre")).unwrap();
    let source = &json["mana"][0]["source"];
    assert!(source["object"]["entity"].is_u64() && source["object"]["incarnation"].is_u64());
    assert!(source["definition"].is_u64());
}

/// Mana no object produced has no origin; mixing it with a land's mana
/// keeps each unit's own fact, and only the unit spent reaches the receipt.
#[test]
fn receipt_separates_sourceless_mana_from_a_lands() {
    let mut game = game(&["Mountain", "Forest"]);
    game.state.players[0].mana_pool = Mana::parse("2").into();
    game.mana_payment[0] = ManaPayment::Auto;
    game.scenario_refresh_priority().unwrap();
    cast(&mut game);
    assert!(game.pending_choice.is_none());
    let receipt = spell_receipt(&game, "Gray Ogre");
    assert_eq!(receipt.total(), 3);
    assert_eq!(receipt.amounts(), [0, 0, 0, 1, 0, 2]);
    let sources = receipt.mana_sources();
    assert_eq!(sources.len(), 1);
    assert_eq!(game.state[sources[0].object].name, "Mountain");
    assert_eq!(untapped(&game), ["Forest"]);
}
