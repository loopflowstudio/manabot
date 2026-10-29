//! Avatar: The Last Airbender (`src/cardsets/tla.rs`).
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::{KeywordData, Observation},
    },
    flow::turn::StepKind,
    state::{
        game_object::{PermanentId, PlayerId, Target},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

use super::{
    helpers::Scenario,
    support::{
        activate, can_activate, cast, choose, counters, counters_of, has_action, pass_until,
        play_land, settle, setup, tapped, to_hand,
    },
};

/// The TLA locations with "enters tapped unless you control a basic land",
/// each with a one-drop it can pay for alone.
const LOCATIONS: [(&str, &str, usize); 4] = [
    ("Abandoned Air Temple", "Healer's Hawk", 0),
    ("Agna Qel'a", "Wind Drake", 2),
    ("Ba Sing Se", "Llanowar Elves", 0),
    ("Realm of Koh", "Typhoid Rats", 0),
];

/// CR 614.1c, 614.12: the condition reads the entering player's battlefield
/// as the land moves. Opposing basics and nonbasic lands do not count.
#[test]
fn locations_enter_tapped_unless_you_control_a_basic_land() {
    for (name, _, _) in LOCATIONS {
        let mut s = setup(&[name, "Agna Qel'a"]);
        let alone = play_land(&mut s, name);
        assert!(tapped(&s, alone), "{name} enters tapped with no lands");

        let mut s = setup(&[name, "Kyoshi Village"]);
        s.force_permanent_on_battlefield(1, "Forest");
        s.force_permanent_on_battlefield(0, "Kyoshi Village");
        let without_basic = play_land(&mut s, name);
        assert!(
            tapped(&s, without_basic),
            "{name}: an opposing basic or own nonbasic land does not count"
        );

        let mut s = setup(&[name]);
        s.force_permanent_on_battlefield(0, "Plains");
        let with_basic = play_land(&mut s, name);
        assert!(!tapped(&s, with_basic), "{name} enters untapped");
    }
}

#[test]
fn locations_tap_for_their_color() {
    for (name, spell, generic) in LOCATIONS {
        let mut s = setup(&[name, spell]);
        let basic = s.force_permanent_on_battlefield(0, "Plains");
        let land = play_land(&mut s, name);
        s.game_mut().state.permanents[basic]
            .as_mut()
            .unwrap()
            .tapped = true;
        for _ in 0..generic {
            s.force_permanent_on_battlefield(0, "Plains");
            s.force_permanent_on_battlefield(0, "Plains");
        }
        let id = cast(&mut s, spell);
        settle(&mut s);
        assert!(tapped(&s, land), "{name} pays for {spell}");
        assert_eq!(
            s.game().state.zones.zone_of(id),
            Some(ZoneType::Battlefield)
        );
    }
}

#[test]
fn abandoned_air_temple_counters_each_creature_you_control_only() {
    let mut s = setup(&["Abandoned Air Temple", "Gray Ogre"]);
    let temple = s.force_permanent_on_battlefield(0, "Abandoned Air Temple");
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "2W");
    assert!(!can_activate(&mut s, temple, 0), "{{3}}{{W}} is required");
    s.set_player_mana_pool(0, "3W");
    activate(&mut s, temple, 0);
    assert!(tapped(&s, temple));
    settle(&mut s);
    assert_eq!(counters(&s, first), 1);
    assert_eq!(counters(&s, second), 1);
    assert_eq!(counters(&s, theirs), 0);
    assert_eq!(counters(&s, temple), 0, "a noncreature land gets none");
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn agna_qel_a_draws_then_discards() {
    let mut s = setup(&["Agna Qel'a", "Gray Ogre"]);
    let agna = s.force_permanent_on_battlefield(0, "Agna Qel'a");
    let kept = to_hand(&mut s, "Gray Ogre");
    let top = *s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .last()
        .unwrap();
    s.set_player_mana_pool(0, "2U");
    activate(&mut s, agna, 0);
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.game().state.zones.zone_of(top), Some(ZoneType::Hand));
    // The drawn card is itself a legal discard; discard the older card.
    assert!(has_action(
        &s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == top)
    ));
    assert!(!has_action(&s, |a| matches!(a, Action::Decline { .. })));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == kept),
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(kept),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
}

/// CR 602.5d: "activate only as a sorcery" needs your main phase and an
/// empty stack.
#[test]
fn ba_sing_se_earthbends_only_at_sorcery_timing() {
    let mut s = setup(&["Ba Sing Se", "Abandon Attachments"]);
    let city = s.force_permanent_on_battlefield(0, "Ba Sing Se");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.set_player_mana_pool(0, "2GUU");

    cast(&mut s, "Abandon Attachments");
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert!(
        !can_activate(&mut s, city, 0),
        "not with a spell on the stack"
    );
    settle(&mut s);

    s.set_player_mana_pool(0, "2G");
    assert!(can_activate(&mut s, city, 0));
    activate(&mut s, city, 0);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert!(tapped(&s, city));
    assert_eq!(counters(&s, forest), 2);
    let land = s.game().state.permanents[forest].as_ref().unwrap();
    assert!(land.animated, "the land becomes a creature");

    let mut s = setup(&["Ba Sing Se"]);
    let city = s.force_permanent_on_battlefield(0, "Ba Sing Se");
    s.force_permanent_on_battlefield(0, "Forest");
    s.advance_to_active_step(1, StepKind::Main);
    if s.action_space().player == Some(PlayerId(1)) {
        s.pass_priority();
    }
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "2G");
    assert!(!can_activate(&mut s, city, 0), "not on an opponent's turn");
}

#[test]
fn realm_of_koh_creates_a_spirit() {
    let mut s = setup(&["Realm of Koh"]);
    let realm = s.force_permanent_on_battlefield(0, "Realm of Koh");
    s.set_player_mana_pool(0, "3B");
    activate(&mut s, realm, 0);
    settle(&mut s);
    assert!(tapped(&s, realm));
    assert_eq!(s.battlefield_permanents_named(0, "Spirit").len(), 1);
    assert!(s.battlefield_permanents_named(1, "Spirit").is_empty());
}

/// The keywords an observer sees on a battlefield permanent.
fn observed_keywords(s: &Scenario, id: PermanentId) -> KeywordData {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    let observation = Observation::for_player(s.game(), PlayerId(0));
    observation
        .agent_permanents
        .iter()
        .chain(&observation.opponent_permanents)
        .find(|p| p.id == permanent.id.0 as i32)
        .map(|p| p.keywords.clone())
        .unwrap()
}

fn ready(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
}

fn mark_damage(s: &mut Scenario, id: PermanentId, damage: i32) {
    s.game_mut().state.permanents[id].as_mut().unwrap().damage = damage;
}

/// CR 702.12b, 704.5g: lethal damage does not destroy an indestructible
/// permanent. The grant covers only other permanents its controller controls
/// and ends when the Walls leave.
#[test]
fn the_walls_of_ba_sing_se_protect_other_permanents_you_control() {
    let mut s = setup(&["The Walls of Ba Sing Se", "Gray Ogre"]);
    let walls = s.force_permanent_on_battlefield(0, "The Walls of Ba Sing Se");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert!(observed_keywords(&s, walls).defender);
    assert!(!observed_keywords(&s, walls).indestructible);
    assert!(observed_keywords(&s, land).indestructible);
    assert!(!observed_keywords(&s, theirs).indestructible);

    let ogre_card = s.game().state.permanents[ogre].as_ref().unwrap().card;
    let theirs_card = s.game().state.permanents[theirs].as_ref().unwrap().card;
    mark_damage(&mut s, ogre, 5);
    mark_damage(&mut s, theirs, 5);
    s.pass_priority();
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Battlefield)
    );
    assert_eq!(
        s.game().state.zones.zone_of(theirs_card),
        Some(ZoneType::Graveyard)
    );

    let walls_card = s.game().state.permanents[walls].as_ref().unwrap().card;
    s.game_mut().move_card(walls_card, ZoneType::Graveyard);
    mark_damage(&mut s, ogre, 5);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Graveyard),
        "the grant ends with its source"
    );
}

/// The anthem reads effective flying, including a temporary grant.
#[test]
fn air_nomad_legacy_makes_a_clue_and_pumps_your_fliers() {
    let mut s = setup(&["Air Nomad Legacy", "Wind Drake", "Gray Ogre"]);
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let granted = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.game_mut().state.permanents[granted]
        .as_mut()
        .unwrap()
        .temp_keywords
        .flying = true;
    s.set_player_mana_pool(0, "WU");
    cast(&mut s, "Air Nomad Legacy");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    assert_eq!(s.game().effective_pt(drake), (3, 3));
    assert_eq!(s.game().effective_pt(granted), (3, 3));
    assert_eq!(s.game().effective_pt(ogre), (2, 2));
    assert_eq!(s.game().effective_pt(theirs), (2, 2));
}

/// Earthbend's counters bring the land under the trample grant; a creature
/// without counters, including Badgermole, or an opponent's does not have it.
#[test]
fn badgermole_earthbends_and_grants_trample_to_countered_creatures() {
    let mut s = setup(&["Badgermole", "Gray Ogre"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let countered = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [countered, theirs] {
        s.game_mut().state.permanents[id]
            .as_mut()
            .unwrap()
            .plus1_counters = 1;
    }
    s.set_player_mana_pool(0, "4G");
    let card = cast(&mut s, "Badgermole");
    for _ in 0..4 {
        if s.choose_target(Target::Permanent(forest)) {
            break;
        }
        s.pass_priority();
    }
    settle(&mut s);
    let badgermole = s.game().state.card_to_permanent[card].unwrap();
    assert_eq!(counters(&s, forest), 2);
    assert!(s.game().state.permanents[forest].as_ref().unwrap().animated);
    assert!(observed_keywords(&s, forest).trample);
    assert!(observed_keywords(&s, countered).trample);
    assert!(!observed_keywords(&s, ogre).trample);
    assert!(!observed_keywords(&s, badgermole).trample);
    assert!(!observed_keywords(&s, theirs).trample);
}

/// CR 205.3m: the bonus counts each creature type once, including granted
/// types, caps at ten, and skips Humans, opponents and animated lands'
/// land types.
#[test]
fn diligent_zookeeper_counts_creature_types_of_non_humans() {
    let mut s = setup(&["Diligent Zookeeper", "Gray Ogre", "Momo, Friendly Flier"]);
    let keeper = s.force_permanent_on_battlefield(0, "Diligent Zookeeper");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let momo = s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let many = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[many]
        .as_mut()
        .unwrap()
        .temp_subtypes = (0..12).map(|i| format!("Type{i}")).collect();
    {
        let land = s.game_mut().state.permanents[land].as_mut().unwrap();
        land.animated = true;
        land.plus1_counters = 1;
    }
    assert_eq!(s.game().effective_pt(keeper), (4, 4));
    assert_eq!(s.game().effective_pt(ogre), (3, 3));
    assert_eq!(s.game().effective_pt(momo), (4, 4), "Lemur, Bat and Ally");
    assert_eq!(s.game().effective_pt(many), (12, 12), "capped at ten");
    assert_eq!(
        s.game().effective_pt(land),
        (1, 1),
        "Forest is not a creature type"
    );
    assert_eq!(s.game().effective_pt(theirs), (2, 2));

    s.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .temp_subtypes = vec!["Ally".into(), "Ogre".into()];
    assert_eq!(
        s.game().effective_pt(ogre),
        (4, 4),
        "a granted type counts once"
    );
}

/// The keywords track the controller's draw count this turn.
#[test]
fn foggy_swamp_hunters_gain_lifelink_and_menace_after_two_draws() {
    let mut s = setup(&["Foggy Swamp Hunters"]);
    let hunters = s.force_permanent_on_battlefield(0, "Foggy Swamp Hunters");
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 1;
    let keywords = observed_keywords(&s, hunters);
    assert!(!keywords.lifelink && !keywords.menace);
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 2;
    let keywords = observed_keywords(&s, hunters);
    assert!(keywords.lifelink && keywords.menace);
    s.game_mut().state.turn.cards_drawn_this_turn[1] = 5;
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 0;
    assert!(
        !observed_keywords(&s, hunters).lifelink,
        "only its controller's draws count"
    );
}

/// CR 122.1d: the stun counter replaces the target's next untap, then is gone.
#[test]
fn rowdy_snowballers_stun_replaces_the_next_untap() {
    let mut s = setup(&["Rowdy Snowballers", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "2U");
    cast(&mut s, "Rowdy Snowballers");
    s.pass_priority();
    s.choose_target(Target::Permanent(ogre));
    settle(&mut s);
    assert!(tapped(&s, ogre));
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 1);

    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(
        tapped(&s, ogre),
        "the untap step removes the stun counter instead"
    );
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 0);

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!tapped(&s, ogre), "the next untap step untaps it");
}

/// CR 702.177a, 122.1b: exhaust activates once; the double strike counter
/// grants double strike for as long as it remains.
#[test]
fn mai_jaded_edge_exhausts_into_a_double_strike_counter() {
    let mut s = setup(&["Mai, Jaded Edge"]);
    let mai = s.force_permanent_on_battlefield(0, "Mai, Jaded Edge");
    assert!(!observed_keywords(&s, mai).double_strike);
    s.set_player_mana_pool(0, "3");
    activate(&mut s, mai, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, mai, CounterKind::DoubleStrike), 1);
    assert!(observed_keywords(&s, mai).double_strike);

    s.set_player_mana_pool(0, "3");
    assert!(!can_activate(&mut s, mai, 0), "exhaust activates only once");

    ready(&mut s, mai);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::End);
    s.assert_life(1, 18);
}

fn choose_mode(s: &mut Scenario, mode: usize) {
    choose(
        s,
        |a| matches!(a, Action::ChooseMode { mode: m, .. } if *m == mode),
    );
}

fn add_counters(s: &mut Scenario, id: PermanentId, kind: CounterKind, count: i32) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .unwrap()
        .change_counters(kind, count);
}

#[test]
fn heartless_act_destroys_only_a_creature_with_no_counters() {
    let mut s = setup(&["Heartless Act", "Gray Ogre"]);
    let plain = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let stunned = s.force_permanent_on_battlefield(1, "Gray Ogre");
    add_counters(&mut s, stunned, CounterKind::Stun, 1);
    s.set_player_mana_pool(0, "1B");
    cast(&mut s, "Heartless Act");
    choose_mode(&mut s, 0);
    assert!(
        !s.choose_target(Target::Permanent(stunned)),
        "a stun counter is a counter"
    );
    assert!(s.choose_target(Target::Permanent(plain)));
    settle(&mut s);
    assert!(s.game().state.permanents[plain].is_none());
    assert!(s.game().state.permanents[stunned].is_some());
}

/// The caster chooses each removed counter's kind and may stop early.
#[test]
fn heartless_act_removes_up_to_three_counters_of_chosen_kinds() {
    let mut s = setup(&["Heartless Act", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
    add_counters(&mut s, ogre, CounterKind::Stun, 2);
    s.set_player_mana_pool(0, "1B");
    cast(&mut s, "Heartless Act");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    pass_until(&mut s, ActionSpaceKind::Modal);
    // Modes: +1/+1, stun, stop.
    assert_eq!(s.action_space().actions.len(), 3);
    choose_mode(&mut s, 1);
    choose_mode(&mut s, 1);
    // Stun counters are gone; only +1/+1 and stop remain.
    assert_eq!(s.action_space().actions.len(), 2);
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 0);
    assert_eq!(counters(&s, ogre), 2);

    let mut s = setup(&["Heartless Act", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 5);
    s.set_player_mana_pool(0, "1B");
    cast(&mut s, "Heartless Act");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    pass_until(&mut s, ActionSpaceKind::Modal);
    for _ in 0..3 {
        choose_mode(&mut s, 0);
    }
    assert_eq!(
        s.action_space().kind,
        ActionSpaceKind::Priority,
        "three is the limit"
    );
    assert_eq!(counters(&s, ogre), 2);
}

/// CR 613.4b-c: the waterbend setter replaces the printed base; a +1/+1
/// counter still applies on top of it, and the setter ends at cleanup.
#[test]
fn flexible_waterbender_sets_base_pt_under_counters_until_end_of_turn() {
    let mut s = setup(&["Flexible Waterbender", "Gray Ogre"]);
    let bender = s.force_permanent_on_battlefield(0, "Flexible Waterbender");
    let helpers = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    add_counters(&mut s, bender, CounterKind::PlusOnePlusOne, 1);
    assert_eq!(s.game().effective_pt(bender), (3, 6));

    s.set_player_mana_pool(0, "1");
    activate(&mut s, bender, 0);
    for helper in helpers {
        assert!(s.choose_waterbend_tap(helper));
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(helpers.iter().all(|helper| tapped(&s, *helper)));
    assert_eq!(s.game().effective_pt(bender), (6, 3));

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(bender), (3, 6));
}

/// Only base power changes; toughness keeps its printed base, and trample
/// lasts as long as the setter.
#[test]
fn turtle_duck_sets_base_power_and_gains_trample_until_end_of_turn() {
    let mut s = setup(&["Turtle-Duck"]);
    let duck = s.force_permanent_on_battlefield(0, "Turtle-Duck");
    add_counters(&mut s, duck, CounterKind::PlusOnePlusOne, 1);
    assert!(!observed_keywords(&s, duck).trample);

    s.set_player_mana_pool(0, "3");
    activate(&mut s, duck, 0);
    settle(&mut s);
    assert_eq!(s.game().effective_pt(duck), (5, 5));
    assert!(observed_keywords(&s, duck).trample);

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(duck), (1, 5));
    assert!(!observed_keywords(&s, duck).trample);
}

fn equip(s: &mut Scenario, equipment: PermanentId, creature: PermanentId) {
    activate(s, equipment, 0);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(s);
}

fn battlefield_token(s: &Scenario, player: usize) -> PermanentId {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Battlefield, PlayerId(player))
        .iter()
        .find(|card| s.game().state.cards[**card].is_token)
        .and_then(|card| s.game().state.card_to_permanent[*card])
        .expect("a token")
}

/// CR 702.6a, 301.5: the token enters, then the Fan attaches to it; equip
/// moves it at sorcery timing only, and the bonus follows the Equipment.
#[test]
fn kyoshi_battle_fan_attaches_to_its_ally_token_then_equips_at_sorcery_speed() {
    let mut s = setup(&["Kyoshi Battle Fan", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "2");
    let fan = cast(&mut s, "Kyoshi Battle Fan");
    settle(&mut s);
    let fan = s.game().state.card_to_permanent[fan].expect("fan entered");
    let ally = battlefield_token(&s, 0);
    assert_eq!(s.game().effective_pt(ally), (2, 1));

    s.set_player_mana_pool(0, "2");
    equip(&mut s, fan, ogre);
    assert_eq!(s.game().effective_pt(ally), (1, 1));
    assert_eq!(s.game().effective_pt(ogre), (3, 2));

    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(0, "2");
    assert!(
        !can_activate(&mut s, fan, 0),
        "equip only as a sorcery on your own turn"
    );
}

/// CR 704.5n: an Equipment whose creature leaves stays on the battlefield,
/// unattached.
#[test]
fn meteor_sword_destroys_a_permanent_and_stays_when_its_creature_dies() {
    let mut s = setup(&["Meteor Sword", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "7");
    let sword = cast(&mut s, "Meteor Sword");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert!(s.game().state.permanents[theirs].is_none());
    let sword = s.game().state.card_to_permanent[sword].expect("sword entered");

    s.set_player_mana_pool(0, "3");
    equip(&mut s, sword, ours);
    assert_eq!(s.game().effective_pt(ours), (5, 5));

    mark_damage(&mut s, ours, 5);
    s.pass_priority();
    assert!(s.game().state.permanents[ours].is_none(), "lethal damage");
    let sword = s.game().state.permanents[sword].as_ref().expect("stays");
    assert_eq!(sword.attached_to, None);
}

/// Flash; the enters trigger attaches to a creature you control and grants
/// double strike until end of turn, while +1/+1 lasts while attached.
#[test]
fn twin_blades_attaches_on_entry_and_grants_double_strike_this_turn() {
    let mut s = setup(&["Twin Blades", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, ours);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.set_player_mana_pool(0, "2R");
    cast(&mut s, "Twin Blades");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(
        !s.choose_target(Target::Permanent(theirs)),
        "only a creature you control"
    );
    assert!(s.choose_target(Target::Permanent(ours)));
    settle(&mut s);
    assert_eq!(s.game().effective_pt(ours), (3, 3));
    assert!(observed_keywords(&s, ours).double_strike);

    s.game_mut().state.permanents[theirs]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::End);
    s.assert_life(1, 14);

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(ours), (3, 3));
    assert!(!observed_keywords(&s, ours).double_strike);
}
