//! Avatar: The Last Airbender (`src/cardsets/tla.rs`).
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::{KeywordData, Observation, PermanentData},
    },
    flow::{event::GameEvent, turn::StepKind},
    state::{
        game_object::{PermanentId, PlayerId, Target},
        mana::Mana,
        permanent::CounterKind,
        zone::ZoneType,
    },
};

use super::{
    helpers::Scenario,
    support::{
        activate, can_activate, card_of, cast, choose, counters, counters_of, has_action,
        library_card, pass_until, play_land, settle, setup, tapped, to_hand,
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
fn observed(s: &Scenario, id: PermanentId) -> PermanentData {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    let observation = Observation::for_player(s.game(), PlayerId(0));
    observation
        .agent_permanents
        .iter()
        .chain(&observation.opponent_permanents)
        .find(|p| p.id == permanent.id.0 as i32)
        .cloned()
        .unwrap()
}

fn observed_keywords(s: &Scenario, id: PermanentId) -> KeywordData {
    observed(s, id).keywords
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

    let ogre_card = card_of(&s, ogre);
    let theirs_card = card_of(&s, theirs);
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

    let walls_card = card_of(&s, walls);
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

/// Cast an Aura from player 0's hand onto `creature` and resolve it.
fn enchant(s: &mut Scenario, name: &str, mana: &str, creature: PermanentId) -> PermanentId {
    s.set_player_mana_pool(0, mana);
    let aura = cast(s, name);
    pass_until(s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(s);
    let aura = s.game().state.card_to_permanent[aura].expect("aura entered");
    assert_eq!(
        s.game().state.permanents[aura]
            .as_ref()
            .unwrap()
            .attached_to,
        s.game().permanent_object_ref(creature),
        "CR 303.4f: the Aura enters attached to its target"
    );
    aura
}

/// Whether the current combat decision offers `id` as an attacker or blocker.
fn offered_in_combat(s: &Scenario, id: PermanentId) -> bool {
    has_action(s, |a| {
        matches!(a, Action::DeclareAttacker { permanent, .. } if *permanent == id)
            || matches!(a, Action::DeclareBlocker { blocker, .. } if *blocker == id)
    })
}

/// CR 508.1c, 509.1b: the enchanted creature is never offered as an attacker
/// or blocker. The sacrifice ability reads the creature it enchanted after
/// the Aura has left as a cost, and only on its controller's turn.
#[test]
fn path_to_redemption_pacifies_then_exiles_for_an_ally_on_your_turn() {
    let mut s = setup(&["Path to Redemption", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let free = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    for id in [theirs, free, ours] {
        ready(&mut s, id);
    }
    let path = enchant(&mut s, "Path to Redemption", "1W", theirs);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(offered_in_combat(&s, free));
    assert!(!offered_in_combat(&s, theirs), "can't block");

    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(0, "5");
    assert!(!can_activate(&mut s, path, 0), "only during your turn");
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    assert!(offered_in_combat(&s, free));
    assert!(!offered_in_combat(&s, theirs), "can't attack");

    s.advance_to_active_step(0, StepKind::Main);
    let theirs_card = card_of(&s, theirs);
    let path_card = card_of(&s, path);
    s.set_player_mana_pool(0, "5");
    activate(&mut s, path, 0);
    settle(&mut s);
    let zones = &s.game().state.zones;
    assert_eq!(zones.zone_of(theirs_card), Some(ZoneType::Exile));
    assert_eq!(zones.zone_of(path_card), Some(ZoneType::Graveyard));
    let ally = battlefield_token(&s, 0);
    assert_eq!(s.game().effective_pt(ally), (1, 1));
}

/// CR 704.5m: an Aura whose creature leaves is put into its owner's
/// graveyard.
#[test]
fn path_to_redemption_follows_its_creature_to_the_graveyard() {
    let mut s = setup(&["Path to Redemption", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let path = enchant(&mut s, "Path to Redemption", "1W", theirs);
    let path_card = card_of(&s, path);
    mark_damage(&mut s, theirs, 2);
    s.pass_priority();
    assert!(s.game().state.permanents[theirs].is_none(), "lethal damage");
    assert_eq!(
        s.game().state.zones.zone_of(path_card),
        Some(ZoneType::Graveyard)
    );
}

/// CR 608.3b: an Aura spell whose target is gone doesn't resolve and never
/// reaches the battlefield.
#[test]
fn path_to_redemption_without_its_target_does_not_resolve() {
    let mut s = setup(&["Path to Redemption", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "1W");
    let spell = cast(&mut s, "Path to Redemption");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(theirs)));
    mark_damage(&mut s, theirs, 2);
    s.drain_events();
    settle(&mut s);
    assert!(s.game().state.permanents[theirs].is_none(), "lethal damage");
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    assert!(s
        .drain_events()
        .iter()
        .any(|event| matches!(event, GameEvent::SpellCountered { card, .. } if *card == spell)));
}

/// CR 502.3: the enchanted creature stays tapped through its controller's
/// untap step. Waterbend {5} shuffles it into its owner's library, and the
/// unattached Aura then goes to the graveyard (704.5m).
#[test]
fn watery_grasp_holds_a_creature_tapped_then_shuffles_it_away() {
    let mut s = setup(&["Watery Grasp", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let helpers = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    s.game_mut().state.permanents[theirs]
        .as_mut()
        .unwrap()
        .tapped = true;
    let grasp = enchant(&mut s, "Watery Grasp", "U", theirs);
    let grasp_card = card_of(&s, grasp);

    s.advance_to_active_step(1, StepKind::Main);
    assert!(tapped(&s, theirs), "doesn't untap");
    assert!(helpers.iter().all(|helper| !tapped(&s, *helper)));

    s.advance_to_active_step(0, StepKind::Main);
    let theirs_card = card_of(&s, theirs);
    let library = s.zone_size(1, ZoneType::Library);
    s.set_player_mana_pool(0, "3");
    activate(&mut s, grasp, 0);
    for helper in helpers {
        assert!(s.choose_waterbend_tap(helper));
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(theirs_card),
        Some(ZoneType::Library)
    );
    assert_eq!(s.zone_size(1, ZoneType::Library), library + 1);
    assert_eq!(
        s.game().state.zones.zone_of(grasp_card),
        Some(ZoneType::Graveyard)
    );
}

/// CR 601.2f: the reduction depends on the chosen target, so with only
/// {1}{B} the spell is castable only onto a flier and offers no other
/// target; at full cost any creature may be enchanted.
#[test]
fn swampsnare_trap_costs_less_onto_a_flier_and_shrinks_its_creature() {
    let mut s = setup(&["Swampsnare Trap", "Wind Drake", "Craw Wurm"]);
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    s.set_player_mana_pool(0, "1B");
    let trap = to_hand(&mut s, "Swampsnare Trap");
    assert!(
        !has_action(
            &s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == trap)
        ),
        "no flier to earn the reduction"
    );

    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == trap),
    );
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ChooseTarget { target, .. } if *target == Target::Permanent(wurm))
    }));
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    assert!(s.game().state.permanents[drake].is_none(), "2/2 gets -5/-3");
    assert_eq!(
        s.game().state.zones.zone_of(trap),
        Some(ZoneType::Graveyard)
    );

    enchant(&mut s, "Swampsnare Trap", "2B", wurm);
    assert_eq!(s.game().effective_pt(wurm), (1, 1));
}

fn is_creature(s: &Scenario, id: PermanentId) -> bool {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    permanent.is_creature(&s.game().state.cards[permanent.card])
}

/// Tap `creature` toward the pending crew cost.
fn crew_with(s: &mut Scenario, creature: PermanentId) {
    let card = card_of(s, creature);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    choose(
        s,
        |a| matches!(a, Action::SelectCard { card: c, .. } if *c == card),
    );
}

/// CR 702.122a: crew taps other creatures whose total power reaches N; the
/// Vehicle is an artifact creature with its printed P/T until end of turn,
/// then a noncreature artifact again. It dies as a creature and leaves a Clue.
#[test]
fn fire_nation_warship_crews_until_end_of_turn_and_leaves_a_clue() {
    let mut s = setup(&["Fire Nation Warship", "Llanowar Elves"]);
    let ship = s.force_permanent_on_battlefield(0, "Fire Nation Warship");
    let first = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    ready(&mut s, ship);
    assert!(!is_creature(&s, ship));
    assert!(!can_activate(&mut s, ship, 0), "power 1 can't crew 2");

    let second = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    activate(&mut s, ship, 0);
    assert!(
        !has_action(&s, |a| matches!(a, Action::SelectCard { card, .. }
            if *card == card_of(&s, ship))),
        "a Vehicle doesn't crew itself"
    );
    crew_with(&mut s, first);
    crew_with(&mut s, second);
    settle(&mut s);
    assert!(tapped(&s, first) && tapped(&s, second));
    assert!(is_creature(&s, ship));
    assert!(
        observed(&s, ship).is_animated,
        "the policy sees the crewed Vehicle"
    );
    assert_eq!(s.game().effective_pt(ship), (4, 4));
    assert!(observed_keywords(&s, ship).reach);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(offered_in_combat(&s, ship));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!is_creature(&s, ship), "crew lasts until end of turn");

    s.advance_to_active_step(0, StepKind::Main);
    activate(&mut s, ship, 0);
    crew_with(&mut s, first);
    crew_with(&mut s, second);
    settle(&mut s);
    let ship_card = card_of(&s, ship);
    mark_damage(&mut s, ship, 4);
    s.pass_priority();
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(ship_card),
        Some(ZoneType::Graveyard)
    );
    let clue = battlefield_token(&s, 0);
    assert_eq!(s.game().state.cards[card_of(&s, clue)].name, "Clue");
}

/// CR 302.6: a Vehicle that came under your control this turn can't attack
/// when crewed, while a summoning-sick creature can crew (crew is not {T}).
/// The enters trigger makes a creature you control indestructible, and the
/// crewed Tank's firebending adds {R} when it attacks.
#[test]
fn tundra_tank_protects_on_entry_and_firebends_once_crewed() {
    let mut s = setup(&["Tundra Tank", "Gray Ogre", "Llanowar Elves"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    ready(&mut s, ogre);
    s.set_player_mana_pool(0, "2B");
    let tank = cast(&mut s, "Tundra Tank");
    settle(&mut s);
    let tank = s.game().state.card_to_permanent[tank].expect("tank entered");
    assert!(observed_keywords(&s, ogre).indestructible);

    activate(&mut s, tank, 0);
    crew_with(&mut s, elves);
    settle(&mut s);
    assert!(is_creature(&s, tank));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!offered_in_combat(&s, tank), "summoning sick");

    s.advance_to_active_step(0, StepKind::Main);
    activate(&mut s, tank, 0);
    crew_with(&mut s, elves);
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == tank),
    );
    s.advance_until(
        |s| s.game().state.players[0].mana_pool.mana[3] == 1,
        "firebending".into(),
    );
}

/// CR 613.1d: three fire counters make the Balloon an artifact creature for
/// as long as they stay; two do not.
#[test]
fn war_balloon_flies_once_it_has_three_fire_counters() {
    let mut s = setup(&["War Balloon"]);
    let balloon = s.force_permanent_on_battlefield(0, "War Balloon");
    for count in 1..=3 {
        assert!(!is_creature(&s, balloon));
        s.set_player_mana_pool(0, "1");
        activate(&mut s, balloon, 1);
        settle(&mut s);
        assert_eq!(counters_of(&s, balloon, CounterKind::Fire), count);
    }
    assert!(is_creature(&s, balloon));
    assert_eq!(s.game().effective_pt(balloon), (4, 3));
    assert!(observed_keywords(&s, balloon).flying);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(is_creature(&s, balloon), "not limited to this turn");
}

/// "Another target": the entering Submersible can return any other nonland
/// permanent but not itself. Its exhaust waterbend makes it a 3/3 artifact
/// creature for as long as it remains, only once.
#[test]
fn invasion_submersible_bounces_another_permanent_then_surfaces_once() {
    let mut s = setup(&["Invasion Submersible", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let theirs_card = card_of(&s, theirs);
    let helpers = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    s.set_player_mana_pool(0, "2U");
    let sub = cast(&mut s, "Invasion Submersible");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let sub = s.game().state.card_to_permanent[sub].expect("entered");
    assert!(!s.choose_target(Target::Permanent(sub)), "another target");
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(theirs_card),
        Some(ZoneType::Hand)
    );
    assert!(!is_creature(&s, sub));

    s.set_player_mana_pool(0, "1");
    activate(&mut s, sub, 0);
    for helper in helpers {
        assert!(s.choose_waterbend_tap(helper));
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(is_creature(&s, sub));
    assert_eq!(s.game().effective_pt(sub), (3, 3));
    s.advance_to_active_step(0, StepKind::Main);
    assert!(is_creature(&s, sub), "no duration");
    s.set_player_mana_pool(0, "3");
    assert!(!can_activate(&mut s, sub, 0), "exhaust");
}

fn pool(s: &Scenario) -> Mana {
    s.game().state.players[0].mana_pool.clone()
}

fn untap(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id].as_mut().unwrap().tapped = false;
}

/// CR 613.1f: lands and Allies you control have the granted ability while the
/// Guide remains; an opponent's land does not, and a summoning-sick Ally
/// can't use it (CR 302.6). The chosen color pays a real cost.
#[test]
fn great_divide_guide_grants_lands_and_allies_any_color() {
    let mut s = setup(&["Great Divide Guide", "Healer's Hawk"]);
    let guide = s.force_permanent_on_battlefield(0, "Great Divide Guide");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    assert!(!can_activate(&mut s, guide, 0), "summoning sick");

    activate(&mut s, forest, 0);
    choose_mode(&mut s, 0);
    assert_eq!(pool(&s), Mana::parse("W"));
    cast(&mut s, "Healer's Hawk");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Healer's Hawk").len(), 1);
    assert_eq!(pool(&s).total(), 0);

    ready(&mut s, guide);
    assert!(can_activate(&mut s, guide, 0));
    untap(&mut s, forest);
    let guide_card = card_of(&s, guide);
    s.game_mut().move_card(guide_card, ZoneType::Graveyard);
    assert!(
        !can_activate(&mut s, forest, 0),
        "the grant ends with its source"
    );

    let mut s = setup(&["Great Divide Guide"]);
    s.force_permanent_on_battlefield(0, "Great Divide Guide");
    let theirs = s.force_permanent_on_battlefield(1, "Forest");
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!can_activate(&mut s, theirs, 0), "only lands you control");
}

/// The output reads effective power as the ability resolves (CR 605.3a), so
/// a creature grown to four by counters doubles it; an opponent's does not.
/// Payment for a spell counts the doubled output.
#[test]
fn raucous_audience_adds_more_with_a_four_power_creature() {
    let mut s = setup(&["Raucous Audience", "Gray Ogre"]);
    let audience = s.force_permanent_on_battlefield(0, "Raucous Audience");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    ready(&mut s, audience);
    add_counters(&mut s, theirs, CounterKind::PlusOnePlusOne, 2);
    activate(&mut s, audience, 0);
    choose_mode(&mut s, 0);
    assert_eq!(pool(&s), Mana::parse("G"));

    untap(&mut s, audience);
    s.set_player_mana_pool(0, "0");
    to_hand(&mut s, "Raucous Audience");
    assert!(
        !has_action(&s, |a| matches!(a, Action::CastSpell { .. })),
        "{{G}} alone can't pay {{1}}{{G}}"
    );
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(&mut s, |a| matches!(a, Action::CastSpell { .. }));
    for _ in 0..4 {
        match s.action_space().kind {
            ActionSpaceKind::Priority => break,
            ActionSpaceKind::Modal => choose_mode(&mut s, 0),
            _ if has_action(&s, |a| matches!(a, Action::PayCost { .. })) => {
                choose(&mut s, |a| matches!(a, Action::PayCost { .. }))
            }
            _ => choose(
                &mut s,
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == audience),
            ),
        }
    }
    settle(&mut s);
    assert!(tapped(&s, audience));
    assert_eq!(
        s.battlefield_permanents_named(0, "Raucous Audience").len(),
        2
    );
    assert_eq!(pool(&s).total(), 0);
}

/// The Tile enters tapped; X counts the largest group of your creatures
/// sharing a creature type (CR 205.3m), including a granted type, and all of
/// it is one chosen color. With no creatures it adds nothing.
#[test]
fn white_lotus_tile_adds_one_color_for_the_largest_shared_type() {
    let mut s = setup(&["White Lotus Tile", "Gray Ogre", "Llanowar Elves"]);
    s.set_player_mana_pool(0, "4");
    let tile = cast(&mut s, "White Lotus Tile");
    settle(&mut s);
    let tile = s.game().state.card_to_permanent[tile].expect("entered");
    assert!(tapped(&s, tile));

    untap(&mut s, tile);
    activate(&mut s, tile, 0);
    choose_mode(&mut s, 0);
    assert!(tapped(&s, tile));
    assert_eq!(pool(&s).total(), 0, "no creatures");

    for _ in 0..2 {
        s.force_permanent_on_battlefield(0, "Gray Ogre");
    }
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    for _ in 0..3 {
        s.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    untap(&mut s, tile);
    activate(&mut s, tile, 0);
    choose_mode(&mut s, 1);
    assert_eq!(pool(&s), Mana::parse("UU"));

    s.game_mut().state.permanents[elves]
        .as_mut()
        .unwrap()
        .temp_subtypes = vec!["Ogre".into()];
    s.set_player_mana_pool(0, "0");
    untap(&mut s, tile);
    activate(&mut s, tile, 0);
    choose_mode(&mut s, 3);
    assert_eq!(pool(&s), Mana::parse("RRR"));
}

/// CR 508.1c, 509.1b: the Lion-Turtle can neither attack nor block until
/// three Lesson cards are in its controller's graveyard; vigilance keeps it
/// untapped once it can attack. It gains 3 life on entry and taps for any
/// color.
#[test]
fn the_lion_turtle_waits_for_three_lessons_to_fight() {
    let mut s = setup(&["The Lion-Turtle", "Waterbending Lesson", "Gray Ogre"]);
    let attacker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    ready(&mut s, attacker);
    s.set_player_mana_pool(0, "1GU");
    let turtle = cast(&mut s, "The Lion-Turtle");
    settle(&mut s);
    let turtle = s.game().state.card_to_permanent[turtle].expect("entered");
    assert_eq!(s.game().state.players[0].life, 23);
    ready(&mut s, turtle);
    activate(&mut s, turtle, 0);
    choose_mode(&mut s, 2);
    assert_eq!(pool(&s), Mana::parse("B"));
    untap(&mut s, turtle);

    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    assert!(!offered_in_combat(&s, turtle), "can't attack");
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(!offered_in_combat(&s, turtle), "can't block");

    for _ in 0..3 {
        let lesson = library_card(&s, 0, "Waterbending Lesson");
        s.game_mut().move_card(lesson, ZoneType::Graveyard);
    }
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(offered_in_combat(&s, turtle), "blocks with three Lessons");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == turtle),
    );
    assert!(!tapped(&s, turtle), "vigilance");
}

/// CR 105.2: the bonus counts each color among Allies you control once,
/// including its own green; a non-Ally or an opponent's Ally adds nothing.
/// The five-color activation earthbends 5.
#[test]
fn earthen_ally_grows_with_ally_colors_and_earthbends_five() {
    let mut s = setup(&["Earthen Ally", "Toph, the First Metalbender", "Gray Ogre"]);
    let ally = s.force_permanent_on_battlefield(0, "Earthen Ally");
    assert_eq!(s.game().effective_pt(ally), (1, 2));
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Toph, the First Metalbender");
    assert_eq!(s.game().effective_pt(ally), (1, 2));
    s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
    assert_eq!(s.game().effective_pt(ally), (3, 2), "green, red and white");

    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.set_player_mana_pool(0, "2WUBRG");
    activate(&mut s, ally, 0);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 5);
    assert_eq!(s.game().effective_pt(forest), (5, 5));
}

/// Declare every offered attacker.
fn attack_with_all(s: &mut Scenario) {
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(s, |a| {
            matches!(a, Action::DeclareAttacker { attack: true, .. })
        });
    }
}

/// Whether the current blocker may block `attacker`.
fn may_block(s: &Scenario, attacker: PermanentId) -> bool {
    has_action(
        s,
        |a| matches!(a, Action::DeclareBlocker { attacker: Some(target), .. } if *target == attacker),
    )
}

/// The entry trigger counters each creature you control. CR 509.1b: a
/// creature you control with effective power 4 or more takes one blocker at
/// most; a smaller one can still be blocked by two.
#[test]
fn flopsie_counters_your_team_and_limits_big_attackers_to_one_blocker() {
    let mut s = setup(&["Flopsie, Bumi's Buddy", "Gray Ogre", "Llanowar Elves"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "4GG");
    let flopsie = cast(&mut s, "Flopsie, Bumi's Buddy");
    settle(&mut s);
    let flopsie = s.game().state.card_to_permanent[flopsie].expect("entered");
    assert_eq!(counters(&s, flopsie), 1);
    assert_eq!(counters(&s, ogre), 1);
    assert_eq!(counters(&s, theirs), 0);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(1, "Llanowar Elves");
    }
    ready(&mut s, flopsie);
    ready(&mut s, ogre);

    attack_with_all(&mut s);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(may_block(&s, flopsie));
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareBlocker { attacker: Some(target), .. } if *target == flopsie),
    );
    assert!(!may_block(&s, flopsie), "a second blocker is refused");
    assert!(may_block(&s, ogre));
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareBlocker { attacker: Some(target), .. } if *target == ogre),
    );
    assert!(may_block(&s, ogre), "a 3/3 can be blocked by two");
}

/// X counts creatures you control with effective power 4 or more as the
/// trigger resolves, including The Boulder and a countered Ogre.
#[test]
fn the_boulder_earthbends_for_each_four_power_creature_on_attack() {
    let mut s = setup(&["The Boulder, Ready to Rumble", "Gray Ogre"]);
    let boulder = s.force_permanent_on_battlefield(0, "The Boulder, Ready to Rumble");
    let big = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    add_counters(&mut s, big, CounterKind::PlusOnePlusOne, 2);
    add_counters(&mut s, theirs, CounterKind::PlusOnePlusOne, 2);
    ready(&mut s, boulder);

    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareAttacker { permanent, attack, .. }
                if *attack == (*permanent == boulder))
        });
    }
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 2);
}

/// CR 702.3b: defender stops the Bear attacking until a Lesson card is in
/// your graveyard; it still has defender. Its entry mills two.
#[test]
fn platypus_bear_mills_two_and_attacks_once_a_lesson_is_in_your_graveyard() {
    let mut s = setup(&["Platypus-Bear", "Abandon Attachments"]);
    s.set_player_mana_pool(0, "1G");
    let bear = cast(&mut s, "Platypus-Bear");
    let library = s.game().state.zones.size(ZoneType::Library, PlayerId(0));
    settle(&mut s);
    let bear = s.game().state.card_to_permanent[bear].expect("entered");
    assert_eq!(
        s.game().state.zones.size(ZoneType::Library, PlayerId(0)),
        library - 2
    );
    let graveyard = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Graveyard, PlayerId(0))
        .to_vec();
    assert_eq!(graveyard.len(), 2);
    for card in graveyard {
        s.game_mut().move_card(card, ZoneType::Exile);
    }
    ready(&mut s, bear);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!offered_in_combat(&s, bear), "defender");

    let lesson = library_card(&s, 0, "Abandon Attachments");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(offered_in_combat(&s, bear));
    assert!(observed_keywords(&s, bear).defender);
}

/// CR 205.1b, 613.1d: your nontoken artifacts are also lands, so earthbend
/// can target them, and an earthbent one is a 0/0 before counters. A token
/// artifact and an opponent's artifact are not lands. The end-step trigger
/// is only yours.
#[test]
fn toph_makes_your_nontoken_artifacts_lands_and_earthbends_at_your_end_step() {
    let mut s = setup(&["Toph, the First Metalbender", "White Lotus Tile"]);
    s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    let theirs = s.force_permanent_on_battlefield(1, "White Lotus Tile");
    let clue = s
        .game_mut()
        .create_token("Clue", PlayerId(0), false)
        .expect("clue");
    let forest = s.force_permanent_on_battlefield(0, "Forest");

    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::ChooseTarget,
        "end step trigger".into(),
    );
    assert_eq!(s.current_step(), StepKind::End);
    let offered = |s: &Scenario, id: PermanentId| {
        has_action(s, |a| {
            matches!(a, Action::ChooseTarget { target, .. }
            if *target == Target::Permanent(id))
        })
    };
    assert!(offered(&s, tile) && offered(&s, forest));
    assert!(!offered(&s, clue), "tokens stay nonland");
    assert!(!offered(&s, theirs), "only artifacts you control");
    assert!(s.choose_target(Target::Permanent(tile)));
    settle(&mut s);
    assert!(is_creature(&s, tile));
    assert_eq!(s.game().effective_pt(tile), (2, 2));

    s.advance_to_active_step(1, StepKind::End);
    assert_ne!(
        s.action_space().kind,
        ActionSpaceKind::ChooseTarget,
        "not on an opponent's end step"
    );
}

/// A noncreature spell makes a Clue; a creature spell does not. Until end of
/// turn the tapped Mechanist makes an artifact token you control a 3/1
/// Construct artifact creature with flying; nontoken artifacts can't be
/// chosen.
#[test]
fn the_mechanist_makes_clues_and_flies_them_until_end_of_turn() {
    let mut s = setup(&[
        "The Mechanist, Aerial Artisan",
        "Abandon Attachments",
        "Llanowar Elves",
        "White Lotus Tile",
    ]);
    let mechanist = s.force_permanent_on_battlefield(0, "The Mechanist, Aerial Artisan");
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    ready(&mut s, mechanist);
    s.set_player_mana_pool(0, "1U");
    cast(&mut s, "Abandon Attachments");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    s.set_player_mana_pool(0, "G");
    cast(&mut s, "Llanowar Elves");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);

    let clue = battlefield_token(&s, 0);
    activate(&mut s, mechanist, 0);
    assert!(!s.choose_target(Target::Permanent(tile)));
    assert!(s.choose_target(Target::Permanent(clue)));
    settle(&mut s);
    assert!(tapped(&s, mechanist));
    assert!(is_creature(&s, clue));
    assert_eq!(s.game().effective_pt(clue), (3, 1));
    assert!(observed_keywords(&s, clue).flying);
    assert!(observed(&s, clue)
        .temp_subtypes
        .contains(&"Construct".to_owned()));

    s.advance_to_active_step(1, StepKind::Main);
    assert!(!is_creature(&s, clue), "until end of turn");
    assert!(!observed_keywords(&s, clue).flying);
}
