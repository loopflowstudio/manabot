//! Tests for cards filed in `src/cmr.rs`, in card order.

use crate::support::{
    game::{five_color_setup, move_to_hand, pin_top, Scenario},
    inspect::{assert_choice_contract, zone_of},
    play::{cast_card, choose, settle},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        card::StaticAbility,
        game_object::{ObjectId, PlayerId, Target},
        zone::ZoneType,
    },
};

const DECK: [&str; 5] = [
    "Volcanic Torrent",
    "Hill Giant",
    "Gray Ogre",
    "Rending Volley",
    "Drannith Magistrate",
];

/// Cast Volcanic Torrent from hand over `top` (first on top) and resolve
/// cascade's trigger up to its first decision.
fn cascade_over(s: &mut Scenario, top: &[&str]) -> Vec<ObjectId> {
    let torrent = move_to_hand(s, "Volcanic Torrent");
    let pinned = pin_top(s, top);
    cast_card(s, torrent);
    while s.action_space().kind == ActionSpaceKind::Priority
        && !s.game().state.stack_objects.is_empty()
        && s.game().state.stack_objects.len() > 1
    {
        s.pass_priority();
    }
    pinned
}

fn bottom(s: &Scenario, count: usize) -> Vec<ObjectId> {
    let mut objects = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0))[..count]
        .to_vec();
    objects.sort();
    objects
}

fn sorted(mut objects: Vec<ObjectId>) -> Vec<ObjectId> {
    objects.sort();
    objects
}

/// Cascade passes a land and a nonland of equal mana value, offers the first
/// cheaper card, casts it free and bottoms the rest; the Torrent then counts
/// both spells cast this turn.
#[test]
fn volcanic_torrent_cascades_past_costly_cards_and_counts_spells_cast() {
    let mut s = five_color_setup(&DECK);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let pinned = cascade_over(&mut s, &["Forest", "Volcanic Torrent", "Hill Giant"]);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_choice_contract(&s);
    let hit = pinned[2];
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { object, .. } if *object == hit),
    );
    assert_eq!(zone_of(&s, hit), Some(ZoneType::Stack));
    assert_eq!(bottom(&s, 2), sorted(pinned[..2].to_vec()));
    settle(&mut s);
    assert_eq!(zone_of(&s, hit), Some(ZoneType::Battlefield));
    assert!(
        s.game().state.permanents[ogre].is_none(),
        "2 damage kills a 2/2"
    );
    assert_eq!(s.game().state.permanents[giant].as_ref().unwrap().damage, 2);
}

/// Declining the cast bottoms the hit with every other exiled card.
#[test]
fn volcanic_torrent_cascade_may_be_declined() {
    let mut s = five_color_setup(&DECK);
    let pinned = cascade_over(&mut s, &["Mountain", "Gray Ogre"]);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    assert_eq!(bottom(&s, 2), sorted(pinned));
    assert_eq!(s.game().state.zones.size(ZoneType::Exile, PlayerId(0)), 0);
}

/// A hit with no legal target can't be cast, and cascade stops there anyway;
/// a library with no cheaper nonland is exiled whole and returned.
#[test]
fn volcanic_torrent_cascade_without_a_castable_hit_bottoms_everything() {
    let mut s = five_color_setup(&DECK);
    let pinned = cascade_over(&mut s, &["Rending Volley", "Gray Ogre"]);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(bottom(&s, 1), vec![pinned[0]]);
    assert_eq!(zone_of(&s, pinned[1]), Some(ZoneType::Library));
    settle(&mut s);

    let mut s = five_color_setup(&DECK);
    let torrent = move_to_hand(&mut s, "Volcanic Torrent");
    let kept = pin_top(&mut s, &["Island", "Volcanic Torrent"]);
    s.game_mut()
        .state
        .zones
        .zone_objects_mut(ZoneType::Library, PlayerId(0))
        .retain(|object| kept.contains(object));
    cast_card(&mut s, torrent);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(bottom(&s, 2), sorted(kept));
}

/// Cascade's free cast is a cast from exile, which Drannith Magistrate
/// forbids; the hit goes to the bottom.
#[test]
fn volcanic_torrent_cascade_respects_hand_only_casting() {
    let mut s = five_color_setup(&DECK);
    s.force_permanent_on_battlefield(1, "Drannith Magistrate");
    let pinned = cascade_over(&mut s, &["Hill Giant"]);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(bottom(&s, 1), pinned);
}

/// Sakashima offers only creatures its controller controls, enters as a
/// copy that keeps the legend-rule exemption (CR 707.9b), so both legendary
/// Rhys stay without a legend-rule choice.
#[test]
fn sakashima_copies_a_legend_you_control_and_both_stay() {
    let mut s = five_color_setup(&[
        "Sakashima of a Thousand Faces",
        "Rhys the Redeemed",
        "Gray Ogre",
    ]);
    let rhys = s.force_permanent_on_battlefield(0, "Rhys the Redeemed");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let sakashima = move_to_hand(&mut s, "Sakashima of a Thousand Faces");
    cast_card(&mut s, sakashima);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let offered: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::ChooseTarget { target, .. } => Some(*target),
            _ => None,
        })
        .collect();
    assert_eq!(offered, vec![Target::Permanent(rhys)]);
    assert!(s.choose_target(Target::Permanent(rhys)));
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "Rhys the Redeemed").len(),
        2
    );
    let copy = s.game().state.object_to_permanent[sakashima].expect("entered");
    assert!(s
        .game()
        .characteristics(copy)
        .static_abilities
        .contains(&StaticAbility::LegendRuleDoesntApply));
}
