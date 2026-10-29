use crate::support::helpers::*;
use managym::{
    agent::action::Action,
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::collections::BTreeMap;

pub(crate) fn deck() -> BTreeMap<String, usize> {
    [
        ("Island", 16),
        ("Mountain", 8),
        ("Gran-Gran", 3),
        ("Proft's Eidetic Memory", 3),
        ("Combustion Technique", 4),
        ("Accumulate Wisdom", 4),
        ("Pop Quiz", 2),
        ("Otter-Penguin", 2),
        ("Water Tribe Rallier", 2),
        ("Lightning Bolt", 2),
    ]
    .into_iter()
    .map(|(name, n)| (name.into(), n))
    .collect()
}

pub(crate) fn setup() -> Scenario {
    let mut s = Scenario::new(deck(), deck(), 88);
    s.advance_to_active_step(0, StepKind::Main);
    s
}

pub(crate) fn available(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(i, card)| {
            (card.owner == PlayerId(player)
                && card.name == name
                && matches!(
                    s.game().state.zones.zone_of(CardId(i)),
                    Some(ZoneType::Hand | ZoneType::Library)
                ))
            .then_some(CardId(i))
        })
        .unwrap_or_else(|| panic!("no available {name}"))
}

pub(crate) fn put(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let card = available(s, player, name);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card].unwrap()
}

pub(crate) fn cast(s: &mut Scenario, name: &str) -> CardId {
    let card = available(s, 0, name);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::CastSpell { card: c, .. } if *c == card))
        .unwrap_or_else(|| panic!("cannot cast {name}: {:?}", s.action_space()));
    s.step_action(index);
    card
}

pub(crate) fn resolve(s: &mut Scenario) {
    s.pass_priority();
    s.pass_priority();
}

pub(crate) fn counters(s: &Scenario, id: PermanentId) -> i32 {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .plus1_counters
}

pub(crate) fn card_of(s: &Scenario, id: PermanentId) -> CardId {
    s.game().state.permanents[id].as_ref().unwrap().card
}

pub(crate) fn plant_lessons(s: &mut Scenario, player: usize, n: usize) {
    for _ in 0..n {
        let card = available(s, player, "Accumulate Wisdom");
        s.game_mut().move_card(card, ZoneType::Graveyard);
    }
}

pub(crate) fn combustion(lessons: usize) -> (Scenario, CardId, PermanentId) {
    let mut s = setup();
    plant_lessons(&mut s, 0, lessons);
    plant_lessons(&mut s, 1, 3); // opponent's Lessons are irrelevant
    put(&mut s, 0, "Mountain");
    put(&mut s, 0, "Mountain");
    let victim = put(&mut s, 1, "Water Tribe Rallier"); // 3/2
    s.game_mut().state.permanents[victim]
        .as_mut()
        .unwrap()
        .plus1_counters = 5;
    let spell = cast(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(victim)));
    resolve(&mut s);
    (s, spell, victim)
}
