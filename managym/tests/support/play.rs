// play.rs
// Directed play: take actions, cast and activate, choose targets, modes and
// cards, pass priority and drive resolution, and step through combat.
//
// Driving helpers stop on different conditions, so each has its own name:
// `settle` passes or declines until the stack and suspended decisions are
// empty; `resolve_everything` uses default actions until triggers, the stack
// and decisions drain; `resolve_until_priority` uses default actions until the
// stack, pending events and triggers drain at a priority decision.

use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::{
        decision::Decision,
        turn::{PhaseKind, StepKind},
    },
    state::{
        game_object::{ObjectId, PermanentId, PlayerId},
        payment::{How, Toward},
        player::ManaPayment,
        target::Target,
        zone::ZoneType,
    },
};

use crate::support::{
    game::{
        clear_hand, five_color_setup, five_color_unfunded_setup, force_lands,
        force_new_card_in_hand, move_to_hand, Scenario,
    },
    inspect::{
        can_cast, card_of, find_owned_card, has_action, library, library_card, now, offered_cards,
        permanent_named, tapped, top_spell, zone_of,
    },
};

const MAX_SCENARIO_ACTIONS: usize = 20_000;

// ---------------------------------------------------------------------------
// Actions and turn structure
// ---------------------------------------------------------------------------

impl Scenario {
    pub fn step_action(&mut self, index: usize) {
        self.game.step(index).expect("scenario step should succeed");
    }

    pub fn take_action_by_type(&mut self, action_type: ActionType) -> bool {
        let Some(index) = self.action_index_by_type(action_type) else {
            return false;
        };
        self.step_action(index);
        true
    }

    /// Complete a pending fixed-source payment automatically, then pass priority.
    pub fn pass_priority(&mut self) {
        if self.action_space().kind == ActionSpaceKind::PayOrNot
            && self
                .action_space()
                .actions
                .iter()
                .any(|a| matches!(a, Action::TapForMana { .. }))
        {
            assert!(self.take_action_by_type(ActionType::PayCost));
        }
        assert!(
            self.take_action_by_type(ActionType::PriorityPassPriority),
            "priority pass action should be available"
        );
    }

    pub fn advance_default_action(&mut self) {
        let space = self.action_space().clone();
        let index = match space.kind {
            ActionSpaceKind::Priority => space
                .actions
                .iter()
                .position(|action| action.action_type() == ActionType::PriorityPassPriority)
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::DeclareAttacker => space
                .actions
                .iter()
                .position(|action| matches!(action, Action::DeclareAttacker { attack: None, .. }))
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::DeclareBlocker => space
                .actions
                .iter()
                .position(|action| matches!(action, Action::DeclareBlocker { attacker: None, .. }))
                .unwrap_or(space.actions.len().saturating_sub(1)),
            // Picks first legal target arbitrarily — tests that care about targeting
            // should use choose_target_named or explicit action selection instead.
            ActionSpaceKind::ChooseTarget | ActionSpaceKind::OrderTriggers => 0,
            ActionSpaceKind::GameOver => 0,
            // Mid-resolution / cost decisions default to declining (the
            // last action is Decline for optional choices); tests that care
            // pick explicitly.
            ActionSpaceKind::Scry => 0,
            ActionSpaceKind::LookAndSelect | ActionSpaceKind::PayOrNot | ActionSpaceKind::Learn => {
                space.actions.len().saturating_sub(1)
            }
            ActionSpaceKind::Modal => 0,
            ActionSpaceKind::Waterbend | ActionSpaceKind::Discard | ActionSpaceKind::LegendRule => {
                0
            }
        };
        self.step_action(index);
    }

    pub fn advance_until<F>(&mut self, mut predicate: F, failure_message: String)
    where
        F: FnMut(&Scenario) -> bool,
    {
        for _ in 0..MAX_SCENARIO_ACTIONS {
            if self.game.is_game_over() || predicate(self) {
                return;
            }
            self.advance_default_action();
        }
        panic!("{failure_message}");
    }

    pub fn advance_to_step(&mut self, target: StepKind) {
        self.advance_until(
            |scenario| scenario.current_step() == target,
            format!("failed to reach step {target:?}"),
        );
    }

    pub fn advance_to_phase(&mut self, target: PhaseKind) {
        self.advance_until(
            |scenario| scenario.current_phase() == target,
            format!("failed to reach phase {target:?}"),
        );
    }

    pub fn advance_to_active_step(&mut self, active_player: usize, target: StepKind) {
        self.advance_until(
            |scenario| {
                scenario.active_player() == active_player && scenario.current_step() == target
            },
            format!("failed to reach step {target:?} for player {active_player}"),
        );
    }

    // -----------------------------------------------------------------------
    // Targets and costs
    // -----------------------------------------------------------------------

    pub fn choose_target(&mut self, target: Target) -> bool {
        let Some(index) = self.action_space().actions.iter().position(
            |action| matches!(action, Action::ChooseTarget { target: t, .. } if *t == target),
        ) else {
            return false;
        };
        self.step_action(index);
        true
    }

    pub fn choose_target_named(&mut self, card_name: &str) {
        let choose_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| {
                let Action::ChooseTarget { target, .. } = action else {
                    return false;
                };
                let Target::Permanent(permanent_id) = target else {
                    return false;
                };
                let Some(permanent) = self.game.state.permanents[*permanent_id].as_ref() else {
                    return false;
                };
                let object = &self.game.state[permanent.object];
                object.name == card_name
            })
            .expect("target choice for named permanent should exist");
        self.step_action(choose_index);
    }

    /// Tap a specific permanent to pay {1} of a pending waterbend cost.
    pub fn choose_waterbend_tap(&mut self, permanent: PermanentId) -> bool {
        let Some(index) = self.action_space().actions.iter().position(
            |action| matches!(action, Action::WaterbendTap { permanent: p, .. } if *p == permanent),
        ) else {
            return false;
        };
        self.step_action(index);
        true
    }

    /// Play a land and cast a creature from a player's hand in their main phase.
    /// Resolves the spell (both players pass priority) and returns with the creature on
    /// the battlefield. Expects the player already has the named cards in hand.
    pub fn play_land_and_cast_creature(
        &mut self,
        player: usize,
        land_name: &str,
        creature_name: &str,
    ) {
        self.advance_to_active_step(player, StepKind::Main);
        self.force_card_in_hand(player, land_name);
        self.force_card_in_hand(player, creature_name);
        assert!(self.take_action_by_type(ActionType::PriorityPlayLand));
        assert!(self.take_action_by_type(ActionType::PriorityCastSpell));
        self.pass_priority();
        self.pass_priority();
    }

    // -----------------------------------------------------------------------
    // Combat
    // -----------------------------------------------------------------------

    /// Set up a creature that has survived summoning sickness and is ready to attack.
    /// Casts the creature on player 0's first turn, advances through player 1's turn,
    /// and arrives at player 0's declare attackers step.
    pub fn setup_attacker_ready(&mut self) {
        self.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
        self.advance_to_active_step(1, StepKind::Main);
        self.advance_to_active_step(0, StepKind::DeclareAttackers);
    }

    /// Declare the first available creature as an attacker.
    pub fn declare_attack(&mut self) {
        let attack_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| {
                matches!(
                    action,
                    Action::DeclareAttacker {
                        attack: Some(_),
                        ..
                    }
                )
            })
            .expect("attack action should exist");
        self.step_action(attack_index);
    }

    /// Decline to attack with the first available creature.
    pub fn decline_attack(&mut self) {
        let decline_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| matches!(action, Action::DeclareAttacker { attack: None, .. }))
            .expect("decline-attack action should exist");
        self.step_action(decline_index);
    }

    /// Assign the first available blocker to an attacker.
    pub fn declare_block(&mut self) {
        let block_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| {
                matches!(
                    action,
                    Action::DeclareBlocker {
                        attacker: Some(_),
                        ..
                    }
                )
            })
            .expect("block action should exist");
        self.step_action(block_index);
    }

    /// Decline to block with the first available creature.
    pub fn decline_block(&mut self) {
        let decline_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| matches!(action, Action::DeclareBlocker { attacker: None, .. }))
            .expect("no-block action should exist");
        self.step_action(decline_index);
    }
}

/// Attack with `attacker` and decline every other offered attacker.
pub fn attack_with(s: &mut Scenario, attacker: PermanentId) {
    attack_with_group(s, &[attacker]);
}

/// Declare every requested attacker and decline the others.
pub fn attack_with_group(s: &mut Scenario, attackers: &[PermanentId]) {
    let mut declared = 0;
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let index = s.action_space().actions.iter().position(|action| {
            matches!(action, Action::DeclareAttacker { permanent, attack: Some(_), .. } if attackers.contains(permanent))
        });
        match index {
            Some(index) => {
                s.step_action(index);
                declared += 1;
            }
            None => s.decline_attack(),
        }
    }
    assert_eq!(
        declared,
        attackers.len(),
        "expected every attacker to attack"
    );
}

// ---------------------------------------------------------------------------
// Choosing actions
// ---------------------------------------------------------------------------

/// Take the first offered action matching `predicate`; panic if none matches.
pub fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}

pub fn choose_mode(s: &mut Scenario, wanted: usize) {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseMode { mode, .. } if *mode == wanted))
        .unwrap();
    s.step_action(index);
}

/// Submit a native spending offer matching its public description. A color
/// predicate must not accidentally select an "Add" offer during production.
pub fn spend_named(s: &mut Scenario, predicate: impl Fn(&str) -> bool) {
    assert!(
        matches!(
            s.game()
                .state
                .suspended_decision
                .as_ref()
                .map(|pending| &pending.decision),
            Some(Decision::ManaSpend { .. })
        ),
        "expected a pending mana payment"
    );
    let frame = s.game().semantic_decision_frame().unwrap();
    let index = frame
        .offers
        .iter()
        .position(|offer| predicate(&offer.label))
        .unwrap_or_else(|| panic!("missing spend: {:?}", frame.offers));
    s.step_action(index);
}

/// Pay the pending generic pips with ordinary, step-expiring colorless mana.
/// Call explicitly in funded fixtures that intend to preserve their colors.
/// Stops at the next non-spending decision; never chooses production or riders.
pub fn spend_colorless(s: &mut Scenario) {
    for _ in 0..256 {
        if !matches!(
            s.game()
                .state
                .suspended_decision
                .as_ref()
                .map(|pending| &pending.decision),
            Some(Decision::ManaSpend { .. })
        ) {
            return;
        }
        spend_named(s, |label| label == "Spend {C} (expires at step end)");
    }
    panic!("colorless payment did not complete");
}

/// Select the card behind `object` wherever it is offered.
pub fn select_card(s: &mut Scenario, object: ObjectId) {
    let object = now(s, object);
    choose(
        s,
        |a| matches!(a,Action::SelectCard {object: selected,..} if *selected==object),
    );
}

/// A copy of the scenario in which the offered action at `index` was taken:
/// one line of play from a choice, leaving `s` at the choice.
pub fn fork_stepping(s: &Scenario, index: usize) -> Scenario {
    let mut fork = s.clone();
    fork.step_action(index);
    fork
}

/// `fork_stepping` the first offered action matching `predicate`.
pub fn fork_choosing(s: &Scenario, predicate: impl Fn(&Action) -> bool) -> Scenario {
    let index = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("no such action is offered: {:?}", s.action_space()));
    fork_stepping(s, index)
}

/// `fork_stepping` the selection of the offered `object`.
pub fn fork_selecting(s: &Scenario, object: ObjectId) -> Scenario {
    fork_choosing(
        s,
        |a| matches!(a, Action::SelectCard { object: offered, .. } if *offered == object),
    )
}

/// Tap `source` for mana while a cost is being paid or at priority. A source
/// whose mana carries a rider confirms its one output.
pub fn tap_for_mana(s: &mut Scenario, source: PermanentId) {
    choose(s, |a| {
        matches!(
            a,
            Action::ActivateAbility { permanent, .. } | Action::TapForMana { permanent, .. }
                if *permanent == source
        )
    });
    if s.action_space().kind == ActionSpaceKind::Modal {
        choose(s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    }
}

/// Finish a manual payment whose remaining choices do not matter: pay, then
/// spend the first unit offered until the spell is cast.
pub fn pay_rest(s: &mut Scenario) {
    let stack = s.game().state.stack_objects.len();
    for _ in 0..32 {
        if s.game().state.stack_objects.len() > stack {
            return;
        }
        let pay = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PayCost { .. }))
            .unwrap_or(0);
        s.step_action(pay);
    }
    panic!("the payment did not finish: {:?}", s.action_space());
}

pub fn target_opponent(s: &mut Scenario) {
    choose(s, |a| {
        matches!(
            a,
            Action::ChooseTarget {
                target: Target::Player(PlayerId(1)),
                ..
            }
        )
    });
}

// ---------------------------------------------------------------------------
// Casting and activating
// ---------------------------------------------------------------------------

/// Move player 0's library copy into the hand through the zone-change
/// authority and cast that copy. Returns the card's object once the action
/// returns: the spell, or the hand card while a casting choice is pending.
pub fn cast(s: &mut Scenario, name: &str) -> ObjectId {
    let id = move_to_hand(s, name);
    cast_card(s, id);
    now(s, id)
}

pub fn cast_clone_as(s: &mut Scenario, creature: PermanentId) -> ObjectId {
    let clone = cast(s, "Clone");
    spend_colorless(s);
    resolve_top(s);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(s);
    clone
}

/// Cast `id` from whichever zone currently permits it, refreshing priority.
pub fn cast_card(s: &mut Scenario, id: ObjectId) {
    s.game_mut().scenario_refresh_priority().unwrap();
    let id = now(s, id);
    choose(
        s,
        |a| matches!(a, Action::CastSpell { object, .. } if *object == id),
    );
}

/// Refresh priority, then whether `id` may be cast.
pub fn castable(s: &mut Scenario, id: ObjectId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    can_cast(s, id)
}

/// Play player 0's library copy of a land with the ordinary land action.
pub fn play_land(s: &mut Scenario, name: &str) -> PermanentId {
    let id = move_to_hand(s, name);
    choose(
        s,
        |a| matches!(a, Action::PlayLand { object, .. } if *object == id),
    );
    s.game()
        .state
        .permanent_of(now(s, id))
        .expect("land entered")
}

/// Play player 0's two-color land `name`, which enters tapped, and check it
/// pays a one-colored-pip spell of each color in `colors` and of no other.
/// `entered` settles and checks whatever the land does as it enters.
pub fn tapped_dual_land_pays_either_color(
    name: &str,
    colors: &str,
    entered: impl Fn(&mut Scenario, PermanentId),
) {
    let spells = [
        ('W', "Healer's Hawk"),
        ('U', "Wind Drake"),
        ('B', "Typhoid Rats"),
        ('R', "Raging Goblin"),
        ('G', "Llanowar Elves"),
    ];
    for (color, spell) in spells {
        let mut s = five_color_setup(&[name, spell]);
        let land = play_land(&mut s, name);
        assert!(tapped(&s, land), "{name} enters tapped");
        entered(&mut s, land);
        // Wind Drake is {2}{U}: the pool pays its generic part.
        s.set_player_mana_pool(0, if color == 'U' { "2" } else { "0" });
        let id = move_to_hand(&mut s, spell);
        assert!(!castable(&mut s, id), "{name} is tapped");
        s.game_mut().state.permanents[land].as_mut().unwrap().tapped = false;
        assert_eq!(
            castable(&mut s, id),
            colors.contains(color),
            "{name} for {color}"
        );
        if colors.contains(color) {
            cast_card(&mut s, id);
            settle(&mut s);
            assert!(tapped(&s, land), "{name} pays {color}");
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        }
    }
}

/// A Temple is a `tapped_dual_land_pays_either_color` that scries 1 as it
/// enters: the top card stays, or goes to the bottom, as chosen.
pub fn temple_scries_one_and_pays_either_color(name: &str, colors: &str) {
    for bottom in [false, true] {
        tapped_dual_land_pays_either_color(name, colors, |s, _| {
            pass_until(s, ActionSpaceKind::Scry);
            let before = library(s, 0);
            assert_eq!(s.action_space().actions.len(), 2);
            choose(
                s,
                |a| matches!(a, Action::ScryCard { to_bottom, .. } if *to_bottom == bottom),
            );
            settle(s);
            let mut expected = before;
            if bottom {
                expected.rotate_right(1);
            }
            assert_eq!(library(s, 0), expected, "{name}");
        });
    }
}

/// Cast the red spell `name`, whose additional cost is "discard a card", with
/// `mountains` for its mana, and return it settled. The discard is player 0's
/// pick under either payment setting, from a Gray Ogre and a Hill Giant and
/// never the spell itself; the Giant is discarded and the receipt names it.
pub fn cast_discarding_a_chosen_card(name: &str, mountains: usize, auto: bool) -> Scenario {
    let mut s = five_color_unfunded_setup(&[name, "Gray Ogre", "Hill Giant"]);
    force_lands(&mut s, 0, "Mountain", mountains);
    if auto {
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
    }
    clear_hand(&mut s, 0);
    let spell = force_new_card_in_hand(&mut s, 0, name);
    assert!(!castable(&mut s, spell), "no other card to discard");
    let ogre = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let giant = force_new_card_in_hand(&mut s, 0, "Hill Giant");
    assert!(castable(&mut s, spell));
    cast_card(&mut s, spell);
    assert_eq!(
        s.action_space().kind,
        ActionSpaceKind::Discard,
        "auto={auto}"
    );
    let mut offered = offered_cards(&s);
    offered.sort();
    let mut expected = vec![ogre, giant];
    expected.sort();
    assert_eq!(offered, expected);
    assert!(!has_action(&s, |a| matches!(a, Action::PayCost { .. })));
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert!(
        context.instruction.contains("to cast this spell"),
        "{}",
        context.instruction
    );
    let fork = fork_selecting(&s, ogre);
    assert_eq!(zone_of(&fork, ogre), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&fork, giant), Some(ZoneType::Hand));
    // The Mountains pay the rest without a further choice.
    select_card(&mut s, giant);
    assert_eq!(zone_of(&s, giant), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Hand));
    let cast = top_spell(&s);
    assert_eq!(usize::from(cast.spent.total()), mountains);
    assert_eq!(
        cast.spent
            .contributed(How::Discarded, Toward::Printed)
            .count(),
        1
    );
    settle(&mut s);
    s
}

/// `name` has flash: player 0 may cast it while a spell is on the stack, where
/// a Gray Ogre may not be cast.
pub fn casts_at_instant_speed(name: &str) {
    let mut s = five_color_unfunded_setup(&[name, "Gray Ogre"]);
    force_lands(&mut s, 0, "Mountain", 8);
    force_lands(&mut s, 0, "Plains", 8);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let first = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let second = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let flashed = force_new_card_in_hand(&mut s, 0, name);
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(&mut s, first);
    assert!(!s.game().state.stack_objects.is_empty());
    assert!(can_cast(&s, flashed));
    assert!(!can_cast(&s, second));
}

/// The creature `name` pays {1}, from one `land`, and itself to destroy an
/// artifact or enchantment: it leaves as the cost is paid, and a creature is
/// no target.
pub fn sacrifices_itself_to_destroy_an_artifact(name: &str, land: &str) {
    let mut s = five_color_unfunded_setup(&[name, "Sol Ring", "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, name);
    let source_card = card_of(&s, source);
    force_lands(&mut s, 0, land, 1);
    let target = s.force_permanent_on_battlefield(1, "Sol Ring");
    let target_card = card_of(&s, target);
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    activate(&mut s, source, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ChooseTarget { target: Target::Permanent(id), .. } if *id == other)
    }));
    assert!(s.choose_target(Target::Permanent(target)));
    assert_eq!(zone_of(&s, source_card), Some(ZoneType::Graveyard));
    assert_eq!(s.untapped_permanents_named(0, land), 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, target_card), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[other].is_some());
}

/// Like `cast`, but the copy may already be in player 0's hand.
pub fn cast_owned(s: &mut Scenario, name: &str) -> ObjectId {
    let object = find_owned_card(s, 0, name);
    let object = s.game_mut().move_object(object, ZoneType::Hand);
    cast_card(s, object);
    now(s, object)
}

/// Take the first offered cast action.
pub fn cast_only(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
}

/// Take the first offered cast action, then both players pass.
pub fn cast_and_resolve(s: &mut Scenario) {
    cast_only(s);
    resolve_top(s);
}

/// Refresh priority, then whether ability `index` of `source` is offered.
pub fn offers_ability(s: &mut Scenario, source: PermanentId, index: usize) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    has_action(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
            if *permanent == source && *ability_index == index)
    })
}

/// Activate ability `index` of a permanent, refreshing priority only at a
/// priority decision.
pub fn activate(s: &mut Scenario, source: PermanentId, index: usize) {
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
        if *permanent == source && *ability_index == index)
    });
}

/// Activate ability `index` of a card in its hand or graveyard, refreshing
/// priority first.
pub fn activate_card(s: &mut Scenario, source: ObjectId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    let source = now(s, source);
    choose(s, |a| {
        matches!(a, Action::ActivateCardAbility { object, ability_index, .. }
        if *object == source && *ability_index == index)
    });
}

/// Activate the first offered ability of a permanent, refreshing priority only
/// at a priority decision.
pub fn activate_first(s: &mut Scenario, id: PermanentId) {
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |a| matches!(a,Action::ActivateAbility {permanent,..} if *permanent==id),
    );
}

/// Activate ability `index` of player 0's first permanent with this name,
/// refreshing priority only at a priority decision.
pub fn activate_named(s: &mut Scenario, name: &str, index: usize) {
    let source = permanent_named(s, name);
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |action| matches!(action, Action::ActivateAbility { permanent, ability_index, .. } if *permanent == source && *ability_index == index),
    );
}

/// Put player 0's library copy into play and `settle`: a land moves straight
/// onto the battlefield, anything else is cast.
pub fn play_and_settle(s: &mut Scenario, name: &str) -> ObjectId {
    let id = library_card(s, 0, name);
    if s.game().state[id].types.is_land() {
        s.game_mut().move_object(id, ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
    } else {
        cast(s, name);
    }
    settle(s);
    now(s, id)
}

/// Move player 0's library copy onto the battlefield through the zone-change
/// authority, pass once if it triggered, and `settle`.
pub fn enter_and_settle(s: &mut Scenario, name: &str) -> ObjectId {
    let id = library_card(s, 0, name);
    move_and_settle(s, id, ZoneType::Battlefield)
}

/// Move player 0's library copy onto the battlefield through the zone-change
/// authority and stop at its entry trigger's target choice. Returns the
/// library appearance; `now` follows it to the battlefield.
pub fn enter_until_target(s: &mut Scenario, name: &str) -> ObjectId {
    let id = library_card(s, 0, name);
    crate::support::game::move_card(s, id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(s, ActionSpaceKind::ChooseTarget);
    id
}

/// Resolve any triggers from moving the card to its destination.
pub fn move_and_settle(s: &mut Scenario, id: ObjectId, zone: ZoneType) -> ObjectId {
    crate::support::game::move_card(s, id, zone);
    s.game_mut().scenario_refresh_priority().unwrap();
    if !s.game().state.pending_triggers.is_empty() {
        s.pass_priority();
    }
    settle(s);
    now(s, id)
}

// ---------------------------------------------------------------------------
// Passing and resolving
// ---------------------------------------------------------------------------

/// Player 0 casts a Lightning Bolt at `target` (floating {R} when its pool
/// is empty) and both players pass, so it resolves; triggers it causes wait
/// for the caller.
pub fn bolt(s: &mut Scenario, target: PermanentId) {
    if s.game().state.players[0].mana_pool.total() == 0 {
        s.set_player_mana_pool(0, "R");
    }
    let card = force_new_card_in_hand(s, 0, "Lightning Bolt");
    cast_card(s, card);
    assert!(s.choose_target(Target::Permanent(target)));
    s.pass_priority();
    s.pass_priority();
}

/// Both players pass priority once.
pub fn resolve_top(s: &mut Scenario) {
    s.pass_priority();
    s.pass_priority();
}

/// Pass or decline until the stack and suspended decisions are empty at a
/// priority decision; otherwise take the first action.
pub fn settle(s: &mut Scenario) {
    for _ in 0..100 {
        if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.suspended_decision.is_none()
        {
            return;
        }
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. } | Action::Decline { .. }))
            .unwrap_or(0);
        s.step_action(index);
    }
    panic!("resolution did not settle")
}

/// Pass priority (ordering triggers first-offered) until a decision of `kind`.
pub fn pass_until(s: &mut Scenario, kind: ActionSpaceKind) {
    for _ in 0..10 {
        if s.action_space().kind == kind {
            return;
        }
        if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            // This fixture driver deliberately chooses the first offered trigger.
            s.step_action(0);
            continue;
        }
        s.pass_priority();
    }
    assert_eq!(s.action_space().kind, kind);
}

/// Resolve damage triggers with a fixed target; fail if the stack never settles.
pub fn settle_targeting_opponent(s: &mut Scenario) {
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            target_opponent(s);
        } else if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        } else if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("damage triggers did not settle");
}

/// Advance with default actions until the stack, pending triggers, and
/// decisions are drained.
pub fn resolve_everything(s: &mut Scenario) {
    s.advance_until(
        |sc| {
            sc.game().state.pending_triggers.is_empty()
                && sc.game().state.trigger_placement.is_none()
                && sc.game().state.stack_objects.is_empty()
                && sc.game().state.suspended_decision.is_none()
        },
        "stack and triggers should drain".to_string(),
    );
}

/// Advance with default actions until the stack, pending events, and pending
/// triggers are empty at a priority decision.
pub fn resolve_until_priority(s: &mut Scenario) {
    s.advance_until(
        |s| {
            s.game().state.stack_objects.is_empty()
                && s.game().state.pending_events.is_empty()
                && s.game().state.pending_triggers.is_empty()
                && s.action_space().kind == ActionSpaceKind::Priority
        },
        "stack should empty out".to_string(),
    );
}

/// Commit the chosen legal action through the public semantic Command API,
/// checking that a fork commits identically, the source is untouched until
/// it commits, and the spent Command is then stale. Return the committed fork.
pub fn command(
    s: &mut Scenario,
    predicate: impl Fn(&Action) -> bool,
) -> (Scenario, managym::decision::Command) {
    use managym::{decision::Command, search_state::witness};
    let index = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .expect("legal action");
    let frame = s.game().semantic_decision_frame().unwrap();
    let command = Command {
        command_id: format!("scenario-{}", frame.revision),
        expected_revision: frame.revision,
        offer_id: frame.offers[index].id.0,
        answers: Vec::new(),
        object_preconditions: Vec::new(),
    };
    let before = witness(s.game());
    let mut fork = s.game().clone();
    fork.execute_semantic_command(&command).unwrap();
    assert_eq!(witness(s.game()), before);
    s.game_mut().execute_semantic_command(&command).unwrap();
    assert_eq!(witness(s.game()), witness(&fork));
    assert!(
        s.game_mut().execute_semantic_command(&command).is_err(),
        "stale Command rejected"
    );
    assert_eq!(
        witness(s.game()),
        witness(&fork),
        "stale Command changed state"
    );
    (Scenario { game: fork }, command)
}

/// Answer the topmost stacked ability with a scenario permanent's counter
/// ability.
pub fn counter_stacked_ability(s: &mut Scenario) {
    use managym::state::{
        ability::{Effect, ObjectSelector, TargetSpec},
        card::ActivatedAbilityDefinition,
    };
    let top = Target::StackAbility(s.game().state.stack_objects.last().unwrap().id());
    let seat = s.action_space().player.unwrap().0;
    let source = s.force_permanent_on_battlefield(seat, "Gray Ogre");
    let object = card_of(s, source);
    s.game_mut().state[object]
        .activated_abilities
        .push(ActivatedAbilityDefinition {
            effects: vec![Effect::CounterAbility {
                what: ObjectSelector::Target(TargetSpec::Ability),
            }],
            ..Default::default()
        });
    activate(s, source, 0);
    assert!(crate::support::inspect::offered_targets(s).contains(&top));
    assert!(s.choose_target(top));
    settle(s);
}

/// Select a printed spell on the card's current appearance.
pub fn cast_profile(
    s: &mut Scenario,
    object: ObjectId,
    profile: managym::state::card::SpellProfile,
) {
    s.game_mut().scenario_refresh_priority().unwrap();
    let object = now(s, object);
    choose(
        s,
        |action| matches!(action, Action::CastSpell { object: legal, profile: offered, .. } if *legal == object && *offered == profile),
    );
}
