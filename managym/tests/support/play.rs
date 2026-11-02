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
    flow::turn::{PhaseKind, StepKind},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        target::Target,
        zone::ZoneType,
    },
};

use crate::support::{
    game::{move_to_hand, Scenario},
    inspect::{can_cast, find_owned_card, has_action, library_card, permanent_named},
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

    pub fn pass_priority(&mut self) {
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
                .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
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
                let card = &self.game.state.cards[permanent.card];
                card.name == card_name
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
            .position(|action| matches!(action, Action::DeclareAttacker { attack: true, .. }))
            .expect("attack action should exist");
        self.step_action(attack_index);
    }

    /// Decline to attack with the first available creature.
    pub fn decline_attack(&mut self) {
        let decline_index = self
            .action_space()
            .actions
            .iter()
            .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
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
    let mut attacked = false;
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let space = s.action_space().clone();
        let index = space.actions.iter().position(|action| {
            matches!(action, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == attacker)
        });
        match index {
            Some(index) => {
                s.step_action(index);
                attacked = true;
            }
            None => s.decline_attack(),
        }
    }
    assert!(attacked, "expected to attack");
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

pub fn select_card(s: &mut Scenario, card: CardId) {
    choose(
        s,
        |a| matches!(a,Action::SelectCard {card: selected,..} if *selected==card),
    );
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
/// authority and cast that copy.
pub fn cast(s: &mut Scenario, name: &str) -> CardId {
    let id = move_to_hand(s, name);
    cast_card(s, id);
    id
}

/// Cast `id` from whichever zone currently permits it, refreshing priority.
pub fn cast_card(s: &mut Scenario, id: CardId) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == id),
    );
}

/// Refresh priority, then whether `id` may be cast.
pub fn castable(s: &mut Scenario, id: CardId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    can_cast(s, id)
}

/// Play player 0's library copy of a land with the ordinary land action.
pub fn play_land(s: &mut Scenario, name: &str) -> PermanentId {
    let id = move_to_hand(s, name);
    choose(
        s,
        |a| matches!(a, Action::PlayLand { card, .. } if *card == id),
    );
    s.game().state.card_to_permanent[id].expect("land entered")
}

/// Like `cast`, but the copy may already be in player 0's hand.
pub fn cast_owned(s: &mut Scenario, name: &str) -> CardId {
    let card = find_owned_card(s, 0, name);
    s.game_mut().move_card(card, ZoneType::Hand);
    cast_card(s, card);
    card
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
pub fn activate_card(s: &mut Scenario, source: CardId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateCardAbility { card, ability_index, .. }
        if *card == source && *ability_index == index)
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
pub fn play_and_settle(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    if s.game().state.cards[id].types.is_land() {
        s.game_mut().move_card(id, ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
    } else {
        cast(s, name);
    }
    settle(s);
    id
}

/// Move player 0's library copy onto the battlefield through the zone-change
/// authority, pass once if it triggered, and `settle`.
pub fn enter_and_settle(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    if !s.game().state.pending_triggers.is_empty() {
        s.pass_priority();
    }
    settle(s);
    id
}

// ---------------------------------------------------------------------------
// Passing and resolving
// ---------------------------------------------------------------------------

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
