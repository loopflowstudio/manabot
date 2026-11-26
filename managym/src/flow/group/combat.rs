//! Combat declaration kinds (`DeclareAttackers`, `DeclareBlockers`) under the
//! shared grouped-input driver.
use std::collections::HashSet;

use crate::agent::action::{Action, ActionSpace, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::combat::AttackTarget;
use crate::flow::event::GameEvent;
use crate::flow::game::Game;
use crate::flow::group::{
    AttackerItem, BlockerItem, ChoiceCheck, DecisionChoice, DecisionOutcome, DecisionRefusal,
    DecisionSpec, GroupKind, NextStep, PlannedAttack, PlannedBlock,
};
use crate::state::game_object::{PermanentId, PlayerId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CombatRole {
    Attackers,
    Blockers,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CombatGroup {
    role: CombatRole,
}

impl CombatGroup {
    pub(crate) const fn attackers() -> Self {
        Self {
            role: CombatRole::Attackers,
        }
    }

    pub(crate) const fn blockers() -> Self {
        Self {
            role: CombatRole::Blockers,
        }
    }
}

pub(crate) struct CombatProgress {
    remaining: Vec<PermanentId>,
    offset: usize,
}

#[derive(Clone, Copy, Debug)]
enum Answer {
    Attack(Option<AttackTarget>),
    Block(Option<PermanentId>),
}

impl GroupKind for CombatGroup {
    type Progress = CombatProgress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<DecisionSpec> {
        let order = game.declaration_order(self.role, viewer).ok()?;
        if order.is_empty() {
            return None;
        }
        match self.role {
            CombatRole::Attackers => {
                let can_attack: Vec<ObjectRenderId> = game
                    .declaration_targets(game.active_player())
                    .filter_map(|target| match target {
                        AttackTarget::Permanent(object) => {
                            game.object_event_ref(object).map(Into::into)
                        }
                        _ => None,
                    })
                    .collect();
                Some(DecisionSpec::DeclareAttackers {
                    declarable: order
                        .iter()
                        .map(|creature| AttackerItem {
                            item: game.permanent_render_id(*creature),
                            label: game.permanent_label(*creature),
                            can_attack: can_attack.clone(),
                            must_attack: game.must_attack(*creature),
                        })
                        .collect(),
                    committed: game.declared_attackers(),
                })
            }
            CombatRole::Blockers => Some(DecisionSpec::DeclareBlockers {
                declarable: order
                    .iter()
                    .map(|creature| {
                        let others: Vec<_> = order
                            .iter()
                            .copied()
                            .filter(|id| *id != *creature)
                            .collect();
                        let options = game.block_options(*creature, &others);
                        let must_block = !options.contains(&None);
                        BlockerItem {
                            item: game.permanent_render_id(*creature),
                            label: game.permanent_label(*creature),
                            can_block: options
                                .into_iter()
                                .flatten()
                                .map(|id| game.permanent_render_id(id))
                                .collect(),
                            must_block,
                        }
                    })
                    .collect(),
                committed: game.declared_blocks(),
            }),
        }
    }

    fn validate(&self, game: &Game, spec: &DecisionSpec, choice: &DecisionChoice) -> ChoiceCheck {
        let refuse = |id: ObjectRenderId, reason: &str| {
            let label = game.permanent_render_label(id);
            ChoiceCheck::Refused(DecisionRefusal {
                item: Some(id),
                reason: format!("{label} {reason}"),
                label,
            })
        };
        match (spec, choice) {
            (
                DecisionSpec::DeclareAttackers {
                    declarable,
                    committed,
                },
                DecisionChoice::DeclareAttackers { attackers },
            ) => {
                let asked: HashSet<_> = declarable.iter().map(|c| c.item).collect();
                let mut seen = HashSet::new();
                for attack in attackers {
                    if attack
                        .defender
                        .is_some_and(|id| game.permanent_by_render_id(id).is_none())
                    {
                        return refuse(attack.creature, "can't attack a defender that has left.");
                    }
                    if !asked.contains(&attack.creature) && !committed.contains(attack) {
                        let reason = match attack.defender {
                            Some(defender) => {
                                format!("can't attack {}.", game.permanent_render_label(defender))
                            }
                            None => "can't attack now.".to_string(),
                        };
                        return refuse(attack.creature, &reason);
                    }
                    if !seen.insert(attack.creature) {
                        return refuse(attack.creature, "can attack only one defender.");
                    }
                }
                ChoiceCheck::Ready
            }
            (
                DecisionSpec::DeclareBlockers {
                    declarable,
                    committed,
                },
                DecisionChoice::DeclareBlockers { blocks },
            ) => {
                let asked: HashSet<_> = declarable.iter().map(|c| c.item).collect();
                let mut seen = HashSet::new();
                for block in blocks {
                    if !asked.contains(&block.blocker) && !committed.contains(block) {
                        let reason = match committed
                            .iter()
                            .find(|other| other.blocker == block.blocker)
                        {
                            Some(other) => {
                                format!(
                                    "already blocks {}.",
                                    game.permanent_render_label(other.attacker)
                                )
                            }
                            None => "can't block now.".to_string(),
                        };
                        return refuse(block.blocker, &reason);
                    }
                    if !seen.insert(block.blocker) {
                        return refuse(block.blocker, "can block only one attacker.");
                    }
                }
                ChoiceCheck::Ready
            }
            _ => unreachable!("matches_choice already verified variant"),
        }
    }

    fn start(
        &self,
        game: &Game,
        viewer: PlayerId,
        _choice: &DecisionChoice,
    ) -> Result<Self::Progress, AgentError> {
        let remaining = game.declaration_order(self.role, viewer)?;
        Ok(CombatProgress {
            remaining,
            offset: 0,
        })
    }

    fn continues(
        &self,
        position: &Game,
        viewer: PlayerId,
        _spec: &DecisionSpec,
        progress: &Self::Progress,
    ) -> bool {
        position
            .asked_creature(self.role, viewer)
            .is_some_and(|asked| progress.remaining[progress.offset..].contains(&asked))
    }

    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        viewer: PlayerId,
        choice: &DecisionChoice,
        progress: &mut Self::Progress,
    ) -> NextStep {
        let Some(asked) = position.asked_creature(self.role, viewer) else {
            return NextStep::Done;
        };
        let Some(at) = progress.remaining[progress.offset..]
            .iter()
            .position(|creature| *creature == asked)
        else {
            return NextStep::Done;
        };
        progress.offset += at + 1;
        let wanted = live.planned_combat_answer(self.role, asked, choice);
        let space = position.action_space().expect("asked decision");
        let index = space
            .actions
            .iter()
            .position(|action| match (action, wanted) {
                (Action::DeclareAttacker { attack, .. }, Answer::Attack(want)) => *attack == want,
                (Action::DeclareBlocker { attacker, .. }, Answer::Block(want)) => *attacker == want,
                _ => false,
            });
        match index {
            Some(action_index) => NextStep::Step {
                action_index,
                item: Some(live.permanent_render_id(asked)),
            },
            None => NextStep::Refuse(live.combat_refusal(asked, wanted, space)),
        }
    }

    fn finish(
        &self,
        live: &Game,
        position: &Game,
        obs_events_from: usize,
        _spec: &DecisionSpec,
        choice: &DecisionChoice,
        _progress: &Self::Progress,
    ) -> (DecisionOutcome, Option<DecisionRefusal>) {
        let outcome = live.read_combat_outcome(self.role, position, obs_events_from);
        let refused = live.combat_undeclared(self.role, choice, &outcome);
        (outcome, refused)
    }
}

impl Game {
    /// The creature asked now, then those the engine will ask after it.
    fn declaration_order(
        &self,
        role: CombatRole,
        viewer: PlayerId,
    ) -> Result<Vec<PermanentId>, AgentError> {
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or_else(|| AgentError("declaration without combat".into()))?;
        let current = self
            .asked_creature(role, viewer)
            .ok_or_else(|| AgentError("declaration names no creature".into()))?;
        let queue = match role {
            CombatRole::Attackers => &combat.attackers_to_declare,
            CombatRole::Blockers => &combat.blockers_to_declare,
        };
        Ok(std::iter::once(current)
            .chain(queue.iter().rev().copied())
            .collect())
    }

    fn asked_creature(&self, role: CombatRole, viewer: PlayerId) -> Option<PermanentId> {
        let space = self.action_space()?;
        if space.player != Some(viewer) {
            return None;
        }
        space
            .actions
            .iter()
            .find_map(|action| match (action, role) {
                (Action::DeclareAttacker { permanent, .. }, CombatRole::Attackers) => {
                    Some(*permanent)
                }
                (Action::DeclareBlocker { blocker, .. }, CombatRole::Blockers) => Some(*blocker),
                _ => None,
            })
    }

    fn permanent_render_id(&self, permanent: PermanentId) -> ObjectRenderId {
        self.permanent_event_ref(permanent)
            .expect("declared creature is on the battlefield")
            .into()
    }

    fn permanent_label(&self, permanent: PermanentId) -> String {
        self.permanent_object(permanent)
            .map_or_else(String::new, |object| self.observed_object_name(object))
    }

    /// Attackers already declared in this combat, before the current
    /// decision (a creature that must attack and was offered no choice).
    fn declared_attackers(&self) -> Vec<PlannedAttack> {
        self.state
            .combat
            .iter()
            .flat_map(|combat| &combat.attack_targets[combat.already_attacking..])
            .map(|&(attacker, target)| PlannedAttack {
                creature: self.permanent_render_id(attacker),
                defender: match target {
                    Some(AttackTarget::Permanent(object)) => {
                        self.object_event_ref(object).map(Into::into)
                    }
                    _ => None,
                },
            })
            .collect()
    }

    /// Blocks already declared in this combat, before the current decision
    /// (a creature that must block and was offered no choice).
    fn declared_blocks(&self) -> Vec<PlannedBlock> {
        self.state
            .combat
            .iter()
            .flat_map(|combat| &combat.attacker_to_blockers)
            .flat_map(|(attacker, blockers)| {
                blockers.iter().map(|blocker| PlannedBlock {
                    blocker: self.permanent_render_id(*blocker),
                    attacker: self.permanent_render_id(*attacker),
                })
            })
            .collect()
    }

    /// The permanent's label, or "That creature" once it is gone.
    pub(crate) fn permanent_render_label(&self, id: ObjectRenderId) -> String {
        self.permanent_by_render_id(id).map_or_else(
            || "That creature".to_string(),
            |permanent| self.permanent_label(permanent),
        )
    }

    fn permanent_by_render_id(&self, id: ObjectRenderId) -> Option<PermanentId> {
        self.state
            .permanents
            .iter()
            .enumerate()
            .filter_map(|(slot, permanent)| permanent.as_ref().map(|_| PermanentId(slot)))
            .find(|permanent| {
                self.permanent_event_ref(*permanent)
                    .is_some_and(|event| ObjectRenderId::from(event) == id)
            })
    }

    fn planned_combat_answer(
        &self,
        role: CombatRole,
        creature: PermanentId,
        choice: &DecisionChoice,
    ) -> Answer {
        let id = self.permanent_render_id(creature);
        match (role, choice) {
            (CombatRole::Attackers, DecisionChoice::DeclareAttackers { attackers }) => {
                Answer::Attack(attackers.iter().find(|a| a.creature == id).map(|a| {
                    match a.defender {
                        Some(object) => AttackTarget::Permanent(
                            self.permanent_object(self.permanent_by_render_id(object).unwrap())
                                .unwrap(),
                        ),
                        None => AttackTarget::Player(self.non_active_player()),
                    }
                }))
            }
            (CombatRole::Blockers, DecisionChoice::DeclareBlockers { blocks }) => Answer::Block(
                blocks
                    .iter()
                    .find(|block| block.blocker == id)
                    .and_then(|block| self.permanent_by_render_id(block.attacker)),
            ),
            _ => unreachable!("matches_choice already verified variant"),
        }
    }

    fn combat_refusal(
        &self,
        creature: PermanentId,
        wanted: Answer,
        space: &ActionSpace,
    ) -> DecisionRefusal {
        let label = self.permanent_label(creature);
        let offered: Vec<_> = space
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::DeclareBlocker {
                    attacker: Some(attacker),
                    ..
                } => Some(*attacker),
                _ => None,
            })
            .collect();
        let reason = match wanted {
            Answer::Block(None) => match offered.as_slice() {
                [attacker] => format!("{label} must block {}.", self.permanent_label(*attacker)),
                _ => format!("{label} must block an attacker that requires it."),
            },
            Answer::Block(Some(attacker)) => {
                format!("{label} can't block {}.", self.permanent_label(attacker))
            }
            Answer::Attack(None) => format!("{label} must attack."),
            Answer::Attack(Some(AttackTarget::Permanent(object))) => {
                format!(
                    "{label} can't attack {}.",
                    self.observed_object_name(object)
                )
            }
            Answer::Attack(Some(AttackTarget::Player(_))) => {
                let can_attack_other = space.actions.iter().any(|action| {
                    matches!(
                        action,
                        Action::DeclareAttacker {
                            attack: Some(_),
                            ..
                        }
                    )
                });
                if can_attack_other {
                    format!("{label} can't attack the defending player.")
                } else {
                    format!("{label} can't attack now.")
                }
            }
        };
        DecisionRefusal {
            item: Some(self.permanent_render_id(creature)),
            label,
            reason,
        }
    }

    /// Read the completed declaration from the clone's own declaration event,
    /// before any later turn could declare again.
    fn read_combat_outcome(
        &self,
        role: CombatRole,
        position: &Game,
        from: usize,
    ) -> DecisionOutcome {
        for event in &position.state.observation_events[from..] {
            match event {
                GameEvent::TurnStarted { .. } => break,
                GameEvent::CombatAttackersDeclared {
                    attackers,
                    permanent_defenders,
                    ..
                } if role == CombatRole::Attackers => {
                    return DecisionOutcome::DeclareAttackers {
                        attackers: attackers
                            .iter()
                            .map(|event| PlannedAttack {
                                creature: (*event).into(),
                                defender: permanent_defenders
                                    .as_deref()
                                    .into_iter()
                                    .flatten()
                                    .find(|(a, _)| a == event)
                                    .map(|(_, d)| (*d).into()),
                            })
                            .collect(),
                    };
                }
                GameEvent::BlockersDeclared { assignments } if role == CombatRole::Blockers => {
                    return DecisionOutcome::DeclareBlockers {
                        blocks: assignments
                            .iter()
                            .flat_map(|(attacker, blockers)| {
                                blockers.iter().map(move |blocker| PlannedBlock {
                                    blocker: (*blocker).into(),
                                    attacker: (*attacker).into(),
                                })
                            })
                            .collect(),
                    };
                }
                _ => {}
            }
        }
        match role {
            CombatRole::Attackers => DecisionOutcome::DeclareAttackers {
                attackers: Vec::new(),
            },
            CombatRole::Blockers => DecisionOutcome::DeclareBlockers { blocks: Vec::new() },
        }
    }

    /// The engine finished the declaration but did not declare the plan:
    /// the engine answered a forced declaration for a skipped creature.
    fn combat_undeclared(
        &self,
        role: CombatRole,
        choice: &DecisionChoice,
        outcome: &DecisionOutcome,
    ) -> Option<DecisionRefusal> {
        let named = |id: ObjectRenderId| self.permanent_render_label(id);
        match (role, choice, outcome) {
            (
                CombatRole::Attackers,
                DecisionChoice::DeclareAttackers { attackers },
                DecisionOutcome::DeclareAttackers {
                    attackers: declared_attackers,
                },
            ) => {
                let mut planned = attackers.clone();
                planned.extend(
                    self.declared_attackers()
                        .into_iter()
                        .filter(|id| !attackers.contains(id)),
                );
                let mut declared = declared_attackers.clone();
                planned.sort_by_key(|a| (a.creature.entity, a.creature.incarnation));
                declared.sort_by_key(|a| (a.creature.entity, a.creature.incarnation));
                (planned != declared).then(|| {
                    let creature = planned
                        .iter()
                        .find(|id| !declared_attackers.contains(id))
                        .or_else(|| declared_attackers.iter().find(|id| !planned.contains(id)))
                        .copied()
                        .expect("differing attacker sets");
                    let label = named(creature.creature);
                    let must_attack = self
                        .permanent_by_render_id(creature.creature)
                        .is_some_and(|attacker| self.must_attack(attacker));
                    let reason = if must_attack && !planned.contains(&creature) {
                        format!("{label} must attack.")
                    } else {
                        format!("The engine would not declare {label} as planned.")
                    };
                    DecisionRefusal {
                        item: Some(creature.creature),
                        reason,
                        label,
                    }
                })
            }
            (
                CombatRole::Blockers,
                DecisionChoice::DeclareBlockers { blocks },
                DecisionOutcome::DeclareBlockers {
                    blocks: declared_blocks,
                },
            ) => {
                let mut planned = blocks.clone();
                planned.extend(
                    self.declared_blocks()
                        .into_iter()
                        .filter(|block| !blocks.contains(block)),
                );
                let missing = planned
                    .iter()
                    .find(|block| !declared_blocks.contains(block));
                if let Some(block) = missing {
                    let label = named(block.blocker);
                    let attacker = named(block.attacker);
                    return Some(DecisionRefusal {
                        item: Some(block.blocker),
                        reason: format!("{label} can't block {attacker} in this declaration."),
                        label,
                    });
                }
                declared_blocks
                    .iter()
                    .find(|block| !planned.contains(block))
                    .map(|block| {
                        let label = named(block.blocker);
                        DecisionRefusal {
                            item: Some(block.blocker),
                            reason: format!("{label} must block {}.", named(block.attacker)),
                            label,
                        }
                    })
            }
            _ => unreachable!("matching combat role, choice, and outcome"),
        }
    }
}
