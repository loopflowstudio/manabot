//! Combat declaration kinds (`DeclareAttackers`, `DeclareBlockers`) under the
//! shared grouped-input driver.
use std::collections::HashSet;

use crate::agent::action::{Action, ActionSpace, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::event::GameEvent;
use crate::flow::game::Game;
use crate::flow::group::{
    AttackerItem, BlockerItem, GroupKind, GroupOutcome, GroupPlan, GroupRefusal, GroupSpec,
    NextStep, PlanCheck, PlannedBlock,
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
    Attack(bool),
    Block(Option<PermanentId>),
}

impl GroupKind for CombatGroup {
    type Progress = CombatProgress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<GroupSpec> {
        let order = game.declaration_order(self.role, viewer).ok()?;
        if order.is_empty() {
            return None;
        }
        match self.role {
            CombatRole::Attackers => Some(GroupSpec::DeclareAttackers {
                declarable: order
                    .iter()
                    .map(|creature| AttackerItem {
                        item: game.permanent_render_id(*creature),
                        label: game.permanent_label(*creature),
                        must_attack: game.must_attack(*creature),
                    })
                    .collect(),
                committed: game.declared_attackers(),
            }),
            CombatRole::Blockers => Some(GroupSpec::DeclareBlockers {
                declarable: order
                    .iter()
                    .map(|creature| {
                        let (can_block, must_block) = game.block_options(*creature);
                        BlockerItem {
                            item: game.permanent_render_id(*creature),
                            label: game.permanent_label(*creature),
                            can_block: can_block
                                .into_iter()
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

    fn matches_plan(&self, plan: &GroupPlan) -> bool {
        matches!(
            (self.role, plan),
            (CombatRole::Attackers, GroupPlan::DeclareAttackers { .. })
                | (CombatRole::Blockers, GroupPlan::DeclareBlockers { .. })
        )
    }

    fn validate(&self, game: &Game, spec: &GroupSpec, plan: &GroupPlan) -> PlanCheck {
        let refuse = |id: ObjectRenderId, reason: &str| {
            let label = game.permanent_render_label(id);
            PlanCheck::Refused(GroupRefusal {
                item: Some(id),
                reason: format!("{label} {reason}"),
                label,
            })
        };
        match (spec, plan) {
            (
                GroupSpec::DeclareAttackers {
                    declarable,
                    committed,
                },
                GroupPlan::DeclareAttackers { attackers },
            ) => {
                let asked: HashSet<_> = declarable.iter().map(|c| c.item).collect();
                let mut seen = HashSet::new();
                for &id in attackers {
                    if !asked.contains(&id) && !committed.contains(&id) {
                        return refuse(id, "can't attack now.");
                    }
                    if !seen.insert(id) {
                        return refuse(id, "can only attack once.");
                    }
                }
                PlanCheck::Ready
            }
            (
                GroupSpec::DeclareBlockers {
                    declarable,
                    committed,
                },
                GroupPlan::DeclareBlockers { blocks },
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
                PlanCheck::Ready
            }
            _ => unreachable!("matches_plan already verified variant"),
        }
    }

    fn start(
        &self,
        game: &Game,
        viewer: PlayerId,
        _spec: &GroupSpec,
        _plan: &GroupPlan,
    ) -> Result<Self::Progress, AgentError> {
        let remaining = game.declaration_order(self.role, viewer)?;
        Ok(CombatProgress {
            remaining,
            offset: 0,
        })
    }

    fn continues(&self, position: &Game, viewer: PlayerId, progress: &Self::Progress) -> bool {
        position
            .asked_creature(self.role, viewer)
            .is_some_and(|asked| progress.remaining[progress.offset..].contains(&asked))
    }

    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        viewer: PlayerId,
        plan: &GroupPlan,
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
        let wanted = live.planned_combat_answer(self.role, asked, plan);
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
        plan: &GroupPlan,
        _progress: &Self::Progress,
    ) -> (GroupOutcome, Option<GroupRefusal>) {
        let outcome = live.read_combat_outcome(self.role, position, obs_events_from);
        let refused = live.combat_undeclared(self.role, plan, &outcome);
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

    /// The attackers `blocker` may block now, and whether it must block.
    fn block_options(&self, blocker: PermanentId) -> (Vec<PermanentId>, bool) {
        let Some(combat) = self.state.combat.as_ref() else {
            return (Vec::new(), false);
        };
        let options: Vec<_> = combat
            .attackers
            .iter()
            .copied()
            .filter(|attacker| self.blocker_can_block_attacker(blocker, *attacker))
            .collect();
        let required: Vec<_> = options
            .iter()
            .copied()
            .filter(|attacker| self.must_be_blocked_by_all_able(*attacker))
            .collect();
        if required.is_empty() {
            (options, false)
        } else {
            (required, true)
        }
    }

    /// Attackers already declared in this combat, before the current
    /// decision (a creature that must attack and was offered no choice).
    fn declared_attackers(&self) -> Vec<ObjectRenderId> {
        self.state
            .combat
            .iter()
            .flat_map(|combat| &combat.attackers)
            .map(|attacker| self.permanent_render_id(*attacker))
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
        plan: &GroupPlan,
    ) -> Answer {
        let id = self.permanent_render_id(creature);
        match (role, plan) {
            (CombatRole::Attackers, GroupPlan::DeclareAttackers { attackers }) => {
                Answer::Attack(attackers.contains(&id))
            }
            (CombatRole::Blockers, GroupPlan::DeclareBlockers { blocks }) => Answer::Block(
                blocks
                    .iter()
                    .find(|block| block.blocker == id)
                    .and_then(|block| self.permanent_by_render_id(block.attacker)),
            ),
            _ => unreachable!("matches_plan already verified variant"),
        }
    }

    fn combat_refusal(
        &self,
        creature: PermanentId,
        wanted: Answer,
        space: &ActionSpace,
    ) -> GroupRefusal {
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
            Answer::Attack(false) => format!("{label} must attack."),
            Answer::Attack(true) => format!("{label} can't attack now."),
        };
        GroupRefusal {
            item: Some(self.permanent_render_id(creature)),
            label,
            reason,
        }
    }

    /// Read the completed declaration from the clone's own declaration event,
    /// before any later turn could declare again.
    fn read_combat_outcome(&self, role: CombatRole, position: &Game, from: usize) -> GroupOutcome {
        for event in &position.state.observation_events[from..] {
            match event {
                GameEvent::TurnStarted { .. } => break,
                GameEvent::CombatAttackersDeclared { attackers, .. }
                    if role == CombatRole::Attackers =>
                {
                    return GroupOutcome::DeclareAttackers {
                        attackers: attackers
                            .iter()
                            .map(|event| ObjectRenderId::from(*event))
                            .collect(),
                    };
                }
                GameEvent::BlockersDeclared { assignments } if role == CombatRole::Blockers => {
                    return GroupOutcome::DeclareBlockers {
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
            CombatRole::Attackers => GroupOutcome::DeclareAttackers {
                attackers: Vec::new(),
            },
            CombatRole::Blockers => GroupOutcome::DeclareBlockers { blocks: Vec::new() },
        }
    }

    /// The engine finished the declaration but did not declare the plan:
    /// a requirement answered a skipped creature, or a restriction such as
    /// menace (CR 702.110b) removed a block once every blocker was declared.
    fn combat_undeclared(
        &self,
        role: CombatRole,
        plan: &GroupPlan,
        outcome: &GroupOutcome,
    ) -> Option<GroupRefusal> {
        let named = |id: ObjectRenderId| self.permanent_render_label(id);
        match (role, plan, outcome) {
            (
                CombatRole::Attackers,
                GroupPlan::DeclareAttackers { attackers },
                GroupOutcome::DeclareAttackers {
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
                planned.sort_by_key(|id| (id.entity, id.incarnation));
                declared.sort_by_key(|id| (id.entity, id.incarnation));
                (planned != declared).then(|| {
                    let creature = planned
                        .iter()
                        .find(|id| !declared_attackers.contains(id))
                        .or_else(|| declared_attackers.iter().find(|id| !planned.contains(id)))
                        .copied()
                        .expect("differing attacker sets");
                    let label = named(creature);
                    let must_attack = self
                        .permanent_by_render_id(creature)
                        .is_some_and(|attacker| self.must_attack(attacker));
                    let reason = if must_attack && !planned.contains(&creature) {
                        format!("{label} must attack.")
                    } else {
                        format!("The engine would not declare {label} as planned.")
                    };
                    GroupRefusal {
                        item: Some(creature),
                        reason,
                        label,
                    }
                })
            }
            (
                CombatRole::Blockers,
                GroupPlan::DeclareBlockers { blocks },
                GroupOutcome::DeclareBlockers {
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
                    return Some(GroupRefusal {
                        item: Some(block.blocker),
                        reason: format!(
                            "{attacker} can't be blocked that way (menace or a blocking limit), so the engine removes {label}'s block."
                        ),
                        label,
                    });
                }
                declared_blocks
                    .iter()
                    .find(|block| !planned.contains(block))
                    .map(|block| {
                        let label = named(block.blocker);
                        GroupRefusal {
                            item: Some(block.blocker),
                            reason: format!("{label} must block {}.", named(block.attacker)),
                            label,
                        }
                    })
            }
            _ => unreachable!("matching combat role, plan, and outcome"),
        }
    }
}
