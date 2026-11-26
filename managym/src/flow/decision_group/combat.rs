//! Combat declaration kinds (`DeclareAttackers`, `DeclareBlockers`) under the
//! shared grouped-input driver.
use std::collections::HashSet;

use crate::agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::combat::AttackTarget;
use crate::flow::decision_group::{
    AttackerItem, BlockerItem, ChoiceCheck, DecisionChoice, DecisionGroupKind, DecisionOutcome,
    DecisionRefusal, DecisionSpec, NextStep, PlannedAttack, PlannedBlock,
};
use crate::flow::event::GameEvent;
use crate::flow::game::Game;
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

pub(crate) enum CombatProgress {
    Attackers {
        rules_player: PlayerId,
        normalizer: AttackPlanNormalizer,
    },
    Blockers {
        rules_player: PlayerId,
        normalizer: BlockPlanNormalizer,
    },
}

impl CombatProgress {
    fn remaining(&self) -> &[PermanentId] {
        match self {
            Self::Attackers { normalizer, .. } => normalizer.remaining(),
            Self::Blockers { normalizer, .. } => normalizer.remaining(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Answer {
    Attack(Option<AttackTarget>),
    Block(Option<PermanentId>),
}

impl DecisionGroupKind for CombatGroup {
    type Progress = CombatProgress;

    fn describe(&self, game: &Game, viewer: PlayerId) -> Option<DecisionSpec> {
        let order = game.declaration_order(self.role, viewer).ok()?;
        let combat = game.state.combat.as_deref()?;
        let declined = match self.role {
            CombatRole::Attackers => combat.declined_attackers.as_slice(),
            CombatRole::Blockers => combat.declined_blockers.as_slice(),
        };
        let active_order: Vec<PermanentId> = if declined.is_empty() {
            order
        } else {
            let suppressed = game.interchangeable_with_any_permanent(declined, &order);
            order
                .into_iter()
                .zip(suppressed)
                .filter_map(|(creature, is_suppressed)| {
                    let keep = !is_suppressed
                        || (self.role == CombatRole::Attackers && game.must_attack(creature));
                    keep.then_some(creature)
                })
                .collect()
        };
        if active_order.is_empty() {
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
                    declarable: active_order
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
                declarable: active_order
                    .iter()
                    .map(|creature| {
                        let others: Vec<_> = active_order
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
        choice: &DecisionChoice,
    ) -> Result<Self::Progress, AgentError> {
        let order = game.declaration_order(self.role, viewer)?;
        let rules_player = game
            .action_space()
            .and_then(|space| space.actions.first().map(Action::player))
            .ok_or_else(|| AgentError("declaration has no legal actions".into()))?;
        match (self.role, choice) {
            (CombatRole::Attackers, DecisionChoice::DeclareAttackers { attackers }) => {
                let to_entry = |attack: PlannedAttack, force_committed: bool| {
                    let perm = game
                        .permanent_by_render_id(attack.creature)
                        .expect("verified attacker");
                    let defender = match attack.defender {
                        Some(object) => AttackTarget::Permanent(
                            game.permanent_object(
                                game.permanent_by_render_id(object)
                                    .expect("verified defender"),
                            )
                            .expect("battlefield defender"),
                        ),
                        None => AttackTarget::Player(game.non_active_player()),
                    };
                    PlannedAttackEntry {
                        attacker: perm,
                        defender,
                        raw_defender: attack.defender,
                        committed: force_committed || !order.contains(&perm),
                    }
                };
                let working_attackers = attackers
                    .iter()
                    .copied()
                    .map(|attack| to_entry(attack, false))
                    .chain(
                        game.declared_attackers()
                            .into_iter()
                            .filter(|attack| !attackers.contains(attack))
                            .map(|attack| to_entry(attack, true)),
                    )
                    .collect();
                Ok(CombatProgress::Attackers {
                    rules_player,
                    normalizer: AttackPlanNormalizer::new(order, working_attackers),
                })
            }
            (CombatRole::Blockers, DecisionChoice::DeclareBlockers { blocks }) => {
                let to_entry = |block: PlannedBlock, force_committed: bool| {
                    let blocker = game
                        .permanent_by_render_id(block.blocker)
                        .expect("verified blocker");
                    PlannedBlockEntry {
                        blocker,
                        attacker: game.permanent_by_render_id(block.attacker),
                        raw_attacker: Some(block.attacker),
                        committed: force_committed || !order.contains(&blocker),
                    }
                };
                let working_blocks = blocks
                    .iter()
                    .copied()
                    .map(|block| to_entry(block, false))
                    .chain(
                        game.declared_blocks()
                            .into_iter()
                            .filter(|block| !blocks.contains(block))
                            .map(|block| to_entry(block, true)),
                    )
                    .collect();
                Ok(CombatProgress::Blockers {
                    rules_player,
                    normalizer: BlockPlanNormalizer::new(order, working_blocks),
                })
            }
            _ => unreachable!("matches_choice already verified variant"),
        }
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
            .is_some_and(|asked| progress.remaining().contains(&asked))
    }

    fn next_step(
        &self,
        live: &Game,
        position: &Game,
        viewer: PlayerId,
        _choice: &DecisionChoice,
        progress: &mut Self::Progress,
    ) -> NextStep {
        if position
            .action_space()
            .is_none_or(|space| space.player != Some(viewer))
        {
            return NextStep::Done;
        }
        match progress {
            CombatProgress::Attackers {
                rules_player,
                normalizer,
            } => {
                let Some(step) = normalizer.next_step(position, *rules_player) else {
                    return NextStep::Done;
                };
                let space = position.action_space().expect("asked decision");
                match step.action_index {
                    Some(action_index) => NextStep::Step {
                        action_index,
                        item: Some(live.permanent_render_id(step.permanent)),
                    },
                    None => NextStep::Refuse(live.combat_refusal(
                        step.permanent,
                        Answer::Attack(step.attack),
                        space,
                    )),
                }
            }
            CombatProgress::Blockers {
                rules_player,
                normalizer,
            } => {
                let Some(step) = normalizer.next_step(position, *rules_player) else {
                    return NextStep::Done;
                };
                let space = position.action_space().expect("asked decision");
                match step.action_index {
                    Some(action_index) => NextStep::Step {
                        action_index,
                        item: Some(live.permanent_render_id(step.blocker)),
                    },
                    None => NextStep::Refuse(live.combat_refusal(
                        step.blocker,
                        Answer::Block(step.wanted_attacker),
                        space,
                    )),
                }
            }
        }
    }

    fn finish(
        &self,
        live: &Game,
        position: &Game,
        obs_events_from: usize,
        _spec: &DecisionSpec,
        _choice: &DecisionChoice,
        progress: &Self::Progress,
    ) -> (DecisionOutcome, Option<DecisionRefusal>) {
        let outcome = live.read_combat_outcome(self.role, position, obs_events_from);
        let refused = match (progress, &outcome) {
            (
                CombatProgress::Attackers { normalizer, .. },
                DecisionOutcome::DeclareAttackers {
                    attackers: declared_attackers,
                },
            ) => live.undeclared_attackers(normalizer.planned(), declared_attackers),
            (
                CombatProgress::Blockers { normalizer, .. },
                DecisionOutcome::DeclareBlockers {
                    blocks: declared_blocks,
                },
            ) => live.undeclared_blocks(normalizer.planned(), declared_blocks),
            _ => unreachable!("matching combat progress and outcome"),
        };
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
        let rules_player = space.actions.first().map(Action::player)?;
        self.asked_creature_for_rules_player(role, rules_player)
    }

    fn asked_creature_for_rules_player(
        &self,
        role: CombatRole,
        rules_player: PlayerId,
    ) -> Option<PermanentId> {
        let space = self.action_space()?;
        let expected_kind = match role {
            CombatRole::Attackers => ActionSpaceKind::DeclareAttacker,
            CombatRole::Blockers => ActionSpaceKind::DeclareBlocker,
        };
        if space.kind != expected_kind {
            return None;
        }
        space
            .actions
            .iter()
            .find_map(|action| match (action, role) {
                (
                    Action::DeclareAttacker {
                        player, permanent, ..
                    },
                    CombatRole::Attackers,
                ) if *player == rules_player => Some(*permanent),
                (
                    Action::DeclareBlocker {
                        player, blocker, ..
                    },
                    CombatRole::Blockers,
                ) if *player == rules_player => Some(*blocker),
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
            Answer::Block(None) => {
                let coupled: Vec<_> = offered
                    .iter()
                    .copied()
                    .filter(|&attacker| self.blocks_are_coupled(attacker))
                    .collect();
                match (offered.as_slice(), coupled.as_slice()) {
                    ([attacker], _) | (_, [attacker]) => {
                        format!("{label} must block {}.", self.permanent_label(*attacker))
                    }
                    _ => format!("{label} must block an attacker that requires it."),
                }
            }
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

    /// The engine finished the attacker declaration but did not declare the
    /// normalized plan (e.g., a forced attack on a skipped creature).
    fn undeclared_attackers(
        &self,
        working_attackers: &[PlannedAttackEntry],
        declared_attackers: &[PlannedAttack],
    ) -> Option<DecisionRefusal> {
        let mut planned: Vec<PlannedAttack> = working_attackers
            .iter()
            .map(|entry| PlannedAttack {
                creature: self.permanent_render_id(entry.attacker),
                defender: entry.raw_defender,
            })
            .collect();
        let mut declared = declared_attackers.to_vec();
        planned.sort_by_key(|a| (a.creature.entity, a.creature.incarnation));
        declared.sort_by_key(|a| (a.creature.entity, a.creature.incarnation));
        (planned != declared).then(|| {
            let creature = planned
                .iter()
                .find(|id| !declared_attackers.contains(id))
                .or_else(|| declared_attackers.iter().find(|id| !planned.contains(id)))
                .copied()
                .expect("differing attacker sets");
            let label = self.permanent_render_label(creature.creature);
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

    /// The engine finished the blocker declaration but did not declare the
    /// normalized plan (a forced block on a skipped creature or unresolvable target).
    fn undeclared_blocks(
        &self,
        working_blocks: &[PlannedBlockEntry],
        declared_blocks: &[PlannedBlock],
    ) -> Option<DecisionRefusal> {
        let planned: Vec<PlannedBlock> = working_blocks
            .iter()
            .map(|block| PlannedBlock {
                blocker: self.permanent_render_id(block.blocker),
                attacker: block
                    .raw_attacker
                    .expect("combat preview blocks carry render ids"),
            })
            .collect();
        if let Some(block) = planned
            .iter()
            .find(|block| !declared_blocks.contains(block))
        {
            let label = self.permanent_render_label(block.blocker);
            let attacker = self.permanent_render_label(block.attacker);
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
                let label = self.permanent_render_label(block.blocker);
                DecisionRefusal {
                    item: Some(block.blocker),
                    reason: format!(
                        "{label} must block {}.",
                        self.permanent_render_label(block.attacker)
                    ),
                    label,
                }
            })
    }
}

/// Cursor tracking the engine's sequential declaration order past creatures
/// auto-resolved by `skip_trivial`.
struct DeclarationCursor {
    order: Vec<PermanentId>,
    cursor: usize,
    skipped: usize,
}

impl DeclarationCursor {
    fn new(order: Vec<PermanentId>) -> Self {
        Self {
            order,
            cursor: 0,
            skipped: 0,
        }
    }

    fn remaining(&self) -> &[PermanentId] {
        &self.order[self.cursor..]
    }

    fn advance(
        &mut self,
        game: &Game,
        role: CombatRole,
        player: PlayerId,
    ) -> Option<(PermanentId, &[PermanentId])> {
        let asked = game.asked_creature_for_rules_player(role, player)?;
        let rel = self.order[self.cursor..]
            .iter()
            .position(|creature| *creature == asked)?;
        self.skipped += rel;
        self.cursor += rel + 1;
        Some((asked, &self.order[self.cursor..]))
    }

    fn all_asked(&self) -> bool {
        self.skipped == 0 && self.cursor == self.order.len()
    }
}

/// One planned attack entry tracked through sequential attacker declaration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlannedAttackEntry {
    pub(crate) attacker: PermanentId,
    pub(crate) defender: AttackTarget,
    pub(crate) raw_defender: Option<ObjectRenderId>,
    pub(crate) committed: bool,
}

/// One planned block entry tracked through sequential blocker declaration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlannedBlockEntry {
    pub(crate) blocker: PermanentId,
    pub(crate) attacker: Option<PermanentId>,
    pub(crate) raw_attacker: Option<ObjectRenderId>,
    pub(crate) committed: bool,
}

/// Normalized answer for the currently asked creature in an attacker declaration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NormalizedAttackStep {
    pub(crate) permanent: PermanentId,
    pub(crate) attack: Option<AttackTarget>,
    pub(crate) action_index: Option<usize>,
}

/// Normalized answer for the currently asked creature in a blocker declaration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NormalizedBlockStep {
    pub(crate) blocker: PermanentId,
    pub(crate) wanted_attacker: Option<PermanentId>,
    pub(crate) action_index: Option<usize>,
}

/// Shared attack-plan normalizer for grouped combat previews and compound lowering.
///
/// Advances past creatures auto-resolved by `skip_trivial` and re-homes
/// planned marks from later interchangeable donors onto the creature being
/// asked so non-canonical subsets (like `[A:no, B:yes]`) normalize cleanly
/// under the declined-earlier symmetry rule.
pub(crate) struct AttackPlanNormalizer {
    cursor: DeclarationCursor,
    planned: Vec<PlannedAttackEntry>,
}

impl AttackPlanNormalizer {
    pub(crate) fn new(order: Vec<PermanentId>, planned: Vec<PlannedAttackEntry>) -> Self {
        Self {
            cursor: DeclarationCursor::new(order),
            planned,
        }
    }

    pub(crate) fn remaining(&self) -> &[PermanentId] {
        self.cursor.remaining()
    }

    pub(crate) fn next_step(
        &mut self,
        game: &Game,
        player: PlayerId,
    ) -> Option<NormalizedAttackStep> {
        let (asked, remaining) = self.cursor.advance(game, CombatRole::Attackers, player)?;
        let attack = if let Some(entry) = self
            .planned
            .iter_mut()
            .find(|entry| !entry.committed && entry.attacker == asked)
        {
            entry.committed = true;
            Some(entry.defender)
        } else if let Some(pos) = remaining.iter().find_map(|&other| {
            let pos = self
                .planned
                .iter()
                .position(|entry| !entry.committed && entry.attacker == other)?;
            game.interchangeable_permanents(asked, other).then_some(pos)
        }) {
            self.planned[pos].attacker = asked;
            self.planned[pos].committed = true;
            Some(self.planned[pos].defender)
        } else {
            None
        };
        let space = game.action_space().expect("asked decision");
        let wanted = Action::DeclareAttacker {
            player,
            permanent: asked,
            attack,
        };
        let action_index = space.actions.iter().position(|action| *action == wanted);
        Some(NormalizedAttackStep {
            permanent: asked,
            attack,
            action_index,
        })
    }

    pub(crate) fn all_committed(&self) -> bool {
        self.planned.iter().all(|entry| entry.committed)
    }

    pub(crate) fn all_asked(&self) -> bool {
        self.cursor.all_asked()
    }

    pub(crate) fn planned(&self) -> &[PlannedAttackEntry] {
        &self.planned
    }
}

/// Shared block-plan normalizer for grouped combat previews and compound lowering.
///
/// Advances past blockers auto-resolved by `skip_trivial`, re-homes planned
/// blocks from later interchangeable donors onto the blocker being asked,
/// resolves folded attacker aliases through `Game::resolve_action_index`, and
/// transposes `want <-> canon` across remaining uncommitted blocks so
/// split-blocks stay split and double-blocks stay paired.
pub(crate) struct BlockPlanNormalizer {
    cursor: DeclarationCursor,
    planned: Vec<PlannedBlockEntry>,
}

impl BlockPlanNormalizer {
    pub(crate) fn new(order: Vec<PermanentId>, planned: Vec<PlannedBlockEntry>) -> Self {
        Self {
            cursor: DeclarationCursor::new(order),
            planned,
        }
    }

    pub(crate) fn remaining(&self) -> &[PermanentId] {
        self.cursor.remaining()
    }

    pub(crate) fn next_step(
        &mut self,
        game: &Game,
        player: PlayerId,
    ) -> Option<NormalizedBlockStep> {
        let (asked, remaining) = self.cursor.advance(game, CombatRole::Blockers, player)?;

        let entry_pos = if let Some(pos) = self
            .planned
            .iter()
            .position(|entry| !entry.committed && entry.blocker == asked)
        {
            Some(pos)
        } else if let Some(pos) = remaining.iter().find_map(|&other| {
            let pos = self.planned.iter().position(|entry| {
                !entry.committed && entry.blocker == other && entry.attacker.is_some()
            })?;
            game.interchangeable_permanents(asked, other).then_some(pos)
        }) {
            self.planned[pos].blocker = asked;
            Some(pos)
        } else {
            None
        };

        let wanted_attacker = entry_pos.and_then(|pos| self.planned[pos].attacker);
        let space = game.action_space().expect("asked decision");
        let wanted_action = Action::DeclareBlocker {
            player,
            blocker: asked,
            attacker: wanted_attacker,
        };
        let action_index = game.resolve_action_index(space, &wanted_action);
        let resolved_attacker = action_index.and_then(|idx| match space.actions[idx] {
            Action::DeclareBlocker { attacker, .. } => attacker,
            _ => None,
        });

        if let Some(pos) = entry_pos {
            if self.planned[pos].attacker.is_some() {
                self.planned[pos].committed = true;
            }
            if let (Some(want), Some(canon)) = (wanted_attacker, resolved_attacker) {
                if canon != want {
                    let has_render = self.planned[pos].raw_attacker.is_some();
                    let canon_render = has_render.then(|| game.permanent_render_id(canon));
                    let want_render = has_render.then(|| game.permanent_render_id(want));
                    let assign = |entry: &mut PlannedBlockEntry,
                                  perm: PermanentId,
                                  render: Option<ObjectRenderId>| {
                        entry.attacker = Some(perm);
                        if render.is_some() {
                            entry.raw_attacker = render;
                        }
                    };
                    assign(&mut self.planned[pos], canon, canon_render);
                    // Transpose want <-> canon across uncommitted blocks so
                    // split-blocks stay split and double-blocks stay paired on
                    // the canonical attacker.
                    for (i, other) in self.planned.iter_mut().enumerate() {
                        if i != pos && !other.committed {
                            if other.attacker == Some(want) {
                                assign(other, canon, canon_render);
                            } else if other.attacker == Some(canon) {
                                assign(other, want, want_render);
                            }
                        }
                    }
                }
            }
        }

        Some(NormalizedBlockStep {
            blocker: asked,
            wanted_attacker,
            action_index,
        })
    }

    pub(crate) fn all_committed(&self) -> bool {
        self.planned.iter().all(|entry| entry.committed)
    }

    pub(crate) fn all_asked(&self) -> bool {
        self.cursor.all_asked()
    }

    pub(crate) fn planned(&self) -> &[PlannedBlockEntry] {
        &self.planned
    }
}
