//! Read-only preview of a whole combat declaration.
//!
//! The engine declares attackers and blockers one creature at a time
//! (`flow/tick.rs`). `Game::combat_preview` takes the current declaration
//! decision and a proposed plan (a set of attackers, or blocker → attacker
//! pairs) and replays the engine's own per-creature answers on a clone. It
//! returns the exact sequence of offers that declares the plan, each at the
//! revision and fingerprint it will be asked at, or the first creature whose
//! answer the engine refuses and why. It also says which creatures can attack
//! or block from here.
//!
//! Nothing is committed: the live game is only read, and its offers, events,
//! observations, hashed state and decision progression are unchanged. A
//! caller sends the returned steps as ordinary per-creature Commands, so
//! recordings, replay and forks see ordinary decisions.
use serde::{Deserialize, Serialize};

use crate::agent::action::{Action, ActionSpaceKind, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::event::GameEvent;
use crate::flow::game::Game;
use crate::state::game_object::{PermanentId, PlayerId};

/// One proposed block: `blocker` blocks `attacker`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlannedBlock {
    pub blocker: ObjectRenderId,
    pub attacker: ObjectRenderId,
}

/// The whole declaration a person drafts. An attack plan lists attackers;
/// a block plan lists blocks. A creature left out declines.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CombatPlan {
    #[serde(default)]
    pub attackers: Vec<ObjectRenderId>,
    #[serde(default)]
    pub blocks: Vec<PlannedBlock>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclarationKind {
    Attackers,
    Blockers,
}

/// A creature the declaration will ask about, in the engine's asking order.
#[derive(Clone, Debug, Serialize)]
pub struct DeclarableCreature {
    pub creature: ObjectRenderId,
    pub label: String,
    /// Blockers only: the attackers it may block from here.
    pub can_block: Vec<ObjectRenderId>,
    /// Blockers only: a requirement (CR 509.1c) removes the option to not block.
    pub must_block: bool,
    /// Attackers only: a requirement (CR 508.1d) removes the option to stay home.
    pub must_attack: bool,
}

/// One per-creature answer the engine will ask for.
#[derive(Clone, Debug, Serialize)]
pub struct DeclarationStep {
    /// The decision this answer is given at.
    pub revision: u64,
    pub fingerprint: String,
    pub creature: ObjectRenderId,
    /// The legal action to choose at that decision.
    pub action_index: usize,
    /// Attackers: whether it attacks. Blockers: the attacker it blocks.
    pub attack: bool,
    pub blocks: Option<ObjectRenderId>,
}

/// The first creature whose planned answer the engine does not give.
#[derive(Clone, Debug, Serialize)]
pub struct DeclarationRefusal {
    pub creature: ObjectRenderId,
    pub label: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct CombatPreview {
    pub kind: DeclarationKind,
    pub declarable: Vec<DeclarableCreature>,
    /// The Commands that declare the plan, in order. Complete only when
    /// `refused` is absent; otherwise the steps before the refusal.
    pub steps: Vec<DeclarationStep>,
    pub refused: Option<DeclarationRefusal>,
    /// What the engine declared once the sequence ended.
    pub declared_attackers: Vec<ObjectRenderId>,
    pub declared_blocks: Vec<PlannedBlock>,
}

impl Game {
    /// Preview `plan` at the current declaration, bound to its exact semantic
    /// address. Fails on a stale or non-actor address or when the decision is
    /// not a declaration; a plan the engine would not declare is a
    /// `refused` result, not an error.
    pub fn combat_preview(
        &self,
        viewer: PlayerId,
        revision: u64,
        fingerprint: &str,
        plan: &CombatPlan,
    ) -> Result<CombatPreview, AgentError> {
        let frame = self
            .semantic_decision_frame()
            .map_err(|error| AgentError(error.to_string()))?;
        if frame.actor as usize != viewer.0
            || frame.revision != revision
            || frame.fingerprint != fingerprint
        {
            return Err(AgentError("stale or unauthorized combat preview".into()));
        }
        let space = self
            .action_space()
            .ok_or_else(|| AgentError("no decision".into()))?;
        let kind = match space.kind {
            ActionSpaceKind::DeclareAttacker => DeclarationKind::Attackers,
            ActionSpaceKind::DeclareBlocker => DeclarationKind::Blockers,
            _ => return Err(AgentError("not a combat declaration".into())),
        };
        let order = self.declaration_order(kind, viewer)?;
        let declarable = order
            .iter()
            .map(|creature| self.declarable(kind, *creature, &order))
            .collect::<Vec<_>>();
        let mut preview = CombatPreview {
            kind,
            declarable,
            steps: Vec::new(),
            refused: None,
            declared_attackers: Vec::new(),
            declared_blocks: Vec::new(),
        };
        if let Some(refusal) = self.unaskable(kind, &order, plan) {
            preview.refused = Some(refusal);
            return Ok(preview);
        }

        let rules_player = space
            .actions
            .first()
            .map(Action::player)
            .ok_or_else(|| AgentError("declaration has no legal actions".into()))?;
        let mut position = self.clone();
        let events_from = position.state.observation_events.len();
        // A creature with one legal answer is answered by the engine without
        // asking (`skip_trivial`); the declaration ends when the decision is
        // about none of the creatures still to declare.
        match kind {
            DeclarationKind::Attackers => {
                let mut working_attackers: Vec<(PermanentId, bool)> = Vec::new();
                for id in &plan.attackers {
                    let perm = self
                        .permanent_by_render_id(*id)
                        .expect("verified by unaskable");
                    working_attackers.push((perm, !order.contains(&perm)));
                }
                for id in self.declared_attackers() {
                    if !plan.attackers.contains(&id) {
                        let perm = self.permanent_by_render_id(id).expect("declared attacker");
                        working_attackers.push((perm, true));
                    }
                }
                let mut normalizer = AttackPlanNormalizer::new(order, working_attackers);
                while position
                    .action_space()
                    .is_some_and(|space| space.player == Some(viewer))
                {
                    let Some(step) = normalizer.next_step(&position, rules_player) else {
                        break;
                    };
                    let space = position.action_space().expect("asked decision");
                    let Some(index) = step.action_index else {
                        preview.refused =
                            Some(self.refusal(step.permanent, Answer::Attack(step.attack), space));
                        return Ok(preview);
                    };
                    let frame = position
                        .semantic_decision_frame()
                        .map_err(|error| AgentError(error.to_string()))?;
                    preview.steps.push(DeclarationStep {
                        revision: frame.revision,
                        fingerprint: frame.fingerprint,
                        creature: self.render_id(step.permanent),
                        action_index: index,
                        attack: step.attack,
                        blocks: None,
                    });
                    position.step(index)?;
                    if position.is_game_over() {
                        break;
                    }
                }
                self.read_declaration(&position, events_from, &mut preview);
                if preview.refused.is_none() {
                    preview.refused = self.undeclared_attackers(normalizer.planned(), &preview);
                }
            }
            DeclarationKind::Blockers => {
                let mut working_blocks: Vec<PlannedBlockEntry> = Vec::new();
                for block in &plan.blocks {
                    let blocker = self
                        .permanent_by_render_id(block.blocker)
                        .expect("verified by unaskable");
                    working_blocks.push(PlannedBlockEntry {
                        blocker,
                        attacker: self.permanent_by_render_id(block.attacker),
                        raw_attacker: Some(block.attacker),
                        committed: !order.contains(&blocker),
                    });
                }
                for block in self.declared_blocks() {
                    if !plan.blocks.contains(&block) {
                        let blocker = self
                            .permanent_by_render_id(block.blocker)
                            .expect("declared blocker");
                        working_blocks.push(PlannedBlockEntry {
                            blocker,
                            attacker: self.permanent_by_render_id(block.attacker),
                            raw_attacker: Some(block.attacker),
                            committed: true,
                        });
                    }
                }
                let mut normalizer = BlockPlanNormalizer::new(order, working_blocks);
                while position
                    .action_space()
                    .is_some_and(|space| space.player == Some(viewer))
                {
                    let Some(step) = normalizer.next_step(&position, rules_player) else {
                        break;
                    };
                    let space = position.action_space().expect("asked decision");
                    let Some(index) = step.action_index else {
                        preview.refused = Some(self.refusal(
                            step.blocker,
                            Answer::Block(step.wanted_attacker),
                            space,
                        ));
                        return Ok(preview);
                    };
                    let frame = position
                        .semantic_decision_frame()
                        .map_err(|error| AgentError(error.to_string()))?;
                    preview.steps.push(DeclarationStep {
                        revision: frame.revision,
                        fingerprint: frame.fingerprint,
                        creature: self.render_id(step.blocker),
                        action_index: index,
                        attack: false,
                        blocks: step
                            .resolved_attacker
                            .map(|attacker| self.render_id(attacker)),
                    });
                    position.step(index)?;
                    if position.is_game_over() {
                        break;
                    }
                }
                self.read_declaration(&position, events_from, &mut preview);
                if preview.refused.is_none() {
                    preview.refused = self.undeclared_blocks(normalizer.planned(), &preview);
                }
            }
        }
        Ok(preview)
    }

    /// The creature asked now, then those the engine will ask after it.
    fn declaration_order(
        &self,
        kind: DeclarationKind,
        viewer: PlayerId,
    ) -> Result<Vec<PermanentId>, AgentError> {
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or_else(|| AgentError("declaration without combat".into()))?;
        let current = self
            .asked_creature(kind, viewer)
            .ok_or_else(|| AgentError("declaration names no creature".into()))?;
        let queue = match kind {
            DeclarationKind::Attackers => &combat.attackers_to_declare,
            DeclarationKind::Blockers => &combat.blockers_to_declare,
        };
        Ok(std::iter::once(current)
            .chain(queue.iter().rev().copied())
            .collect())
    }

    fn asked_creature(&self, kind: DeclarationKind, viewer: PlayerId) -> Option<PermanentId> {
        let space = self.action_space()?;
        if space.player != Some(viewer) {
            return None;
        }
        let rules_player = space.actions.first().map(Action::player)?;
        self.asked_creature_for_rules_player(kind, rules_player)
    }

    pub(crate) fn asked_creature_for_rules_player(
        &self,
        kind: DeclarationKind,
        rules_player: PlayerId,
    ) -> Option<PermanentId> {
        let space = self.action_space()?;
        let expected_kind = match kind {
            DeclarationKind::Attackers => ActionSpaceKind::DeclareAttacker,
            DeclarationKind::Blockers => ActionSpaceKind::DeclareBlocker,
        };
        if space.kind != expected_kind {
            return None;
        }
        space
            .actions
            .iter()
            .find_map(|action| match (action, kind) {
                (
                    Action::DeclareAttacker {
                        player, permanent, ..
                    },
                    DeclarationKind::Attackers,
                ) if *player == rules_player => Some(*permanent),
                (
                    Action::DeclareBlocker {
                        player, blocker, ..
                    },
                    DeclarationKind::Blockers,
                ) if *player == rules_player => Some(*blocker),
                _ => None,
            })
    }

    fn render_id(&self, permanent: PermanentId) -> ObjectRenderId {
        self.permanent_event_ref(permanent)
            .expect("declared creature is on the battlefield")
            .into()
    }

    fn label(&self, permanent: PermanentId) -> String {
        self.permanent_object(permanent)
            .map_or_else(String::new, |object| self.observed_object_name(object))
    }

    fn declarable(
        &self,
        kind: DeclarationKind,
        creature: PermanentId,
        order: &[PermanentId],
    ) -> DeclarableCreature {
        let (can_block, must_block) = match kind {
            DeclarationKind::Attackers => (Vec::new(), false),
            DeclarationKind::Blockers => {
                let others: Vec<_> = order.iter().copied().filter(|id| *id != creature).collect();
                let options = self.block_options(creature, &others);
                let must_block = !options.contains(&None);
                (options.into_iter().flatten().collect(), must_block)
            }
        };
        DeclarableCreature {
            creature: self.render_id(creature),
            label: self.label(creature),
            can_block: can_block.into_iter().map(|id| self.render_id(id)).collect(),
            must_block,
            must_attack: kind == DeclarationKind::Attackers && self.must_attack(creature),
        }
    }

    /// Attackers already declared in this combat, before the current
    /// decision (a creature that must attack and was offered no choice).
    fn declared_attackers(&self) -> Vec<ObjectRenderId> {
        self.state
            .combat
            .iter()
            .flat_map(|combat| &combat.attackers)
            .map(|attacker| self.render_id(*attacker))
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
                    blocker: self.render_id(*blocker),
                    attacker: self.render_id(*attacker),
                })
            })
            .collect()
    }

    /// The permanent's label, or "That creature" once it is gone.
    fn render_label(&self, id: ObjectRenderId) -> String {
        self.permanent_by_render_id(id).map_or_else(
            || "That creature".to_string(),
            |permanent| self.label(permanent),
        )
    }

    fn permanent_by_render_id(&self, id: ObjectRenderId) -> Option<PermanentId> {
        self.state
            .permanents
            .iter()
            .enumerate()
            .filter_map(|(slot, permanent)| permanent.as_ref().map(|_| PermanentId(slot)))
            .find(|permanent| {
                self.permanent_event_ref(*permanent).is_some_and(|event| {
                    event.entity.0 == id.entity && event.incarnation == id.incarnation
                })
            })
    }

    /// A planned creature the declaration never asks about, or a blocker
    /// planned twice, refuses the plan before anything is replayed.
    fn unaskable(
        &self,
        kind: DeclarationKind,
        order: &[PermanentId],
        plan: &CombatPlan,
    ) -> Option<DeclarationRefusal> {
        let refuse = |id: ObjectRenderId, reason: &str| {
            let label = self.render_label(id);
            Some(DeclarationRefusal {
                creature: id,
                reason: format!("{label} {reason}"),
                label,
            })
        };
        let asked = |id: ObjectRenderId| {
            self.permanent_by_render_id(id)
                .is_some_and(|permanent| order.contains(&permanent))
        };
        match kind {
            DeclarationKind::Attackers => {
                if !plan.blocks.is_empty() {
                    return refuse(
                        plan.blocks[0].blocker,
                        "is not blocking now: this is an attack.",
                    );
                }
                let declared = self.declared_attackers();
                plan.attackers
                    .iter()
                    .find(|id| !asked(**id) && !declared.contains(id))
                    .and_then(|id| refuse(*id, "can't attack now."))
            }
            DeclarationKind::Blockers => {
                if let Some(id) = plan.attackers.first() {
                    return refuse(*id, "is not attacking now: this is a block.");
                }
                let declared = self.declared_blocks();
                for (index, block) in plan.blocks.iter().enumerate() {
                    if !asked(block.blocker) && !declared.contains(block) {
                        let reason =
                            match declared.iter().find(|other| other.blocker == block.blocker) {
                                Some(other) => {
                                    format!("already blocks {}.", self.render_label(other.attacker))
                                }
                                None => "can't block now.".to_string(),
                            };
                        return refuse(block.blocker, &reason);
                    }
                    if plan.blocks[..index]
                        .iter()
                        .any(|other| other.blocker == block.blocker)
                    {
                        return refuse(block.blocker, "can block only one attacker.");
                    }
                }
                None
            }
        }
    }

    fn refusal(
        &self,
        creature: PermanentId,
        wanted: Answer,
        space: &crate::agent::action::ActionSpace,
    ) -> DeclarationRefusal {
        let label = self.label(creature);
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
                        format!("{label} must block {}.", self.label(*attacker))
                    }
                    _ => format!("{label} must block an attacker that requires it."),
                }
            }
            Answer::Block(Some(attacker)) => {
                format!("{label} can't block {}.", self.label(attacker))
            }
            Answer::Attack(false) => format!("{label} must attack."),
            Answer::Attack(true) => format!("{label} can't attack now."),
        };
        DeclarationRefusal {
            creature: self.render_id(creature),
            label,
            reason,
        }
    }

    /// Read the completed declaration from the clone's own declaration event,
    /// before any later turn could declare again.
    fn read_declaration(&self, position: &Game, from: usize, preview: &mut CombatPreview) {
        for event in &position.state.observation_events[from..] {
            match event {
                GameEvent::TurnStarted { .. } => return,
                GameEvent::CombatAttackersDeclared { attackers, .. }
                    if preview.kind == DeclarationKind::Attackers =>
                {
                    preview.declared_attackers = attackers
                        .iter()
                        .map(|event| ObjectRenderId::from(*event))
                        .collect();
                    return;
                }
                GameEvent::BlockersDeclared { assignments }
                    if preview.kind == DeclarationKind::Blockers =>
                {
                    preview.declared_blocks = assignments
                        .iter()
                        .flat_map(|(attacker, blockers)| {
                            blockers.iter().map(move |blocker| PlannedBlock {
                                blocker: (*blocker).into(),
                                attacker: (*attacker).into(),
                            })
                        })
                        .collect();
                    return;
                }
                _ => {}
            }
        }
    }

    /// The engine finished the attacker declaration but did not declare the
    /// normalized plan (e.g., a forced attack on a skipped creature).
    fn undeclared_attackers(
        &self,
        working_attackers: &[(PermanentId, bool)],
        preview: &CombatPreview,
    ) -> Option<DeclarationRefusal> {
        let mut planned: Vec<ObjectRenderId> = working_attackers
            .iter()
            .map(|(perm, _)| self.render_id(*perm))
            .collect();
        let mut declared = preview.declared_attackers.clone();
        planned.sort_by_key(|id| (id.entity, id.incarnation));
        declared.sort_by_key(|id| (id.entity, id.incarnation));
        (planned != declared).then(|| {
            let creature = planned
                .iter()
                .find(|id| !preview.declared_attackers.contains(id))
                .or_else(|| {
                    preview
                        .declared_attackers
                        .iter()
                        .find(|id| !planned.contains(id))
                })
                .copied()
                .expect("differing attacker sets");
            let label = self.render_label(creature);
            let must_attack = self
                .permanent_by_render_id(creature)
                .is_some_and(|attacker| self.must_attack(attacker));
            let reason = if must_attack && !planned.contains(&creature) {
                format!("{label} must attack.")
            } else {
                format!("The engine would not declare {label} as planned.")
            };
            DeclarationRefusal {
                creature,
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
        preview: &CombatPreview,
    ) -> Option<DeclarationRefusal> {
        let planned: Vec<PlannedBlock> = working_blocks
            .iter()
            .map(|block| PlannedBlock {
                blocker: self.render_id(block.blocker),
                attacker: block
                    .raw_attacker
                    .expect("combat preview blocks carry render ids"),
            })
            .collect();
        if let Some(block) = planned
            .iter()
            .find(|block| !preview.declared_blocks.contains(block))
        {
            let label = self.render_label(block.blocker);
            let attacker = self.render_label(block.attacker);
            return Some(DeclarationRefusal {
                creature: block.blocker,
                reason: format!("{label} can't block {attacker} in this declaration."),
                label,
            });
        }
        preview
            .declared_blocks
            .iter()
            .find(|block| !planned.contains(block))
            .map(|block| {
                let label = self.render_label(block.blocker);
                DeclarationRefusal {
                    creature: block.blocker,
                    reason: format!("{label} must block {}.", self.render_label(block.attacker)),
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

    fn advance(
        &mut self,
        game: &Game,
        kind: DeclarationKind,
        player: PlayerId,
    ) -> Option<(PermanentId, &[PermanentId])> {
        let asked = game.asked_creature_for_rules_player(kind, player)?;
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
    pub(crate) attack: bool,
    pub(crate) action_index: Option<usize>,
}

/// Normalized answer for the currently asked creature in a blocker declaration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NormalizedBlockStep {
    pub(crate) blocker: PermanentId,
    pub(crate) wanted_attacker: Option<PermanentId>,
    pub(crate) resolved_attacker: Option<PermanentId>,
    pub(crate) action_index: Option<usize>,
}

/// Shared attack-plan normalizer for `combat_preview` and compound lowering.
///
/// Advances past creatures auto-resolved by `skip_trivial` and re-homes
/// planned marks from later interchangeable donors onto the creature being
/// asked so non-canonical subsets (like `[A:no, B:yes]`) normalize cleanly
/// under the declined-earlier symmetry rule.
pub(crate) struct AttackPlanNormalizer {
    cursor: DeclarationCursor,
    planned: Vec<(PermanentId, bool)>,
}

impl AttackPlanNormalizer {
    pub(crate) fn new(order: Vec<PermanentId>, planned: Vec<(PermanentId, bool)>) -> Self {
        Self {
            cursor: DeclarationCursor::new(order),
            planned,
        }
    }

    pub(crate) fn next_step(
        &mut self,
        game: &Game,
        player: PlayerId,
    ) -> Option<NormalizedAttackStep> {
        let (asked, remaining) = self
            .cursor
            .advance(game, DeclarationKind::Attackers, player)?;
        let attack = if let Some(entry) = self
            .planned
            .iter_mut()
            .find(|(perm, committed)| !*committed && *perm == asked)
        {
            entry.1 = true;
            true
        } else if let Some(pos) = remaining.iter().find_map(|&other| {
            let pos = self
                .planned
                .iter()
                .position(|(perm, committed)| !*committed && *perm == other)?;
            game.interchangeable_permanents(asked, other).then_some(pos)
        }) {
            self.planned[pos] = (asked, true);
            true
        } else {
            false
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
        self.planned.iter().all(|(_, committed)| *committed)
    }

    pub(crate) fn all_asked(&self) -> bool {
        self.cursor.all_asked()
    }

    pub(crate) fn planned(&self) -> &[(PermanentId, bool)] {
        &self.planned
    }
}

/// Shared block-plan normalizer for `combat_preview` and compound lowering.
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

    pub(crate) fn next_step(
        &mut self,
        game: &Game,
        player: PlayerId,
    ) -> Option<NormalizedBlockStep> {
        let (asked, remaining) = self
            .cursor
            .advance(game, DeclarationKind::Blockers, player)?;

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
                    let canon_render = has_render.then(|| game.render_id(canon));
                    let want_render = has_render.then(|| game.render_id(want));
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
            resolved_attacker,
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

#[derive(Clone, Copy, Debug)]
enum Answer {
    Attack(bool),
    Block(Option<PermanentId>),
}
