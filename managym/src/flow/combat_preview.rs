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
    /// What the engine declared once the sequence ended (after, for
    /// example, removing a lone block of a creature with menace).
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
            .map(|creature| self.declarable(kind, *creature))
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

        let mut position = self.clone();
        let events_from = position.state.observation_events.len();
        let mut remaining = order.as_slice();
        // A creature with one legal answer is answered by the engine without
        // asking (`skip_trivial`); the declaration ends when the decision is
        // about none of the creatures still to declare.
        while let Some(asked) = position.asked_creature(kind, viewer) {
            let Some(at) = remaining.iter().position(|creature| *creature == asked) else {
                break;
            };
            remaining = &remaining[at + 1..];
            let wanted = self.planned_answer(kind, asked, plan);
            let space = position.action_space().expect("asked decision");
            let index = space
                .actions
                .iter()
                .position(|action| match (action, wanted) {
                    (Action::DeclareAttacker { attack, .. }, Answer::Attack(want)) => {
                        *attack == want
                    }
                    (Action::DeclareBlocker { attacker, .. }, Answer::Block(want)) => {
                        *attacker == want
                    }
                    _ => false,
                });
            let Some(index) = index else {
                preview.refused = Some(self.refusal(asked, wanted, space));
                return Ok(preview);
            };
            let frame = position
                .semantic_decision_frame()
                .map_err(|error| AgentError(error.to_string()))?;
            preview.steps.push(DeclarationStep {
                revision: frame.revision,
                fingerprint: frame.fingerprint,
                creature: self.render_id(asked),
                action_index: index,
                attack: matches!(wanted, Answer::Attack(true)),
                blocks: match wanted {
                    Answer::Block(Some(attacker)) => Some(self.render_id(attacker)),
                    _ => None,
                },
            });
            position.step(index)?;
            if position.is_game_over() {
                break;
            }
        }
        self.read_declaration(&position, events_from, &mut preview);
        if preview.refused.is_none() {
            preview.refused = self.undeclared(kind, plan, &preview);
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
        space
            .actions
            .iter()
            .find_map(|action| match (action, kind) {
                (Action::DeclareAttacker { permanent, .. }, DeclarationKind::Attackers) => {
                    Some(*permanent)
                }
                (Action::DeclareBlocker { blocker, .. }, DeclarationKind::Blockers) => {
                    Some(*blocker)
                }
                _ => None,
            })
    }

    fn render_id(&self, permanent: PermanentId) -> ObjectRenderId {
        let event = self
            .permanent_event_ref(permanent)
            .expect("declared creature is on the battlefield");
        ObjectRenderId {
            entity: event.entity.0,
            incarnation: event.incarnation,
        }
    }

    fn label(&self, permanent: PermanentId) -> String {
        self.permanent_object(permanent)
            .map_or_else(String::new, |object| self.observed_object_name(object))
    }

    /// The attackers `blocker` may block now, and whether it must block.
    fn block_options(&self, blocker: PermanentId) -> (Vec<PermanentId>, bool) {
        let attackers = self
            .state
            .combat
            .as_ref()
            .map(|combat| combat.attackers.clone())
            .unwrap_or_default();
        let options: Vec<_> = attackers
            .into_iter()
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

    fn declarable(&self, kind: DeclarationKind, creature: PermanentId) -> DeclarableCreature {
        let (can_block, must_block) = match kind {
            DeclarationKind::Attackers => (Vec::new(), false),
            DeclarationKind::Blockers => self.block_options(creature),
        };
        DeclarableCreature {
            creature: self.render_id(creature),
            label: self.label(creature),
            can_block: can_block.into_iter().map(|id| self.render_id(id)).collect(),
            must_block,
        }
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
            let label = self.permanent_by_render_id(id).map_or_else(
                || "That creature".to_string(),
                |permanent| self.label(permanent),
            );
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
                plan.attackers
                    .iter()
                    .find(|id| !asked(**id))
                    .and_then(|id| refuse(*id, "can't attack now."))
            }
            DeclarationKind::Blockers => {
                if let Some(id) = plan.attackers.first() {
                    return refuse(*id, "is not attacking now: this is a block.");
                }
                for (index, block) in plan.blocks.iter().enumerate() {
                    if !asked(block.blocker) {
                        return refuse(block.blocker, "can't block now.");
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

    fn planned_answer(
        &self,
        kind: DeclarationKind,
        creature: PermanentId,
        plan: &CombatPlan,
    ) -> Answer {
        let id = self.render_id(creature);
        match kind {
            DeclarationKind::Attackers => Answer::Attack(plan.attackers.contains(&id)),
            DeclarationKind::Blockers => Answer::Block(
                plan.blocks
                    .iter()
                    .find(|block| block.blocker == id)
                    .and_then(|block| self.permanent_by_render_id(block.attacker)),
            ),
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
            Answer::Block(None) => match offered.as_slice() {
                [attacker] => format!("{label} must block {}.", self.label(*attacker)),
                _ => format!("{label} must block an attacker that requires it."),
            },
            Answer::Block(Some(attacker)) => {
                format!("{label} can't block {}.", self.label(attacker))
            }
            Answer::Attack(_) => format!("{label} has no such attack answer."),
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
                        .map(|event| ObjectRenderId {
                            entity: event.entity.0,
                            incarnation: event.incarnation,
                        })
                        .collect();
                    return;
                }
                GameEvent::BlockersDeclared { assignments }
                    if preview.kind == DeclarationKind::Blockers =>
                {
                    let id = |event: &crate::flow::event::ObjectEventRef| ObjectRenderId {
                        entity: event.entity.0,
                        incarnation: event.incarnation,
                    };
                    preview.declared_blocks = assignments
                        .iter()
                        .flat_map(|(attacker, blockers)| {
                            blockers.iter().map(move |blocker| PlannedBlock {
                                blocker: id(blocker),
                                attacker: id(attacker),
                            })
                        })
                        .collect();
                    return;
                }
                _ => {}
            }
        }
    }

    /// The engine finished the declaration but did not declare the plan:
    /// a requirement answered a skipped creature, or a restriction such as
    /// menace (CR 702.110b) removed a block once every blocker was declared.
    fn undeclared(
        &self,
        kind: DeclarationKind,
        plan: &CombatPlan,
        preview: &CombatPreview,
    ) -> Option<DeclarationRefusal> {
        let named = |id: ObjectRenderId| {
            self.permanent_by_render_id(id).map_or_else(
                || "That creature".to_string(),
                |permanent| self.label(permanent),
            )
        };
        match kind {
            DeclarationKind::Attackers => {
                let mut planned = plan.attackers.clone();
                let mut declared = preview.declared_attackers.clone();
                planned.sort_by_key(|id| (id.entity, id.incarnation));
                declared.sort_by_key(|id| (id.entity, id.incarnation));
                (planned != declared).then(|| {
                    let creature = plan
                        .attackers
                        .iter()
                        .find(|id| !preview.declared_attackers.contains(id))
                        .or_else(|| {
                            preview
                                .declared_attackers
                                .iter()
                                .find(|id| !plan.attackers.contains(id))
                        })
                        .copied()
                        .expect("differing attacker sets");
                    let label = named(creature);
                    DeclarationRefusal {
                        creature,
                        reason: format!("The engine would not declare {label} as planned."),
                        label,
                    }
                })
            }
            DeclarationKind::Blockers => {
                let missing = plan
                    .blocks
                    .iter()
                    .find(|block| !preview.declared_blocks.contains(block));
                if let Some(block) = missing {
                    let label = named(block.blocker);
                    let attacker = named(block.attacker);
                    return Some(DeclarationRefusal {
                        creature: block.blocker,
                        reason: format!(
                            "{attacker} can't be blocked that way (menace or a blocking limit), so the engine removes {label}'s block."
                        ),
                        label,
                    });
                }
                preview
                    .declared_blocks
                    .iter()
                    .find(|block| !plan.blocks.contains(block))
                    .map(|block| {
                        let label = named(block.blocker);
                        DeclarationRefusal {
                            creature: block.blocker,
                            reason: format!("{label} must block {}.", named(block.attacker)),
                            label,
                        }
                    })
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Answer {
    Attack(bool),
    Block(Option<PermanentId>),
}
