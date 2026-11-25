//! Deprecated combat-preview wrapper over `flow::group` (removed in 0.3.0).
//!
//! Preserves the `0.2.6` `CombatPlan` / `CombatPreview` surface by delegating
//! to [`Game::preview_group`]. New callers should use [`Game::group`],
//! [`Game::preview_group`] and [`Game::apply_group`].
use serde::{Deserialize, Serialize};

use crate::agent::action::{ActionSpaceKind, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::game::Game;
use crate::flow::group::{GroupOutcome, GroupPlan, GroupSpec};
use crate::state::game_object::PlayerId;

pub use crate::flow::group::PlannedBlock;

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
    /// What the engine declared once the sequence ended (after, for
    /// example, removing a lone block of a creature with menace).
    pub declared_attackers: Vec<ObjectRenderId>,
    pub declared_blocks: Vec<PlannedBlock>,
}

fn declarable_from_spec(spec: &GroupSpec) -> Vec<DeclarableCreature> {
    match spec {
        GroupSpec::DeclareAttackers { declarable, .. } => declarable
            .iter()
            .map(|item| DeclarableCreature {
                creature: item.item,
                label: item.label.clone(),
                can_block: Vec::new(),
                must_block: false,
                must_attack: item.must_attack,
            })
            .collect(),
        GroupSpec::DeclareBlockers { declarable, .. } => declarable
            .iter()
            .map(|item| DeclarableCreature {
                creature: item.item,
                label: item.label.clone(),
                can_block: item.can_block.clone(),
                must_block: item.must_block,
                must_attack: false,
            })
            .collect(),
        _ => Vec::new(),
    }
}

impl Game {
    /// Preview `plan` at the current declaration, bound to its exact semantic
    /// address. Deprecated wrapper over [`Game::preview_group`] (removed in 0.3.0).
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

        // Preserve cross-field refusal messages from the flat `CombatPlan` shape.
        if kind == DeclarationKind::Attackers && !plan.blocks.is_empty() {
            let group = self
                .group(viewer)
                .ok_or_else(|| AgentError("not a combat declaration".into()))?;
            let id = plan.blocks[0].blocker;
            let label = self.permanent_render_label(id);
            return Ok(CombatPreview {
                kind,
                declarable: declarable_from_spec(&group.spec),
                steps: Vec::new(),
                refused: Some(DeclarationRefusal {
                    creature: id,
                    reason: format!("{label} is not blocking now: this is an attack."),
                    label,
                }),
                declared_attackers: Vec::new(),
                declared_blocks: Vec::new(),
            });
        }
        if kind == DeclarationKind::Blockers && !plan.attackers.is_empty() {
            let group = self
                .group(viewer)
                .ok_or_else(|| AgentError("not a combat declaration".into()))?;
            let id = plan.attackers[0];
            let label = self.permanent_render_label(id);
            return Ok(CombatPreview {
                kind,
                declarable: declarable_from_spec(&group.spec),
                steps: Vec::new(),
                refused: Some(DeclarationRefusal {
                    creature: id,
                    reason: format!("{label} is not attacking now: this is a block."),
                    label,
                }),
                declared_attackers: Vec::new(),
                declared_blocks: Vec::new(),
            });
        }

        let group_plan = match kind {
            DeclarationKind::Attackers => GroupPlan::DeclareAttackers {
                attackers: plan.attackers.clone(),
            },
            DeclarationKind::Blockers => GroupPlan::DeclareBlockers {
                blocks: plan.blocks.clone(),
            },
        };
        let preview = self.preview_group(viewer, revision, fingerprint, &group_plan)?;
        let declarable = declarable_from_spec(&preview.group.spec);
        let steps = preview
            .steps
            .into_iter()
            .map(|step| {
                let creature = step.item.expect("combat step names a creature");
                DeclarationStep {
                    revision: step.revision,
                    fingerprint: step.fingerprint,
                    creature,
                    action_index: step.action_index,
                    attack: kind == DeclarationKind::Attackers
                        && plan.attackers.contains(&creature),
                    blocks: (kind == DeclarationKind::Blockers)
                        .then(|| {
                            plan.blocks
                                .iter()
                                .find(|block| block.blocker == creature)
                                .map(|block| block.attacker)
                        })
                        .flatten(),
                }
            })
            .collect();
        let refused = preview.refused.map(|refusal| DeclarationRefusal {
            creature: refusal.item.expect("combat refusal names a creature"),
            label: refusal.label,
            reason: refusal.reason,
        });
        let (declared_attackers, declared_blocks) = match preview.outcome {
            Some(GroupOutcome::DeclareAttackers { attackers }) => (attackers, Vec::new()),
            Some(GroupOutcome::DeclareBlockers { blocks }) => (Vec::new(), blocks),
            _ => (Vec::new(), Vec::new()),
        };
        Ok(CombatPreview {
            kind,
            declarable,
            steps,
            refused,
            declared_attackers,
            declared_blocks,
        })
    }
}
