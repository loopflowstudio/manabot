//! Deprecated combat-preview wrapper over `flow::group` (removed in 0.3.0).
//!
//! Preserves the `0.2.6` `CombatPlan` / `CombatPreview` surface by delegating
//! to [`Game::preview`]. New callers should use [`Game::decision`],
//! [`Game::preview`] and [`Game::submit`].
use serde::{Deserialize, Serialize};

use crate::agent::action::{ActionSpaceKind, AgentError};
use crate::experience::ObjectRenderId;
use crate::flow::game::Game;
use crate::flow::group::{DecisionAddress, DecisionChoice, DecisionOutcome, DecisionSpec};
use crate::state::game_object::PlayerId;

pub use crate::flow::group::{PlannedAttack, PlannedBlock};

/// The whole declaration a person drafts. An attack plan lists attackers;
/// a block plan lists blocks. A creature left out declines.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CombatPlan {
    #[serde(default)]
    pub attackers: Vec<PlannedAttack>,
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
    /// Attackers only: permanent defenders (planeswalkers) it may attack.
    pub can_attack: Vec<ObjectRenderId>,
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
    pub defender: Option<ObjectRenderId>,
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
    pub declared_attackers: Vec<PlannedAttack>,
    pub declared_blocks: Vec<PlannedBlock>,
}

fn declarable_from_spec(spec: &DecisionSpec) -> Vec<DeclarableCreature> {
    match spec {
        DecisionSpec::DeclareAttackers { declarable, .. } => declarable
            .iter()
            .map(|item| DeclarableCreature {
                creature: item.item,
                label: item.label.clone(),
                can_block: Vec::new(),
                can_attack: item.can_attack.clone(),
                must_block: false,
                must_attack: item.must_attack,
            })
            .collect(),
        DecisionSpec::DeclareBlockers { declarable, .. } => declarable
            .iter()
            .map(|item| DeclarableCreature {
                creature: item.item,
                label: item.label.clone(),
                can_block: item.can_block.clone(),
                can_attack: Vec::new(),
                must_block: item.must_block,
                must_attack: false,
            })
            .collect(),
        _ => Vec::new(),
    }
}

impl Game {
    /// Preview `plan` at the current declaration, bound to its exact semantic
    /// address. Deprecated wrapper over [`Game::preview`] (removed in 0.3.0).
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
        let wrong_field = match kind {
            DeclarationKind::Attackers => plan
                .blocks
                .first()
                .map(|block| (block.blocker, "is not blocking now: this is an attack.")),
            DeclarationKind::Blockers => plan
                .attackers
                .first()
                .map(|attack| (attack.creature, "is not attacking now: this is a block.")),
        };
        if let Some((id, suffix)) = wrong_field {
            let decision = self
                .decision(Some(viewer))
                .ok_or_else(|| AgentError("not a combat declaration".into()))?;
            let label = self.permanent_render_label(id);
            return Ok(CombatPreview {
                kind,
                declarable: declarable_from_spec(&decision.spec),
                steps: Vec::new(),
                refused: Some(DeclarationRefusal {
                    creature: id,
                    reason: format!("{label} {suffix}"),
                    label,
                }),
                declared_attackers: Vec::new(),
                declared_blocks: Vec::new(),
            });
        }

        let choice = match kind {
            DeclarationKind::Attackers => DecisionChoice::DeclareAttackers {
                attackers: plan.attackers.clone(),
            },
            DeclarationKind::Blockers => DecisionChoice::DeclareBlockers {
                blocks: plan.blocks.clone(),
            },
        };
        let address = DecisionAddress {
            viewer: viewer.0,
            revision,
            fingerprint: fingerprint.to_string(),
        };
        let preview = self.preview(Some(&address), &choice)?;
        let declarable = declarable_from_spec(&preview.decision.spec);
        let steps = preview
            .steps
            .into_iter()
            .map(|step| {
                let creature = step.item.expect("combat step names a creature");
                let planned_attack = (kind == DeclarationKind::Attackers)
                    .then(|| plan.attackers.iter().find(|a| a.creature == creature))
                    .flatten();
                DeclarationStep {
                    revision: step.revision,
                    fingerprint: step.fingerprint,
                    creature,
                    action_index: step.action_index,
                    attack: planned_attack.is_some(),
                    defender: planned_attack.and_then(|a| a.defender),
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
            Some(DecisionOutcome::DeclareAttackers { attackers }) => (attackers, Vec::new()),
            Some(DecisionOutcome::DeclareBlockers { blocks }) => (Vec::new(), blocks),
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
