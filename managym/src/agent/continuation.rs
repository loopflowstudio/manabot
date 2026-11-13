//! Current stack and unfinished decisions, projected once from rules authority.
//! References link only to exact, visible incarnations. Tensor rows and portable
//! text evidence consume these same facts; no strings or JSON enter encoding.

use super::observation_layout::relation;

use crate::state::ability::LibraryEnd;
use serde::{Deserialize, Serialize};

use crate::{
    agent::observation::{KeywordData, KEYWORD_COUNT},
    flow::{
        decision::{Decision, EffectFrame, ResolutionRef, SuspendedResolution},
        game::{ActivationSource, Game, Payment, PendingChoice},
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        ability::{DelayedExpiry, DelayedFrequency, PaymentCost},
        game_object::{ObjectId, ObjectRef, PlayerId, Target},
        mana::ManaCost,
        stack_object::StackObject,
        zone::ZoneType,
    },
};

/// Up to 64 relation rows, columns named by `observation_layout::relation`.
/// Overflow is an admission error, never truncation. Zero is padding/absence
/// for selectors. Granted keywords follow `KeywordData` field order.
pub const RELATION_ROWS: usize = 64;
pub const RELATION_DIM: usize = relation::WIDTH;
const _: () = assert!(relation::THIS_TURN == relation::FLYING + KEYWORD_COUNT);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct VisibleReference {
    pub object: Option<i32>,
    pub definition: Option<u32>,
    pub player: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct ClauseTarget {
    pub reference: VisibleReference,
    pub clause: usize,
    /// Retained CR 608.2b result, not current presence. Missing historical data is unknown.
    #[serde(default)]
    pub legal_at_resolution_start: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct ManaOutput {
    pub mode: usize,
    pub mana: [u8; 6],
    /// Spend-restricted output by color, whatever its restriction.
    pub lesson: [u8; 6],
}

/// Public stack entries precede the viewer's private announcement/resolution
/// frames; public waiting delayed triggers follow them. Parent indices retain
/// interrupted payment relationships.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct ContinuationFrame {
    /// Complete originating program when no catalog join can identify it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub program: Vec<(u16, i32)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program_description: Option<String>,
    /// Effect execution tokens, a separate ordered occurrence from the source program.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effect_execution: Vec<(u16, i32)>,
    /// Kind-10 relation rows link the one-based slots in effect execution tokens.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effect_execution_references: Vec<VisibleReference>,
    /// Every retained fact as `name: value` text. Filled only by
    /// `Game::describe_effect_execution` (decision context and capture); a policy
    /// observation leaves it absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_execution_description: Option<String>,
    /// Plain-language pending work for the same viewer-safe Academy decision,
    /// filled with `effect_execution_description`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effect_execution_summary: Vec<String>,
    /// Viewer-safe exact subjects retained by a delayed ability, separate from event context.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remembered: Vec<VisibleReference>,
    /// Source's captured ability lender, retained after source departure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inherited_from: Option<VisibleReference>,
    /// The public attacker restriction of a scheduled/current extra combat.
    /// Its token is a fact; its readable form comes from the rules predicate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attackers: Option<String>,
    /// Transform count captured for this ability's source (CR 701.27f).
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub source_transforms: u32,
    /// 1 spell, 2 activated stack ability, 3 triggered stack ability,
    /// 4 announcement, 5 resolution, 6 obligation, 7 waiting trigger,
    /// 8 trigger being placed, 9 two-pass placement cursor, 10 waiting
    /// delayed trigger (CR 603.7; `decision` names its event, see
    /// [`delayed_event_code`]), 11 scheduled turn, 12 additional combat,
    /// 13 player control (context: controlled player, decision: 1 combat/2 turn,
    /// mode: 0 pending/1 active; creation order), 14 delegated decision
    /// (controller: chooser, context: player whose decision it is).
    pub kind: u8,
    /// Kind 6: the obligation's `decision_code`. Kinds 7 and 8: trigger
    /// ordering class (1 ordinary, 2 ability-triggered). Kind 9: placement
    /// group plus one. Kind 10: delayed event code. Zero otherwise.
    pub decision: u8,
    pub parent: Option<usize>,
    pub controller: usize,
    pub source: VisibleReference,
    pub mode: Option<usize>,
    pub ability: Option<usize>,
    pub kicked: bool,
    pub active_clause: Option<usize>,
    pub targets: Vec<ClauseTarget>,
    pub context: Option<VisibleReference>,
    pub sacrifice: Option<VisibleReference>,
    pub cost: Option<[u8; 7]>,
    pub hybrid: Vec<[u8; 2]>,
    /// The payment has a purpose (a cast spell or an activation), so some
    /// spend-restricted mana may pay it. Named for the first restriction.
    pub lesson: bool,
    pub waterbend: bool,
    pub can_decline: bool,
    pub mana_outputs: Vec<ManaOutput>,
    pub resolutions_this_turn: u32,
    /// The announced value of X (CR 107.3) of the spell or ability this frame
    /// casts, activates or resolves; zero when it has none.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    /// Keywords a static grants this spell on the stack (a Lesson's lifelink),
    /// in `KeywordData` field order.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub granted_keywords: [bool; KEYWORD_COUNT],
    /// Waiting frequency: 1 once, 2 each match. Missing historical data is unknown.
    /// Encoded as a kind-9 relation row with the code in `relation::DECISION`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delayed_frequency: Option<u8>,
    /// A waiting delayed trigger that ends in this turn's cleanup step.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub this_turn: bool,
}

pub const MAX_DELAYED_EVENT_CODE: u8 = 7;

/// A waiting delayed trigger's event as its frame's `decision`: 1 the watched
/// object dies, 2 it dies or is exiled, 3 any creature dies, 4 the end
/// step, 5 its controller casts a matching spell, 6 upkeep,
/// 7 its controller's end step. Frequency is projected separately.
pub fn delayed_event_code(event: &DelayedTriggerEvent) -> u8 {
    match event {
        DelayedTriggerEvent::Leaves {
            or_exiled: false, ..
        } => 1,
        DelayedTriggerEvent::Leaves {
            or_exiled: true, ..
        } => 2,
        DelayedTriggerEvent::CreatureDies => 3,
        DelayedTriggerEvent::EndStep {
            your_turn: false, ..
        } => 4,
        DelayedTriggerEvent::EndStep {
            your_turn: true, ..
        } => 7,
        DelayedTriggerEvent::YouCastSpell(_) => 5,
        DelayedTriggerEvent::Upkeep => 6,
    }
}

impl ContinuationFrame {
    fn program(&mut self, effects: &[crate::state::ability::Effect]) {
        if effects.is_empty() {
            return;
        }
        self.program = crate::cards::tokens::rules_tokens(
            &effects,
            crate::cards::content_pack::default_content_pack(),
        )
        .expect("rules program tokens");
        self.program_description = Some(super::ability_description::effects_description(effects));
    }
    fn cost(&mut self, cost: &ManaCost) {
        self.cost = Some(cost.cost);
        self.hybrid = cost
            .hybrid
            .iter()
            .map(|pair| [pair[0] as u8, pair[1] as u8])
            .collect();
        self.lesson = !cost.purpose.is_resolving();
    }
}

impl Game {
    /// Whether retained inputs can recover an activation from catalog joins.
    /// Execution always consumes the captured effects, regardless of this answer.
    pub(crate) fn activation_catalog_program(
        &self,
        object: ObjectId,
        face: crate::state::card::CardDefId,
        inherited: Option<crate::state::card::CardDefId>,
        index: usize,
        grant: Option<crate::state::card::AbilityGrant>,
        effects: &[crate::state::ability::Effect],
    ) -> bool {
        let definition = if let Some(grant) = grant {
            self.state.objects[ObjectId::from(grant.grantor.entity)]
                .attached_grants()
                .nth(grant.index)
        } else {
            self.source_face(object, face).and_then(|source| {
                source.activated_abilities.get(index).or_else(|| {
                    crate::cards::content_pack::default_content_pack()
                        .definition(inherited?)?
                        .activated_abilities
                        .get(index.checked_sub(source.activated_abilities.len())?)
                })
            })
        };
        definition.is_some_and(|definition| definition.effects == effects)
    }
    pub(crate) fn visible_reference(
        &self,
        object: ObjectRef,
        viewer: PlayerId,
    ) -> VisibleReference {
        let object_id = ObjectId::from(object.entity);
        if self.current_object_ref(object_id) != Some(object) {
            return VisibleReference {
                definition: self.object_lki(object).map(|lki| lki.definition_id.0),
                ..VisibleReference::default()
            };
        }
        let record = &self.state.objects[object_id];
        let zone = self.state.zones.zone_of(object_id);
        // Ordinary visibility needs no walk of retained event batches. Only
        // otherwise hidden objects need the prospective-entry permission.
        let visible = matches!(
            zone,
            Some(
                ZoneType::Battlefield
                    | ZoneType::Graveyard
                    | ZoneType::Stack
                    | ZoneType::Exile
                    | ZoneType::Command
            )
        ) || (zone == Some(ZoneType::Hand)
            && self.shares_information(viewer, record.owner))
            || (matches!(zone, Some(ZoneType::Hand | ZoneType::Library))
                && (self.searches_private_zones(viewer, record.owner)
                    || self.decision_reveals(object_id, viewer)))
            || self.pending_public_entries().contains(&object);
        if !visible
            || (self.is_foretold(object_id) && !self.shares_information(viewer, record.owner))
        {
            return VisibleReference::default();
        }
        VisibleReference {
            object: Some(
                self.lookup_current_permanent(object)
                    .ok()
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(record.id.0, |p| p.id.0) as i32,
            ),
            definition: Some(record.definition_id.0),
            player: None,
        }
    }

    /// A hidden-zone card a suspended decision shows its chooser: the exact
    /// cards it looks at, orders or has revealed, or the whole zone a card
    /// selection reads. The observation lists these same cards for that viewer,
    /// so their reference slots join its rows instead of reading as unrevealed.
    fn decision_reveals(&self, object: ObjectId, viewer: PlayerId) -> bool {
        let Some(suspended) = &self.state.suspended_decision else {
            return false;
        };
        let (owner, zone) = (
            self.state.objects[object].owner,
            self.state.zones.zone_of(object),
        );
        suspended.decisions().any(|decision| {
            self.shares_information(viewer, decision.player())
                && (decision.revealed_objects().contains(&object)
                    || matches!(decision, Decision::SelectCards { from: Some(source), .. }
                        if source.owner == owner && Some(source.zone) == zone))
        })
    }

    pub(crate) fn object_reference(
        &self,
        object_id: impl Into<ObjectId>,
        viewer: PlayerId,
    ) -> VisibleReference {
        self.current_object_ref(object_id)
            .map(|object| self.visible_reference(object, viewer))
            .unwrap_or_default()
    }

    fn activation_reference(&self, source: ActivationSource, viewer: PlayerId) -> VisibleReference {
        match source {
            ActivationSource::Permanent(permanent) => {
                self.target_reference(Target::Permanent(permanent), viewer)
            }
            ActivationSource::Card(card) => self.object_reference(card, viewer),
        }
    }

    pub(crate) fn target_reference(&self, target: Target, viewer: PlayerId) -> VisibleReference {
        match target {
            Target::Player(player) => VisibleReference {
                player: Some(player.0),
                ..VisibleReference::default()
            },
            Target::Permanent(id) => self
                .permanent_object_ref(id)
                .map(|object| self.visible_reference(object, viewer))
                .unwrap_or_default(),
            Target::StackSpell(object_id) => {
                if self.state.zones.zone_of(object_id) == Some(ZoneType::Stack) {
                    self.object_reference(object_id, viewer)
                } else {
                    VisibleReference::default()
                }
            }
            Target::StackAbility(id) => self
                .stack_ability_source(id)
                .map(|object_id| self.object_reference(object_id, viewer))
                .unwrap_or_default(),
            Target::GraveyardCard(object) => self.visible_reference(object, viewer),
        }
    }

    fn clause_targets(
        &self,
        targets: &[Target],
        clauses: &[usize],
        viewer: PlayerId,
    ) -> Vec<ClauseTarget> {
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| ClauseTarget {
                reference: self.target_reference(*target, viewer),
                clause: clauses.get(index).copied().unwrap_or(0),
                legal_at_resolution_start: None,
            })
            .collect()
    }

    /// The native observation owns the result. Nonacting viewers receive only
    /// public stack facts, never another player's announced private choices.
    pub fn continuation_frames(&self, viewer: PlayerId) -> Vec<ContinuationFrame> {
        // Same top-to-bottom ordering as Observation.stack_objects.
        let mut frames = Vec::new();
        for object in self.state.stack_objects.iter().rev() {
            frames.push(self.stack_frame(object, viewer));
        }
        if self.current_action_space.as_ref().and_then(|s| s.player) == Some(viewer)
            && !self.is_game_over()
        {
            self.append_private_frames(viewer, &mut frames);
        }
        frames.extend(
            self.state
                .delayed_triggers
                .iter()
                .map(|trigger| self.delayed_trigger_frame(trigger, viewer)),
        );
        // Scheduled phases/turns are public. Newest insertions happen first;
        // mode is the ordinal, not a raw object identity. A resume turn is
        // retained after queued extra turns, and current combat uses mode 0.
        for (index, player) in self
            .state
            .turn
            .extra_turns
            .iter()
            .rev()
            .chain(self.state.turn.resume_with.iter())
            .enumerate()
        {
            frames.push(ContinuationFrame {
                kind: 11,
                controller: player.0,
                mode: Some(index),
                ..Default::default()
            });
        }
        if let Some(attackers) = &self.state.turn.combat_attackers {
            frames.push(ContinuationFrame {
                kind: 12,
                controller: self.active_player().0,
                mode: Some(0),
                attackers: Some(super::ability_description::predicate_description(
                    attackers,
                    "creatures",
                )),
                ..Default::default()
            });
        }
        for (index, combat) in self.state.turn.additional_combats.iter().rev().enumerate() {
            frames.push(ContinuationFrame {
                kind: 12,
                controller: self.active_player().0,
                mode: Some(index + 1),
                attackers: combat.attackers.as_ref().map(|predicate| {
                    super::ability_description::predicate_description(predicate, "creatures")
                }),
                ..Default::default()
            });
        }
        for control in &self.state.turn.player_controls {
            frames.push(ContinuationFrame {
                kind: 13,
                controller: control.controller.0,
                context: Some(VisibleReference {
                    player: Some(control.player.0),
                    ..Default::default()
                }),
                decision: match control.duration {
                    crate::state::ability::PlayerControlDuration::NextCombat => 1,
                    crate::state::ability::PlayerControlDuration::NextTurn => 2,
                },
                mode: Some(usize::from(control.active)),
                ..Default::default()
            });
        }
        if let Some((controller, action)) = self
            .action_space()
            .filter(|_| !self.is_game_over())
            .and_then(|space| space.player.zip(space.actions.first()))
        {
            if action.player() != controller {
                frames.push(ContinuationFrame {
                    kind: 14,
                    controller: controller.0,
                    context: Some(VisibleReference {
                        player: Some(action.player().0),
                        ..Default::default()
                    }),
                    ..Default::default()
                });
            }
        }
        frames
    }

    /// A waiting delayed trigger is public: its source, controller, event,
    /// watched object or remembered context, and whether it lasts this turn.
    fn delayed_trigger_frame(
        &self,
        trigger: &DelayedTrigger,
        viewer: PlayerId,
    ) -> ContinuationFrame {
        let mut source = trigger
            .source_ref
            .map(|r| self.visible_reference(r, viewer))
            .unwrap_or_default();
        source.definition = Some(self.state.objects[trigger.source_object].definition_id.0);
        let context = match &trigger.event {
            DelayedTriggerEvent::Leaves { watched, .. } => {
                Some(self.visible_reference(*watched, viewer))
            }
            DelayedTriggerEvent::EndStep { .. }
            | DelayedTriggerEvent::CreatureDies
            | DelayedTriggerEvent::YouCastSpell(_)
            | DelayedTriggerEvent::Upkeep => None,
        };
        ContinuationFrame {
            remembered: self.remembered_references(&trigger.remembered, viewer),
            kind: 10,
            decision: delayed_event_code(&trigger.event),
            controller: trigger.controller.0,
            source,
            context,
            delayed_frequency: Some(match trigger.frequency {
                DelayedFrequency::Once => 1,
                DelayedFrequency::EachMatch => 2,
            }),
            this_turn: trigger.expiry == DelayedExpiry::ThisTurn,
            ..ContinuationFrame::default()
        }
    }

    fn remembered_references(
        &self,
        objects: &[crate::state::binding::ObjectBinding],
        viewer: PlayerId,
    ) -> Vec<VisibleReference> {
        objects
            .iter()
            .map(|binding| self.visible_reference(binding.object, viewer))
            .collect()
    }

    /// The acting viewer's announcement, resolution and placement frames.
    fn append_private_frames(&self, viewer: PlayerId, frames: &mut Vec<ContinuationFrame>) {
        if let Some(pending) = &self.pending_choice {
            frames.push(self.announcement_frame(pending, viewer));
        }
        if let Some(suspended) = &self.state.suspended_decision {
            self.append_suspended_frames(suspended, viewer, frames);
        }
        if let Some(placement) = &self.state.trigger_placement {
            let parent = frames.len();
            frames.push(ContinuationFrame {
                kind: 9,
                decision: placement.group + 1,
                controller: placement.controller().0,
                ..ContinuationFrame::default()
            });
            for trigger in placement
                .remaining
                .iter()
                .chain(placement.placing.iter())
                .filter(|t| self.decision_controller(t.controller) == viewer)
            {
                let mut source = trigger
                    .source_ref
                    .map(|r| self.visible_reference(r, viewer))
                    .unwrap_or_default();
                source.definition = Some(trigger.source_lki.as_ref().map_or(
                    self.state.objects[trigger.source_object].definition_id.0,
                    |lki| lki.definition_id.0,
                ));
                let mut frame = ContinuationFrame {
                    kind: if placement
                        .placing
                        .as_ref()
                        .is_some_and(|t| t.id == trigger.id)
                    {
                        8
                    } else {
                        7
                    },
                    parent: Some(parent),
                    controller: trigger.controller.0,
                    source,
                    decision: match trigger.ordering_class {
                        crate::flow::trigger::TriggerOrderingClass::Ordinary => 1,
                        crate::flow::trigger::TriggerOrderingClass::AbilityTriggered => 2,
                    },
                    ability: Some(trigger.ability_index),
                    x: trigger.x,
                    source_transforms: trigger.source_lki.as_ref().map_or(0, |lki| lki.transforms),
                    active_clause: Some(trigger.requirement_index),
                    targets: self.clause_targets(
                        &trigger.targets,
                        &trigger.target_req_indices,
                        viewer,
                    ),
                    remembered: self.remembered_references(&trigger.remembered, viewer),
                    context: trigger.context.map(|t| self.target_reference(t, viewer)),
                    ..ContinuationFrame::default()
                };
                if trigger.program {
                    frame.program(trigger.inline_effects.as_deref().unwrap_or_default());
                }
                frames.push(frame);
            }
        }
    }

    pub(crate) fn trigger_action_frame(
        &self,
        action: &super::action::Action,
        viewer: PlayerId,
    ) -> Option<usize> {
        let super::action::Action::ChooseTrigger { trigger_id, player } = action else {
            return None;
        };
        if *player != viewer {
            return None;
        }
        let placement = self.state.trigger_placement.as_ref()?;
        let index = placement
            .remaining
            .iter()
            .filter(|t| self.decision_controller(t.controller) == viewer)
            .position(|t| t.id == *trigger_id)?;
        Some(
            self.state.stack_objects.len() + usize::from(self.pending_choice.is_some()) + 1 + index,
        )
    }

    fn stack_frame(&self, object: &StackObject, viewer: PlayerId) -> ContinuationFrame {
        let mut frame = ContinuationFrame {
            controller: object.controller().0,
            ability: object.ability_index(),
            ..ContinuationFrame::default()
        };
        let clauses: &[usize] = match object {
            StackObject::Spell(spell) => {
                frame.kind = 1;
                frame.source = self.object_reference(spell.object, viewer);
                frame.mode = spell.mode;
                frame.kicked = spell.kicked;
                frame.x = spell.x;
                frame.granted_keywords = KeywordData::from_keywords(
                    &self.spell_keywords(spell.object, spell.controller),
                )
                .flags();
                &spell.target_req_indices
            }
            StackObject::ActivatedAbility(ability) => {
                frame.kind = 2;
                if ability.program {
                    frame.program(&ability.effects);
                }
                frame.source = self.visible_reference(ability.source_ref, viewer);
                frame.x = ability.x;
                frame.inherited_from = ability.inherited_from.map(|id| VisibleReference {
                    definition: Some(id.0),
                    ..Default::default()
                });
                frame.source_transforms = ability.source_transforms;
                &ability.target_req_indices
            }
            StackObject::TriggeredAbility(ability) => {
                frame.kind = 3;
                if ability.program {
                    frame.program(ability.inline_effects.as_deref().unwrap_or_default());
                }
                frame.inherited_from = ability
                    .source_lki
                    .as_ref()
                    .and_then(|lki| lki.inherited_from)
                    .map(|id| VisibleReference {
                        definition: Some(id.0),
                        ..Default::default()
                    });
                frame.x = ability.x;
                frame.source_transforms = ability.source_transforms;
                frame.source = ability
                    .source_ref
                    .map(|r| self.visible_reference(r, viewer))
                    .unwrap_or_default();
                frame.remembered = self.remembered_references(&ability.remembered, viewer);
                frame.context = ability.context.map(|t| self.target_reference(t, viewer));
                &ability.target_req_indices
            }
        };
        // Stack definitions remain public after the original source departs.
        frame.source.definition = Some(object.source_definition_id().0);
        frame.targets = self.clause_targets(object.targets(), clauses, viewer);
        frame
    }

    fn announcement_frame(&self, pending: &PendingChoice, viewer: PlayerId) -> ContinuationFrame {
        let mut frame = ContinuationFrame {
            kind: 4,
            controller: self
                .current_action_space
                .as_ref()
                .and_then(|space| space.actions.first().map(|action| action.player()))
                .expect("announcement has a player")
                .0,
            ..ContinuationFrame::default()
        };
        match pending {
            PendingChoice::ManaPayment {
                player,
                payment,
                sacrifice,
                alternative,
            } => {
                self.payment_frame(payment, viewer, &mut frame);
                frame.sacrifice =
                    sacrifice.map(|id| self.target_reference(Target::Permanent(id), viewer));
                frame.cost(&self.payment_mana_cost(*player, payment, alternative.as_ref()));
            }
            PendingChoice::SacrificePayment { payment, .. }
            | PendingChoice::ExileFromHandPayment { payment, .. } => {
                self.payment_frame(payment, viewer, &mut frame)
            }
            PendingChoice::ChooseTargets {
                object,
                cast,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                ..
            } => {
                frame.source = self.object_reference(*object, viewer);
                frame.kicked = cast.kicked;
                frame.x = cast.x;
                frame.mode = *mode;
                frame.active_clause = Some(*requirement_index);
                frame.targets = self.clause_targets(chosen, chosen_req_indices, viewer);
            }
            PendingChoice::SpellMode { object, cast, .. } => {
                frame.source = self.object_reference(*object, viewer);
                frame.kicked = cast.kicked;
                frame.x = cast.x;
            }
            PendingChoice::KickerChoice { object, .. }
            | PendingChoice::AnnounceX { object, .. } => {
                frame.source = self.object_reference(*object, viewer)
            }
            PendingChoice::ActivatedTarget {
                source,
                ability_index,
                chosen,
                requirement,
                ..
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
                frame.active_clause = Some(*requirement);
                let clauses = self
                    .activation_ability(*source, *ability_index)
                    .map(|ability| ability.target_clause_indices(chosen.len()))
                    .unwrap_or_default();
                frame.targets = self.clause_targets(chosen, &clauses, viewer);
            }
            PendingChoice::DiscardPayment {
                source,
                ability_index,
                ..
            }
            | PendingChoice::TapPayment {
                source,
                ability_index,
                ..
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
            }
            PendingChoice::ActivationX {
                permanent,
                ability_index,
                ..
            }
            | PendingChoice::Crew {
                permanent,
                ability_index,
                ..
            } => {
                frame.source = self.target_reference(Target::Permanent(*permanent), viewer);
                frame.ability = Some(*ability_index);
            }
            PendingChoice::Waterbend {
                player,
                payment,
                tapped,
                ..
            } => {
                self.payment_frame(payment, viewer, &mut frame);
                frame.waterbend = true;
                let owed = self
                    .payment_mana_cost(*player, payment, None)
                    .reduced_generic(*tapped);
                frame.cost(&owed);
            }
        }
        if let PendingChoice::TapPayment {
            remaining,
            source,
            ability_index,
            ..
        } = pending
        {
            if self
                .activation_ability(*source, *ability_index)
                .and_then(|a| a.tap_cost)
                .is_some_and(|cost| cost.count > 1)
            {
                frame.mode = Some(*remaining);
            }
        }
        frame
    }

    fn append_suspended_frames(
        &self,
        suspended: &SuspendedResolution,
        viewer: PlayerId,
        frames: &mut Vec<ContinuationFrame>,
    ) {
        let mut parent = self.pending_choice.as_ref().map(|_| frames.len() - 1);
        for part in suspended.continuation() {
            let mut frame = match part {
                ResolutionRef::Frame(effect) => self.effect_frame(effect, viewer),
                ResolutionRef::Decision(decision) => {
                    self.resolution_decision_frame(decision, viewer)
                }
            };
            frame.parent = parent;
            frames.push(frame);
            parent = Some(frames.len() - 1);
        }
    }

    fn effect_frame(&self, effect: &EffectFrame, viewer: PlayerId) -> ContinuationFrame {
        let mut frame = ContinuationFrame {
            kind: 5,
            controller: effect.controller.0,
            source: effect
                .source_ref
                .map(|r| self.visible_reference(r, viewer))
                .unwrap_or_else(|| {
                    effect
                        .source
                        .map(|object| self.object_reference(object, viewer))
                        .unwrap_or_default()
                }),
            kicked: effect.kicked,
            inherited_from: effect.inherited_from.map(|id| VisibleReference {
                definition: Some(id.0),
                ..Default::default()
            }),
            x: effect.x,
            source_transforms: effect.source_transforms,
            resolutions_this_turn: effect.resolutions_this_turn,
            active_clause: Some(effect.active_requirement),
            targets: effect
                .targets
                .iter()
                .map(|slot| ClauseTarget {
                    reference: self.target_reference(slot.target, viewer),
                    clause: slot.requirement,
                    legal_at_resolution_start: Some(slot.legal),
                })
                .collect(),
            remembered: self.remembered_references(&effect.remembered, viewer),
            context: effect
                .context_target
                .map(|t| self.target_reference(t, viewer)),
            ..ContinuationFrame::default()
        };
        frame.program(&effect.program);
        super::effect_execution::project_frame(self, effect, viewer, &mut frame, false);
        frame
    }

    /// Render the viewer's own suspended frames' effect execution as text:
    /// the complete `name: value` description and plain-language summaries.
    /// This is a presentation product of decision context and capture; the
    /// policy observation leaves both fields empty and never builds strings.
    pub(crate) fn describe_effect_execution(
        &self,
        viewer: PlayerId,
        frames: &mut [ContinuationFrame],
    ) {
        if self.current_action_space.as_ref().and_then(|s| s.player) != Some(viewer)
            || self.is_game_over()
        {
            return;
        }
        let Some(suspended) = &self.state.suspended_decision else {
            return;
        };
        // Kinds 5 and 6 come only from `append_suspended_frames`, in this order.
        let mut parts = suspended.continuation().into_iter();
        for frame in frames.iter_mut().filter(|f| matches!(f.kind, 5 | 6)) {
            match parts.next() {
                Some(ResolutionRef::Frame(effect)) => {
                    debug_assert_eq!(frame.kind, 5);
                    super::effect_execution::project_frame(self, effect, viewer, frame, true);
                }
                Some(ResolutionRef::Decision(decision)) => {
                    debug_assert_eq!(frame.kind, 6);
                    super::effect_execution::project_decision(self, decision, viewer, frame, true);
                }
                None => unreachable!("suspended frames outnumber the retained caller chain"),
            }
        }
    }

    fn resolution_decision_frame(
        &self,
        decision: &Decision,
        viewer: PlayerId,
    ) -> ContinuationFrame {
        let mut frame = ContinuationFrame {
            kind: 6,
            controller: decision.player().0,
            ..ContinuationFrame::default()
        };
        frame.decision = decision_code(decision);
        match decision {
            Decision::ChoosePaymentX { .. } => {
                frame.can_decline = true;
            }
            Decision::SearchNamed {
                owner, remaining, ..
            } => {
                frame.can_decline = true;
                frame.mode = Some(*remaining);
                frame.context = Some(self.target_reference(Target::Player(*owner), viewer));
            }
            Decision::Payment(progress) => {
                frame.cost(&progress.remaining.mana_cost());
                frame.waterbend = matches!(progress.remaining, PaymentCost::Waterbend { .. });
                frame.can_decline = progress.can_decline();
            }
            Decision::ManaUpfront {
                permanent,
                ability_index,
                tap_remaining,
                ..
            } => {
                frame.source = self.target_reference(Target::Permanent(*permanent), viewer);
                frame.ability = Some(*ability_index);
                if *tap_remaining > 0 {
                    frame.mode = Some(*tap_remaining);
                }
                frame.cost(
                    &self.characteristics(*permanent).activated_abilities[*ability_index].mana_cost,
                );
            }
            Decision::ManaChoice { choices, .. } => {
                frame.mana_outputs = choices
                    .iter()
                    .enumerate()
                    .map(|(mode, output)| ManaOutput {
                        mode,
                        mana: output.mana,
                        lesson: output.restricted_colors(),
                    })
                    .collect();
            }
            Decision::OrderLibrary {
                next,
                end: LibraryEnd::Top,
                ..
            } => {
                // The visible library segment is top-first. Its first
                // `next` cards have already been ordered by the player.
                frame.mode = Some(*next);
            }
            Decision::SelectCards {
                selected, min, max, ..
            } => {
                frame.can_decline = selected.len() >= *min;
                frame.mode = Some(max.saturating_sub(selected.len()));
            }
            Decision::ChooseObject { optional, .. } => {
                frame.can_decline = *optional;
            }
            _ => {}
        };
        super::effect_execution::project_decision(self, decision, viewer, &mut frame, false);
        frame
    }

    fn payment_frame(&self, payment: &Payment, viewer: PlayerId, frame: &mut ContinuationFrame) {
        match payment {
            Payment::Spell {
                object,
                cast,
                mode,
                targets,
                target_req_indices,
                ..
            } => {
                frame.source = self.object_reference(*object, viewer);
                frame.kicked = cast.kicked;
                frame.x = cast.x;
                frame.mode = *mode;
                frame.targets = self.clause_targets(targets, target_req_indices, viewer);
            }
            Payment::Activation {
                source,
                ability_index,
                targets,
                x,
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
                frame.x = *x;
                frame.targets = self.clause_targets(targets, &[], viewer);
            }
        }
    }
}

/// Highest obligation code `decision_code` assigns. Captured continuation
/// frames are validated against it; raise it with each new code.
pub const MAX_DECISION_CODE: u8 = 30;

/// A suspended obligation's stable code in `ContinuationFrame::decision`
/// (1..=`MAX_DECISION_CODE`). Codes are part of retained captures: append
/// new ones, never renumber.
pub fn decision_code(decision: &Decision) -> u8 {
    match decision {
        Decision::OrderReplacement { .. } => 29,
        Decision::EnterAsCopy { .. } => 30,
        Decision::ChoosePaymentX { .. } => 24,
        Decision::SearchNamed { .. } => 25,
        Decision::Payment(_) => 1,
        Decision::ManaUpfront { .. } => 2,
        Decision::ManaChoice { .. } => 3,
        Decision::ChooseObject { .. } => 4,
        Decision::SelectCards { .. } => 28,
        Decision::Scry { .. } => 8,
        Decision::LookAndSelect { .. } => 9,
        Decision::Modal { .. } | Decision::NameCard { .. } => 10,
        Decision::Learn { .. } => 11,
        Decision::Discard { .. } => 12,
        Decision::LegendRule { .. } => 13,
        Decision::SearchDistinctNames { .. } => 14,
        Decision::OpponentSplitsRevealed { .. } => 15,
        Decision::DiscardFromRevealedHand { .. } => 16,
        Decision::CastDuringResolution { .. } => 17,
        Decision::ExileLinked { .. } => 18,
        Decision::DistributeCounters { .. } => 19,
        Decision::KeepCreatures { .. } => 20,
        Decision::Retarget { .. } => 21,
        Decision::Surveil { .. } => 22,
        Decision::OrderLibrary {
            end: LibraryEnd::Bottom,
            ..
        } => 23,
        Decision::OrderLibrary {
            end: LibraryEnd::Top,
            ..
        } => 27,
        Decision::ChooseToExile { .. } => 26,
    }
}

/// Write directly into the caller's reusable tensor. References use the same
/// padded object table as action focus; absent/expired links encode as zero.
pub fn encode_into(
    frames: &[ContinuationFrame],
    viewer: usize,
    objects: &std::collections::HashMap<i32, i32>,
    out: &mut [f32],
) -> Result<(), super::observation_encoder::ObservationEncodeError> {
    use super::observation_encoder::ObservationEncodeError;
    if out.len() != RELATION_ROWS * RELATION_DIM {
        return Err(ObservationEncodeError::InvalidLength {
            field: "relations",
            expected: RELATION_ROWS * RELATION_DIM,
            actual: out.len(),
        });
    }
    let count: usize = frames
        .iter()
        .map(|f| {
            1 + f.effect_execution_references.len()
                + f.targets.len()
                + usize::from(f.context.is_some())
                + usize::from(f.sacrifice.is_some())
                + usize::from(f.inherited_from.is_some())
                + f.remembered.len()
                + usize::from(f.delayed_frequency.is_some())
                + f.hybrid.len()
                + 2 * f.mana_outputs.len()
        })
        .sum();
    if count > RELATION_ROWS {
        return Err(ObservationEncodeError::Capacity {
            field: "continuation relations",
            capacity: RELATION_ROWS,
            actual: count,
        });
    }
    out.fill(0.0);
    let mut rows = out.as_chunks_mut::<RELATION_DIM>().0.iter_mut();
    let link = |reference: VisibleReference| {
        reference
            .object
            .and_then(|id| objects.get(&id))
            .map_or(0.0, |index| (index + 1) as f32)
    };
    let definition =
        |reference: VisibleReference| reference.definition.map_or(0.0, |id| (id + 1) as f32);
    for (index, frame) in frames.iter().enumerate() {
        let mut header = [0.0; RELATION_DIM];
        header[relation::KIND] = 1.0;
        header[relation::FRAME] = (index + 1) as f32;
        header[relation::PARENT] = frame.parent.map_or(0.0, |p| (p + 1) as f32);
        header[relation::FRAME_KIND] = frame.kind as f32;
        header[relation::CONTROLLER] = if frame.controller == viewer { 1.0 } else { 2.0 };
        header[relation::SOURCE_DEFINITION] = definition(frame.source);
        header[relation::SOURCE] = link(frame.source);
        header[relation::MODE] = frame.mode.map_or(0.0, |v| (v + 1) as f32);
        header[relation::ABILITY] = frame.ability.map_or(0.0, |v| (v + 1) as f32);
        header[relation::KICKED] = f32::from(frame.kicked);
        header[relation::ACTIVE_CLAUSE] = frame.active_clause.map_or(0.0, |v| (v + 1) as f32);
        header[relation::HAS_COST] = f32::from(frame.cost.is_some());
        if let Some(cost) = frame.cost {
            for (i, amount) in cost.iter().enumerate() {
                header[relation::MANA_WHITE + i] = *amount as f32;
            }
        }
        header[relation::RESTRICTED] = f32::from(frame.lesson);
        header[relation::WATERBEND] = f32::from(frame.waterbend);
        header[relation::CAN_DECLINE] = f32::from(frame.can_decline);
        header[relation::DECISION] = frame.decision as f32;
        header[relation::RESOLUTIONS_THIS_TURN] = frame.resolutions_this_turn as f32;
        header[relation::VALID] = 1.0;
        header[relation::X] = f32::from(frame.x);
        for (index, granted) in frame.granted_keywords.iter().enumerate() {
            header[relation::FLYING + index] = f32::from(*granted);
        }
        header[relation::THIS_TURN] = f32::from(frame.this_turn);
        rows.next().unwrap().copy_from_slice(&header);
        let mut reference_row = |kind: f32,
                                 reference: VisibleReference,
                                 clause: usize,
                                 ordinal: usize,
                                 legal: Option<bool>| {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header);
            row[relation::KIND] = kind;
            row[relation::LEGAL_AT_RESOLUTION_START] =
                legal.map_or(0.0, |value| if value { 1.0 } else { 2.0 });
            row[relation::CLAUSE] = (clause + 1) as f32;
            row[relation::ORDINAL] = (ordinal + 1) as f32;
            row[relation::REFERENCE_DEFINITION] = definition(reference);
            row[relation::REFERENCE] = link(reference);
            row[relation::REFERENCE_PLAYER] =
                reference
                    .player
                    .map_or(0.0, |p| if p == viewer { 1.0 } else { 2.0 });
            if let Some(player) = reference.player {
                row[relation::REFERENCE] = if player == viewer { 1.0 } else { 2.0 };
            }
        };
        for (ordinal, target) in frame.targets.iter().enumerate() {
            reference_row(
                2.0,
                target.reference,
                target.clause,
                ordinal,
                target.legal_at_resolution_start,
            );
        }
        if let Some(target) = frame.context {
            reference_row(3.0, target, 0, 0, None);
        }
        if let Some(target) = frame.sacrifice {
            reference_row(4.0, target, 0, 0, None);
        }
        for (ordinal, reference) in frame.effect_execution_references.iter().enumerate() {
            reference_row(10.0, *reference, 0, ordinal, None);
        }
        if let Some(definition) = frame.inherited_from {
            reference_row(7.0, definition, 0, 0, None);
        }
        for (ordinal, reference) in frame.remembered.iter().enumerate() {
            reference_row(8.0, *reference, 0, ordinal, None);
        }
        if let Some(frequency) = frame.delayed_frequency {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header);
            row[relation::KIND] = 9.0;
            row[relation::DECISION] = f32::from(frequency);
        }
        for pair in &frame.hybrid {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header);
            row[relation::KIND] = 5.0;
            row[relation::HYBRID_FIRST] = (pair[0] + 1) as f32;
            row[relation::HYBRID_SECOND] = (pair[1] + 1) as f32;
        }
        for output in &frame.mana_outputs {
            for (restricted, amounts) in [(false, output.mana), (true, output.lesson)] {
                let row = rows.next().unwrap();
                row.copy_from_slice(&header);
                row[relation::KIND] = 6.0;
                row[relation::MODE] = (output.mode + 1) as f32;
                for (index, amount) in amounts.iter().enumerate() {
                    row[relation::MANA_WHITE + index] = *amount as f32;
                }
                row[relation::RESTRICTED] = f32::from(restricted);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ability::DistinctSearchEnd;
    use std::collections::HashMap;

    #[test]
    fn departed_sources_and_stack_targets_never_follow_new_incarnations() {
        use crate::state::player::PlayerConfig;
        let deck = [("Forest".into(), 20), ("Gray Ogre".into(), 20)].into();
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck),
                PlayerConfig::new("b", [("Forest".into(), 40)].into()),
            ],
            23,
            false,
        );
        let permanent = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .unwrap();
        let reference = game.permanent_object_ref(permanent).unwrap();
        assert!(game
            .visible_reference(reference, PlayerId(0))
            .object
            .is_some());
        let object_id = ObjectId::from(reference.entity);
        game.move_object(object_id, ZoneType::Hand);
        assert!(game
            .visible_reference(reference, PlayerId(0))
            .object
            .is_none());
        assert!(game
            .target_reference(Target::StackSpell(object_id), PlayerId(0))
            .object
            .is_none());
        assert!(game
            .object_reference(object_id, PlayerId(1))
            .object
            .is_none());
    }

    #[test]
    fn mana_output_is_a_child_of_the_payment_it_interrupts() {
        use crate::{
            flow::decision::PaymentProgress,
            state::{mana::Mana, player::PlayerConfig},
        };
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", [("Forest".into(), 40)].into()),
                PlayerConfig::new("b", [("Forest".into(), 40)].into()),
            ],
            23,
            false,
        );
        let actor = game.agent_player();
        game.suspend_rule_decision(Decision::ManaChoice {
            player: actor,
            choices: vec![Mana::parse("U")],
            resume: Some(Box::new(Decision::Payment(PaymentProgress::new(
                actor,
                PaymentCost::Waterbend { generic: 2 },
                vec![],
                vec![],
            )))),
        });
        let frames = game.continuation_frames(actor);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1].decision, 1);
        assert_eq!(frames[1].cost.unwrap()[6], 2);
        assert_eq!(frames[1].parent, Some(0));
        assert_eq!(frames[2].decision, 3);
        assert_eq!(frames[2].parent, Some(1));
        assert_eq!(frames[2].mana_outputs[0].mana, [0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn every_decision_code_is_within_the_validated_bound() {
        use crate::{
            flow::decision::{PaymentProgress, ResolutionCastOffer},
            state::{
                ability::LinkedGrant, game_object::PermanentId, permanent::CounterKind,
                predicate::CardPredicate,
            },
        };
        let player = PlayerId(0);
        let predicate = CardPredicate::default();
        // One value per variant. `decision_code`'s exhaustive match makes a
        // new variant fail to compile until it has a code; add it here too.
        let decisions = [
            Decision::OrderReplacement {
                player: PlayerId(0),
                candidates: Vec::new(),
            },
            Decision::ChoosePaymentX {
                player,
                if_paid: Vec::new(),
            },
            Decision::SearchNamed {
                player,
                owner: player,
                name: "Gray Ogre".into(),
                remaining: 4,
            },
            Decision::Payment(PaymentProgress::new(
                player,
                PaymentCost::Waterbend { generic: 1 },
                vec![],
                vec![],
            )),
            Decision::ManaUpfront {
                player,
                permanent: PermanentId(0),
                ability_index: 0,
                tap_remaining: 0,
                resume: None,
            },
            Decision::ManaChoice {
                player,
                choices: vec![],
                resume: None,
            },
            Decision::ChooseObject {
                player,
                predicate: predicate.clone(),
                excluded: None,
                optional: false,
                effect: Box::new(crate::state::ability::Effect::tap(
                    crate::state::ability::ObjectSelector::chosen(predicate.clone()),
                )),
            },
            Decision::DistributeCounters {
                player,
                kind: CounterKind::PlusOnePlusOne,
                remaining: 1,
            },
            Decision::SelectCards {
                origin: crate::flow::decision::SelectionOrigin::Other,
                player,
                from: None,
                candidates: vec![],
                selected: vec![],
                min: 0,
                max: 1,
                effects: vec![],
                if_declined: vec![],
            },
            Decision::Scry {
                player,
                looked: vec![],
                remaining: vec![],
                top: vec![],
                bottom: vec![],
            },
            Decision::Surveil {
                player,
                looked: vec![],
                graveyard: vec![],
            },
            Decision::OrderLibrary {
                player,
                object_ids: vec![],
                next: 0,
                end: LibraryEnd::Bottom,
            },
            Decision::OrderLibrary {
                player,
                object_ids: vec![],
                next: 0,
                end: LibraryEnd::Top,
            },
            Decision::LookAndSelect {
                to_battlefield: false,
                random_order: true,
                reveal: false,
                player,
                looked: vec![],
                predicate: predicate.clone(),
                selected: 0,
                min_select: 0,
                max_select: 1,
            },
            Decision::SearchDistinctNames {
                player,
                predicate: predicate.clone(),
                remaining: 1,
                selected: vec![],
                end: DistinctSearchEnd::OpponentSplits { picks: 1 },
            },
            Decision::DiscardFromRevealedHand {
                player,
                owner: player,
                predicate: predicate.clone(),
            },
            Decision::OpponentSplitsRevealed {
                player,
                owner: player,
                revealed: vec![],
                remaining: 1,
            },
            Decision::CastDuringResolution {
                player,
                offer: ResolutionCastOffer::Hand {
                    predicate: predicate.clone(),
                },
                paid: false,
                spends: None,
            },
            Decision::ExileLinked {
                player,
                owner: player,
                zone: ZoneType::Graveyard,
                predicate,
                up_to: true,
                grant: LinkedGrant::FlashAnyMana,
                grantee: player,
                source: None,
            },
            Decision::Modal {
                player,
                modes: vec![],
            },
            Decision::Learn { player },
            Decision::Discard {
                player,
                remaining: 1,
            },
            Decision::KeepCreatures {
                player,
                max_total_power: 0,
                kept: vec![],
            },
            Decision::ChooseToExile {
                player,
                controller: player,
                predicate: CardPredicate::default(),
                chosen: vec![],
            },
            Decision::Retarget {
                player,
                object: Target::Player(player),
                slot: 0,
                may_keep: false,
            },
            Decision::LegendRule {
                player,
                groups: vec![],
                keep: vec![],
            },
            Decision::EnterAsCopy {
                player,
                object_id: ObjectId(0),
                candidates: vec![],
            },
        ];
        let codes: Vec<u8> = decisions.iter().map(decision_code).collect();
        assert!(codes
            .iter()
            .all(|code| (1..=MAX_DECISION_CODE).contains(code)));
        assert_eq!(codes.iter().max(), Some(&MAX_DECISION_CODE));
    }

    #[test]
    fn relation_capacity_rejects_instead_of_truncating() {
        let frames = vec![ContinuationFrame::default(); RELATION_ROWS + 1];
        let mut out = vec![0.0; RELATION_ROWS * RELATION_DIM];
        assert!(matches!(
            encode_into(&frames, 0, &HashMap::new(), &mut out),
            Err(super::super::observation_encoder::ObservationEncodeError::Capacity { .. })
        ));
    }

    #[test]
    fn target_order_clauses_and_payment_are_preserved() {
        let mut frames = vec![ContinuationFrame {
            kind: 4,
            cost: Some([0, 1, 0, 0, 0, 0, 3]),
            hybrid: vec![[0, 2]],
            lesson: true,
            targets: vec![
                ClauseTarget {
                    reference: VisibleReference {
                        player: Some(0),
                        ..VisibleReference::default()
                    },
                    clause: 0,
                    legal_at_resolution_start: None,
                },
                ClauseTarget {
                    reference: VisibleReference {
                        player: Some(1),
                        ..VisibleReference::default()
                    },
                    clause: 1,
                    legal_at_resolution_start: None,
                },
            ],
            ..ContinuationFrame::default()
        }];
        let mut left = vec![0.0; RELATION_ROWS * RELATION_DIM];
        encode_into(&frames, 0, &HashMap::new(), &mut left).unwrap();
        frames[0].targets.reverse();
        let mut right = left.clone();
        encode_into(&frames, 0, &HashMap::new(), &mut right).unwrap();
        assert_ne!(left, right);
        assert_eq!(&left[12..19], &[0., 1., 0., 0., 0., 0., 3.]);
        assert_eq!(left[19], 1.);
        assert_eq!(
            &left[3 * RELATION_DIM + 26..3 * RELATION_DIM + 28],
            &[1., 3.]
        );
        encode_into(&[], 0, &HashMap::new(), &mut right).unwrap();
        assert!(right.iter().all(|value| *value == 0.));
    }
}
