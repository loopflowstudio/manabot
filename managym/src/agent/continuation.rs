//! Current stack and unfinished decisions, projected once from rules authority.
//! References link only to exact, visible incarnations. Tensor rows and portable
//! text evidence consume these same facts; no strings or JSON enter encoding.

use serde::{Deserialize, Serialize};

use crate::{
    agent::observation::{KeywordData, KEYWORD_COUNT},
    flow::{
        decision::{Decision, SuspendedResolution},
        game::{ActivationSource, Game, Payment, PendingChoice},
        trigger::{DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        ability::{DelayedExpiry, DelayedFrequency, PaymentCost},
        game_object::{CardId, ObjectRef, PlayerId, Target},
        mana::ManaCost,
        stack_object::StackObject,
        zone::ZoneType,
    },
};

/// Policy inputs v2/v3/v4/v5: up to 64 relation rows, 32/33/49 floats per row.
/// Overflow is an admission error, never truncation. Zero is padding/absence
/// for selectors. Validity stays at column 31 in every width.
pub const RELATION_ROWS: usize = 64;
pub const LEGACY_RELATION_DIM: usize = 32;
pub const RELATION_DIM: usize = 33;
/// V5 appends the frame's X (column 33), its granted keywords (34..48, in
/// `KeywordData` field order) and a waiting delayed trigger's "this turn"
/// (48) after v4's eligibility column.
pub const RELATION_DIM_V5: usize = 49;
pub const RELATION_X_INDEX: usize = 33;
pub const RELATION_KEYWORDS_INDEX: usize = 34;
pub const RELATION_THIS_TURN_INDEX: usize = RELATION_KEYWORDS_INDEX + KEYWORD_COUNT;

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
    /// V5/v6 encode it as a kind-9 relation row with the code in column 28.
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
        self.lesson = !cost.purpose.is_ordinary();
    }
}

impl Game {
    /// Whether retained inputs can recover an activation from catalog joins.
    /// Execution always consumes the captured effects, regardless of this answer.
    pub(crate) fn activation_catalog_program(
        &self,
        card: CardId,
        face: crate::state::card::CardDefId,
        inherited: Option<crate::state::card::CardDefId>,
        index: usize,
        grant: Option<crate::state::card::AbilityGrant>,
        effects: &[crate::state::ability::Effect],
    ) -> bool {
        let definition = if let Some(grant) = grant {
            self.state.cards[CardId::from(grant.grantor.entity)]
                .attached_grants()
                .nth(grant.index)
        } else {
            self.source_face(card, face).and_then(|source| {
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
        let card_id = CardId::from(object.entity);
        if self.current_object_ref(card_id) != Some(object) {
            return VisibleReference {
                definition: self.object_lki(object).map(|lki| lki.definition_id.0),
                ..VisibleReference::default()
            };
        }
        let card = &self.state.cards[card_id];
        let zone = self.state.zones.zone_of(card_id);
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
            && self.shares_information(viewer, card.owner))
            || (matches!(zone, Some(ZoneType::Hand | ZoneType::Library))
                && self.searches_private_zones(viewer, card.owner));
        if !visible || (self.is_foretold(card_id) && !self.shares_information(viewer, card.owner)) {
            return VisibleReference::default();
        }
        VisibleReference {
            object: Some(
                self.lookup_current_permanent(object)
                    .ok()
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(card.id.0, |p| p.id.0) as i32,
            ),
            definition: Some(card.definition_id.0),
            player: None,
        }
    }

    fn card_reference(&self, card: CardId, viewer: PlayerId) -> VisibleReference {
        self.current_object_ref(card)
            .map(|object| self.visible_reference(object, viewer))
            .unwrap_or_default()
    }

    fn activation_reference(&self, source: ActivationSource, viewer: PlayerId) -> VisibleReference {
        match source {
            ActivationSource::Permanent(permanent) => {
                self.target_reference(Target::Permanent(permanent), viewer)
            }
            ActivationSource::Card(card) => self.card_reference(card, viewer),
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
            Target::StackSpell(card) => {
                if self.state.zones.zone_of(card) == Some(ZoneType::Stack) {
                    self.card_reference(card, viewer)
                } else {
                    VisibleReference::default()
                }
            }
            Target::StackAbility(id) => self
                .stack_ability_source(id)
                .map(|card| self.card_reference(card, viewer))
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
        source.definition = Some(self.state.cards[trigger.source_card].definition_id.0);
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
        objects: &[ObjectRef],
        viewer: PlayerId,
    ) -> Vec<VisibleReference> {
        objects
            .iter()
            .map(|object| self.visible_reference(*object, viewer))
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
                    self.state.cards[trigger.source_card].definition_id.0,
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
                frame.source = self.card_reference(spell.card, viewer);
                frame.mode = spell.mode;
                frame.kicked = spell.kicked;
                frame.x = spell.x;
                frame.granted_keywords =
                    KeywordData::from_keywords(&self.spell_keywords(spell.card, spell.controller))
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
                &[]
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
            PendingChoice::SacrificePayment { payment, .. } => {
                self.payment_frame(payment, viewer, &mut frame)
            }
            PendingChoice::ChooseTargets {
                card,
                cast,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                ..
            } => {
                frame.source = self.card_reference(*card, viewer);
                frame.kicked = cast.kicked;
                frame.x = cast.x;
                frame.mode = *mode;
                frame.active_clause = Some(*requirement_index);
                frame.targets = self.clause_targets(chosen, chosen_req_indices, viewer);
            }
            PendingChoice::SpellMode { card, cast, .. } => {
                frame.source = self.card_reference(*card, viewer);
                frame.kicked = cast.kicked;
                frame.x = cast.x;
            }
            PendingChoice::KickerChoice { card, .. } | PendingChoice::AnnounceX { card, .. } => {
                frame.source = self.card_reference(*card, viewer)
            }
            PendingChoice::ActivatedTarget {
                source,
                ability_index,
                chosen,
                ..
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
                frame.active_clause = Some(0);
                frame.targets = self.clause_targets(chosen, &[], viewer);
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
        frame
    }

    fn append_suspended_frames(
        &self,
        suspended: &SuspendedResolution,
        viewer: PlayerId,
        frames: &mut Vec<ContinuationFrame>,
    ) {
        let effect = &suspended.frame;
        let parent = if self.pending_choice.is_some() {
            Some(frames.len() - 1)
        } else {
            None
        };
        let mut frame = ContinuationFrame {
            kind: 5,
            parent,
            controller: effect.controller.0,
            source: effect
                .source_ref
                .map(|r| self.visible_reference(r, viewer))
                .unwrap_or_else(|| {
                    effect
                        .source
                        .map(|card| self.card_reference(card, viewer))
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
        // The native chain starts at the active choice and follows resume
        // obligations. Reverse its projection so each interrupted obligation
        // precedes its child and the current decision is the deepest frame.
        frames.push(frame);
        let first_decision = frames.len();
        let mut decision = &suspended.decision;
        loop {
            let mut frame = ContinuationFrame {
                kind: 6,
                controller: decision.player().0,
                ..ContinuationFrame::default()
            };
            frame.decision = decision_code(decision);
            let resume = match decision {
                Decision::ChoosePaymentX { .. } => {
                    frame.can_decline = true;
                    None
                }
                Decision::SearchNamed {
                    owner, remaining, ..
                } => {
                    frame.can_decline = true;
                    frame.mode = Some(*remaining);
                    frame.context = Some(self.target_reference(Target::Player(*owner), viewer));
                    None
                }
                Decision::Payment(progress) => {
                    frame.cost(&progress.remaining.mana_cost());
                    frame.waterbend = matches!(progress.remaining, PaymentCost::Waterbend { .. });
                    frame.can_decline = progress.can_decline();
                    None
                }
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    resume,
                    ..
                } => {
                    frame.source = self.target_reference(Target::Permanent(*permanent), viewer);
                    frame.ability = Some(*ability_index);
                    frame.cost(
                        &self.characteristics(*permanent).activated_abilities[*ability_index]
                            .mana_cost,
                    );
                    resume.as_deref()
                }
                Decision::ManaChoice {
                    choices, resume, ..
                } => {
                    frame.mana_outputs = choices
                        .iter()
                        .enumerate()
                        .map(|(mode, output)| ManaOutput {
                            mode,
                            mana: output.mana,
                            lesson: output.restricted_colors(),
                        })
                        .collect();
                    resume.as_deref()
                }
                Decision::ChooseObject { optional, .. } => {
                    frame.can_decline = *optional;
                    None
                }
                _ => None,
            };
            frames.push(frame);
            let Some(next) = resume else {
                break;
            };
            decision = next;
        }
        frames[first_decision..].reverse();
        for (index, frame) in frames.iter_mut().enumerate().skip(first_decision) {
            frame.parent = Some(index - 1);
        }
    }

    fn payment_frame(&self, payment: &Payment, viewer: PlayerId, frame: &mut ContinuationFrame) {
        match payment {
            Payment::Spell {
                card,
                cast,
                mode,
                targets,
                target_req_indices,
                ..
            } => {
                frame.source = self.card_reference(*card, viewer);
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
pub const MAX_DECISION_CODE: u8 = 26;

/// A suspended obligation's stable code in `ContinuationFrame::decision`
/// (1..=`MAX_DECISION_CODE`). Codes are part of retained captures: append
/// new ones, never renumber.
pub fn decision_code(decision: &Decision) -> u8 {
    match decision {
        Decision::ChoosePaymentX { .. } => 24,
        Decision::SearchNamed { .. } => 25,
        Decision::Payment(_) => 1,
        Decision::ManaUpfront { .. } => 2,
        Decision::ManaChoice { .. } => 3,
        Decision::ChooseObject { .. } => 4,
        Decision::SelectMilled { .. } => 5,
        Decision::SearchBasicLand { .. } | Decision::SearchLibraryToHand { .. } => 6,
        Decision::OptionalDiscardDraw { .. } => 7,
        Decision::Scry { .. } => 8,
        Decision::LookAndSelect { .. } => 9,
        Decision::Modal { .. } => 10,
        Decision::Learn { .. } => 11,
        Decision::Discard { .. } => 12,
        Decision::LegendRule { .. } => 13,
        Decision::SearchDistinctLands { .. } => 14,
        Decision::OpponentSplitsRevealed { .. } => 15,
        Decision::DiscardFromRevealedHand { .. } => 16,
        Decision::CastDuringResolution { .. } => 17,
        Decision::ExileLinked { .. } => 18,
        Decision::DistributeCounters { .. } => 19,
        Decision::KeepCreatures { .. } => 20,
        Decision::Retarget { .. } => 21,
        Decision::Surveil { .. } => 22,
        Decision::OrderLibraryBottom { .. } => 23,
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
    let dim = out.len() / RELATION_ROWS;
    if ![LEGACY_RELATION_DIM, RELATION_DIM, RELATION_DIM_V5].contains(&dim)
        || out.len() != RELATION_ROWS * dim
    {
        return Err(ObservationEncodeError::InvalidLength {
            field: "relations",
            expected: RELATION_ROWS * RELATION_DIM_V5,
            actual: out.len(),
        });
    }
    // Delayed triggers and scheduling are v5 facts; frozen v2-v4 inputs keep their
    // rows. They follow every other frame, so no index moves.
    let frames = if dim == RELATION_DIM_V5 {
        frames
    } else {
        let shown = frames
            .iter()
            .position(|f| f.kind >= 10)
            .unwrap_or(frames.len());
        &frames[..shown]
    };
    let count: usize = frames
        .iter()
        .map(|f| {
            1 + f.targets.len()
                + usize::from(f.context.is_some())
                + usize::from(f.sacrifice.is_some())
                + if dim == RELATION_DIM_V5 {
                    usize::from(f.inherited_from.is_some())
                        + f.remembered.len()
                        + usize::from(f.delayed_frequency.is_some())
                } else {
                    0
                }
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
    let mut rows = out.chunks_exact_mut(dim);
    let link = |reference: VisibleReference| {
        reference
            .object
            .and_then(|id| objects.get(&id))
            .map_or(0.0, |index| (index + 1) as f32)
    };
    let definition =
        |reference: VisibleReference| reference.definition.map_or(0.0, |id| (id + 1) as f32);
    for (index, frame) in frames.iter().enumerate() {
        let mut header = [0.0; RELATION_DIM_V5];
        header[0] = 1.0;
        header[1] = (index + 1) as f32;
        header[2] = frame.parent.map_or(0.0, |p| (p + 1) as f32);
        header[3] = frame.kind as f32;
        header[4] = if frame.controller == viewer { 1.0 } else { 2.0 };
        header[5] = definition(frame.source);
        header[6] = link(frame.source);
        header[7] = frame.mode.map_or(0.0, |v| (v + 1) as f32);
        header[8] = frame.ability.map_or(0.0, |v| (v + 1) as f32);
        header[9] = f32::from(frame.kicked);
        header[10] = frame.active_clause.map_or(0.0, |v| (v + 1) as f32);
        header[11] = f32::from(frame.cost.is_some());
        if let Some(cost) = frame.cost {
            for (i, amount) in cost.iter().enumerate() {
                header[12 + i] = *amount as f32;
            }
        }
        header[19] = f32::from(frame.lesson);
        header[20] = f32::from(frame.waterbend);
        header[21] = f32::from(frame.can_decline);
        header[28] = frame.decision as f32;
        header[30] = frame.resolutions_this_turn as f32;
        header[31] = 1.0;
        header[RELATION_X_INDEX] = f32::from(frame.x);
        for (index, granted) in frame.granted_keywords.iter().enumerate() {
            header[RELATION_KEYWORDS_INDEX + index] = f32::from(*granted);
        }
        header[RELATION_THIS_TURN_INDEX] = f32::from(frame.this_turn);
        rows.next().unwrap().copy_from_slice(&header[..dim]);
        let mut reference_row = |kind: f32,
                                 reference: VisibleReference,
                                 clause: usize,
                                 ordinal: usize,
                                 legal: Option<bool>| {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header[..dim]);
            row[0] = kind;
            if dim >= RELATION_DIM {
                row[32] = legal.map_or(0.0, |value| if value { 1.0 } else { 2.0 });
            }
            row[22] = (clause + 1) as f32;
            row[29] = (ordinal + 1) as f32;
            row[23] = definition(reference);
            row[24] = link(reference);
            row[25] = reference
                .player
                .map_or(0.0, |p| if p == viewer { 1.0 } else { 2.0 });
            if let Some(player) = reference.player {
                row[24] = if player == viewer { 1.0 } else { 2.0 };
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
        if dim == RELATION_DIM_V5 {
            if let Some(definition) = frame.inherited_from {
                reference_row(7.0, definition, 0, 0, None);
            }
            for (ordinal, reference) in frame.remembered.iter().enumerate() {
                reference_row(8.0, *reference, 0, ordinal, None);
            }
            if let Some(frequency) = frame.delayed_frequency {
                let row = rows.next().unwrap();
                row.copy_from_slice(&header);
                row[0] = 9.0;
                row[28] = f32::from(frequency);
            }
        }
        for pair in &frame.hybrid {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header[..dim]);
            row[0] = 5.0;
            row[26] = (pair[0] + 1) as f32;
            row[27] = (pair[1] + 1) as f32;
        }
        for output in &frame.mana_outputs {
            for (restricted, amounts) in [(false, output.mana), (true, output.lesson)] {
                let row = rows.next().unwrap();
                row.copy_from_slice(&header[..dim]);
                row[0] = 6.0;
                row[7] = (output.mode + 1) as f32;
                for (index, amount) in amounts.iter().enumerate() {
                    row[12 + index] = *amount as f32;
                }
                row[19] = f32::from(restricted);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let card = CardId::from(reference.entity);
        game.move_card(card, ZoneType::Hand);
        assert!(game
            .visible_reference(reference, PlayerId(0))
            .object
            .is_none());
        assert!(game
            .target_reference(Target::StackSpell(card), PlayerId(0))
            .object
            .is_none());
        assert!(game.card_reference(card, PlayerId(1)).object.is_none());
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
            Decision::SelectMilled {
                player,
                cards: vec![],
                if_declined: vec![],
                to_battlefield: false,
            },
            Decision::SearchBasicLand {
                player,
                subtype: None,
                remaining: 1,
                tapped: true,
                untap_if_lands: None,
                bottom_source: false,
                to_hand: false,
                selected: vec![],
            },
            Decision::SearchLibraryToHand {
                player,
                predicate: predicate.clone(),
            },
            Decision::OptionalDiscardDraw {
                player,
                count: 1,
                remaining: 1,
                discarded: 0,
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
            },
            Decision::OrderLibraryBottom {
                player,
                remaining: vec![],
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
            Decision::SearchDistinctLands {
                player,
                remaining: 1,
                selected: vec![],
                opponent_picks: 1,
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
                then: None,
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
