//! Viewer-safe effect execution, derived from the native frame and event owners.
//!
//! One exhaustive walk over the suspended caller chain writes the policy
//! occurrence: learning tokens carrying rules payloads and scalar progress, with
//! object identities replaced by slots in an explicit visible-reference table
//! that v8 relation rows join to the same objects legal actions use. The same
//! walk renders the complete text and plain-language summaries, but only when
//! a decision context or capture asks for them: every policy observation runs
//! it without strings. Neither raw game serialization nor storage IDs enter
//! the policy or text view.

use serde::Serialize;

use crate::{
    agent::{
        ability_description::effects_description,
        continuation::{ContinuationFrame, VisibleReference},
    },
    cards::{content_pack::default_content_pack, tokens::TokenWriter},
    flow::{
        decision::{Decision, EffectFrame, FrameCompletion, SelectionOrigin},
        event::GameEvent,
        game::{ActivationSource, Game, Payment},
        proposed_event::{
            EventBatch, EventBinding, ProposedDamageTarget, ProposedEvent, TokenDefinition,
            ZoneEntry,
        },
    },
    state::{
        ability::Effect,
        game_object::{CardId, ObjectLki, ObjectRef, PlayerId, Target},
    },
};

/// The text products of a walk: every fact in `name: value` form for the LLM
/// view, and the plain-language pending summaries Academy shows.
#[derive(Default)]
struct Text {
    notes: Vec<String>,
    summary: Vec<String>,
}

struct Projection<'a> {
    game: &'a Game,
    viewer: PlayerId,
    tokens: TokenWriter<'static>,
    /// Whether any retained fact has been written. Nothing retained means no
    /// execution occurrence at all: no marker, holder or relation row.
    started: bool,
    references: Vec<VisibleReference>,
    text: Option<Text>,
    last_known: Vec<ObjectRef>,
}

impl<'a> Projection<'a> {
    fn new(game: &'a Game, viewer: PlayerId, text: bool) -> Self {
        Self {
            game,
            viewer,
            tokens: TokenWriter::new(default_content_pack()),
            started: false,
            references: Vec::new(),
            text: text.then(Text::default),
            last_known: Vec::new(),
        }
    }

    fn note(&mut self, line: impl FnOnce() -> String) {
        if let Some(text) = &mut self.text {
            text.notes.push(line());
        }
    }

    fn summary(&mut self, line: impl FnOnce() -> String) {
        if let Some(text) = &mut self.text {
            text.summary.push(line());
        }
    }

    /// One retained fact: its name token, then the rules value itself. Serde
    /// values delimit themselves, so no tuple wrapper is needed. The first
    /// fact opens the occurrence with a marker that separates it from the
    /// frame's captured program, which shares the fact id.
    fn field<T: Serialize + std::fmt::Debug>(&mut self, name: &str, value: &T) {
        if !self.started {
            self.started = true;
            self.field("effect execution", &true);
        }
        self.note(|| format!("{name}: {value:?}"));
        self.tokens
            .append_rules(&name)
            .and_then(|_| self.tokens.append_rules(value))
            .expect("projected rules tokens");
    }

    fn reference_slot(&mut self, reference: VisibleReference) -> usize {
        self.references
            .iter()
            .position(|existing| *existing == reference)
            .map(|index| index + 1)
            .unwrap_or_else(|| {
                self.references.push(reference);
                self.references.len()
            })
    }

    fn reference(&mut self, name: &str, reference: VisibleReference) {
        let slot = self.reference_slot(reference);
        self.field(name, &slot);
        self.note(|| format!("{name}: E{slot}"));
    }

    fn label(reference: VisibleReference) -> String {
        if let Some(player) = reference.player {
            return format!("Seat {}", player + 1);
        }
        reference
            .definition
            .and_then(|id| {
                default_content_pack()
                    .definition(crate::state::card::CardDefId(id))
                    .map(|definition| definition.name.clone())
            })
            .unwrap_or_else(|| "an unrevealed object".into())
    }

    fn object(&mut self, name: &str, object: ObjectRef) {
        self.reference(name, self.game.visible_reference(object, self.viewer));
        self.retained_object(object);
    }

    fn retained_object(&mut self, object: ObjectRef) {
        if let Some(lki) = self.game.object_lki(object) {
            self.lki(lki);
        }
    }

    fn retained_target(&mut self, target: Target) {
        if let Target::Permanent(id) = target {
            if let Some(lki) = self.game.departed_permanent_lki(id) {
                self.lki(lki);
            }
        }
    }

    fn card(&mut self, name: &str, card: CardId) {
        self.reference(name, self.game.card_reference(card, self.viewer));
    }

    fn target(&mut self, name: &str, target: Target) {
        self.reference(name, self.game.target_reference(target, self.viewer));
    }

    fn objects(&mut self, name: &str, objects: &[ObjectRef]) {
        if objects.is_empty() {
            return;
        }
        let (game, viewer) = (self.game, self.viewer);
        self.summary(|| {
            format!(
                "{name}: {}.",
                objects
                    .iter()
                    .map(|object| Self::label(game.visible_reference(*object, viewer)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });
        self.field(name, &objects.len());
        for object in objects {
            self.object(name, *object);
        }
    }

    fn cards(&mut self, name: &str, cards: &[CardId]) {
        if cards.is_empty() {
            return;
        }
        let (game, viewer) = (self.game, self.viewer);
        self.summary(|| {
            format!(
                "{name}: {}.",
                cards
                    .iter()
                    .map(|card| Self::label(game.card_reference(*card, viewer)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });
        self.field(name, &cards.len());
        for card in cards {
            self.card(name, *card);
        }
    }

    fn effects(&mut self, name: &str, effects: &[Effect]) {
        if !effects.is_empty() {
            if let Some(text) = &mut self.text {
                let description = format!("{name}: {}", effects_description(effects));
                text.summary.push(description.clone());
                text.notes.push(description);
            }
            self.field(name, &effects);
        }
    }

    fn finish(mut self, frame: &mut ContinuationFrame) {
        if !self.started {
            return;
        }
        // Fact programs are routed to the controller's player row. Explicitly
        // join that holder even for focus-free choices with no source/targets,
        // so continuation-to-action context can read the retained program.
        self.reference(
            "execution holder",
            VisibleReference {
                player: Some(frame.controller),
                ..Default::default()
            },
        );
        let tokens = self.tokens.into_rules_tokens();
        debug_assert!(
            frame.effect_execution.is_empty() || frame.effect_execution == tokens,
            "the text pass must reproduce the policy occurrence"
        );
        frame.effect_execution = tokens;
        frame.effect_execution_references = self.references;
        if let Some(text) = self.text {
            frame.effect_execution_summary = text.summary;
            frame.effect_execution_description = Some(text.notes.join("; "));
        }
    }

    fn completion(&mut self, completion: &FrameCompletion) {
        // Caller frames themselves appear once in the borrowed root-first walk.
        // Restarting priority is the default return of a root ability and
        // retains nothing; every other destination is a fact.
        match completion {
            FrameCompletion::Priority => {}
            FrameCompletion::Continue => self.field("return", &"turn action"),
            FrameCompletion::CombatNormal => {
                self.summary(|| "Normal combat damage follows these choices.".into());
                self.field("return", &"normal combat damage");
            }
            FrameCompletion::StateBased { next } => {
                self.summary(|| "Finish the pending state-based actions before priority.".into());
                self.field("return", &"finish state-based actions");
                self.completion(next);
            }
            FrameCompletion::Frame(_) => self.field("return", &"caller frame"),
            FrameCompletion::Decision(_) => self.field("return", &"saved decision"),
            FrameCompletion::Event { event, .. } => {
                self.summary(|| {
                    format!("Complete the {:?} event after these choices.", event.kind())
                });
                self.field("return", &"commit completion event");
                // These are precisely the operation events retained by
                // complete_with_event, not arbitrary serialized game events.
                match event {
                    GameEvent::Scried { player }
                    | GameEvent::Surveilled { player }
                    | GameEvent::LibrarySearched { player } => {
                        self.field("event", event);
                        self.note(|| {
                            format!("On completion: {:?} for player {}", event.kind(), player.0)
                        });
                    }
                    _ => unreachable!("only operation completion events are retained"),
                }
            }
            FrameCompletion::Spell { card, attach_to } => {
                self.field("return", &"finish spell");
                self.card("spell", *card);
                if let Some(object) = attach_to {
                    self.object("attach to", *object);
                }
            }
            FrameCompletion::Paid { player, payment } => {
                self.paid_summary();
                self.field("return", &"paid announcement");
                self.field("payer", player);
                match payment {
                    Payment::Spell {
                        card,
                        cast,
                        mode,
                        targets,
                        target_req_indices,
                    } => {
                        self.card("paid spell", *card);
                        self.field("cast choices", cast);
                        self.field("mode", mode);
                        self.field("target clauses", target_req_indices);
                        for target in targets {
                            self.target("target", *target);
                        }
                    }
                    Payment::Activation {
                        source,
                        ability_index,
                        targets,
                        x,
                    } => {
                        match source {
                            ActivationSource::Permanent(id) => {
                                self.target("paid source", Target::Permanent(*id))
                            }
                            ActivationSource::Card(card) => self.card("paid source", *card),
                        }
                        self.field("ability", ability_index);
                        self.field("X", x);
                        for target in targets {
                            self.target("target", *target);
                        }
                    }
                }
            }
            FrameCompletion::Activated {
                ability,
                source_after_cost,
            } => {
                self.paid_summary();
                self.field("return", &"paid activation");
                self.object("activation source", ability.source_ref);
                self.field("controller", &ability.controller);
                self.field("ability", &ability.ability_index);
                self.field("X", &ability.x);
                self.field("source transforms", &ability.source_transforms);
                self.field("inherited definition", &ability.inherited_from);
                self.effects("captured activation", &ability.effects);
                for target in &ability.targets {
                    self.target("target", *target);
                }
                if let Some(object) = ability.attached {
                    self.object("attached at activation", object);
                }
                if let Some(grant) = ability.granted_by {
                    self.object("grantor", grant.grantor);
                    self.field("grant index", &grant.index);
                }
                if let Some(card) = source_after_cost {
                    self.card("source after cost", *card);
                }
            }
        }
    }

    fn paid_summary(&mut self) {
        self.summary(|| {
            "Paid costs stay committed; finish the announcement after these choices.".into()
        });
    }

    fn lki(&mut self, lki: &ObjectLki) {
        if self.last_known.contains(&lki.object_ref) {
            return;
        }
        self.last_known.push(lki.object_ref);
        self.reference(
            "pre-event object",
            self.game.visible_reference(lki.object_ref, self.viewer),
        );
        // While the exact object still has this view, its ordinary object row
        // and catalog already carry the snapshot. After departure or a changed
        // view, retain the non-identity facts explicitly.
        if self.game.snapshot_current_permanent(lki.card).as_ref() == Some(lki) {
            self.field("snapshot", &"current characteristics");
            return;
        }
        let (game, viewer) = (self.game, self.viewer);
        self.summary(|| {
            format!(
                "Last known {}: power {}; counters {:?}.",
                Self::label(game.visible_reference(lki.object_ref, viewer)),
                lki.power,
                lki.counters
            )
        });
        let ObjectLki {
            transforms,
            inherited_from,
            granted_abilities,
            keywords,
            creature_types,
            permanent: _,
            animated_creature,
            abilities_lost,
            dealt_damage_this_turn,
            power,
            counters,
            object_ref: _,
            card: _,
            from_zone,
            owner,
            controller,
            definition_id,
            presentation_id: _,
        } = lki;
        self.field("last known definition", definition_id);
        let counter_values: Vec<_> = counters.iter().collect();
        self.field(
            "last known characteristics",
            &(
                (
                    transforms,
                    inherited_from,
                    granted_abilities,
                    keywords,
                    creature_types,
                ),
                (
                    animated_creature,
                    abilities_lost,
                    dealt_damage_this_turn,
                    power,
                    counter_values,
                    from_zone,
                    owner,
                    controller,
                ),
            ),
        );
    }

    fn entry(&mut self, entry: &ZoneEntry) {
        // ZoneEntry's default terms have no effect. Encode only present terms,
        // preserving the same defaults instead of repeating six zero values
        // for every member of a simultaneous token group.
        if entry.tapped {
            self.field("enters tapped", &true);
        }
        if !entry.counters.is_empty() {
            let counters: Vec<_> = entry.counters.iter().collect();
            self.field("entry counters", &counters);
        }
        if entry.x != 0 {
            self.field("entry X", &entry.x);
        }
        if let Some(controller) = entry.controller {
            self.field("entry controller", &controller);
        }
        if entry.face_down {
            self.field("enters face down", &true);
        }
        if entry.transformed {
            self.field("enters transformed", &true);
        }
    }

    fn batch(&mut self, batch: &EventBatch) {
        self.field("batch", &(batch.cursor, batch.events.len()));
        self.note(|| {
            format!(
                "{} proposed events; {} replacements completed",
                batch.events.len(),
                batch.cursor,
            )
        });
        for pending in &batch.events {
            self.field("event ordinal", &pending.ordinal);
            match &pending.event {
                ProposedEvent::Damage {
                    source,
                    target,
                    amount,
                    combat,
                } => {
                    self.field("damage", &(amount, combat));
                    if let Some(source) = source {
                        self.object("damage source", *source);
                    }
                    match target {
                        ProposedDamageTarget::Player(player) => {
                            self.field("damaged player", player)
                        }
                        ProposedDamageTarget::Object(object) => {
                            self.object("damage target", *object)
                        }
                    }
                    self.note(|| {
                        format!(
                            "Proposed {}damage: {amount}",
                            if *combat { "combat " } else { "" }
                        )
                    });
                }
                ProposedEvent::LifeChange { player, delta } => {
                    self.field("life change", &(player, delta));
                    self.note(|| format!("Player {} life change {delta}", player.0));
                }
                ProposedEvent::CounterChange {
                    provenance,
                    target,
                    kind,
                    delta,
                    by,
                } => {
                    self.object("counter recipient", *target);
                    self.field("counter placement", &(kind, delta, by, provenance));
                    let (game, viewer) = (self.game, self.viewer);
                    self.summary(|| format!("Pending: put {delta} {kind:?} counters on {}. No counters in this group are committed until replacement choices finish.", Self::label(game.visible_reference(*target, viewer))));
                    self.note(|| {
                        format!(
                            "{delta} {kind:?} counters; cause {:?}; modified by a replacement {}",
                            provenance.cause, provenance.replaced
                        )
                    });
                }
                ProposedEvent::ZoneMove {
                    card,
                    object,
                    from,
                    to,
                    entry,
                } => {
                    if let Some(object) = object {
                        self.object("moving object", *object);
                    } else {
                        self.reference(
                            "entering definition",
                            VisibleReference {
                                definition: Some(self.game.state.cards[*card].definition_id.0),
                                ..Default::default()
                            },
                        );
                    }
                    self.field("zone move", &(from, to));
                    self.entry(entry);
                    self.note(|| {
                        format!(
                            "Move {from:?} to {to:?}; tapped {}; entry counters {:?}",
                            entry.tapped, entry.counters
                        )
                    });
                }
                ProposedEvent::CreateTokens {
                    definition,
                    controller,
                    count,
                    entry,
                } => {
                    self.field("create tokens", &(controller, count));
                    self.entry(entry);
                    match definition.as_ref() {
                        TokenDefinition::Registered(name) => {
                            self.reference(
                                "token definition",
                                VisibleReference {
                                    definition: default_content_pack()
                                        .definition_id(name)
                                        .map(|id| id.0),
                                    ..Default::default()
                                },
                            );
                            self.summary(|| format!("Pending: create {count} {name} tokens."));
                        }
                        TokenDefinition::Copy {
                            definition_id,
                            definition,
                            front,
                            copy,
                            exceptions,
                        } => {
                            self.reference(
                                "copy definition",
                                VisibleReference {
                                    definition: Some(definition_id.0),
                                    ..Default::default()
                                },
                            );
                            self.tokens
                                .append_definition(definition)
                                .expect("copiable definition tokens");
                            if let Some((front, _)) = front {
                                self.reference(
                                    "copy front face",
                                    VisibleReference {
                                        definition: Some(front.0),
                                        ..Default::default()
                                    },
                                );
                            }
                            self.field("copy exceptions", &(copy, exceptions));
                            self.note(|| {
                                format!(
                                    "Create {count} token copies of {} with {exceptions:?}",
                                    definition.name
                                )
                            });
                        }
                    }
                }
                ProposedEvent::Destroy { .. } | ProposedEvent::Sacrifice { .. } => {
                    unreachable!("normalized batch")
                }
            }
            if let Some((card, controller)) = pending.sacrifice {
                self.card("sacrificed object", card);
                self.field("sacrificing player", &controller);
            }
            // One ordered rules record per identity, reusing visible source
            // slots. Empty means no replacement has applied; it needs no tokens.
            if !pending.applied.is_empty() {
                let applied: Vec<_> = pending
                    .applied
                    .iter()
                    .map(|identity| {
                        let source = self.reference_slot(
                            self.game.visible_reference(identity.source, self.viewer),
                        );
                        (source, identity.definition_index, identity.static_ability)
                    })
                    .collect();
                self.field("applied replacements", &applied);
            }
        }
        match &batch.binding {
            EventBinding::Moved {
                linked,
                shuffle,
                library_end,
                reveal,
            } => {
                self.field("bind moved", &(shuffle, library_end, reveal));
                if let Some((source, returns)) = linked {
                    self.object("linked source", *source);
                    self.field("linked return", returns);
                }
            }
            // These variants contain only rules values and player identities.
            binding => self.field("bind result", binding),
        }
        self.field("end batch", &true);
    }
}

/// Project a suspended effect frame's retained work. `text` additionally
/// renders its description and summaries; the policy occurrence is identical
/// either way.
pub(crate) fn project_frame(
    game: &Game,
    effect: &EffectFrame,
    viewer: PlayerId,
    frame: &mut ContinuationFrame,
    text: bool,
) {
    let mut out = Projection::new(game, viewer, text);
    // Source/trigger quantities can read departed characteristics after another
    // instruction has suspended. Their definition join alone cannot carry LKI.
    if let Some(source) = effect.source_ref {
        out.retained_object(source);
    }
    if let Some(target) = effect.context_target {
        out.retained_target(target);
    }
    for target in &effect.targets {
        out.retained_target(target.target);
    }
    if !effect.queue.is_empty() {
        out.field("remaining instructions", &effect.queue);
        out.summary(|| {
            let (first, second) = effect.queue.as_slices();
            format!(
                "After this choice: {} {}",
                effects_description(first),
                effects_description(second)
            )
            .trim()
            .to_string()
        });
    }
    out.objects("selected", &effect.selected);
    out.objects("moved", &effect.moved);
    out.objects("this way", &effect.this_way);
    out.objects("created", &effect.created);
    if let Some(source) = &effect.selected_from {
        out.field("selection domain", source);
    }
    if effect.excess_damage != 0 {
        out.field("excess damage", &effect.excess_damage);
    }
    if let Some(player) = effect.triggering_player {
        out.field("triggering player", &player);
    }
    if let Some(ability) = effect.ability {
        out.field("ability", &ability);
    }
    if let Some(object) = effect.chosen {
        out.object("chosen object", object);
    }
    if let Some(object) = effect.grantor {
        out.object("grantor", object);
    }
    if let Some(iteration) = &effect.iteration {
        out.field("remaining iteration subjects", &iteration.remaining.len());
        for target in &iteration.remaining {
            out.retained_target(target.target);
            out.target("remaining target", target.target);
            out.field("target eligibility", &(target.requirement, target.legal));
        }
        out.effects("iteration instructions", &iteration.effects);
    }
    out.completion(&effect.completion);
    if let Some(batch) = &effect.batch {
        out.batch(batch);
    }
    // Ordinary catalog execution keeps its frozen projection: v1-v7 never
    // encoded earlier instructions' results (selected, moved, created), so a
    // later choice in the same program stays admissible. These retained domains
    // have no predecessor representation and must be admitted explicitly.
    frame.effect_execution_requires_v8 = effect.batch.is_some()
        || effect.iteration.is_some()
        || completion_requires_v8(game, &effect.completion);
    out.finish(frame);
}

fn completion_requires_v8(game: &Game, completion: &FrameCompletion) -> bool {
    match completion {
        FrameCompletion::Paid { .. }
        | FrameCompletion::Activated { .. }
        | FrameCompletion::StateBased { .. }
        | FrameCompletion::CombatNormal => true,
        FrameCompletion::Event { event, .. } => {
            // The frozen obligation already names these operation completions.
            // A different saved event under the same current offers (e.g. a
            // nested modal choice) is not implied and needs the v8 payload.
            let decision = game.state.suspended_decision.as_ref().map(|s| &s.decision);
            !matches!((event, decision),
                (GameEvent::Scried { player }, Some(Decision::Scry { player: chooser, .. }))
                | (GameEvent::Surveilled { player }, Some(Decision::Surveil { player: chooser, .. }))
                | (GameEvent::LibrarySearched { player }, Some(Decision::SelectCards {
                    player: chooser, origin: SelectionOrigin::Library, ..
                })) if player == chooser)
        }
        _ => false,
    }
}

/// Project a suspended decision's retained progress; see [`project_frame`].
pub(crate) fn project_decision(
    game: &Game,
    decision: &Decision,
    viewer: PlayerId,
    frame: &mut ContinuationFrame,
    text: bool,
) {
    let mut out = Projection::new(game, viewer, text);
    match decision {
        Decision::SelectCards {
            selected,
            min,
            max,
            effects,
            if_declined,
            ..
        } => {
            out.objects("selected so far", selected);
            out.field("selection limits", &(min, max));
            out.summary(|| {
                format!(
                    "Selected {} of {min}–{max} permitted cards.",
                    selected.len()
                )
            });
            out.effects("after selection", effects);
            out.effects("if none selected", if_declined);
            // A migrated domain keeps its frozen obligation code across picks:
            // its predecessor never encoded the picks already made either.
            frame.effect_execution_requires_v8 = frame.legacy_decision.is_none();
        }
        Decision::OrderReplacement { candidates, .. } => {
            for (index, candidate) in candidates.iter().enumerate() {
                out.field("replacement option", &index);
                out.object("replacement source", candidate.source());
                out.field("replacement", &candidate.effect);
                out.note(|| candidate.description());
            }
            frame.effect_execution_requires_v8 = true;
        }
        Decision::Scry {
            looked,
            remaining,
            top,
            bottom,
            ..
        } => {
            out.cards("inspected", looked);
            out.cards("remaining", remaining);
            out.cards("kept on top", top);
            out.cards("placed on bottom", bottom);
        }
        Decision::Surveil {
            looked, graveyard, ..
        } => {
            out.cards("inspected", looked);
            out.cards("selected for graveyard", graveyard);
        }
        Decision::OrderLibrary {
            cards, next, end, ..
        } => {
            out.cards("library order", cards);
            out.field("ordered prefix", next);
            out.field("library end", end);
        }
        Decision::LookAndSelect {
            random_order,
            reveal,
            to_battlefield,
            looked,
            predicate,
            selected,
            min_select,
            max_select,
            ..
        } => {
            out.cards("inspected", looked);
            out.field("selection", &(predicate, selected, min_select, max_select));
            out.field("placement", &(random_order, reveal, to_battlefield));
        }
        Decision::Payment(progress) => {
            out.field("remaining payment", &progress.remaining);
            out.objects("waterbend contributions", &progress.waterbend_paid);
            out.effects("if paid", &progress.if_paid);
            out.effects("if declined", &progress.if_declined);
            if let Some(key) = &progress.spends {
                out.object("once each turn source", key.0);
                out.field("once each turn ability", &key.1);
            }
            if let Some(target) = progress.counters_on_decline {
                out.target("counter on decline", target);
            }
        }
        Decision::ChoosePaymentX { if_paid, .. } => out.effects("if paid", if_paid),
        Decision::ChooseObject {
            predicate,
            excluded,
            optional,
            effect,
            ..
        } => {
            out.field("choice", &(predicate, optional));
            out.effects("chosen instruction", std::slice::from_ref(effect));
            if let Some(id) = excluded {
                out.target("excluded", Target::Permanent(*id));
            }
        }
        Decision::DistributeCounters {
            kind, remaining, ..
        } => out.field("counters remaining", &(kind, remaining)),
        Decision::SearchNamed {
            owner,
            name,
            remaining,
            ..
        } => out.field("named search", &(owner, name, remaining)),
        Decision::SearchDistinctLands {
            remaining,
            selected,
            opponent_picks,
            ..
        } => {
            out.cards("selected lands", selected);
            out.field("land choices remaining", &(remaining, opponent_picks));
        }
        Decision::OpponentSplitsRevealed {
            owner,
            revealed,
            remaining,
            ..
        } => {
            out.cards("revealed lands", revealed);
            out.field("graveyard choices remaining", &(owner, remaining));
        }
        Decision::DiscardFromRevealedHand {
            owner, predicate, ..
        } => out.field("discard from hand", &(owner, predicate)),
        Decision::Modal { modes, .. } => {
            for (index, mode) in modes.iter().enumerate() {
                out.field("mode", &index);
                out.effects("instructions", mode);
            }
        }
        Decision::Discard { remaining, .. } => out.field("discards remaining", remaining),
        Decision::KeepCreatures {
            max_total_power,
            kept,
            ..
        } => {
            out.field("maximum total power", max_total_power);
            for id in kept {
                out.target("kept creature", Target::Permanent(*id));
            }
        }
        Decision::ChooseToExile {
            controller,
            predicate,
            chosen,
            ..
        } => {
            out.field("exile choice", &(controller, predicate));
            for id in chosen {
                out.target("chosen to exile", Target::Permanent(*id));
            }
        }
        Decision::LegendRule { groups, keep, .. } => {
            for group in groups {
                out.cards("legend group", group);
            }
            out.cards("legends kept", keep);
        }
        Decision::Retarget {
            object,
            slot,
            may_keep,
            ..
        } => {
            out.target("retargeted stack object", *object);
            out.field("retarget", &(slot, may_keep));
        }
        Decision::ExileLinked {
            owner,
            zone,
            predicate,
            up_to,
            grant,
            grantee,
            source,
            ..
        } => {
            out.field(
                "linked exile",
                &(owner, zone, predicate, up_to, grant, grantee),
            );
            if let Some(source) = source {
                out.object("linked source", *source);
            }
        }
        Decision::CastDuringResolution {
            offer,
            paid,
            spends,
            ..
        } => {
            use crate::flow::decision::ResolutionCastOffer;
            out.field("pay casting costs", paid);
            match offer {
                ResolutionCastOffer::Hand { predicate } => out.field("cast from hand", predicate),
                ResolutionCastOffer::Exiled { card } => out.card("cast from exile", *card),
                ResolutionCastOffer::LibraryTop { card } => out.card("cast from library", *card),
                ResolutionCastOffer::ExiledWith { source, predicate } => {
                    out.object("exiled with", *source);
                    out.field("cast predicate", predicate);
                }
            }
            if let Some(key) = spends {
                out.object("once each turn source", key.0);
                out.field("once each turn ability", &key.1);
            }
        }
        // Mana output/cost and caller chains are already explicit relation frames;
        // a naming scope is published only through v9's name candidates.
        Decision::ManaUpfront { .. }
        | Decision::ManaChoice { .. }
        | Decision::Learn { .. }
        | Decision::NameCard { .. } => return,
    }
    out.finish(frame);
}
