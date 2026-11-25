//! Viewer-safe effect execution, derived from the native frame and event owners.
//!
//! One exhaustive walk over the suspended caller chain writes the policy
//! occurrence: learning tokens carrying rules payloads and scalar progress, with
//! object identities replaced by slots in an explicit visible-reference table
//! that relation rows join to the same objects legal actions use. The same
//! walk renders the complete text and plain-language summaries, but only when
//! a decision context or capture asks for them: every policy observation runs
//! it without strings. Neither raw game serialization nor storage IDs enter
//! the policy or text view.

use serde::Serialize;

use crate::{
    agent::{
        ability_description::{effects_description, mana_description},
        continuation::{ContinuationFrame, VisibleReference},
    },
    cards::{content_pack::default_content_pack, tokens::TokenWriter},
    flow::{
        decision::{Decision, EffectFrame, FrameCompletion, SpendCompletion},
        event::GameEvent,
        game::{ActivationSource, CastChoices, Game},
        payment::{CostComponent, Operation, Payment},
        proposed_event::{
            EventBatch, EventBinding, ProposedDamageTarget, ProposedEvent, TokenSource, ZoneEntry,
        },
    },
    state::{
        ability::{Ability, Effect},
        card::CardDefId,
        game_object::{ObjectId, ObjectLki, PlayerId, Target},
        mana::Mana,
        payment::{Contribution, How, PaymentReceipt, SpentMana, Toward},
        stack_object::SpellLki,
    },
};

/// Names emitted by the retained-execution projection, including its return points.
/// Keep this vocabulary beside the writer: fresh catalog policies reject unknown tokens.
pub(crate) const EXECUTION_TEXT: &[&str] = &[
    "X",
    "ability",
    "activation source",
    "after selection",
    "applied replacements",
    "attach to",
    "batch",
    "bind moved",
    "bind result",
    "caller frame",
    "captured activation",
    "cascade exiled",
    "cascade mana value limit",
    "spell mana value limit",
    "cast choices",
    "quota type",
    "cast from exile",
    "cast from hand",
    "cast from graveyard",
    "exile instants and sorceries",
    "cast from library",
    "cast predicate",
    "choice",
    "chosen instruction",
    "chosen object",
    "chosen to exile",
    "commit completion event",
    "contribution",
    "controller",
    "copy definition",
    "copy exceptions",
    "copy front face",
    "counter on decline",
    "counter placement",
    "counter recipient",
    "counters remaining",
    "create tokens",
    "created",
    "current characteristics",
    "damage",
    "damage source",
    "damage target",
    "damaged player",
    "discard from hand",
    "discards remaining",
    "distinct name search",
    "effect execution",
    "end batch",
    "end the turn",
    "entering card",
    "entering definition",
    "enters face down",
    "enters tapped",
    "enters transformed",
    "entry X",
    "entry controller",
    "entry counters",
    "event",
    "event ordinal",
    "excess damage",
    "excluded",
    "execution holder",
    "exile choice",
    "exiled with",
    "finish spell",
    "finish state-based actions",
    "grant index",
    "grantor",
    "graveyard choices remaining",
    "if declined",
    "if none selected",
    "if paid",
    "inherited definition",
    "inspected",
    "instructions",
    "iteration instructions",
    "kept creature",
    "kept on top",
    "last known attacking",
    "last known characteristics",
    "last known creature",
    "last known in combat",
    "last known land",
    "last known tapped",
    "last known attachment",
    "last known definition",
    "last known spell",
    "last known spell profile",
    "last known kicker",
    "legend group",
    "legends kept",
    "library end",
    "library order",
    "life change",
    "life lost this way",
    "life paid",
    "linked exile",
    "linked return",
    "linked source",
    "mana spent",
    "spending option",
    "cost sacrifice",
    "maximum total power",
    "creatures each player sacrifices",
    "may copy",
    "mode",
    "moved",
    "moving object",
    "named search",
    "normal combat damage",
    "once each turn ability",
    "once each turn source",
    "ordered prefix",
    "paid activation",
    "paid announcement",
    "paid by",
    "paid source",
    "paid spell",
    "pay casting costs",
    "payer",
    "placed on bottom",
    "placement",
    "pre-event object",
    "combat damage source",
    "another damage step",
    "first-strike participant",
    "minimum assignment",
    "last blocker",
    "assigned combat damage",
    "remaining",
    "remaining instructions",
    "remaining iteration subjects",
    "remaining payment",
    "remaining target",
    "replacement",
    "replacement option",
    "replacement source",
    "retarget",
    "retargeted stack object",
    "return",
    "revealed lands",
    "sacrificed object",
    "sacrificing player",
    "saved decision",
    "selected",
    "selected distinct names",
    "selected for graveyard",
    "selected so far",
    "selection",
    "selection domain",
    "selection limits",
    "snapshot",
    "source after cost",
    "source transforms",
    "spell",
    "target",
    "target clauses",
    "target eligibility",
    "this way",
    "token definition",
    "triggering player",
    "turn action",
    "zone move",
];

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
    last_known: Vec<ObjectId>,
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
        debug_assert!(
            EXECUTION_TEXT.contains(&name),
            "unregistered execution field: {name}"
        );
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

    fn object(&mut self, name: &str, object: ObjectId) {
        self.reference(name, self.game.visible_reference(object, self.viewer));
        self.retained_object(object);
    }

    fn retained_object(&mut self, object: ObjectId) {
        if let Some(lki) = self.game.object_lki(object) {
            self.lki(lki);
        }
    }

    /// The same event object follows an occurrence from placement to the stack.
    fn trigger_context(
        &mut self,
        context: Option<Target>,
        source: ObjectId,
        definition: CardDefId,
        index: usize,
        inline: Option<&[Effect]>,
    ) {
        let Some(target) = context else { return };
        if self.game.trigger_reads(
            source,
            definition,
            index,
            inline,
            Effect::reads_triggering_object,
            Ability::reads_triggering_object,
        ) {
            self.retained_target(target);
        }
    }

    fn retained_target(&mut self, target: Target) {
        match target {
            Target::Permanent(id) => {
                if let Some(lki) = self.game.departed_permanent_lki(id) {
                    self.lki(lki);
                }
            }
            // A live spell's stack row already carries these facts.
            Target::StackSpell(object) => {
                if let Some(spell) = self.game.state.spell_lki.get(&object) {
                    self.spell_lki(spell);
                }
            }
            _ => {}
        }
    }

    /// A departed spell as it last existed on the stack (CR 608.2h): what a
    /// trigger reading "that spell" will see after it is countered.
    fn spell_lki(&mut self, spell: &SpellLki) {
        if self.last_known.contains(&spell.object) {
            return;
        }
        self.last_known.push(spell.object);
        let definition = VisibleReference {
            definition: Some(spell.definition_id.0),
            ..Default::default()
        };
        self.reference("pre-event object", definition);
        self.field("last known definition", &spell.definition_id);
        self.field(
            "last known spell",
            &(spell.mana_value, &spell.colors, spell.spent),
        );
        if !spell.profile.is_front() {
            self.field("last known spell profile", &spell.profile);
        }
        if let Some(copy) = &spell.copiable {
            self.field("X", &copy.spell.x);
            self.field("last known kicker", &copy.spell.kicked);
            self.field("mode", &copy.spell.mode);
            self.field("target clauses", &copy.spell.target_req_indices);
            for target in &copy.spell.targets {
                self.target("target", *target);
            }
        }
        let game = self.game;
        self.summary(|| {
            let name = if spell.profile.is_front() {
                Self::label(definition)
            } else {
                game.recorded_spell_printing(spell.object, spell.profile)
                    .name
                    .clone()
            };
            format!(
                "Last known spell {}: mana value {}, {} color(s), {} mana spent.",
                name,
                spell.mana_value,
                spell.colors.len(),
                spell.spent
            )
        });
    }

    fn object_id(&mut self, name: &str, object_id: ObjectId) {
        self.reference(name, self.game.object_reference(object_id, self.viewer));
    }

    fn target(&mut self, name: &str, target: Target) {
        self.reference(name, self.game.target_reference(target, self.viewer));
    }

    fn objects(&mut self, name: &str, objects: &[ObjectId]) {
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

    fn object_ids(&mut self, name: &str, object_ids: &[ObjectId]) {
        if object_ids.is_empty() {
            return;
        }
        let (game, viewer) = (self.game, self.viewer);
        self.summary(|| {
            format!(
                "{name}: {}.",
                object_ids
                    .iter()
                    .map(|object_id| Self::label(game.object_reference(*object_id, viewer)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        });
        self.field(name, &object_ids.len());
        for object_id in object_ids {
            self.object_id(name, *object_id);
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
            FrameCompletion::EndTurn => {
                self.summary(|| {
                    "The turn is ending. Cleanup follows the state-based actions.".into()
                });
                self.field("return", &"end the turn");
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
            FrameCompletion::Spell {
                object: object_id,
                attach_to,
            } => {
                self.field("return", &"finish spell");
                self.object_id("spell", *object_id);
                if let Some(object) = attach_to {
                    self.object("attach to", *object);
                }
            }
            FrameCompletion::Paid { player, payment } => {
                self.paid_summary();
                // Unfinished: the source's own printed cost is paid last.
                if !payment.receipt.is_empty() {
                    self.receipt(&payment.receipt, None);
                }
                self.field("return", &"paid announcement");
                self.field("payer", player);
                self.payment(payment);
            }
            FrameCompletion::Activated {
                ability,
                source_after_cost,
            } => {
                self.paid_summary();
                self.field("return", &"paid activation");
                self.object("activation source", ability.source_object());
                self.field("controller", &ability.controller);
                self.field("ability", &ability.ability_index);
                self.field("X", &ability.x);
                self.field("source transforms", &ability.source_transforms);
                self.field("inherited definition", &ability.inherited_from);
                self.effects("captured activation", &ability.effects);
                for target in &ability.targets {
                    self.target("target", *target);
                }
                if let Some(grant) = ability.granted_by {
                    self.object("grantor", grant.grantor);
                    self.field("grant index", &grant.index);
                }
                if let Some(object_id) = source_after_cost {
                    self.object_id("source after cost", *object_id);
                }
            }
        }
    }

    fn payment(&mut self, payment: &Payment) {
        match &payment.operation {
            Operation::Spell {
                object,
                cast,
                mode,
                targets,
                target_req_indices,
            } => {
                self.object_id("paid spell", *object);
                // Storage keys identify the chosen permission only within the game.
                // The payment already gives its cost. Project a quota source as a
                // viewer-safe object reference, including when it left to pay a cost.
                self.field(
                    "cast choices",
                    &CastChoices {
                        route: None,
                        ..*cast
                    },
                );
                if let Some(crate::flow::cast_permission::PlayRoute::GraveyardType {
                    source,
                    kind,
                    ..
                }) = cast.route
                {
                    self.object("grantor", source);
                    self.field("quota type", &kind);
                }
                self.field("mode", mode);
                self.field("target clauses", target_req_indices);
                for target in targets {
                    self.target("target", *target);
                }
            }
            Operation::Activation {
                source,
                ability_index,
                targets,
                x,
            } => {
                match source {
                    ActivationSource::Permanent(id) => {
                        self.target("paid source", Target::Permanent(*id))
                    }
                    ActivationSource::Card(card) => {
                        self.object_id("paid source", self.game.state.object_of(*card))
                    }
                }
                self.field("ability", ability_index);
                self.field("X", x);
                for target in targets {
                    self.target("target", *target);
                }
            }
        }
    }

    /// What a payment took, less what the frame of `source` already says.
    /// Its mana is the amounts actually spent to cast (CR 601.2h) and the
    /// entry counters its spend effects add, the two mana facts a rule reads;
    /// which object produced each unit is in the text only and adds no policy
    /// tokens. Life paid is an amount. Each object that paid otherwise is a
    /// reference, with what was done to it and the part of the cost it paid
    /// (`shown_contributions`).
    fn receipt(&mut self, spent: &PaymentReceipt, source: Option<ObjectId>) {
        if !spent.mana().is_empty() {
            self.mana_spent(spent);
        }
        if spent.life() > 0 {
            self.field("life paid", &spent.life());
            self.summary(|| format!("Life paid: {}.", spent.life()));
        }
        let shown: Vec<Contribution> = shown_contributions(spent, source).copied().collect();
        if shown.is_empty() {
            return;
        }
        for paid in &shown {
            self.field("contribution", &(paid.how, paid.toward));
            self.object("paid by", paid.object);
        }
        let (game, viewer) = (self.game, self.viewer);
        self.summary(|| {
            let paid: Vec<_> = shown
                .iter()
                .map(|paid| contribution_phrase(game, viewer, paid))
                .collect();
            format!("Paid with: {}.", paid.join("; "))
        });
    }

    fn mana_spent(&mut self, spent: &PaymentReceipt) {
        let facts = ManaSpent::of(spent);
        self.field("mana spent", &facts);
        self.summary(|| {
            let mana = mana_description(&Mana {
                mana: facts.amounts,
                ..Default::default()
            });
            let sources = spent
                .mana_sources()
                .iter()
                .filter_map(|origin| default_content_pack().definition(origin.definition))
                .map(|definition| definition.printed.name.as_str())
                .collect::<Vec<_>>();
            let mut text = format!("Mana spent to cast: {mana}");
            if !sources.is_empty() {
                text.push_str(&format!(" (from {})", sources.join(", ")));
            }
            if facts.entry_counters > 0 {
                text.push_str(&format!(
                    "; it enters with {} additional +1/+1 counter(s)",
                    facts.entry_counters
                ));
            }
            for keyword in &facts.gains {
                text.push_str(&format!(
                    "; it gains {} until end of turn",
                    keyword.printed()
                ));
            }
            if facts.copies > 0 {
                text.push_str(&format!("; that mana copies it {} time(s)", facts.copies));
            }
            text + "."
        });
    }

    fn paid_summary(&mut self) {
        self.summary(|| {
            "Paid costs stay committed; finish the announcement after these choices.".into()
        });
    }

    fn lki(&mut self, lki: &ObjectLki) {
        if self.last_known.contains(&lki.object) {
            return;
        }
        self.last_known.push(lki.object);
        self.reference(
            "pre-event object",
            self.game.visible_reference(lki.object, self.viewer),
        );
        // While the exact object still has this view, its ordinary object row
        // and catalog already carry the snapshot. After departure or a changed
        // view, retain the non-identity facts explicitly.
        if self.game.snapshot_current_permanent(lki.object).as_ref() == Some(lki) {
            self.field("snapshot", &"current characteristics");
            return;
        }
        let (game, viewer) = (self.game, self.viewer);
        self.summary(|| {
            format!(
                "Last known {}{}: power {}, toughness {}; counters {:?}.",
                Self::label(game.visible_reference(lki.object, viewer)),
                if lki.was_attacking {
                    ", attacking"
                } else if lki.was_in_combat {
                    ", blocking"
                } else {
                    ""
                },
                lki.power,
                lki.toughness,
                lki.counters
            )
        });
        let ObjectLki {
            copiable,
            attached_to,
            transforms,
            inherited_from,
            granted_abilities,
            keywords,
            renamed: _,
            types,
            subtypes,
            colors,
            is_legendary,
            permanent: _,
            animated_creature,
            was_attacking,
            was_in_combat,
            was_tapped,
            abilities_lost,
            dealt_damage_this_turn,
            power,
            toughness,
            counters,
            object: _,
            from_zone,
            owner,
            controller,
            definition_id,
            presentation_id: _,
        } = lki;
        if let Some(values) = copiable {
            self.field(
                "copy exceptions",
                &values.copy.as_ref().map(|copy| &copy.exceptions),
            );
            if let Some((id, _)) = &values.front {
                self.reference(
                    "copy front face",
                    VisibleReference {
                        definition: Some(id.0),
                        ..Default::default()
                    },
                );
            }
        }
        if lki.was_land() {
            self.field("last known land", &true);
        }
        if lki.was_creature() {
            self.field("last known creature", &true);
        }
        if *was_attacking {
            self.field("last known attacking", &true);
        }
        if *was_in_combat {
            self.field("last known in combat", &true);
        }
        if *was_tapped {
            self.field("last known tapped", &true);
        }
        if let Some(attached) = attached_to {
            self.reference(
                "last known attachment",
                game.visible_reference(*attached, viewer),
            );
        }
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
                    lki.creature_types(),
                    (types, subtypes, colors, is_legendary),
                ),
                (
                    animated_creature,
                    abilities_lost,
                    dealt_damage_this_turn,
                    power,
                    toughness,
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
                    object_id,
                    from,
                    to,
                    entry,
                } => {
                    if from.is_some() {
                        self.object("moving object", *object_id);
                    } else {
                        self.reference(
                            "entering definition",
                            VisibleReference {
                                definition: Some(
                                    self.game.state.entry_object(*object_id).definition_id.0,
                                ),
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
                        TokenSource::Registered(key) => {
                            let content = default_content_pack();
                            self.reference(
                                "token definition",
                                VisibleReference {
                                    definition: content.token_id(key).map(|id| id.0),
                                    ..Default::default()
                                },
                            );
                            let name = content.token_name(key);
                            self.summary(|| format!("Pending: create {count} {name} tokens."));
                        }
                        TokenSource::Copy {
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
            if let Some((object_id, controller)) = pending.sacrifice {
                self.object_id("sacrificed object", object_id);
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

/// A receipt's rule-read mana facts, under the names their tokens have: the
/// amounts by color and what the units' spend riders did. Which object
/// produced each unit is text only and adds no policy tokens.
#[derive(Debug, Serialize)]
struct ManaSpent {
    amounts: [u8; 6],
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    entry_counters: u8,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    gains: Vec<crate::state::card::Keyword>,
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    copies: u8,
}

impl ManaSpent {
    fn of(spent: &PaymentReceipt) -> Self {
        Self {
            amounts: spent.amounts(),
            entry_counters: spent.entry_counters(),
            gains: spent.keywords_gained().iter().collect(),
            copies: spent.copy_triggers().count() as u8,
        }
    }
}

/// What one object did toward a cost, as `viewer` may name it: "tapped Gray
/// Ogre to crew". A departed object is named by what it last was. A chosen
/// sacrifice is carried out as the payment completes (CR 601.2h), so while
/// its object is still where it was the phrase says so, not "sacrificed X".
pub(crate) fn contribution_phrase(game: &Game, viewer: PlayerId, paid: &Contribution) -> String {
    let how = match paid.how {
        How::Tapped => "tapped",
        How::Sacrificed => "sacrificed",
        How::Exiled => "exiled",
        How::Discarded => "discarded",
        How::Returned => "returned",
    };
    let toward = match paid.toward {
        Toward::Printed => "",
        Toward::Waterbend => " to waterbend",
        Toward::Crew => " to crew",
        Toward::Saddle => " to saddle",
        Toward::Alternative => " instead of paying mana",
        Toward::Convoke => " to convoke",
        Toward::Improvise => " to improvise",
        Toward::Delve => " to delve",
        Toward::Bargain => " to bargain",
        Toward::Forage => " to forage",
        Toward::Station => " to station",
    };
    let object = Projection::label(game.visible_reference(paid.object, viewer));
    if paid.how == How::Sacrificed && game.state.live(paid.object).is_some() {
        format!("{object}, sacrificed{toward} when the payment completes")
    } else {
        format!("{how} {object}{toward}")
    }
}

/// One unit of mana spent toward a cost, as `viewer` may name its source:
/// "{G} from Forest". A unit no object produced has no source to name.
pub(crate) fn spent_mana_phrase(game: &Game, viewer: PlayerId, unit: &SpentMana) -> String {
    let mut amounts = [0; 6];
    amounts[unit.color as usize] = 1;
    let mana = mana_description(&Mana {
        mana: amounts,
        ..Default::default()
    });
    match unit.source {
        Some(origin) => format!(
            "{mana} from {}",
            Projection::label(game.visible_reference(origin.object, viewer))
        ),
        None => mana,
    }
}

/// The contributions a frame of `source` shows. The receipt keeps everything
/// paid; the frame omits its own source paying its own printed cost (its
/// {T}, "sacrifice this", "discard this"), which the frame's ability already
/// states. Every other object is shown, and so is the source when it paid
/// under a keyword.
fn shown_contributions(
    spent: &PaymentReceipt,
    source: Option<ObjectId>,
) -> impl Iterator<Item = &Contribution> {
    spent
        .contributions()
        .iter()
        .filter(move |paid| !(paid.toward == Toward::Printed && Some(paid.object) == source))
}

/// Whether the frame of `source`, an object defined by `definition`, shows
/// anything of `spent`, what was paid for it or for the spell that became
/// it. It does when the payment says something the frame does not: another
/// object contributed (`shown_contributions`), life was paid, a spend effect
/// applied (Animal Attendant's counter), or the definition counts the mana
/// spent (Toph, converge). Mana that no rule reads, and the source's own
/// printed cost, stay on the receipt and out of the observation.
pub(crate) fn receipt_matters(
    spent: &PaymentReceipt,
    definition: Option<CardDefId>,
    source: Option<ObjectId>,
) -> bool {
    static READS: std::sync::OnceLock<Vec<bool>> = std::sync::OnceLock::new();
    if spent.is_empty() {
        return false;
    }
    if spent.mana().iter().any(|unit| unit.effect.is_some())
        || shown_contributions(spent, source).next().is_some()
        || spent.life() > 0
    {
        return true;
    }
    let reads = READS.get_or_init(|| {
        // The catalog iterates definitions in CardDefId order.
        default_content_pack()
            .definitions()
            .map(|definition| {
                let json = serde_json::to_string(definition).expect("serializable definition");
                json.contains("ManaSpentToCast") || json.contains("ColorsSpentToCast")
            })
            .collect()
    });
    definition.is_some_and(|id| reads.get(id.0 as usize).copied().unwrap_or(false))
}

/// Project the payment in progress onto its announcement frame: how many
/// counters a removal cost still owes (Quilled Greatwurm), then what the
/// payment has already committed. Writes nothing when neither applies.
pub(crate) fn project_pending_payment(
    game: &Game,
    payment: &Payment,
    viewer: PlayerId,
    frame: &mut ContinuationFrame,
    text: bool,
) {
    let remaining = match payment.owed.first() {
        Some(CostComponent::RemoveCounters { remaining, .. }) => Some(*remaining),
        _ => None,
    };
    if remaining.is_none() && payment.receipt.is_empty() {
        return;
    }
    let mut out = Projection::new(game, viewer, text);
    if let Some(remaining) = remaining {
        out.field("counters remaining", &remaining);
        out.summary(|| match remaining {
            1 => "1 counter left to remove.".into(),
            n => format!("{n} counters left to remove."),
        });
    }
    out.receipt(&payment.receipt, None);
    out.finish(frame);
}

impl Game {
    pub(super) fn trigger_reads(
        &self,
        source: ObjectId,
        definition: CardDefId,
        index: usize,
        inline: Option<&[Effect]>,
        effect_reads: fn(&Effect) -> bool,
        ability_reads: fn(&Ability) -> bool,
    ) -> bool {
        match inline {
            Some(effects) => effects.iter().any(effect_reads),
            None => self
                .source_face(source, definition)
                .and_then(|face| face.abilities.get(index))
                .is_some_and(ability_reads),
        }
    }

    pub(super) fn stack_object_reads(
        &self,
        object: &crate::state::stack_object::StackObject,
        effect_reads: fn(&Effect) -> bool,
        ability_reads: fn(&Ability) -> bool,
    ) -> bool {
        use crate::state::stack_object::StackObject;
        match object {
            StackObject::ActivatedAbility(ability) => ability.effects.iter().any(effect_reads),
            StackObject::TriggeredAbility(ability) => self.trigger_reads(
                ability.source_object(),
                ability.source_definition_id,
                ability.ability_index,
                ability.inline_effects.as_deref(),
                effect_reads,
                ability_reads,
            ),
            StackObject::Spell(_) => false,
        }
    }
}

/// Project the payment and any departed object the trigger reads.
/// Both text and policy read these same public facts.
pub(crate) fn project_stack_object(
    game: &Game,
    object: &crate::state::stack_object::StackObject,
    viewer: PlayerId,
    frame: &mut ContinuationFrame,
    text: bool,
) {
    use crate::state::stack_object::StackObject;
    let mut out = Projection::new(game, viewer, text);
    // "The number of soul counters on this artifact" after it was sacrificed
    // to pay the cost: the source's last known information (CR 608.2h).
    if game.stack_object_reads(object, Effect::reads_source_lki, Ability::reads_source_lki) {
        out.retained_object(object.source_object());
    }
    if let StackObject::TriggeredAbility(ability) = object {
        out.trigger_context(
            ability.context,
            ability.source_object(),
            ability.source_definition_id,
            ability.ability_index,
            ability.inline_effects.as_deref(),
        );
    }
    if let Some(spent) = super::continuation::stack_receipt(object) {
        out.receipt(spent, Some(object.source_object()));
    }
    out.finish(frame);
}

/// Project a pending trigger's departed triggering object, as its
/// stacked ability will (`project_stack_object`).
pub(crate) fn project_pending_trigger(
    game: &Game,
    trigger: &crate::flow::trigger::PendingTrigger,
    viewer: PlayerId,
    frame: &mut ContinuationFrame,
) {
    let mut out = Projection::new(game, viewer, false);
    out.trigger_context(
        trigger.context,
        trigger.source_object(),
        game.pending_trigger_definition(trigger),
        trigger.ability_index,
        trigger.inline_effects.as_deref(),
    );
    out.finish(frame);
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
    if let Some(source) = effect.source_ref() {
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
    if effect.life_lost != 0 {
        out.field("life lost this way", &effect.life_lost);
    }
    let source_definition = effect
        .source()
        .map(|object| game.state.source_base(object).0);
    if receipt_matters(&effect.spent, source_definition, effect.source()) {
        out.receipt(&effect.spent, effect.source());
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
    out.finish(frame);
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
        Decision::ManaSpend {
            player,
            cost,
            spent,
            completion,
        } => {
            out.field("mana spent", &ManaSpent::of(spent));
            out.summary(|| {
                format!(
                    "Mana already spent in this payment: {}.",
                    mana_description(&crate::state::mana::Mana {
                        mana: spent.amounts(),
                        ..Default::default()
                    })
                )
            });
            for (mode, choice) in game.mana_spend_choices(*player, cost).iter().enumerate() {
                out.field("mode", &mode);
                out.field("spending option", &choice.unit);
            }
            match completion.as_ref() {
                SpendCompletion::Announced(payment) => {
                    out.payment(payment);
                    for permanent in payment
                        .receipt
                        .paid_as(How::Sacrificed)
                        .filter_map(|object| game.state.permanent_of(object))
                    {
                        out.target(
                            "cost sacrifice",
                            crate::state::game_object::Target::Permanent(permanent),
                        );
                    }
                }
                SpendCompletion::Resolution(payment) => {
                    out.effects("if paid", &payment.if_paid);
                    out.receipt(&payment.receipt, None);
                    if let Some(key) = payment.spends {
                        out.object("once each turn source", key.0);
                        out.field("once each turn ability", &key.2);
                    }
                }
                SpendCompletion::Foretell(card) => {
                    out.object_id("cast from hand", game.state.object_of(*card))
                }
                SpendCompletion::Upfront(_) => {}
            }
        }
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
        }
        Decision::AssignCombatDamage {
            attacker,
            blocker,
            pass,
            minimum,
            remaining,
            last,
            assigned,
            ..
        } => {
            out.object("combat damage source", *attacker);
            out.object("target", *blocker);
            out.field(
                "another damage step",
                &matches!(pass, crate::flow::game::CombatDamagePass::FirstStrike),
            );
            if let Some(combat) = &game.state.combat {
                for &id in &combat.first_strikers {
                    if let Some(object) = game.permanent_object(id) {
                        out.object("first-strike participant", object);
                    }
                }
            }
            out.field("minimum assignment", minimum);
            out.field("remaining", remaining);
            out.field("last blocker", last);
            out.field("assigned combat damage", assigned);
        }
        Decision::OrderReplacement { candidates, .. } => {
            for (index, candidate) in candidates.iter().enumerate() {
                out.field("replacement option", &index);
                out.object("replacement source", candidate.source());
                out.field("replacement", &candidate.effect);
                out.note(|| candidate.description());
            }
        }
        Decision::Scry {
            looked,
            remaining,
            top,
            bottom,
            ..
        } => {
            out.object_ids("inspected", looked);
            out.object_ids("remaining", remaining);
            out.object_ids("kept on top", top);
            out.object_ids("placed on bottom", bottom);
        }
        Decision::Surveil {
            looked, graveyard, ..
        } => {
            out.object_ids("inspected", looked);
            out.object_ids("selected for graveyard", graveyard);
        }
        Decision::OrderLibrary {
            object_ids,
            next,
            end,
            ..
        } => {
            out.object_ids("library order", object_ids);
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
            out.object_ids("inspected", looked);
            out.field("selection", &(predicate, selected, min_select, max_select));
            out.field("placement", &(random_order, reveal, to_battlefield));
        }
        Decision::Payment(progress) => {
            out.field("remaining payment", &progress.owed);
            out.receipt(&progress.receipt, None);
            out.effects("if paid", &progress.if_paid);
            out.effects("if declined", &progress.if_declined);
            if let Some(key) = &progress.spends {
                out.object("once each turn source", key.0);
                out.field("once each turn ability", &key.2);
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
        Decision::SearchDistinctNames {
            predicate,
            remaining,
            selected,
            end,
            ..
        } => {
            out.object_ids("selected distinct names", selected);
            out.field("distinct name search", &(predicate, remaining, end));
        }
        Decision::OpponentSplitsRevealed {
            owner,
            revealed,
            remaining,
            ..
        } => {
            out.object_ids("revealed lands", revealed);
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
            sacrifice,
            kept,
            ..
        } => {
            match sacrifice {
                Some(count) => out.field("creatures each player sacrifices", count),
                None => out.field("maximum total power", max_total_power),
            }
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
                out.object_ids("legend group", group);
            }
            out.object_ids("legends kept", keep);
        }
        Decision::EnterAsCopy {
            object_id,
            candidates,
            ..
        } => {
            out.object_id("entering card", *object_id);
            for id in candidates {
                out.target("may copy", Target::Permanent(*id));
            }
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
                ResolutionCastOffer::Graveyard {
                    predicate,
                    exile_instants_and_sorceries,
                } => {
                    out.field("cast from graveyard", predicate);
                    out.field("exile instants and sorceries", exile_instants_and_sorceries);
                }
                ResolutionCastOffer::Hand { predicate } => out.field("cast from hand", predicate),
                ResolutionCastOffer::Exiled {
                    object: object_id,
                    below,
                } => {
                    out.field("spell mana value limit", below);
                    out.object_id("cast from exile", *object_id)
                }
                ResolutionCastOffer::AmongExiled { objects } => {
                    for object in objects {
                        out.object_id("cast from exile", *object);
                    }
                }
                ResolutionCastOffer::Cascade {
                    object: object_id,
                    exiled,
                    below,
                } => {
                    out.field("cascade mana value limit", below);
                    out.object_id("cast from exile", *object_id);
                    for exiled in exiled {
                        out.object_id("cascade exiled", *exiled);
                    }
                }
                ResolutionCastOffer::LibraryTop { object: object_id } => {
                    out.object_id("cast from library", *object_id)
                }
                ResolutionCastOffer::ExiledWith { source, predicate } => {
                    out.object("exiled with", *source);
                    out.field("cast predicate", predicate);
                }
            }
            if let Some(key) = spends {
                out.object("once each turn source", key.0);
                out.field("once each turn ability", &key.2);
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
