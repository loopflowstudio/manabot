//! Viewer-owned decision prefixes. Capture precedes execution; only a successful
//! external commit appends. Fixed-size immutable chunks share fork prefixes and
//! keep append/coalescing work bounded independently of the game's length.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind},
    flow::game::Game,
    state::{
        game_object::{ObjectId, PlayerId},
        zone::ZoneType,
    },
};

const CHUNK: usize = 32;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionLog {
    tail: Option<Arc<Chunk>>,
    len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Chunk {
    previous: Option<Arc<Chunk>>,
    rows: Vec<OpponentDecisionRecord>,
}

impl DecisionLog {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Exact projected size without allocating a policy tensor or selecting a
    /// capacity. This cold-path read is for corpus census and diagnostics.
    pub fn detail_count(&self) -> usize {
        self.iter()
            .map(|record| {
                let references = record
                    .context
                    .mana_sources
                    .iter()
                    .chain(&record.response)
                    .chain(&record.choice.references);
                20 + references
                    .map(|reference| {
                        7 + usize::from(reference.definition.is_some())
                            + reference.program.len()
                            + reference.mana.len() * 6
                    })
                    .sum::<usize>()
                    + usize::from(record.choice.kind.is_some())
                    + usize::from(record.choice.mode.is_some())
                    + usize::from(record.choice.ability.is_some())
                    + usize::from(record.choice.name_row.is_some())
                    + usize::from(record.choice.declared.is_some())
            })
            .sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = &OpponentDecisionRecord> {
        let mut chunks = Vec::new();
        let mut next = self.tail.as_deref();
        while let Some(chunk) = next {
            chunks.push(chunk);
            next = chunk.previous.as_deref();
        }
        chunks.into_iter().rev().flat_map(|chunk| chunk.rows.iter())
    }

    fn push(&mut self, mut row: OpponentDecisionRecord) {
        if let Some(tail) = self.tail.as_mut() {
            let last = tail.rows.last().expect("nonempty chunk");
            let position = last
                .last_position
                .checked_add(1)
                .expect("decision position exhausted");
            row.first_position = position;
            row.last_position = position;
            row.ordinal = self.len as u32;
            if row.choice.kind.as_deref() == Some("PassPriority")
                && last.choice == row.choice
                && last.context == row.context
                && last.response == row.response
                && last.kind == row.kind
            {
                let last = Arc::make_mut(tail).rows.last_mut().unwrap();
                last.count = last.count.checked_add(1).expect("pass count exhausted");
                last.last_position = position;
                return;
            }
        }
        if self
            .tail
            .as_ref()
            .is_none_or(|tail| tail.rows.len() == CHUNK)
        {
            self.tail = Some(Arc::new(Chunk {
                previous: self.tail.take(),
                rows: Vec::new(),
            }));
        }
        Arc::make_mut(self.tail.as_mut().unwrap()).rows.push(row);
        self.len += 1;
    }
}

impl Serialize for DecisionLog {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for DecisionLog {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        let rows = Vec::<OpponentDecisionRecord>::deserialize(decoder)?;
        let mut log = Self::default();
        // Retained projections preserve their exact spans and multiplicities.
        for row in rows {
            if log
                .tail
                .as_ref()
                .is_none_or(|tail| tail.rows.len() == CHUNK)
            {
                log.tail = Some(Arc::new(Chunk {
                    previous: log.tail.take(),
                    rows: Vec::new(),
                }));
            }
            Arc::make_mut(log.tail.as_mut().unwrap()).rows.push(row);
            log.len += 1;
        }
        Ok(log)
    }
}

/// A reference owns the characteristics visible at emission, never a future
/// lookup. Definition and appearance are provenance, never numeric features.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionReference {
    pub role: u8,
    pub definition: Option<u32>,
    pub appearance: Option<u32>,
    pub name: Option<String>,
    /// Human-readable native rules snapshot, derived alongside the learning program.
    pub description: String,
    pub power: i32,
    pub toughness: i32,
    pub tapped: bool,
    pub ready: bool,
    pub attacking: bool,
    pub blocking: bool,
    /// Complete effective rules characteristics as existing learning tokens.
    pub program: Vec<(u16, i32)>,
    /// Public production alternatives, W/U/B/R/G/C; no hidden-spell tests.
    pub mana: Vec<[u32; 6]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionContext {
    pub turn: u32,
    pub phase: u8,
    pub step: u8,
    pub life: [i32; 2],
    pub hand: [u32; 2],
    /// Total, tapped, land, creature, artifact, enchantment, planeswalker, battle.
    pub board_counts: [[u32; 8]; 2],
    pub mana_sources: Vec<DecisionReference>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicChoice {
    /// None means unavailable, permanently, even if later revealed.
    pub kind: Option<String>,
    pub mode: Option<u32>,
    pub ability: Option<u32>,
    pub named: Option<String>,
    pub name_row: Option<u32>,
    pub declared: Option<bool>,
    pub references: Vec<DecisionReference>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpponentDecisionRecord {
    pub ordinal: u32,
    pub first_position: u32,
    pub last_position: u32,
    pub count: u32,
    pub kind: String,
    pub context: DecisionContext,
    pub response: Vec<DecisionReference>,
    pub choice: PublicChoice,
}

fn reference(
    game: &Game,
    object: ObjectId,
    viewer: PlayerId,
    committed_public: bool,
) -> DecisionReference {
    let visible = game.state.live(object).is_some()
        && (committed_public || game.visible_reference(object, viewer).definition.is_some());
    if !visible {
        return DecisionReference {
            role: 0,
            definition: None,
            appearance: None,
            name: None,
            description: String::new(),
            power: 0,
            toughness: 0,
            tapped: false,
            ready: false,
            attacking: false,
            blocking: false,
            program: Vec::new(),
            mana: Vec::new(),
        };
    }
    let data = &game.state[object];
    let mut printed = game.state.source_base(object).1.clone();
    let mut mana = Vec::new();
    let mut role = if data.owner == viewer { 1 } else { 2 };
    let mut tapped = false;
    let mut ready = false;
    let mut attacking = false;
    let mut blocking = false;
    if let Some(id) = game.state.permanent_of(object) {
        let permanent = game.state.permanents[id].as_ref().unwrap();
        let view = game.characteristics(id);
        role = if permanent.controller == viewer { 1 } else { 2 };
        tapped = permanent.tapped;
        ready = game.permanent_can_tap(id);
        attacking = permanent.attacking;
        blocking = game.state.combat.as_ref().is_some_and(|combat| {
            combat
                .attacker_to_blockers
                .values()
                .any(|blockers| blockers.contains(&id))
        });
        printed.name = view.name.clone();
        printed.power = Some(view.power);
        printed.toughness = Some(view.toughness);
        printed.types = view.types.clone();
        printed.subtypes = view.subtypes.clone();
        printed.color_override = Some(view.colors.clone());
        printed.keywords = view.keywords;
        printed.mana_abilities = view.mana_abilities.clone();
        printed.triggered_mana_abilities = view.triggered_mana_abilities.clone();
        printed.statics = view.static_abilities.clone();
        printed.abilities = view
            .triggered_abilities
            .iter()
            .map(|trigger| trigger.ability.clone())
            .collect();
        printed.activated_abilities = view.activated_abilities.clone();
        if !tapped {
            for ability in &view.mana_abilities {
                mana.push(ability.mana.mana.map(u32::from));
            }
            for ability in &view.activated_abilities {
                if let Some(choices) = ability.mana_choices() {
                    mana.extend(choices.iter().map(|choice| choice.mana.map(u32::from)));
                }
            }
        }
    }
    let snapshot = crate::state::card::ObjectDefinition::face(printed.clone());
    let program = crate::cards::tokens::definition_tokens(
        &snapshot,
        crate::cards::content_pack::default_content_pack(),
    )
    .expect("registered public characteristics")
    .into_iter()
    .map(|(kind, value)| (kind as u16, value))
    .collect();
    let (power, toughness) = game
        .state
        .permanent_of(object)
        .map(|id| {
            let view = game.characteristics(id);
            (view.power, view.toughness)
        })
        .unwrap_or((0, 0));
    DecisionReference {
        role,
        definition: Some(data.definition_id.0),
        appearance: Some(data.id.0),
        name: Some(printed.name),
        description: serde_json::to_string(&snapshot).expect("snapshot serializes"),
        power,
        toughness,
        tapped,
        ready,
        attacking,
        blocking,
        program,
        mana,
    }
}

fn target_reference(
    game: &Game,
    target: crate::state::game_object::Target,
    viewer: PlayerId,
) -> DecisionReference {
    use crate::state::game_object::Target;
    let object = match target {
        Target::Player(player) => {
            return DecisionReference {
                role: if player == viewer { 1 } else { 2 },
                ..Default::default()
            }
        }
        Target::Permanent(id) => game.permanent_object(id),
        Target::StackSpell(object) | Target::GraveyardCard(object) => Some(object),
        Target::StackAbility(id) => game.stack_ability_source(id),
    };
    object
        .map(|object| reference(game, object, viewer, false))
        .unwrap_or_default()
}

pub const RECORD_DIM: usize = super::observation_layout::opponent_record::WIDTH;
pub const DETAIL_DIM: usize = super::observation_layout::opponent_detail::WIDTH;
/// Names introduced by this projection, registered in the shared catalog.
pub const RECORD_NAMES: &[&str] = &[
    "Priority",
    "DeclareAttacker",
    "DeclareBlocker",
    "ChooseTarget",
    "Scry",
    "LookAndSelect",
    "PayOrNot",
    "Modal",
    "Learn",
    "Waterbend",
    "Discard",
    "LegendRule",
    "OrderTriggers",
    "PassPriority",
    "PriorityPlayLand",
    "PriorityCastSpell",
    "PriorityActivateAbility",
    "PayCost",
    "DeclineChoice",
    "ChooseMode",
    "TapForCost",
    "LearnTakeLesson",
    "LearnDiscard",
    "SelectCard",
    "Life",
    "Hand",
    "Role",
    "Power",
    "Toughness",
    "Tapped",
    "Attacking",
    "Blocking",
    "Program",
    "Mana",
    "Choice",
    "Mode",
    "Declared",
    "Definition",
    "Ability",
    "Named",
    "BoardCounts",
    "Ready",
    "Foretell",
];

/// Exact signed words, shared with fact and execution transport conventions.
fn words(value: i32) -> [f32; 2] {
    let bits = value as u32;
    [(bits >> 16) as f32, (bits & 0xffff) as f32]
}

/// Whole-prefix tensor projection. Headers own disjoint detail slices; no
/// appearance id, private alternative count or current-object lookup is used.
pub fn encode(log: &DecisionLog) -> (Vec<[f32; RECORD_DIM]>, Vec<[f32; DETAIL_DIM]>) {
    use crate::cards::tokens::name_value;
    let mut headers = Vec::with_capacity(log.len());
    let mut details = Vec::new();
    for record in log.iter() {
        let start = details.len();
        let mut detail =
            |group: u32, reference: u32, field: &str, kind: i32, value: i32, ordinal: usize| {
                let name = words(name_value(&format!("OpponentMemory::{field}")));
                let value = words(value);
                details.push([
                    group as f32,
                    reference as f32,
                    name[0],
                    name[1],
                    kind as f32,
                    value[0],
                    value[1],
                    ordinal as f32,
                ]);
            };
        for seat in 0..2 {
            detail(0, seat as u32, "Life", -1, record.context.life[seat], 0);
            detail(
                0,
                seat as u32,
                "Hand",
                -1,
                record.context.hand[seat] as i32,
                0,
            );
        }
        for seat in 0..2 {
            for (kind, &count) in record.context.board_counts[seat].iter().enumerate() {
                detail(0, seat as u32, "BoardCounts", -1, count as i32, kind);
            }
        }
        for (group, references) in [
            (1, &record.context.mana_sources),
            (2, &record.response),
            (3, &record.choice.references),
        ] {
            for (index, reference) in references.iter().enumerate() {
                let index = index as u32;
                for (field, value) in [
                    ("Role", reference.role as i32),
                    ("Power", reference.power),
                    ("Toughness", reference.toughness),
                    ("Tapped", reference.tapped as i32),
                    ("Ready", reference.ready as i32),
                    ("Attacking", reference.attacking as i32),
                    ("Blocking", reference.blocking as i32),
                ] {
                    detail(group, index, field, -1, value, 0);
                }
                if let Some(definition) = reference.definition {
                    detail(group, index, "Definition", -3, (definition + 1) as i32, 0);
                }
                for (ordinal, &(kind, value)) in reference.program.iter().enumerate() {
                    detail(group, index, "Program", kind as i32, value, ordinal);
                }
                for (ordinal, production) in reference.mana.iter().enumerate() {
                    for (color, &amount) in production.iter().enumerate() {
                        detail(group, index, "Mana", -1, amount as i32, ordinal * 6 + color);
                    }
                }
            }
        }
        if let Some(kind) = &record.choice.kind {
            detail(
                4,
                0,
                "Choice",
                -2,
                name_value(&format!("OpponentMemory::{kind}")),
                0,
            );
        }
        if let Some(mode) = record.choice.mode {
            detail(4, 0, "Mode", -1, mode as i32, 0);
        }
        if let Some(ability) = record.choice.ability {
            detail(4, 0, "Ability", -1, ability as i32, 0);
        }
        if let Some(row) = record.choice.name_row {
            detail(4, 0, "Named", -4, (row + 1) as i32, 0);
        }
        if let Some(declared) = record.choice.declared {
            detail(4, 0, "Declared", -1, declared as i32, 0);
        }
        let mut header = [0.0; RECORD_DIM];
        for (offset, value) in [
            (0, record.ordinal),
            (2, record.count),
            (4, record.context.turn),
            (6, record.first_position),
            (8, record.last_position),
        ] {
            header[offset..offset + 2].copy_from_slice(&words(value as i32));
        }
        header[10] = record.context.phase as f32;
        header[11] = record.context.step as f32;
        header[12..14].copy_from_slice(&words(name_value(&format!(
            "OpponentMemory::{}",
            record.kind
        ))));
        header[14] = start as f32;
        header[15] = (details.len() - start) as f32;
        header[16] = 2.0; // chooser's viewer-relative role
        header[17] = 1.0;
        headers.push(header);
    }
    (headers, details)
}

/// Pending records contain no post-choice/private data. Callers discard them
/// on rejection; publication appends only after successful execution.
pub(crate) fn capture(
    game: &Game,
    space: &ActionSpace,
    action: &Action,
) -> Option<(usize, OpponentDecisionRecord)> {
    if space.actions.len() <= 1 {
        return None;
    }
    let chooser = space.player?;
    let viewer = PlayerId((chooser.0 + 1) % 2);
    let players = [viewer, chooser];
    let mut board_counts = [[0; 8]; 2];
    let mut board = Vec::new();
    let mut attacking = Vec::new();
    for (seat, &player) in players.iter().enumerate() {
        for id in game.battlefield_permanents(player) {
            let permanent = game.state.permanents[id].as_ref().unwrap();
            let view = game.characteristics(id);
            let flags = [
                true,
                permanent.tapped,
                view.is_land(),
                view.is_creature(),
                view.types.is_artifact(),
                view.types.is_enchantment(),
                view.types.is_planeswalker(),
                view.types.is_battle(),
            ];
            for (count, flag) in board_counts[seat].iter_mut().zip(flags) {
                *count += u32::from(flag);
            }
            let object = game.permanent_object(id).expect("battlefield object");
            if !permanent.tapped
                && (!view.mana_abilities.is_empty()
                    || view
                        .activated_abilities
                        .iter()
                        .any(|ability| ability.mana_choices().is_some()))
            {
                board.push(reference(game, object, viewer, false));
            }
            if permanent.attacking {
                attacking.push(reference(game, object, viewer, false));
            }
        }
    }
    let hand = players.map(|player| game.state.zones.size(ZoneType::Hand, player) as u32);
    let resource_seat = usize::from(action.player() != viewer);
    if matches!(action, Action::PassPriority { .. })
        && (space.kind != ActionSpaceKind::Priority
            || hand[resource_seat] == 0
            || !board.iter().any(|r| {
                r.role as usize == resource_seat + 1
                    && r.ready
                    && r.mana
                        .iter()
                        .any(|mana| mana.iter().any(|&amount| amount > 0))
            }))
    {
        return None;
    }
    let mut choice = PublicChoice {
        kind: None,
        mode: None,
        ability: None,
        named: None,
        name_row: None,
        declared: None,
        references: Vec::new(),
    };
    // Exhaustive privacy classification. Private choices share one unavailable
    // projection; action indexes and private alternatives are never retained.
    let public = match action {
        Action::PlayLand { object, .. }
        | Action::CastSpell { object, .. }
        | Action::ActivateCardAbility { object, .. } => {
            choice
                .references
                .push(reference(game, *object, viewer, true));
            true
        }
        Action::PassPriority { .. } => true,
        Action::DeclareAttacker { attack, .. } => {
            choice.declared = Some(*attack);
            true
        }
        Action::DeclareBlocker { attacker, .. } => {
            choice.declared = Some(attacker.is_some());
            true
        }
        Action::ChooseTarget { target, .. } => {
            choice
                .references
                .push(target_reference(game, *target, viewer));
            true
        }
        Action::ActivateAbility { ability_index, .. } => {
            choice.ability = Some(*ability_index as u32);
            true
        }
        Action::WaterbendTap { .. } => true,
        Action::ChooseMode { mode, .. } => {
            use crate::flow::decision::Decision;
            match game
                .state
                .suspended_decision
                .as_ref()
                .map(|pending| &pending.decision)
            {
                Some(Decision::NameCard { names, scope, .. }) => {
                    if let Some(crate::cards::names::NameOption::Name(row)) = scope.option(*mode) {
                        choice.name_row = Some(row as u32);
                        choice.named = names.get(row).map(|candidate| candidate.name.clone());
                        true
                    } else {
                        false
                    }
                }
                Some(
                    Decision::Modal { .. }
                    | Decision::ManaChoice { .. }
                    | Decision::ManaSpend { .. }
                    | Decision::ChoosePaymentX { .. },
                ) => {
                    choice.mode = Some(*mode as u32);
                    true
                }
                _ => false,
            }
        }
        Action::PayCost { .. } => true,
        Action::Decline { .. } => matches!(
            space.kind,
            ActionSpaceKind::PayOrNot | ActionSpaceKind::Learn | ActionSpaceKind::Waterbend
        ),
        Action::LearnTakeLesson { object, .. } | Action::LearnDiscard { object, .. } => {
            choice
                .references
                .push(reference(game, *object, viewer, true));
            true
        }
        Action::SelectCard { object, .. }
            if matches!(
                space.kind,
                ActionSpaceKind::Discard | ActionSpaceKind::LegendRule
            ) =>
        {
            choice
                .references
                .push(reference(game, *object, viewer, true));
            true
        }
        Action::Foretell { .. } => {
            choice.references.push(DecisionReference::default());
            true
        }
        Action::ChooseTrigger { .. } | Action::ScryCard { .. } | Action::SelectCard { .. } => false,
    };
    if let Action::ActivateCardAbility { ability_index, .. } = action {
        choice.ability = Some(*ability_index as u32);
    }
    if public {
        choice.kind = Some(match action {
            Action::PassPriority { .. } => "PassPriority".into(),
            Action::Foretell { .. } => "Foretell".into(),
            _ => format!("{:?}", action.action_type()),
        });
        if choice.references.is_empty() {
            choice.references = crate::agent::observation::Observation::action_focus(game, action)
                .iter()
                .filter_map(|id| {
                    game.state
                        .latest_objects()
                        .find(|(_, data)| data.id == *id)
                        .map(|(object, _)| reference(game, object, viewer, false))
                })
                .collect();
        }
    }
    let mut response = Vec::new();
    for stack in game.state.stack_objects.iter().rev() {
        let source = stack
            .source_ref()
            .map(|object| reference(game, object, viewer, false))
            .unwrap_or_default();
        response.push(source);
        response.extend(
            stack
                .targets()
                .iter()
                .map(|&target| target_reference(game, target, viewer)),
        );
    }
    response.extend(attacking);
    Some((
        viewer.0,
        OpponentDecisionRecord {
            ordinal: 0,
            first_position: 0,
            last_position: 0,
            count: 1,
            kind: format!("{:?}", space.kind),
            context: DecisionContext {
                turn: game.state.turn.turn_number,
                phase: game.state.turn.current_phase_kind() as u8,
                step: game.state.turn.current_step_kind() as u8,
                life: players.map(|p| game.state.players[p.0].life),
                hand,
                board_counts,
                mana_sources: board,
            },
            response,
            choice,
        },
    ))
}

pub(crate) fn commit(game: &mut Game, pending: Option<(usize, OpponentDecisionRecord)>) {
    if let Some((viewer, row)) = pending {
        game.opponent_memory[viewer].push(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::{
            observation::Observation,
            observation_encoder::{self, ObservationEncoderConfig},
        },
        state::player::PlayerConfig,
    };

    fn game() -> Game {
        let players = (0..2)
            .map(|seat| {
                PlayerConfig::new(
                    format!("seat{seat}"),
                    [("Mountain".into(), 20), ("Lightning Bolt".into(), 20)].into(),
                )
            })
            .collect();
        let mut game = Game::new(players, 111, true);
        let land = game
            .state
            .latest_objects()
            .find(|(_, data)| data.owner == PlayerId(0) && data.name == "Mountain")
            .unwrap()
            .0;
        game.move_object(land, ZoneType::Battlefield);
        let actions = game.compute_player_actions(PlayerId(0));
        game.publish_action_space(ActionSpace {
            player: Some(PlayerId(0)),
            kind: ActionSpaceKind::Priority,
            actions,
            focus: Vec::new(),
        });
        game
    }

    fn pending_pass(game: &Game) -> Option<(usize, OpponentDecisionRecord)> {
        let space = game.current_action_space.as_ref().unwrap();
        capture(
            game,
            space,
            &Action::PassPriority {
                player: PlayerId(0),
            },
        )
    }

    #[test]
    fn opponent_memory_commit_replay_rejection_and_viewer_identity() {
        let mut first = game();
        assert!(first.step(usize::MAX).is_err());
        assert!(first.opponent_memory.iter().all(DecisionLog::is_empty));
        let pass = first
            .current_action_space
            .as_ref()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .unwrap();
        first.step(pass).unwrap();
        assert!(first.opponent_memory[0].is_empty());
        assert_eq!(first.opponent_memory[1].len(), 1);
        let mut replay = game();
        replay.step(pass).unwrap();
        assert_eq!(first.opponent_memory, replay.opponent_memory);
        let observation = Observation::for_player(&first, PlayerId(1));
        let decoded = crate::agent::observation_json::parse(&observation.to_json()).unwrap();
        assert_eq!(decoded.opponent_memory, observation.opponent_memory);
    }

    #[test]
    fn opponent_memory_prefixes_coalesce_without_mutating_observations_or_forks() {
        let mut game = game();
        let pending = pending_pass(&game);
        commit(&mut game, pending.clone());
        let old = Observation::for_player(&game, PlayerId(1));
        let mut fork = game.clone();
        assert!(Arc::ptr_eq(
            game.opponent_memory[1].tail.as_ref().unwrap(),
            fork.opponent_memory[1].tail.as_ref().unwrap()
        ));
        commit(&mut fork, pending.clone());
        assert_eq!(old.opponent_memory.iter().next().unwrap().count, 1);
        assert_eq!(game.opponent_memory[1].iter().next().unwrap().count, 1);
        assert_eq!(fork.opponent_memory[1].iter().next().unwrap().count, 2);
        fork.state.players[0].life -= 1;
        let changed = pending_pass(&fork);
        commit(&mut fork, changed);
        assert_eq!(fork.opponent_memory[1].len(), 2);
        for turn in 2..100 {
            let mut row = pending.clone().unwrap();
            row.1.context.turn = turn;
            commit(&mut fork, Some(row));
        }
        assert!(fork.opponent_memory[1].len() > CHUNK);
        assert_eq!(old.opponent_memory.len(), 1);
        let rows: Vec<_> = fork.opponent_memory[1].iter().collect();
        assert!(rows
            .windows(2)
            .all(|pair| pair[0].last_position < pair[1].first_position));
    }

    #[test]
    fn opponent_memory_pass_filter_is_public_and_forced_choices_are_omitted() {
        let mut game = game();
        assert!(pending_pass(&game).is_some());
        let mut space = game.current_action_space.as_ref().unwrap().clone();
        space
            .actions
            .retain(|action| matches!(action, Action::PassPriority { .. }));
        assert!(capture(&game, &space, &space.actions[0]).is_none());
        let id = game.battlefield_permanents(PlayerId(0))[0];
        game.state.permanents[id].as_mut().unwrap().tapped = true;
        assert!(pending_pass(&game).is_none());
        game.state.permanents[id].as_mut().unwrap().tapped = false;
        game.scenario_clear_hand(PlayerId(0));
        assert!(pending_pass(&game).is_none());
    }

    #[test]
    fn opponent_memory_private_choices_remain_unavailable_after_reveal() {
        let mut game = game();
        let object = game
            .state
            .zones
            .latest()
            .find(|&id| game.state.zones.zone_of(id) == Some(ZoneType::Library))
            .unwrap();
        let actions: Vec<_> = [false, true]
            .map(|to_bottom| Action::ScryCard {
                player: PlayerId(0),
                object,
                to_bottom,
            })
            .into();
        let space = ActionSpace {
            player: Some(PlayerId(0)),
            kind: ActionSpaceKind::Scry,
            actions,
            focus: Vec::new(),
        };
        let first = capture(&game, &space, &space.actions[0]);
        let second = capture(&game, &space, &space.actions[1]);
        assert_eq!(first, second);
        commit(&mut game, first);
        let before = game.opponent_memory[1].clone();
        game.move_object(object, ZoneType::Graveyard);
        assert_eq!(game.opponent_memory[1], before);
        assert!(before.iter().next().unwrap().choice.kind.is_none());
    }

    #[test]
    fn opponent_memory_capacity_refuses_the_complete_prefix_and_companion_overflow() {
        let mut game = game();
        let pending = pending_pass(&game);
        commit(&mut game, pending);
        let obs = Observation::for_player(&game, PlayerId(1));
        let config = ObservationEncoderConfig {
            max_opponent_records: 1,
            max_opponent_details: 1,
            ..Default::default()
        };
        assert!(matches!(
            observation_encoder::encode(&obs, &config),
            Err(observation_encoder::ObservationEncodeError::Capacity {
                field: "opponent_details",
                ..
            })
        ));
        let (_, details) = encode(&obs.opponent_memory);
        assert_eq!(obs.opponent_memory.detail_count(), details.len());
        let config = ObservationEncoderConfig {
            max_opponent_details: details.len(),
            ..config
        };
        assert!(observation_encoder::encode(&obs, &config).is_ok());
    }

    #[test]
    fn opponent_memory_chooser_is_resolved_once_but_pass_uses_rules_resources() {
        use crate::{flow::turn::PlayerControl, state::ability::PlayerControlDuration};
        let mut game = game();
        game.state.turn.player_controls.push(PlayerControl {
            player: PlayerId(0),
            controller: PlayerId(1),
            duration: PlayerControlDuration::NextTurn,
            active: true,
        });
        let space = game.current_action_space.as_ref().unwrap().clone();
        game.publish_action_space(space);
        let pending = pending_pass(&game).expect("seat zero still has the resources");
        assert_eq!(
            pending.0, 0,
            "the other chooser is the opponent of viewer zero"
        );
        commit(&mut game, Some(pending));
        assert_eq!(game.opponent_memory[0].len(), 1);
        assert!(game.opponent_memory[1].is_empty());
    }
}
