//! Replayable semantic-kernel conformance evidence for the curated two-deck slice.
//!
//! The readable reference path surfaces every singleton action. The optimized
//! production path collapses those forced actions internally. They deliberately
//! share the low-level rules implementation; this module checks the scheduling
//! optimization, deterministic replay, and checked evidence boundary rather
//! than claiming to be a second Comprehensive Rules engine.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::Arc,
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::{
    cardsets::alpha::default_content_pack,
    flow::{game::Game, search::mix_seed},
    state::player::PlayerConfig,
};

pub const CONTRACT_ID: &str = "manabot.semantic-kernel-conformance.v1";
pub const SCHEMA_VERSION: u32 = 1;
pub const REFERENCE_ID: &str = "reference/explicit-step-v1";
pub const OPTIMIZED_ID: &str = "optimized/skip-trivial-v1";
pub const PHASE_REVISION: &str = "553b97bd5c9f1a28bf7a6ebe80f6cb3a0e296c0d";

const MAX_FORCED_ACTIONS: usize = 100_000;
const MATCHUP_SOURCE_BYTES: &[u8] =
    include_bytes!("../../content/semantic/v1/two_deck.source.json");
const COVERAGE_EVIDENCE_BYTES: &[u8] =
    include_bytes!("../../content/semantic/v1/coverage.evidence.json");

#[derive(Clone, Debug, Deserialize)]
struct MatchupSource {
    schema_version: u32,
    pack_key: String,
    decks: Vec<SourceDeck>,
    definitions: Vec<SourceDefinition>,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceDeck {
    key: String,
    card_count: usize,
    cards: Vec<SourceDeckEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceDeckEntry {
    definition: String,
    count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceDefinition {
    key: String,
    registry_name: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CoverageEvidence {
    rule_families: Vec<CoverageFamily>,
}

#[derive(Clone, Debug, Deserialize)]
struct CoverageFamily {
    key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    UrLessonsVsGwAllies,
    GwAlliesVsUrLessons,
}

impl Orientation {
    fn seat_keys(self) -> (&'static str, &'static str) {
        match self {
            Self::UrLessonsVsGwAllies => ("ur_lessons", "gw_allies"),
            Self::GwAlliesVsUrLessons => ("gw_allies", "ur_lessons"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseSpec {
    pub id: String,
    pub orientation: Orientation,
    pub game_seed: u64,
    pub choice_seed: u64,
    pub max_commands: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzSpec {
    pub seed: u64,
    pub cases: usize,
    pub max_commands: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub contract_id: String,
    pub cases: Vec<CaseSpec>,
    pub fuzz: FuzzSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayOutcome {
    Terminal,
    Capped,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub state_hash: String,
    pub action_space_digest: String,
    pub pending_choice_digest: String,
    pub event_boundary: [usize; 3],
    pub game_over: bool,
    pub winner: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayCommand {
    pub diagnostic_index: usize,
    pub prompt_kind: String,
    pub action_type: String,
    pub action: Value,
    pub checkpoint: Checkpoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayCase {
    pub schema_version: u32,
    pub contract_id: String,
    pub case_id: String,
    pub matchup_pack_key: String,
    pub matchup_source_digest: String,
    pub content_digest: String,
    pub reference: String,
    pub optimized: String,
    pub orientation: Orientation,
    pub game_seed: u64,
    pub choice_seed: u64,
    pub max_commands: usize,
    pub initial_checkpoint: Checkpoint,
    pub commands: Vec<ReplayCommand>,
    pub outcome: ReplayOutcome,
    pub final_checkpoint: Checkpoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    Match,
    Mismatch,
    Excluded,
}

impl PhaseStatus {
    fn key(&self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Mismatch => "mismatch",
            Self::Excluded => "excluded",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseRow {
    pub id: String,
    pub rule_family: String,
    pub normalized_assertion: String,
    pub manabot_evidence: Vec<String>,
    pub phase_evidence: Vec<String>,
    pub status: PhaseStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manabot_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_outcome: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusion_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseMatrix {
    pub schema_version: u32,
    pub phase_repository: String,
    pub phase_revision: String,
    pub comparison_mode: String,
    pub limitations: Vec<String>,
    pub rows: Vec<PhaseRow>,
    pub mismatches: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySummary {
    pub case_id: String,
    pub orientation: Orientation,
    pub commands: usize,
    pub outcome: ReplayOutcome,
    pub final_state_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    pub schema_version: u32,
    pub contract_id: String,
    pub matchup_source_digest: String,
    pub content_digest: String,
    pub reference: String,
    pub optimized: String,
    pub limitation: String,
    pub replays: Vec<ReplaySummary>,
    pub exercised_prompt_kinds: Vec<String>,
    pub exercised_action_types: Vec<String>,
    pub property_checks: Vec<String>,
    pub metamorphic_checks: Vec<String>,
    pub fuzz: FuzzSpec,
    pub phase_revision: String,
    pub phase_status_counts: BTreeMap<String, usize>,
    pub phase_mismatches: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureReceipt {
    pub schema_version: u32,
    pub contract_id: String,
    pub matchup_source_digest: String,
    pub content_digest: String,
    pub reference: String,
    pub optimized: String,
    pub orientation: Orientation,
    pub game_seed: u64,
    pub choice_seed: u64,
    pub max_commands: usize,
    pub failing_step: usize,
    pub action_tape: Vec<Value>,
    pub last_matching_checkpoint: Option<Checkpoint>,
    pub error: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum ReplayReceipt {
    Checked(ReplayCase),
    Failure(FailureReceipt),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckSummary {
    pub replay_cases: usize,
    pub replay_commands: usize,
    pub phase_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuzzSummary {
    pub cases: usize,
    pub commands: usize,
}

#[derive(Clone)]
struct ConformancePair {
    reference: Game,
    optimized: Game,
}

impl ConformancePair {
    fn new(orientation: Orientation, game_seed: u64) -> Result<Self, String> {
        let configs = player_configs(orientation)?;
        // Frozen v1 receipts predate the compiled authored pack and include
        // the original content allocation in their checkpoint hash. Replay
        // them against that exact recorded world; ordinary exact-deck matches
        // select the compiled pack through `Game::new`.
        let content = default_content_pack();
        let mut reference =
            Game::new_with_content(configs.clone(), game_seed, false, Arc::clone(&content));
        let optimized = Game::new_with_content(configs, game_seed, true, content);
        normalize_reference(&mut reference)?;
        let pair = Self {
            reference,
            optimized,
        };
        pair.assert_equivalent("initial boundary")?;
        Ok(pair)
    }

    fn checkpoint(&self) -> Result<Checkpoint, String> {
        checkpoint(&self.optimized)
    }

    fn assert_equivalent(&self, boundary: &str) -> Result<Checkpoint, String> {
        let reference = checkpoint(&self.reference)?;
        let optimized = checkpoint(&self.optimized)?;
        if reference != optimized {
            return Err(format!(
                "{boundary}: reference/optimized divergence\nreference={reference:#?}\noptimized={optimized:#?}"
            ));
        }
        Ok(optimized)
    }

    fn apply_semantic_action(&mut self, action: &Value) -> Result<Checkpoint, String> {
        let optimized_index = semantic_action_index(&self.optimized, action)?;
        let reference_index = semantic_action_index(&self.reference, action)?;
        self.optimized
            .step(optimized_index)
            .map_err(|error| format!("optimized apply failed: {error}"))?;
        self.reference
            .step(reference_index)
            .map_err(|error| format!("reference apply failed: {error}"))?;
        normalize_reference(&mut self.reference)?;
        self.assert_equivalent("post-command boundary")
    }
}

fn normalize_reference(game: &mut Game) -> Result<(), String> {
    for _ in 0..MAX_FORCED_ACTIONS {
        if game.is_game_over() {
            return Ok(());
        }
        let action_count = game
            .action_space()
            .ok_or_else(|| "reference has no active action space".to_string())?
            .actions
            .len();
        match action_count {
            0 => return Err("reference reached a nonterminal empty action space".to_string()),
            1 => {
                game.step(0)
                    .map_err(|error| format!("forced reference action failed: {error}"))?;
            }
            _ => return Ok(()),
        }
    }
    Err(format!(
        "reference exceeded {MAX_FORCED_ACTIONS} forced singleton actions"
    ))
}

fn checkpoint(game: &Game) -> Result<Checkpoint, String> {
    Ok(Checkpoint {
        state_hash: game.state.deterministic_hash().to_string(),
        action_space_digest: digest_serialized(&game.current_action_space)?,
        pending_choice_digest: digest_serialized(&game.pending_choice)?,
        event_boundary: [
            game.state.events.len(),
            game.state.pending_events.len(),
            game.state.observation_events.len(),
        ],
        game_over: game.is_game_over(),
        winner: game.winner_index(),
    })
}

fn digest_serialized<T: Serialize + ?Sized>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn semantic_action_index(game: &Game, expected: &Value) -> Result<usize, String> {
    let space = game
        .action_space()
        .ok_or_else(|| "game has no active action space".to_string())?;
    let matches: Vec<_> = space
        .actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            let value = serde_json::to_value(action).ok()?;
            (value == *expected).then_some(index)
        })
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(format!(
            "semantic action is absent from {:?}: {expected}",
            space.kind
        )),
        _ => Err(format!(
            "semantic action is duplicated in {:?}: {expected}",
            space.kind
        )),
    }
}

fn selected_action(game: &Game, index: usize) -> Result<(String, String, Value), String> {
    let space = game
        .action_space()
        .ok_or_else(|| "game has no active action space".to_string())?;
    let action = space.actions.get(index).ok_or_else(|| {
        format!(
            "action {index} out of bounds for {} legal actions",
            space.actions.len()
        )
    })?;
    Ok((
        format!("{:?}", space.kind),
        format!("{:?}", action.action_type()),
        serde_json::to_value(action).map_err(|error| error.to_string())?,
    ))
}

fn matchup_source() -> Result<MatchupSource, String> {
    let source: MatchupSource =
        serde_json::from_slice(MATCHUP_SOURCE_BYTES).map_err(|error| error.to_string())?;
    if source.schema_version != 1 {
        return Err(format!(
            "unsupported matchup source schema {}",
            source.schema_version
        ));
    }
    Ok(source)
}

fn matchup_source_digest() -> String {
    blake3::hash(MATCHUP_SOURCE_BYTES).to_hex().to_string()
}

fn player_configs(orientation: Orientation) -> Result<Vec<PlayerConfig>, String> {
    let source = matchup_source()?;
    let definitions: BTreeMap<_, _> = source
        .definitions
        .into_iter()
        .map(|definition| (definition.key, definition.registry_name))
        .collect();
    let decks: BTreeMap<_, _> = source
        .decks
        .into_iter()
        .map(|deck| (deck.key.clone(), deck))
        .collect();
    let (first, second) = orientation.seat_keys();
    [first, second]
        .into_iter()
        .map(|key| {
            let deck = decks
                .get(key)
                .ok_or_else(|| format!("matchup source is missing deck {key}"))?;
            let mut decklist = BTreeMap::new();
            for entry in &deck.cards {
                let registry_name = definitions.get(&entry.definition).ok_or_else(|| {
                    format!(
                        "deck {key} references unknown definition {}",
                        entry.definition
                    )
                })?;
                if decklist
                    .insert(registry_name.clone(), entry.count)
                    .is_some()
                {
                    return Err(format!("deck {key} repeats {registry_name}"));
                }
            }
            let actual_count: usize = decklist.values().sum();
            if actual_count != deck.card_count {
                return Err(format!(
                    "deck {key} declares {} cards but resolves to {actual_count}",
                    deck.card_count
                ));
            }
            Ok(PlayerConfig::new(key, decklist))
        })
        .collect()
}

fn validate_spec(spec: &CaseSpec) -> Result<(), String> {
    if spec.id.trim().is_empty() {
        return Err("case id must not be empty".to_string());
    }
    if spec.max_commands == 0 || spec.max_commands > 2_000 {
        return Err(format!(
            "case {} max_commands must be within 1..=2000",
            spec.id
        ));
    }
    Ok(())
}

fn generate_case_internal(
    spec: &CaseSpec,
    action_tape: &mut Vec<Value>,
    last_matching_checkpoint: &mut Option<Checkpoint>,
) -> Result<ReplayCase, String> {
    validate_spec(spec)?;
    let source = matchup_source()?;
    let mut pair = ConformancePair::new(spec.orientation, spec.game_seed)?;
    let initial_checkpoint = pair.checkpoint()?;
    *last_matching_checkpoint = Some(initial_checkpoint.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(spec.choice_seed);
    let mut commands = Vec::new();

    while !pair.optimized.is_game_over() && commands.len() < spec.max_commands {
        let action_count = pair
            .optimized
            .action_space()
            .ok_or_else(|| "optimized executor has no active prompt".to_string())?
            .actions
            .len();
        if action_count == 0 {
            return Err("optimized executor reached a nonterminal empty prompt".to_string());
        }
        let diagnostic_index = rng.gen_range(0..action_count);
        let (prompt_kind, action_type, action) =
            selected_action(&pair.optimized, diagnostic_index)?;
        action_tape.push(action.clone());
        let checkpoint = pair.apply_semantic_action(&action)?;
        *last_matching_checkpoint = Some(checkpoint.clone());
        commands.push(ReplayCommand {
            diagnostic_index,
            prompt_kind,
            action_type,
            action,
            checkpoint,
        });
    }

    let outcome = if pair.optimized.is_game_over() {
        ReplayOutcome::Terminal
    } else {
        ReplayOutcome::Capped
    };
    let final_checkpoint = pair.checkpoint()?;
    Ok(ReplayCase {
        schema_version: SCHEMA_VERSION,
        contract_id: CONTRACT_ID.to_string(),
        case_id: spec.id.clone(),
        matchup_pack_key: source.pack_key,
        matchup_source_digest: matchup_source_digest(),
        content_digest: default_content_pack().content_digest(),
        reference: REFERENCE_ID.to_string(),
        optimized: OPTIMIZED_ID.to_string(),
        orientation: spec.orientation,
        game_seed: spec.game_seed,
        choice_seed: spec.choice_seed,
        max_commands: spec.max_commands,
        initial_checkpoint,
        commands,
        outcome,
        final_checkpoint,
    })
}

pub fn generate_case(spec: &CaseSpec) -> Result<ReplayCase, String> {
    generate_case_internal(spec, &mut Vec::new(), &mut None)
}

pub fn replay_case(case: &ReplayCase) -> Result<(), String> {
    if case.schema_version != SCHEMA_VERSION || case.contract_id != CONTRACT_ID {
        return Err(format!(
            "{} has an unsupported replay contract",
            case.case_id
        ));
    }
    if case.matchup_source_digest != matchup_source_digest() {
        return Err(format!("{} matchup source digest drift", case.case_id));
    }
    if case.content_digest != default_content_pack().content_digest() {
        return Err(format!("{} content digest drift", case.case_id));
    }
    let mut pair = ConformancePair::new(case.orientation, case.game_seed)?;
    if pair.checkpoint()? != case.initial_checkpoint {
        return Err(format!("{} initial checkpoint drift", case.case_id));
    }
    for (step, command) in case.commands.iter().enumerate() {
        if pair.optimized.is_game_over() {
            return Err(format!(
                "{} became terminal before command {step}",
                case.case_id
            ));
        }
        let actual = pair.apply_semantic_action(&command.action)?;
        if actual != command.checkpoint {
            return Err(format!(
                "{} checkpoint drift after command {step}\nexpected={:#?}\nactual={actual:#?}",
                case.case_id, command.checkpoint
            ));
        }
    }
    let final_checkpoint = pair.checkpoint()?;
    if final_checkpoint != case.final_checkpoint {
        return Err(format!("{} final checkpoint drift", case.case_id));
    }
    match case.outcome {
        ReplayOutcome::Terminal if !pair.optimized.is_game_over() => Err(format!(
            "{} tape exhausted before expected terminal state",
            case.case_id
        )),
        ReplayOutcome::Capped
            if pair.optimized.is_game_over() || case.commands.len() != case.max_commands =>
        {
            Err(format!(
                "{} does not reproduce its cap outcome",
                case.case_id
            ))
        }
        _ => Ok(()),
    }
}

fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn pretty_json<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, pretty_json(value)?)
        .map_err(|error| format!("{}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| {
        format!(
            "rename {} to {}: {error}",
            temporary.display(),
            path.display()
        )
    })
}

fn load_manifest(root: &Path) -> Result<Manifest, String> {
    let manifest: Manifest = load_json(&root.join("manifest.json"))?;
    if manifest.schema_version != SCHEMA_VERSION || manifest.contract_id != CONTRACT_ID {
        return Err("unsupported conformance manifest contract".to_string());
    }
    if manifest.cases.is_empty() {
        return Err("conformance manifest must contain replay cases".to_string());
    }
    if manifest.fuzz.cases == 0 || manifest.fuzz.max_commands == 0 {
        return Err("conformance fuzz manifest must be bounded and nonempty".to_string());
    }
    let mut ids = BTreeSet::new();
    for spec in &manifest.cases {
        validate_spec(spec)?;
        if !ids.insert(&spec.id) {
            return Err(format!("duplicate case id {}", spec.id));
        }
    }
    Ok(manifest)
}

fn phase_counts(matrix: &PhaseMatrix) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for row in &matrix.rows {
        *counts.entry(row.status.key().to_string()).or_insert(0) += 1;
    }
    counts
}

pub fn validate_phase_matrix(matrix: &PhaseMatrix) -> Result<(), String> {
    if matrix.schema_version != SCHEMA_VERSION {
        return Err("unsupported Phase matrix schema".to_string());
    }
    if matrix.phase_repository != "https://github.com/phase-rs/phase"
        || matrix.phase_revision != PHASE_REVISION
    {
        return Err("Phase matrix repository or revision is not exactly pinned".to_string());
    }
    if matrix.comparison_mode != "pinned_source_inspection_not_runtime_differential" {
        return Err("Phase matrix comparison mode is not explicit".to_string());
    }
    if matrix.limitations.is_empty() {
        return Err("Phase matrix must state its practical limitations".to_string());
    }
    let coverage: CoverageEvidence =
        serde_json::from_slice(COVERAGE_EVIDENCE_BYTES).map_err(|error| error.to_string())?;
    let expected_families: BTreeSet<_> = coverage
        .rule_families
        .into_iter()
        .map(|family| family.key)
        .collect();
    let actual_families: BTreeSet<_> = matrix
        .rows
        .iter()
        .map(|row| row.rule_family.clone())
        .collect();
    if actual_families != expected_families || matrix.rows.len() != expected_families.len() {
        return Err(format!(
            "Phase matrix rule-family closure mismatch; expected={expected_families:?}, actual={actual_families:?}"
        ));
    }
    let mut row_ids = BTreeSet::new();
    let pinned_source_prefix = format!(
        "{}/blob/{}/",
        matrix.phase_repository, matrix.phase_revision
    );
    for row in &matrix.rows {
        if !row_ids.insert(row.id.clone()) {
            return Err(format!("duplicate Phase row id {}", row.id));
        }
        if row.normalized_assertion.trim().is_empty() || row.manabot_evidence.is_empty() {
            return Err(format!("Phase row {} lacks local evidence", row.id));
        }
        for reference in &row.phase_evidence {
            if !reference.starts_with(&pinned_source_prefix) {
                return Err(format!("Phase row {} contains an unpinned URL", row.id));
            }
        }
        match row.status {
            PhaseStatus::Match => {
                if row.phase_evidence.is_empty()
                    || row.manabot_outcome.is_none()
                    || row.phase_outcome.is_none()
                {
                    return Err(format!(
                        "Phase match row {} lacks bilateral evidence",
                        row.id
                    ));
                }
            }
            PhaseStatus::Mismatch => {
                if row.phase_evidence.is_empty()
                    || row.manabot_outcome.is_none()
                    || row.phase_outcome.is_none()
                {
                    return Err(format!("Phase mismatch row {} lacks both outcomes", row.id));
                }
            }
            PhaseStatus::Excluded => {
                if row.exclusion_reason.as_deref().is_none_or(str::is_empty) {
                    return Err(format!("Phase exclusion row {} lacks a reason", row.id));
                }
            }
        }
    }
    let expected_mismatches: Vec<_> = matrix
        .rows
        .iter()
        .filter(|row| row.status == PhaseStatus::Mismatch)
        .map(|row| row.id.clone())
        .collect();
    if expected_mismatches.is_empty() || matrix.mismatches != expected_mismatches {
        return Err("Phase mismatch index is empty or stale".to_string());
    }
    Ok(())
}

fn build_evidence(manifest: &Manifest, cases: &[ReplayCase], matrix: &PhaseMatrix) -> Evidence {
    let mut prompt_kinds = BTreeSet::new();
    let mut action_types = BTreeSet::new();
    for case in cases {
        for command in &case.commands {
            prompt_kinds.insert(command.prompt_kind.clone());
            action_types.insert(command.action_type.clone());
        }
    }
    Evidence {
        schema_version: SCHEMA_VERSION,
        contract_id: CONTRACT_ID.to_string(),
        matchup_source_digest: matchup_source_digest(),
        content_digest: default_content_pack().content_digest(),
        reference: REFERENCE_ID.to_string(),
        optimized: OPTIMIZED_ID.to_string(),
        limitation: "Both executors share low-level rules/card primitives; this is not a second Comprehensive Rules engine.".to_string(),
        replays: cases
            .iter()
            .map(|case| ReplaySummary {
                case_id: case.case_id.clone(),
                orientation: case.orientation,
                commands: case.commands.len(),
                outcome: case.outcome.clone(),
                final_state_hash: case.final_checkpoint.state_hash.clone(),
            })
            .collect(),
        exercised_prompt_kinds: prompt_kinds.into_iter().collect(),
        exercised_action_types: action_types.into_iter().collect(),
        property_checks: vec![
            "fresh_same_seed_semantic_tapes_are_identical".to_string(),
            "semantic_action_lookup_is_independent_of_diagnostic_index".to_string(),
            "nonterminal_prompts_are_nonempty".to_string(),
            "cloned_sibling_does_not_mutate_root".to_string(),
        ],
        metamorphic_checks: vec![
            "explicit_singletons_equal_trivial_step_collapse".to_string(),
            "equal_content_pack_reallocation_preserves_trace".to_string(),
            "compatibility_card_renames_preserve_trace".to_string(),
            "fresh_and_cloned_roots_preserve_trace".to_string(),
        ],
        fuzz: manifest.fuzz.clone(),
        phase_revision: matrix.phase_revision.clone(),
        phase_status_counts: phase_counts(matrix),
        phase_mismatches: matrix.mismatches.clone(),
    }
}

fn replay_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = root.join("replays");
    let mut paths: Vec<_> = fs::read_dir(&directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn regression_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = root.join("regressions");
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<_> = fs::read_dir(&directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    Ok(paths)
}

pub fn record_root(root: &Path) -> Result<CheckSummary, String> {
    let manifest = load_manifest(root)?;
    let matrix: PhaseMatrix = load_json(&root.join("phase-overlap.json"))?;
    validate_phase_matrix(&matrix)?;
    let replay_directory = root.join("replays");
    fs::create_dir_all(&replay_directory)
        .map_err(|error| format!("{}: {error}", replay_directory.display()))?;
    for stale in replay_paths(root)? {
        fs::remove_file(&stale).map_err(|error| format!("{}: {error}", stale.display()))?;
    }
    let mut cases = Vec::new();
    for spec in &manifest.cases {
        let case = generate_case(spec)?;
        atomic_write_json(&replay_directory.join(format!("{}.json", spec.id)), &case)?;
        cases.push(case);
    }
    let evidence = build_evidence(&manifest, &cases, &matrix);
    atomic_write_json(&root.join("evidence.json"), &evidence)?;
    Ok(CheckSummary {
        replay_cases: cases.len(),
        replay_commands: cases.iter().map(|case| case.commands.len()).sum(),
        phase_rows: matrix.rows.len(),
    })
}

pub fn check_root(root: &Path) -> Result<CheckSummary, String> {
    let manifest = load_manifest(root)?;
    let matrix: PhaseMatrix = load_json(&root.join("phase-overlap.json"))?;
    validate_phase_matrix(&matrix)?;
    let paths = replay_paths(root)?;
    if paths.len() != manifest.cases.len() {
        return Err(format!(
            "expected {} replay files, found {}",
            manifest.cases.len(),
            paths.len()
        ));
    }
    let mut cases = Vec::new();
    for spec in &manifest.cases {
        let path = root.join("replays").join(format!("{}.json", spec.id));
        let case: ReplayCase = load_json(&path)?;
        if case.case_id != spec.id
            || case.orientation != spec.orientation
            || case.game_seed != spec.game_seed
            || case.choice_seed != spec.choice_seed
            || case.max_commands != spec.max_commands
        {
            return Err(format!(
                "{} does not match its manifest spec",
                path.display()
            ));
        }
        replay_case(&case)?;
        cases.push(case);
    }
    for path in regression_paths(root)? {
        let regression: ReplayCase = load_json(&path)?;
        replay_case(&regression)?;
    }
    let derived = build_evidence(&manifest, &cases, &matrix);
    let expected_bytes =
        fs::read(root.join("evidence.json")).map_err(|error| format!("evidence.json: {error}"))?;
    if expected_bytes != pretty_json(&derived)? {
        return Err("checked conformance evidence is stale; inspect and run record".to_string());
    }
    Ok(CheckSummary {
        replay_cases: cases.len(),
        replay_commands: cases.iter().map(|case| case.commands.len()).sum(),
        phase_rows: matrix.rows.len(),
    })
}

pub fn validate_contract_root(root: &Path) -> Result<(), String> {
    let _ = load_manifest(root)?;
    let matrix: PhaseMatrix = load_json(&root.join("phase-overlap.json"))?;
    validate_phase_matrix(&matrix)
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic payload".to_string()
    }
}

fn failure_path(directory: &Path, case_index: usize, receipt: &FailureReceipt) -> PathBuf {
    directory.join(format!(
        "case-{case_index:04}-game-{}-choice-{}.json",
        receipt.game_seed, receipt.choice_seed
    ))
}

fn persist_fuzz_failure(
    failure_directory: &Path,
    case_index: usize,
    spec: &CaseSpec,
    action_tape: Vec<Value>,
    last_matching_checkpoint: Option<Checkpoint>,
    message: String,
) -> Result<String, String> {
    let receipt = FailureReceipt {
        schema_version: SCHEMA_VERSION,
        contract_id: CONTRACT_ID.to_string(),
        matchup_source_digest: matchup_source_digest(),
        content_digest: default_content_pack().content_digest(),
        reference: REFERENCE_ID.to_string(),
        optimized: OPTIMIZED_ID.to_string(),
        orientation: spec.orientation,
        game_seed: spec.game_seed,
        choice_seed: spec.choice_seed,
        max_commands: spec.max_commands,
        failing_step: action_tape.len().saturating_sub(1),
        action_tape,
        last_matching_checkpoint,
        error: message.clone(),
    };
    let path = failure_path(failure_directory, case_index, &receipt);
    atomic_write_json(&path, &receipt).map_err(|write_error| {
        format!(
            "fuzz case failed: {message}; additionally could not persist {}: {write_error}",
            path.display()
        )
    })?;
    Ok(format!(
        "fuzz case {case_index} failed: {message}; replay {}",
        path.display()
    ))
}

pub fn fuzz(
    seed: u64,
    cases: usize,
    max_commands: usize,
    failure_directory: &Path,
) -> Result<FuzzSummary, String> {
    if cases == 0 || max_commands == 0 || max_commands > 2_000 {
        return Err(
            "fuzz cases and max_commands must be within bounded nonzero limits".to_string(),
        );
    }
    let mut total_commands = 0;
    for case_index in 0..cases {
        let orientation = if case_index % 2 == 0 {
            Orientation::UrLessonsVsGwAllies
        } else {
            Orientation::GwAlliesVsUrLessons
        };
        let spec = CaseSpec {
            id: format!("fuzz-{case_index:04}"),
            orientation,
            game_seed: mix_seed(seed, case_index as u64 * 2),
            choice_seed: mix_seed(seed, case_index as u64 * 2 + 1),
            max_commands,
        };
        let mut action_tape = Vec::new();
        let mut last_matching_checkpoint = None;
        let result = catch_unwind(AssertUnwindSafe(|| {
            generate_case_internal(&spec, &mut action_tape, &mut last_matching_checkpoint)
        }));
        match result {
            Ok(Ok(case)) => total_commands += case.commands.len(),
            Ok(Err(message)) => {
                return Err(persist_fuzz_failure(
                    failure_directory,
                    case_index,
                    &spec,
                    action_tape,
                    last_matching_checkpoint,
                    message,
                )?);
            }
            Err(payload) => {
                return Err(persist_fuzz_failure(
                    failure_directory,
                    case_index,
                    &spec,
                    action_tape,
                    last_matching_checkpoint,
                    format!("panic: {}", panic_message(payload)),
                )?);
            }
        }
    }
    Ok(FuzzSummary {
        cases,
        commands: total_commands,
    })
}

fn replay_failure_receipt(path: &Path, receipt: &FailureReceipt) -> Result<String, String> {
    if receipt.schema_version != SCHEMA_VERSION || receipt.contract_id != CONTRACT_ID {
        return Err("unsupported failure receipt contract".to_string());
    }
    if receipt.matchup_source_digest != matchup_source_digest()
        || receipt.content_digest != default_content_pack().content_digest()
        || receipt.reference != REFERENCE_ID
        || receipt.optimized != OPTIMIZED_ID
    {
        return Err("failure receipt source, content, or executor identity drift".to_string());
    }
    let spec = CaseSpec {
        id: "failure-replay".to_string(),
        orientation: receipt.orientation,
        game_seed: receipt.game_seed,
        choice_seed: receipt.choice_seed,
        max_commands: receipt.max_commands,
    };
    let mut action_tape = Vec::new();
    let mut last_matching_checkpoint = None;
    let result = catch_unwind(AssertUnwindSafe(|| {
        generate_case_internal(&spec, &mut action_tape, &mut last_matching_checkpoint)
    }));
    let observed_error = match result {
        Ok(Ok(_)) => {
            return Err(format!(
                "{} no longer reproduces: {}",
                path.display(),
                receipt.error
            ));
        }
        Ok(Err(message)) => message,
        Err(payload) => format!("panic: {}", panic_message(payload)),
    };
    let failing_step = action_tape.len().saturating_sub(1);
    if action_tape != receipt.action_tape
        || last_matching_checkpoint != receipt.last_matching_checkpoint
        || failing_step != receipt.failing_step
        || observed_error != receipt.error
    {
        return Err(format!(
            "{} failure reproduction drifted\nexpected_step={}\nactual_step={failing_step}\nexpected_error={}\nactual_error={observed_error}",
            path.display(),
            receipt.failing_step,
            receipt.error
        ));
    }
    Ok(format!(
        "reproduced fuzz failure at step {failing_step}: {observed_error}"
    ))
}

pub fn replay_receipt(path: &Path) -> Result<String, String> {
    match load_json(path)? {
        ReplayReceipt::Checked(case) => {
            let case_id = case.case_id.clone();
            let commands = case.commands.len();
            replay_case(&case)?;
            Ok(format!(
                "replayed checked case {case_id} through {commands} commands"
            ))
        }
        ReplayReceipt::Failure(receipt) => replay_failure_receipt(path, &receipt),
    }
}

pub fn verify_properties(spec: &CaseSpec) -> Result<(), String> {
    let first = generate_case(spec)?;
    let second = generate_case(spec)?;
    if first != second {
        return Err("fresh same-seed traces are not identical".to_string());
    }
    replay_case(&first)?;

    let root = ConformancePair::new(spec.orientation, spec.game_seed)?;
    let root_checkpoint = root.checkpoint()?;
    let mut sibling = root.clone();
    let action = selected_action(&sibling.optimized, 0)?.2;
    sibling.apply_semantic_action(&action)?;
    if root.checkpoint()? != root_checkpoint {
        return Err("mutating a sibling changed the root".to_string());
    }
    Ok(())
}

fn rename_compatibility_cards(pair: &mut ConformancePair) {
    for game in [&mut pair.reference, &mut pair.optimized] {
        for (index, card) in game.state.cards.iter_mut().enumerate() {
            card.name = format!("compatibility-name-{index}");
        }
    }
}

fn reallocate_content(pair: &mut ConformancePair) {
    for game in [&mut pair.reference, &mut pair.optimized] {
        game.state.content = Arc::new((*game.state.content).clone());
    }
}

pub fn verify_metamorphic(
    orientation: Orientation,
    game_seed: u64,
    choice_seed: u64,
    commands: usize,
) -> Result<(), String> {
    let mut baseline = ConformancePair::new(orientation, game_seed)?;
    let mut renamed = baseline.clone();
    rename_compatibility_cards(&mut renamed);
    let mut reallocated = baseline.clone();
    reallocate_content(&mut reallocated);
    if baseline.checkpoint()? != renamed.checkpoint()?
        || baseline.checkpoint()? != reallocated.checkpoint()?
    {
        return Err("metamorphic initial checkpoint changed".to_string());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(choice_seed);
    for step in 0..commands {
        if baseline.optimized.is_game_over() {
            break;
        }
        let action_count = baseline
            .optimized
            .action_space()
            .ok_or_else(|| format!("missing baseline prompt at step {step}"))?
            .actions
            .len();
        let index = rng.gen_range(0..action_count);
        let action = selected_action(&baseline.optimized, index)?.2;
        let expected = baseline.apply_semantic_action(&action)?;
        let renamed_checkpoint = renamed.apply_semantic_action(&action)?;
        let reallocated_checkpoint = reallocated.apply_semantic_action(&action)?;
        if expected != renamed_checkpoint || expected != reallocated_checkpoint {
            return Err(format!("metamorphic trace diverged at step {step}"));
        }
    }
    Ok(())
}
