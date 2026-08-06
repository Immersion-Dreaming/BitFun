//! Lifecycle facts for task-aware context eviction.
//!
//! This module deliberately owns no message rewriting. The execution loop records
//! deterministic facts here; its worker may query the configured fast model with
//! immutable snapshots, then applies the result through a guarded reducer.

use crate::agentic::core::message::{Message, MessageContent, ToolCall};
use crate::infrastructure::ai::get_global_ai_client_factory;
use crate::util::errors::{BitFunError, BitFunResult};
use crate::util::types::Message as AIMessage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

pub(crate) const LIFECYCLE_REGISTRY_METADATA_KEY: &str = "lifecycleEvictionRegistry";

const REGISTRY_SCHEMA_VERSION: u32 = 3;
const RESULT_PREVIEW_CHARS: usize = 600;
const LIFECYCLE_BATCH_SIZE: u64 = 3;
const RECENT_SEGMENTS_TO_PROTECT: u64 = 2;
const ESTIMATOR_TIMEOUT: Duration = Duration::from_secs(30);

fn default_schema_version() -> u32 {
    REGISTRY_SCHEMA_VERSION
}

/// Lifecycle state belongs to an intra-turn work unit. A root objective and its
/// control segments are retained; a segment is only the eventual physical
/// replacement unit and must not independently become evictable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LifecycleTaskState {
    Active,
    Completed,
    Evictable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LifecycleWorkUnitSource {
    RootFallback,
    Todo,
}

impl Default for LifecycleWorkUnitSource {
    fn default() -> Self {
        Self::RootFallback
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LifecycleTodoStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

impl Default for LifecycleTodoStatus {
    fn default() -> Self {
        Self::Pending
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleTodoItem {
    pub(crate) key: String,
    pub(crate) todo_id: String,
    pub(crate) root_task_id: String,
    pub(crate) content: String,
    pub(crate) status: LifecycleTodoStatus,
    pub(crate) last_update_tick: u64,
}

impl Default for LifecycleTaskState {
    fn default() -> Self {
        Self::Active
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleTask {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) root_task_id: String,
    #[serde(default)]
    pub(crate) source: LifecycleWorkUnitSource,
    /// Todo work units are evictable candidates; root/control work units are
    /// retained because they carry the user objective or execution plan.
    #[serde(default)]
    pub(crate) control_only: bool,
    #[serde(default)]
    pub(crate) todo_ids: Vec<String>,
    /// Revision is the optimistic-concurrency token for one task, not for the
    /// entire registry. It is incremented whenever deterministic facts change.
    #[serde(default)]
    pub(crate) revision: u64,
    #[serde(default)]
    pub(crate) title: Option<String>,
    /// Immutable source fact. It is never truncated in persistence.
    #[serde(default)]
    pub(crate) original_user_prompt: String,
    /// Initially equals the complete user prompt. A future estimator may add a
    /// normalized objective but may not overwrite the original prompt.
    #[serde(default)]
    pub(crate) objective: String,
    #[serde(default)]
    pub(crate) acceptance_criteria: Vec<String>,
    #[serde(default)]
    pub(crate) lifecycle: LifecycleTaskState,
    #[serde(default)]
    pub(crate) covered_turn_ids: Vec<String>,
    #[serde(default)]
    pub(crate) segment_ids: Vec<String>,
    #[serde(default)]
    pub(crate) completion_evidence: Vec<String>,
    #[serde(default)]
    pub(crate) unresolved_items: Vec<String>,
    #[serde(default)]
    pub(crate) dependencies: Vec<String>,
    #[serde(default)]
    pub(crate) last_activity_tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleToolFact {
    pub(crate) tool_id: String,
    pub(crate) tool_name: String,
    pub(crate) arguments_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleResultFact {
    pub(crate) tool_id: String,
    pub(crate) tool_name: String,
    pub(crate) is_error: bool,
    pub(crate) char_count: usize,
    pub(crate) content_sha256: String,
    pub(crate) preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleSegment {
    pub(crate) id: String,
    pub(crate) turn_id: String,
    /// Empty only while reading a v1 registry. `normalize_after_load` assigns
    /// a conservative legacy owner before the registry is used.
    #[serde(default)]
    pub(crate) owner_task_id: String,
    pub(crate) tick_seq: u64,
    #[serde(default)]
    pub(crate) tool_call_ids: Vec<String>,
    #[serde(default)]
    pub(crate) tool_names: Vec<String>,
    #[serde(default)]
    pub(crate) tool_facts: Vec<LifecycleToolFact>,
    #[serde(default)]
    pub(crate) result_facts: Vec<LifecycleResultFact>,
    #[serde(default)]
    pub(crate) has_error: bool,
    #[serde(default)]
    pub(crate) has_pending_todo: bool,
    #[serde(default)]
    pub(crate) is_control_segment: bool,
    #[serde(default)]
    pub(crate) todo_ids: Vec<String>,
    #[serde(default)]
    pub(crate) token_estimate: usize,
    #[serde(default)]
    pub(crate) evicted: bool,
}

/// Durable, deterministic lifecycle facts. `registry_revision` is only an
/// audit marker; async reducer correctness will use `LifecycleTask::revision`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleRegistry {
    #[serde(default = "default_schema_version")]
    pub(crate) schema_version: u32,
    #[serde(default)]
    pub(crate) registry_revision: u64,
    #[serde(default)]
    pub(crate) latest_tick_seq: u64,
    #[serde(default)]
    pub(crate) last_snapshot_tick_seq: u64,
    #[serde(default)]
    pub(crate) last_finished_snapshot_tick_seq: u64,
    #[serde(default)]
    pub(crate) next_snapshot_seq: u64,
    #[serde(default)]
    pub(crate) tasks: BTreeMap<String, LifecycleTask>,
    #[serde(default)]
    pub(crate) segments: BTreeMap<String, LifecycleSegment>,
    #[serde(default)]
    pub(crate) todo_ledger: BTreeMap<String, LifecycleTodoItem>,
    /// Maps a dialog turn to its retained root objective work unit. Ordinary
    /// segments may instead belong to Todo-derived work units beneath it.
    #[serde(default)]
    pub(crate) turn_task_ids: BTreeMap<String, String>,
    /// Current work-unit owner for ordinary tool segments in each dialog turn.
    #[serde(default)]
    pub(crate) active_work_unit_ids: BTreeMap<String, String>,
}

impl Default for LifecycleRegistry {
    fn default() -> Self {
        Self {
            schema_version: REGISTRY_SCHEMA_VERSION,
            registry_revision: 0,
            latest_tick_seq: 0,
            last_snapshot_tick_seq: 0,
            last_finished_snapshot_tick_seq: 0,
            next_snapshot_seq: 0,
            tasks: BTreeMap::new(),
            segments: BTreeMap::new(),
            todo_ledger: BTreeMap::new(),
            turn_task_ids: BTreeMap::new(),
            active_work_unit_ids: BTreeMap::new(),
        }
    }
}

/// Immutable estimator input. It contains deterministic task and tool facts,
/// but deliberately contains no raw assistant reasoning content.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleEstimatorSnapshot {
    pub(crate) schema_version: u32,
    pub(crate) snapshot_id: String,
    pub(crate) created_at_tick: u64,
    pub(crate) current_turn_id: String,
    pub(crate) current_task_id: String,
    /// The complete source prompt is exposed only for the current user task.
    pub(crate) current_user_prompt: String,
    pub(crate) todos: Vec<LifecycleTodoItem>,
    pub(crate) tasks: Vec<LifecycleTaskSnapshot>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleTaskSnapshot {
    pub(crate) task_id: String,
    pub(crate) root_task_id: String,
    pub(crate) source: LifecycleWorkUnitSource,
    pub(crate) control_only: bool,
    pub(crate) todo_ids: Vec<String>,
    pub(crate) expected_revision: u64,
    pub(crate) title: Option<String>,
    /// Older tasks do not repeat the raw source prompt. This field is present
    /// only after a future normalized objective differs from that source.
    pub(crate) normalized_objective: Option<String>,
    pub(crate) lifecycle: LifecycleTaskState,
    pub(crate) acceptance_criteria: Vec<String>,
    pub(crate) completion_evidence: Vec<String>,
    pub(crate) unresolved_items: Vec<String>,
    pub(crate) dependencies: Vec<String>,
    pub(crate) last_activity_tick: u64,
    pub(crate) segments: Vec<LifecycleSegment>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleEstimatorDelta {
    pub(crate) snapshot_id: String,
    #[serde(default)]
    pub(crate) task_updates: Vec<LifecycleTaskUpdate>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleTaskUpdate {
    pub(crate) task_id: String,
    pub(crate) expected_revision: u64,
    pub(crate) lifecycle: LifecycleTaskState,
    #[serde(default)]
    pub(crate) completion_evidence: Vec<String>,
    #[serde(default)]
    pub(crate) unresolved_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleReducerRejection {
    pub(crate) task_id: String,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleReducerOutcome {
    pub(crate) applied_task_ids: Vec<String>,
    pub(crate) rejected_updates: Vec<LifecycleReducerRejection>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleWorkUnitProtection {
    pub(crate) task_id: String,
    pub(crate) reasons: Vec<String>,
}

#[derive(Debug, Clone)]
struct LifecycleTodoWrite {
    todos: Vec<(String, String, LifecycleTodoStatus)>,
    replaces_root: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LifecycleScheduleDisposition {
    Started,
    Coalesced,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LifecycleWorkerResult {
    pub(crate) snapshot: LifecycleEstimatorSnapshot,
    pub(crate) response: Result<String, String>,
    pub(crate) duration_ms: u64,
}

struct LifecycleInflightJob {
    snapshot: LifecycleEstimatorSnapshot,
    handle: JoinHandle<LifecycleWorkerResult>,
}

impl Drop for LifecycleInflightJob {
    fn drop(&mut self) {
        // Tokio detaches a task when its JoinHandle is dropped. Lifecycle jobs
        // must instead stop with their scheduler so a destroyed engine cannot
        // leave a paid estimator request running and then recover the same
        // snapshot in a new engine instance.
        self.handle.abort();
    }
}

trait LifecycleEstimatorWorker: Send + Sync {
    fn spawn(&self, snapshot: LifecycleEstimatorSnapshot) -> LifecycleInflightJob;
}

struct FastModelLifecycleEstimatorWorker;

impl LifecycleEstimatorWorker for FastModelLifecycleEstimatorWorker {
    fn spawn(&self, snapshot: LifecycleEstimatorSnapshot) -> LifecycleInflightJob {
        spawn_lifecycle_worker(snapshot, |snapshot| async move {
            estimate_snapshot_with_fast_model(snapshot).await
        })
    }
}

#[derive(Default)]
struct LifecycleSessionScheduler {
    inflight: Option<LifecycleInflightJob>,
    pending: Option<LifecycleEstimatorSnapshot>,
}

/// Process-local asynchronous scheduler. Its eventual engine owner keys jobs
/// by session, so distinct sessions cannot stall each other while a single
/// session retains deterministic result ordering.
pub(crate) struct LifecycleEstimatorScheduler {
    worker: Arc<dyn LifecycleEstimatorWorker>,
    sessions: BTreeMap<String, LifecycleSessionScheduler>,
}

impl LifecycleEstimatorScheduler {
    fn new(worker: Arc<dyn LifecycleEstimatorWorker>) -> Self {
        Self {
            worker,
            sessions: BTreeMap::new(),
        }
    }

    pub(crate) fn with_fast_model_worker() -> Self {
        Self::new(Arc::new(FastModelLifecycleEstimatorWorker))
    }

    pub(crate) fn submit(
        &mut self,
        session_id: &str,
        snapshot: LifecycleEstimatorSnapshot,
    ) -> LifecycleScheduleDisposition {
        let state = self.sessions.entry(session_id.to_string()).or_default();
        if state.inflight.is_some() {
            state.pending = Some(snapshot);
            return LifecycleScheduleDisposition::Coalesced;
        }
        state.inflight = Some(self.worker.spawn(snapshot));
        LifecycleScheduleDisposition::Started
    }

    /// Polls only a completed handle. Awaiting after `is_finished` does not put
    /// an estimator request on the agent critical path.
    pub(crate) async fn poll_ready(&mut self, session_id: &str) -> Option<LifecycleWorkerResult> {
        let mut job = {
            let state = self.sessions.get_mut(session_id)?;
            if !state
                .inflight
                .as_ref()
                .is_some_and(|job| job.handle.is_finished())
            {
                return None;
            }
            state.inflight.take().expect("checked above")
        };
        let result = match (&mut job.handle).await {
            Ok(result) => result,
            Err(error) => LifecycleWorkerResult {
                snapshot: job.snapshot.clone(),
                response: Err(format!("lifecycle estimator worker join error: {error}")),
                duration_ms: 0,
            },
        };
        let remove_session = {
            let state = self
                .sessions
                .get_mut(session_id)
                .expect("session state exists");
            if let Some(next) = state.pending.take() {
                state.inflight = Some(self.worker.spawn(next));
            }
            state.inflight.is_none() && state.pending.is_none()
        };
        if remove_session {
            self.sessions.remove(session_id);
        }
        Some(result)
    }

    pub(crate) fn has_work(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .is_some_and(|state| state.inflight.is_some() || state.pending.is_some())
    }
}

fn spawn_lifecycle_worker<F, Fut>(
    snapshot: LifecycleEstimatorSnapshot,
    worker: F,
) -> LifecycleInflightJob
where
    F: FnOnce(LifecycleEstimatorSnapshot) -> Fut + Send + 'static,
    Fut: Future<Output = Result<String, String>> + Send + 'static,
{
    let job_snapshot = snapshot.clone();
    let handle = tokio::spawn(async move {
        let started = Instant::now();
        let response = worker(snapshot.clone()).await;
        LifecycleWorkerResult {
            snapshot,
            response,
            duration_ms: started.elapsed().as_millis() as u64,
        }
    });
    LifecycleInflightJob {
        snapshot: job_snapshot,
        handle,
    }
}

async fn estimate_snapshot_with_fast_model(
    snapshot: LifecycleEstimatorSnapshot,
) -> Result<String, String> {
    const PROMPT: &str = r#"You are a conservative lifecycle estimator for a software-engineering agent. Return ONLY JSON with this schema: {"snapshotId": string, "taskUpdates": [{"taskId": string, "expectedRevision": number, "lifecycle": "active"|"completed"|"evictable", "completionEvidence": [string], "unresolvedItems": [string]}]}. The currentUserPrompt is the immutable root objective. Each task is a work unit; tasks with source root_fallback or controlOnly true are retained control context and must never be made evictable. Todo tasks refer to current Todo ledger entries by todoIds. Never make a task evictable if a linked Todo is not completed, there is an error, unresolved item, recent activity, a dependency, or uncertainty. A task must first be completed with concrete evidence before it becomes evictable. Use only task IDs and revisions from the input. Prefer no update over a risky update."#;
    let input = serde_json::to_string(&snapshot)
        .map_err(|error| format!("lifecycle snapshot serialization: {error}"))?;
    let request = async {
        let factory = get_global_ai_client_factory()
            .await
            .map_err(|error| format!("lifecycle estimator factory: {error}"))?;
        let client = factory
            .get_client_resolved("fast")
            .await
            .map_err(|error| format!("lifecycle estimator client: {error}"))?;
        let response = client
            .send_message_with_trace(
                vec![
                    AIMessage::system(PROMPT.to_string()),
                    AIMessage::user(input),
                ],
                None,
                None,
            )
            .await
            .map_err(|error| format!("lifecycle estimator request: {error}"))?;
        Ok::<String, String>(response.text)
    };
    tokio::time::timeout(ESTIMATOR_TIMEOUT, request)
        .await
        .map_err(|_| {
            format!(
                "lifecycle estimator timed out after {}s",
                ESTIMATOR_TIMEOUT.as_secs()
            )
        })?
}

pub(crate) fn parse_estimator_delta(text: &str) -> BitFunResult<LifecycleEstimatorDelta> {
    let json = extract_json_object(text)
        .ok_or_else(|| BitFunError::tool("lifecycle estimator response has no JSON object"))?;
    serde_json::from_str(json)
        .map_err(|error| BitFunError::tool(format!("lifecycle estimator invalid delta: {error}")))
}

impl LifecycleRegistry {
    /// Creates one retained root objective for an actual user turn. The source
    /// prompt is kept exactly as supplied; no character cap is applied.
    pub(crate) fn observe_user_turn(
        &mut self,
        turn_id: &str,
        original_user_prompt: &str,
    ) -> String {
        if let Some(task_id) = self.turn_task_ids.get(turn_id) {
            return task_id.clone();
        }
        let task_id = format!("task:{turn_id}");
        self.tasks.insert(
            task_id.clone(),
            LifecycleTask {
                id: task_id.clone(),
                root_task_id: task_id.clone(),
                source: LifecycleWorkUnitSource::RootFallback,
                control_only: true,
                todo_ids: vec![],
                revision: 1,
                title: None,
                original_user_prompt: original_user_prompt.to_string(),
                objective: original_user_prompt.to_string(),
                acceptance_criteria: vec![],
                lifecycle: LifecycleTaskState::Active,
                covered_turn_ids: vec![turn_id.to_string()],
                segment_ids: vec![],
                completion_evidence: vec![],
                unresolved_items: vec![],
                dependencies: vec![],
                last_activity_tick: self.latest_tick_seq,
            },
        );
        self.turn_task_ids
            .insert(turn_id.to_string(), task_id.clone());
        self.active_work_unit_ids
            .insert(turn_id.to_string(), task_id.clone());
        self.registry_revision += 1;
        task_id
    }

    /// Records a completed assistant/tool-result round. It intentionally does
    /// not inspect reasoning content: raw chain-of-thought is not a lifecycle
    /// signal and is never copied into estimator input.
    pub(crate) fn record_segment(
        &mut self,
        messages: &[Message],
        assistant_idx: usize,
        end_idx: usize,
        turn_id: &str,
    ) -> Option<String> {
        let MessageContent::Mixed { tool_calls, .. } = &messages.get(assistant_idx)?.content else {
            return None;
        };
        if tool_calls.is_empty() {
            return None;
        }
        let results = messages.get(assistant_idx + 1..end_idx)?;
        let tool_call_ids: Vec<String> =
            tool_calls.iter().map(|call| call.tool_id.clone()).collect();
        let id = format!("{turn_id}:{}", tool_call_ids.join(","));
        if self.segments.contains_key(&id) {
            return Some(id);
        }

        let root_task_id = self.ensure_task_for_turn(turn_id);
        let todo_writes = todo_writes_from_results(tool_calls, results);
        // A failed TodoWrite is still plan/control context. It must stay on the
        // root even though its requested state is not admitted to the ledger.
        let is_control_segment = tool_calls.iter().any(|call| call.tool_name == "TodoWrite");
        let owner_task_id = if is_control_segment {
            root_task_id.clone()
        } else {
            self.active_work_unit_ids
                .get(turn_id)
                .cloned()
                .unwrap_or_else(|| root_task_id.clone())
        };
        self.latest_tick_seq += 1;
        let tool_facts = tool_calls
            .iter()
            .map(|call| LifecycleToolFact {
                tool_id: call.tool_id.clone(),
                tool_name: call.tool_name.clone(),
                arguments_json: serde_json::to_string(&call.arguments).unwrap_or_default(),
            })
            .collect();
        let result_facts = results.iter().filter_map(result_fact).collect::<Vec<_>>();
        let has_error = result_facts.iter().any(|fact| fact.is_error);
        let has_pending_todo =
            todo_writes
                .iter()
                .flat_map(|write| &write.todos)
                .any(|(_, _, status)| {
                    matches!(
                        status,
                        LifecycleTodoStatus::Pending | LifecycleTodoStatus::InProgress
                    )
                });
        let token_estimate = result_facts
            .iter()
            .map(|fact| fact.char_count)
            .sum::<usize>()
            / 4;

        self.segments.insert(
            id.clone(),
            LifecycleSegment {
                id: id.clone(),
                turn_id: turn_id.to_string(),
                owner_task_id: owner_task_id.clone(),
                tick_seq: self.latest_tick_seq,
                tool_call_ids,
                tool_names: tool_calls
                    .iter()
                    .map(|call| call.tool_name.clone())
                    .collect(),
                tool_facts,
                result_facts,
                has_error,
                has_pending_todo,
                is_control_segment,
                todo_ids: todo_writes
                    .iter()
                    .flat_map(|write| &write.todos)
                    .map(|(todo_id, _, _)| todo_id.clone())
                    .collect(),
                token_estimate,
                evicted: false,
            },
        );
        {
            let task = self
                .tasks
                .get_mut(&owner_task_id)
                .expect("owner task exists");
            if !is_control_segment && task.lifecycle != LifecycleTaskState::Active {
                task.lifecycle = LifecycleTaskState::Active;
                task.completion_evidence.clear();
                task.unresolved_items.clear();
            }
            task.segment_ids.push(id.clone());
            task.last_activity_tick = self.latest_tick_seq;
            task.revision += 1;
        }
        for todo_write in todo_writes {
            self.apply_todo_updates(&root_task_id, todo_write.todos, todo_write.replaces_root);
        }
        self.registry_revision += 1;
        Some(id)
    }

    fn apply_todo_updates(
        &mut self,
        root_task_id: &str,
        updates: Vec<(String, String, LifecycleTodoStatus)>,
        replaces_root: bool,
    ) {
        if replaces_root {
            let updated_keys: BTreeSet<String> = updates
                .iter()
                .map(|(todo_id, _, _)| format!("{root_task_id}:{todo_id}"))
                .collect();
            // A Todo-derived work unit retains a link to its originating item.
            // Do not delete that audit fact when a later whole-list update omits
            // it. An omitted unfinished item is conservatively cancelled; an
            // already completed item stays completed and remains eligible for
            // lifecycle assessment.
            for item in self.todo_ledger.values_mut().filter(|item| {
                item.root_task_id == root_task_id
                    && !updated_keys.contains(&item.key)
                    && item.status != LifecycleTodoStatus::Completed
            }) {
                item.status = LifecycleTodoStatus::Cancelled;
                item.last_update_tick = self.latest_tick_seq;
            }
        }
        for (todo_id, content, status) in updates {
            let key = format!("{root_task_id}:{todo_id}");
            self.todo_ledger.insert(
                key.clone(),
                LifecycleTodoItem {
                    key,
                    todo_id,
                    root_task_id: root_task_id.to_string(),
                    content,
                    status,
                    last_update_tick: self.latest_tick_seq,
                },
            );
        }
        self.refresh_active_work_unit(root_task_id);
    }

    fn refresh_active_work_unit(&mut self, root_task_id: &str) {
        let Some(turn_id) = self
            .tasks
            .get(root_task_id)
            .and_then(|task| task.covered_turn_ids.first())
            .cloned()
        else {
            return;
        };
        let in_progress: Vec<LifecycleTodoItem> = self
            .todo_ledger
            .values()
            .filter(|item| {
                item.root_task_id == root_task_id && item.status == LifecycleTodoStatus::InProgress
            })
            .cloned()
            .collect();
        let active_id = if in_progress.len() == 1 {
            Some(self.ensure_todo_work_unit(root_task_id, &in_progress[0]))
        } else {
            None
        };
        self.active_work_unit_ids.insert(
            turn_id,
            active_id.unwrap_or_else(|| root_task_id.to_string()),
        );
    }

    fn ensure_todo_work_unit(&mut self, root_task_id: &str, item: &LifecycleTodoItem) -> String {
        let task_id = format!("{root_task_id}:todo:{}", item.todo_id);
        let root_turn_id = self
            .tasks
            .get(root_task_id)
            .and_then(|task| task.covered_turn_ids.first())
            .cloned()
            .unwrap_or_default();
        let task = self
            .tasks
            .entry(task_id.clone())
            .or_insert_with(|| LifecycleTask {
                id: task_id.clone(),
                root_task_id: root_task_id.to_string(),
                source: LifecycleWorkUnitSource::Todo,
                control_only: false,
                todo_ids: vec![item.key.clone()],
                revision: 1,
                title: Some(item.content.clone()),
                original_user_prompt: String::new(),
                objective: item.content.clone(),
                acceptance_criteria: vec![],
                lifecycle: LifecycleTaskState::Active,
                covered_turn_ids: vec![root_turn_id],
                segment_ids: vec![],
                completion_evidence: vec![],
                unresolved_items: vec![],
                dependencies: vec![],
                last_activity_tick: self.latest_tick_seq,
            });
        if task.title.as_deref() != Some(item.content.as_str()) {
            task.title = Some(item.content.clone());
            task.objective = item.content.clone();
            task.revision += 1;
        }
        if !task.todo_ids.contains(&item.key) {
            task.todo_ids.push(item.key.clone());
            task.revision += 1;
        }
        task_id
    }

    /// Creates an immutable input every three newly recorded ticks. Scheduling
    /// state is persisted so a future async coordinator can coalesce snapshots
    /// without repeatedly submitting the same tick range.
    pub(crate) fn schedule_snapshot(
        &mut self,
        current_turn_id: &str,
    ) -> BitFunResult<Option<LifecycleEstimatorSnapshot>> {
        self.validate()?;
        if self
            .latest_tick_seq
            .saturating_sub(self.last_snapshot_tick_seq)
            < LIFECYCLE_BATCH_SIZE
        {
            return Ok(None);
        }
        self.create_snapshot(current_turn_id)
    }

    /// Recreates the latest snapshot only when a previous process submitted it
    /// but never observed a worker result. Completed batches are never replayed.
    pub(crate) fn schedule_recovery_snapshot(
        &mut self,
        current_turn_id: &str,
    ) -> BitFunResult<Option<LifecycleEstimatorSnapshot>> {
        self.validate()?;
        if self.last_snapshot_tick_seq <= self.last_finished_snapshot_tick_seq
            || self.latest_tick_seq < LIFECYCLE_BATCH_SIZE
        {
            return Ok(None);
        }
        self.create_snapshot(current_turn_id)
    }

    pub(crate) fn mark_snapshot_finished(&mut self, snapshot: &LifecycleEstimatorSnapshot) {
        self.last_finished_snapshot_tick_seq = self
            .last_finished_snapshot_tick_seq
            .max(snapshot.created_at_tick);
        self.registry_revision += 1;
    }

    fn create_snapshot(
        &mut self,
        current_turn_id: &str,
    ) -> BitFunResult<Option<LifecycleEstimatorSnapshot>> {
        let current_root_task_id = self
            .turn_task_ids
            .get(current_turn_id)
            .cloned()
            .ok_or_else(|| {
                BitFunError::tool(format!(
                    "lifecycle snapshot has no task for current turn {current_turn_id}"
                ))
            })?;
        let current_user_prompt = self
            .tasks
            .get(&current_root_task_id)
            .ok_or_else(|| {
                BitFunError::tool(format!(
                    "lifecycle snapshot current root task disappeared: {current_root_task_id}"
                ))
            })?
            .original_user_prompt
            .clone();
        let current_task_id = self
            .active_work_unit_ids
            .get(current_turn_id)
            .cloned()
            .unwrap_or_else(|| current_root_task_id.clone());

        self.next_snapshot_seq += 1;
        self.last_snapshot_tick_seq = self.latest_tick_seq;
        self.registry_revision += 1;
        let snapshot_id = format!(
            "lifecycle-snapshot-{}-tick-{}",
            self.next_snapshot_seq, self.latest_tick_seq
        );
        let tasks = self
            .tasks
            .values()
            .map(|task| LifecycleTaskSnapshot {
                task_id: task.id.clone(),
                root_task_id: task.root_task_id.clone(),
                source: task.source,
                control_only: task.control_only,
                todo_ids: task.todo_ids.clone(),
                expected_revision: task.revision,
                title: task.title.clone(),
                normalized_objective: (task.id != current_task_id
                    && task.id != task.root_task_id
                    && !task.objective.is_empty())
                .then(|| task.objective.clone()),
                lifecycle: task.lifecycle,
                acceptance_criteria: task.acceptance_criteria.clone(),
                completion_evidence: task.completion_evidence.clone(),
                unresolved_items: task.unresolved_items.clone(),
                dependencies: task.dependencies.clone(),
                last_activity_tick: task.last_activity_tick,
                segments: task
                    .segment_ids
                    .iter()
                    .filter_map(|segment_id| self.segments.get(segment_id).cloned())
                    .collect(),
            })
            .collect();
        Ok(Some(LifecycleEstimatorSnapshot {
            schema_version: REGISTRY_SCHEMA_VERSION,
            snapshot_id,
            created_at_tick: self.latest_tick_seq,
            current_turn_id: current_turn_id.to_string(),
            current_task_id,
            current_user_prompt,
            todos: self.todo_ledger.values().cloned().collect(),
            tasks,
        }))
    }

    /// Applies independently validated task updates. A stale update is rejected
    /// by task ID while unrelated updates in the same estimator response remain
    /// eligible. This is intentionally a pure registry operation: no message,
    /// artifact, session, or cache mutation is permitted here.
    pub(crate) fn reduce_estimator_delta(
        &mut self,
        snapshot: &LifecycleEstimatorSnapshot,
        delta: LifecycleEstimatorDelta,
    ) -> BitFunResult<LifecycleReducerOutcome> {
        if delta.snapshot_id != snapshot.snapshot_id {
            return Err(BitFunError::tool(format!(
                "lifecycle delta snapshot {} does not match {}",
                delta.snapshot_id, snapshot.snapshot_id
            )));
        }
        let expected_revisions: BTreeMap<&str, u64> = snapshot
            .tasks
            .iter()
            .map(|task| (task.task_id.as_str(), task.expected_revision))
            .collect();
        let mut outcome = LifecycleReducerOutcome::default();
        for update in delta.task_updates {
            let Some(snapshot_revision) = expected_revisions.get(update.task_id.as_str()) else {
                outcome.rejected_updates.push(LifecycleReducerRejection {
                    task_id: update.task_id,
                    reason: "unknown_task_for_snapshot".to_string(),
                });
                continue;
            };
            if *snapshot_revision != update.expected_revision {
                outcome.rejected_updates.push(LifecycleReducerRejection {
                    task_id: update.task_id,
                    reason: "expected_revision_does_not_match_snapshot".to_string(),
                });
                continue;
            }
            let Some(current_task) = self.tasks.get(&update.task_id) else {
                outcome.rejected_updates.push(LifecycleReducerRejection {
                    task_id: update.task_id,
                    reason: "task_missing_from_registry".to_string(),
                });
                continue;
            };
            if current_task.revision != update.expected_revision {
                outcome.rejected_updates.push(LifecycleReducerRejection {
                    task_id: update.task_id,
                    reason: "stale_task_revision".to_string(),
                });
                continue;
            }
            if current_task.lifecycle == update.lifecycle
                && current_task.completion_evidence == update.completion_evidence
                && current_task.unresolved_items == update.unresolved_items
            {
                outcome.rejected_updates.push(LifecycleReducerRejection {
                    task_id: update.task_id,
                    reason: "no_effective_task_change".to_string(),
                });
                continue;
            }
            if let Some(reason) = self.rejection_reason_for_update(&update) {
                outcome.rejected_updates.push(LifecycleReducerRejection {
                    task_id: update.task_id,
                    reason,
                });
                continue;
            }

            let task = self
                .tasks
                .get_mut(&update.task_id)
                .expect("task was checked above");
            task.lifecycle = update.lifecycle;
            task.completion_evidence = update.completion_evidence;
            task.unresolved_items = update.unresolved_items;
            task.revision += 1;
            self.registry_revision += 1;
            outcome.applied_task_ids.push(update.task_id);
        }
        Ok(outcome)
    }

    /// Only a task marked evictable by the reducer may be a future physical
    /// candidate. This phase returns IDs for shadow analysis only.
    pub(crate) fn shadow_candidate_segment_ids(&self) -> Vec<String> {
        self.tasks
            .values()
            .filter(|task| {
                task.lifecycle == LifecycleTaskState::Evictable
                    && !self.task_is_deterministically_protected(&task.id)
            })
            .flat_map(|task| task.segment_ids.iter())
            .filter_map(|segment_id| self.segments.get(segment_id))
            .filter(|segment| !segment.evicted)
            .map(|segment| segment.id.clone())
            .collect()
    }

    fn rejection_reason_for_update(&self, update: &LifecycleTaskUpdate) -> Option<String> {
        let task = self.tasks.get(&update.task_id)?;
        match (task.lifecycle, update.lifecycle) {
            (LifecycleTaskState::Active, LifecycleTaskState::Evictable) => {
                return Some("direct_active_to_evictable_transition".to_string());
            }
            (LifecycleTaskState::Completed, LifecycleTaskState::Active)
            | (LifecycleTaskState::Evictable, LifecycleTaskState::Active)
            | (LifecycleTaskState::Evictable, LifecycleTaskState::Completed) => {
                return Some("model_cannot_reopen_or_downgrade_task".to_string());
            }
            _ => {}
        }
        if update.lifecycle == LifecycleTaskState::Completed
            && update.completion_evidence.is_empty()
        {
            return Some("completed_requires_evidence".to_string());
        }
        if update.lifecycle == LifecycleTaskState::Evictable {
            if update.completion_evidence.is_empty() {
                return Some("evictable_requires_completion_evidence".to_string());
            }
            if !update.unresolved_items.is_empty() {
                return Some("evictable_has_unresolved_items".to_string());
            }
            if self.task_is_deterministically_protected(&update.task_id) {
                return Some("evictable_task_is_deterministically_protected".to_string());
            }
        }
        None
    }

    fn task_is_deterministically_protected(&self, task_id: &str) -> bool {
        !self.task_protection_reasons(task_id).is_empty()
    }

    pub(crate) fn work_unit_protection_reasons(&self) -> Vec<LifecycleWorkUnitProtection> {
        self.tasks
            .keys()
            .map(|task_id| LifecycleWorkUnitProtection {
                task_id: task_id.clone(),
                reasons: self.task_protection_reasons(task_id),
            })
            .collect()
    }

    fn task_protection_reasons(&self, task_id: &str) -> Vec<String> {
        let Some(task) = self.tasks.get(task_id) else {
            return vec!["missing_work_unit".to_string()];
        };
        let mut reasons = Vec::new();
        if task.control_only {
            reasons.push("control_work_unit".to_string());
        }
        if self.latest_tick_seq.saturating_sub(task.last_activity_tick) < RECENT_SEGMENTS_TO_PROTECT
        {
            reasons.push("recent_activity".to_string());
        }
        if task.todo_ids.iter().any(|todo_key| {
            self.todo_ledger
                .get(todo_key)
                .is_none_or(|item| item.status != LifecycleTodoStatus::Completed)
        }) {
            reasons.push("todo_not_completed".to_string());
        }
        if task
            .segment_ids
            .iter()
            .filter_map(|id| self.segments.get(id))
            .any(|segment| segment.has_error)
        {
            reasons.push("tool_error".to_string());
        }
        if task
            .segment_ids
            .iter()
            .filter_map(|id| self.segments.get(id))
            .any(|segment| segment.evicted || segment.is_control_segment)
        {
            reasons.push("invalid_work_unit_segment".to_string());
        }
        if self.tasks.values().any(|other| {
            other.id != task_id
                && other.lifecycle == LifecycleTaskState::Active
                && other
                    .dependencies
                    .iter()
                    .any(|dependency| dependency == task_id)
        }) {
            reasons.push("active_dependency".to_string());
        }
        reasons
    }

    /// Converts v1/v2 state into a safe v3 registry. Legacy records
    /// are active and cannot be evicted until a new estimator run supplies task
    /// evidence. No historical user prompt is invented during migration.
    pub(crate) fn normalize_after_load(&mut self) {
        if self.schema_version >= REGISTRY_SCHEMA_VERSION
            && self
                .segments
                .values()
                .all(|segment| !segment.owner_task_id.is_empty())
            && self
                .tasks
                .values()
                .all(|task| !task.root_task_id.is_empty())
        {
            return;
        }
        for task in self.tasks.values_mut() {
            if task.root_task_id.is_empty() {
                task.root_task_id = task.id.clone();
            }
            // Existing v2 tasks were dialog-turn tasks, not validated Todo
            // work units. Keep them conservative and non-evictable.
            task.source = LifecycleWorkUnitSource::RootFallback;
            task.control_only = true;
            task.todo_ids.clear();
        }
        let segment_ids: Vec<String> = self.segments.keys().cloned().collect();
        for segment_id in segment_ids {
            let (turn_id, owner_task_id, tick_seq) = {
                let segment = self.segments.get(&segment_id).expect("segment exists");
                (
                    segment.turn_id.clone(),
                    segment.owner_task_id.clone(),
                    segment.tick_seq,
                )
            };
            let owner_task_id = if owner_task_id.is_empty() {
                let legacy_id = format!("legacy-task:{turn_id}");
                if !self.tasks.contains_key(&legacy_id) {
                    self.tasks.insert(
                        legacy_id.clone(),
                        LifecycleTask {
                            id: legacy_id.clone(),
                            root_task_id: legacy_id.clone(),
                            source: LifecycleWorkUnitSource::RootFallback,
                            control_only: true,
                            todo_ids: vec![],
                            revision: 1,
                            title: Some("Legacy lifecycle record".to_string()),
                            original_user_prompt: String::new(),
                            objective: "Legacy segment-only record; not eligible for eviction until re-estimated.".to_string(),
                            acceptance_criteria: vec![],
                            lifecycle: LifecycleTaskState::Active,
                            covered_turn_ids: vec![turn_id.clone()],
                            segment_ids: vec![],
                            completion_evidence: vec![],
                            unresolved_items: vec![],
                            dependencies: vec![],
                            last_activity_tick: tick_seq,
                        },
                    );
                }
                if let Some(segment) = self.segments.get_mut(&segment_id) {
                    segment.owner_task_id = legacy_id.clone();
                }
                legacy_id
            } else {
                owner_task_id
            };
            self.turn_task_ids
                .entry(turn_id)
                .or_insert_with(|| owner_task_id.clone());
            let task = self
                .tasks
                .entry(owner_task_id.clone())
                .or_insert_with(|| LifecycleTask {
                    id: owner_task_id.clone(),
                    root_task_id: owner_task_id.clone(),
                    source: LifecycleWorkUnitSource::RootFallback,
                    control_only: true,
                    todo_ids: vec![],
                    revision: 1,
                    title: Some("Recovered lifecycle task".to_string()),
                    original_user_prompt: String::new(),
                    objective:
                        "Recovered lifecycle task; not eligible for eviction until re-estimated."
                            .to_string(),
                    acceptance_criteria: vec![],
                    lifecycle: LifecycleTaskState::Active,
                    covered_turn_ids: vec![],
                    segment_ids: vec![],
                    completion_evidence: vec![],
                    unresolved_items: vec![],
                    dependencies: vec![],
                    last_activity_tick: tick_seq,
                });
            if !task.segment_ids.contains(&segment_id) {
                task.segment_ids.push(segment_id);
            }
            task.last_activity_tick = task.last_activity_tick.max(tick_seq);
        }
        for (turn_id, root_task_id) in &self.turn_task_ids {
            self.active_work_unit_ids
                .entry(turn_id.clone())
                .or_insert_with(|| root_task_id.clone());
        }
        self.schema_version = REGISTRY_SCHEMA_VERSION;
        self.registry_revision += 1;
    }

    pub(crate) fn validate(&self) -> BitFunResult<()> {
        if self.schema_version != REGISTRY_SCHEMA_VERSION {
            return Err(BitFunError::tool(format!(
                "unsupported lifecycle registry schema {}",
                self.schema_version
            )));
        }
        for (turn_id, task_id) in &self.turn_task_ids {
            let task = self.tasks.get(task_id).ok_or_else(|| {
                BitFunError::tool(format!(
                    "lifecycle turn {turn_id} references missing task {task_id}"
                ))
            })?;
            if !task.covered_turn_ids.contains(turn_id) {
                return Err(BitFunError::tool(format!(
                    "lifecycle task {task_id} does not cover mapped turn {turn_id}"
                )));
            }
        }
        let mut referenced = BTreeSet::new();
        for (task_id, task) in &self.tasks {
            if task.root_task_id.is_empty() || !self.tasks.contains_key(&task.root_task_id) {
                return Err(BitFunError::tool(format!(
                    "lifecycle work unit {task_id} references missing root {}",
                    task.root_task_id
                )));
            }
            for todo_key in &task.todo_ids {
                let todo = self.todo_ledger.get(todo_key).ok_or_else(|| {
                    BitFunError::tool(format!(
                        "lifecycle work unit {task_id} references missing Todo {todo_key}"
                    ))
                })?;
                if todo.root_task_id != task.root_task_id {
                    return Err(BitFunError::tool(format!(
                        "lifecycle Todo {todo_key} belongs to a different root"
                    )));
                }
            }
            for segment_id in &task.segment_ids {
                if !referenced.insert(segment_id) {
                    return Err(BitFunError::tool(format!(
                        "lifecycle segment {segment_id} belongs to more than one task"
                    )));
                }
                let segment = self.segments.get(segment_id).ok_or_else(|| {
                    BitFunError::tool(format!(
                        "lifecycle task {task_id} references missing segment {segment_id}"
                    ))
                })?;
                if segment.owner_task_id != *task_id {
                    return Err(BitFunError::tool(format!(
                        "lifecycle segment {segment_id} owner does not match task {task_id}"
                    )));
                }
            }
        }
        for (segment_id, segment) in &self.segments {
            if !self.tasks.contains_key(&segment.owner_task_id) || !referenced.contains(segment_id)
            {
                return Err(BitFunError::tool(format!(
                    "lifecycle segment {segment_id} has no valid task ownership"
                )));
            }
        }
        Ok(())
    }

    fn ensure_task_for_turn(&mut self, turn_id: &str) -> String {
        self.turn_task_ids
            .get(turn_id)
            .cloned()
            .unwrap_or_else(|| self.observe_user_turn(turn_id, ""))
    }
}

/// Reads the actual user input for one dialog turn. Internal reminders are not
/// a user-task source even though some are represented as user-role messages.
pub(crate) fn actual_user_prompt(messages: &[Message], turn_id: &str) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| {
            message.metadata.turn_id.as_deref() == Some(turn_id) && message.is_actual_user_message()
        })
        .and_then(|message| match &message.content {
            MessageContent::Text(text) | MessageContent::Multimodal { text, .. } => {
                Some(text.clone())
            }
            _ => None,
        })
}

fn result_fact(message: &Message) -> Option<LifecycleResultFact> {
    let MessageContent::ToolResult {
        tool_id,
        tool_name,
        result_for_assistant,
        is_error,
        ..
    } = &message.content
    else {
        return None;
    };
    let content = result_for_assistant.as_deref().unwrap_or("");
    let content_sha256 = format!("{:x}", Sha256::digest(content.as_bytes()));
    Some(LifecycleResultFact {
        tool_id: tool_id.clone(),
        tool_name: tool_name.clone(),
        is_error: *is_error,
        char_count: content.chars().count(),
        content_sha256,
        preview: truncate_chars(content, RESULT_PREVIEW_CHARS),
    })
}

fn todo_writes_from_results(calls: &[ToolCall], results: &[Message]) -> Vec<LifecycleTodoWrite> {
    calls
        .iter()
        .filter_map(|call| {
            if call.tool_name != "TodoWrite" {
                return None;
            }
            let result = results.iter().find_map(|message| match &message.content {
                MessageContent::ToolResult {
                    tool_id,
                    result,
                    is_error: false,
                    ..
                } if tool_id == &call.tool_id => Some(result),
                _ => None,
            })?;
            let todos = result
                .get("todos")
                .or_else(|| call.arguments.get("todos"))
                .and_then(|value| value.as_array())?;
            let todos = todos
                .iter()
                .enumerate()
                .filter_map(|(index, todo)| {
                    let todo_id = todo
                        .get("id")
                        .and_then(|value| value.as_str())
                        .map(ToString::to_string)
                        .unwrap_or_else(|| format!("anonymous-{index}"));
                    let content = todo
                        .get("content")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let status = match todo.get("status")?.as_str()? {
                        "pending" => LifecycleTodoStatus::Pending,
                        "in_progress" => LifecycleTodoStatus::InProgress,
                        "completed" => LifecycleTodoStatus::Completed,
                        "cancelled" => LifecycleTodoStatus::Cancelled,
                        _ => return None,
                    };
                    Some((todo_id, content, status))
                })
                .collect();
            Some(LifecycleTodoWrite {
                todos,
                // TodoWrite itself returns `merge: false`; defaulting to a
                // replacement keeps its documented whole-list semantics when
                // adapters omit that field.
                replaces_root: !result
                    .get("merge")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
            })
        })
        .collect()
}

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then_some(&text[start..=end])
}

pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let out: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() {
        format!("{out}...")
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::core::message::ToolResult;

    struct TestWorker {
        receivers:
            std::sync::Mutex<std::collections::VecDeque<tokio::sync::oneshot::Receiver<String>>>,
    }

    impl TestWorker {
        fn new(receiver: tokio::sync::oneshot::Receiver<String>) -> Self {
            Self {
                receivers: std::sync::Mutex::new(std::collections::VecDeque::from([receiver])),
            }
        }
    }

    impl LifecycleEstimatorWorker for TestWorker {
        fn spawn(&self, snapshot: LifecycleEstimatorSnapshot) -> LifecycleInflightJob {
            let receiver = self
                .receivers
                .lock()
                .unwrap()
                .pop_front()
                .expect("test worker was asked to start without a queued response");
            spawn_lifecycle_worker(snapshot, move |_| async move {
                receiver.await.map_err(|error| error.to_string())
            })
        }
    }

    struct AbortAwareFuture {
        aborted: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl std::future::Future for AbortAwareFuture {
        type Output = Result<String, String>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            std::task::Poll::Pending
        }
    }

    impl Drop for AbortAwareFuture {
        fn drop(&mut self) {
            self.aborted
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    struct PendingWorker {
        aborted: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl LifecycleEstimatorWorker for PendingWorker {
        fn spawn(&self, snapshot: LifecycleEstimatorSnapshot) -> LifecycleInflightJob {
            let aborted = self.aborted.clone();
            spawn_lifecycle_worker(snapshot, move |_| AbortAwareFuture { aborted })
        }
    }

    fn round(turn: &str, suffix: &str, error: bool, pending: bool) -> Vec<Message> {
        let mut calls = vec![ToolCall {
            tool_id: format!("{suffix}-read"),
            tool_name: "Read".into(),
            arguments: serde_json::json!({"file_path":"src/lib.rs"}),
            ..Default::default()
        }];
        if pending {
            calls.push(ToolCall {
                tool_id: format!("{suffix}-todo"),
                tool_name: "TodoWrite".into(),
                arguments: serde_json::json!({"todos":[{"id":format!("{suffix}-todo"),"content":"finish the work unit","status":"pending"}]}),
                ..Default::default()
            });
        }
        let mut messages = vec![
            Message::assistant_with_tools("inspect".into(), calls).with_turn_id(turn.into()),
            Message::tool_result(ToolResult {
                tool_id: format!("{suffix}-read"),
                tool_name: "Read".into(),
                result: serde_json::json!({}),
                result_for_assistant: Some("full result".into()),
                is_error: error,
                duration_ms: None,
                image_attachments: None,
            })
            .with_turn_id(turn.into()),
        ];
        if pending {
            messages.push(
                Message::tool_result(ToolResult {
                    tool_id: format!("{suffix}-todo"),
                    tool_name: "TodoWrite".into(),
                    result: serde_json::json!({"success": true}),
                    result_for_assistant: Some("todo updated".into()),
                    is_error: false,
                    duration_ms: None,
                    image_attachments: None,
                })
                .with_turn_id(turn.into()),
            );
        }
        messages
    }

    fn todo_round(turn: &str, suffix: &str, todo_id: &str, status: &str) -> Vec<Message> {
        vec![
            Message::assistant_with_tools(
                "update plan".into(),
                vec![ToolCall {
                    tool_id: format!("{suffix}-todo-call"),
                    tool_name: "TodoWrite".into(),
                    arguments: serde_json::json!({
                        "todos": [{
                            "id": todo_id,
                            "content": "finish the work unit",
                            "status": status
                        }]
                    }),
                    ..Default::default()
                }],
            )
            .with_turn_id(turn.into()),
            Message::tool_result(ToolResult {
                tool_id: format!("{suffix}-todo-call"),
                tool_name: "TodoWrite".into(),
                result: serde_json::json!({"success": true}),
                result_for_assistant: Some("todo updated".into()),
                is_error: false,
                duration_ms: None,
                image_attachments: None,
            })
            .with_turn_id(turn.into()),
        ]
    }

    #[test]
    fn preserves_full_user_prompt_and_assigns_a_provisional_task() {
        let mut registry = LifecycleRegistry::default();
        let prompt = format!("objective {}", "x".repeat(2_000));
        let task_id = registry.observe_user_turn("turn-1", &prompt);
        let task = &registry.tasks[&task_id];
        assert_eq!(task.original_user_prompt, prompt);
        assert_eq!(task.objective, prompt);
        assert_eq!(task.lifecycle, LifecycleTaskState::Active);
        registry.validate().unwrap();
    }

    #[test]
    fn records_segments_under_task_without_reasoning_content() {
        let mut registry = LifecycleRegistry::default();
        registry.observe_user_turn("turn-1", "fix the parser");
        let messages = round("turn-1", "one", false, false);
        let segment_id = registry.record_segment(&messages, 0, 2, "turn-1").unwrap();
        let segment = &registry.segments[&segment_id];
        let task = &registry.tasks[&segment.owner_task_id];
        assert_eq!(task.segment_ids, vec![segment_id.clone()]);
        assert_eq!(segment.tool_facts[0].tool_name, "Read");
        assert_eq!(segment.result_facts[0].preview, "full result");
        assert_eq!(registry.latest_tick_seq, 1);
        registry.validate().unwrap();
    }

    #[test]
    fn no_todo_fallback_is_a_control_work_unit_and_never_a_candidate() {
        let mut registry = LifecycleRegistry::default();
        let messages = round("turn-1", "one", false, false);
        let segment_id = registry.record_segment(&messages, 0, 2, "turn-1").unwrap();
        let task_id = registry.segments[&segment_id].owner_task_id.clone();
        assert_eq!(
            registry.tasks[&task_id].source,
            LifecycleWorkUnitSource::RootFallback
        );
        assert!(registry.tasks[&task_id].control_only);
        registry.tasks.get_mut(&task_id).unwrap().lifecycle = LifecycleTaskState::Evictable;
        registry
            .tasks
            .get_mut(&task_id)
            .unwrap()
            .completion_evidence = vec!["done".into()];
        registry.latest_tick_seq = 3;
        assert!(registry.shadow_candidate_segment_ids().is_empty());
    }

    #[test]
    fn todo_ledger_assigns_ordinary_segments_and_releases_completed_work_unit() {
        let mut registry = LifecycleRegistry::default();
        let root_id = registry.observe_user_turn("turn-1", "fix the parser");
        let started = todo_round("turn-1", "start", "one-todo", "in_progress");
        let control_segment = registry.record_segment(&started, 0, 2, "turn-1").unwrap();
        assert_eq!(registry.segments[&control_segment].owner_task_id, root_id);
        assert!(registry.segments[&control_segment].is_control_segment);

        let active_task_id = registry.active_work_unit_ids["turn-1"].clone();
        assert_eq!(
            registry.tasks[&active_task_id].source,
            LifecycleWorkUnitSource::Todo
        );
        assert!(registry
            .work_unit_protection_reasons()
            .iter()
            .find(|entry| entry.task_id == active_task_id)
            .unwrap()
            .reasons
            .contains(&"todo_not_completed".to_string()));
        let work = round("turn-1", "two", false, false);
        let work_segment = registry.record_segment(&work, 0, 2, "turn-1").unwrap();
        assert_eq!(
            registry.segments[&work_segment].owner_task_id,
            active_task_id
        );
        assert!(!registry.segments[&work_segment].is_control_segment);

        let finished = todo_round("turn-1", "finish", "one-todo", "completed");
        let finish_segment = registry.record_segment(&finished, 0, 2, "turn-1").unwrap();
        assert_eq!(
            registry.segments[&finish_segment].owner_task_id,
            "task:turn-1"
        );
        assert_eq!(
            registry.todo_ledger["task:turn-1:one-todo"].status,
            LifecycleTodoStatus::Completed
        );
        assert_eq!(registry.active_work_unit_ids["turn-1"], "task:turn-1");
        assert!(!registry
            .work_unit_protection_reasons()
            .iter()
            .find(|entry| entry.task_id == active_task_id)
            .unwrap()
            .reasons
            .contains(&"todo_not_completed".to_string()));
        registry.validate().unwrap();
    }

    #[test]
    fn latest_successful_todo_write_cancels_omitted_unfinished_items() {
        let mut registry = LifecycleRegistry::default();
        registry.observe_user_turn("turn-1", "fix the parser");
        let first = todo_round("turn-1", "first", "todo-1", "in_progress");
        registry.record_segment(&first, 0, 2, "turn-1").unwrap();

        let second = todo_round("turn-1", "second", "todo-2", "in_progress");
        registry.record_segment(&second, 0, 2, "turn-1").unwrap();

        assert_eq!(registry.todo_ledger.len(), 2);
        assert!(registry.todo_ledger.contains_key("task:turn-1:todo-2"));
        assert_eq!(
            registry.todo_ledger["task:turn-1:todo-1"].status,
            LifecycleTodoStatus::Cancelled
        );
        assert_eq!(
            registry.active_work_unit_ids["turn-1"],
            "task:turn-1:todo:todo-2"
        );
        registry.validate().unwrap();
    }

    #[test]
    fn todo_ledger_uses_generated_id_from_successful_result() {
        let mut registry = LifecycleRegistry::default();
        registry.observe_user_turn("turn-1", "fix the parser");
        let mut messages = todo_round("turn-1", "generated", "request-id", "in_progress");
        if let MessageContent::Mixed { tool_calls, .. } = &mut messages[0].content {
            tool_calls[0]
                .arguments
                .get_mut("todos")
                .and_then(|todos| todos.as_array_mut())
                .unwrap()[0]
                .as_object_mut()
                .unwrap()
                .remove("id");
        }
        if let MessageContent::ToolResult { result, .. } = &mut messages[1].content {
            *result = serde_json::json!({
                "success": true,
                "merge": false,
                "todos": [{
                    "id": "generated-id",
                    "content": "finish the work unit",
                    "status": "in_progress"
                }]
            });
        }
        registry.record_segment(&messages, 0, 2, "turn-1").unwrap();

        assert!(registry
            .todo_ledger
            .contains_key("task:turn-1:generated-id"));
        assert!(!registry.todo_ledger.contains_key("task:turn-1:anonymous-0"));
        registry.validate().unwrap();
    }

    #[test]
    fn snapshot_exposes_current_todo_work_unit_without_repeating_root_objective() {
        let mut registry = LifecycleRegistry::default();
        let root_prompt = format!("repair the parser {}", "x".repeat(1_500));
        registry.observe_user_turn("turn-1", &root_prompt);
        let started = todo_round("turn-1", "start", "todo-1", "in_progress");
        registry.record_segment(&started, 0, 2, "turn-1").unwrap();
        for suffix in ["inspect", "patch"] {
            let messages = round("turn-1", suffix, false, false);
            registry.record_segment(&messages, 0, 2, "turn-1").unwrap();
        }

        let snapshot = registry.schedule_snapshot("turn-1").unwrap().unwrap();
        let work_unit = snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1:todo:todo-1")
            .unwrap();
        assert_eq!(snapshot.current_user_prompt, root_prompt);
        assert_eq!(snapshot.current_task_id, work_unit.task_id);
        assert_eq!(work_unit.source, LifecycleWorkUnitSource::Todo);
        assert!(!work_unit.control_only);
        assert_eq!(work_unit.todo_ids, vec!["task:turn-1:todo-1"]);
        assert_eq!(work_unit.segments.len(), 2);
        assert_eq!(snapshot.todos.len(), 1);
        assert_eq!(snapshot.todos[0].status, LifecycleTodoStatus::InProgress);
        assert!(snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1")
            .unwrap()
            .normalized_objective
            .is_none());
        assert_eq!(
            serde_json::to_string(&snapshot)
                .unwrap()
                .matches(&root_prompt)
                .count(),
            1
        );
    }

    #[test]
    fn completed_todo_work_unit_becomes_candidate_during_later_work_in_same_turn() {
        let mut registry = LifecycleRegistry::default();
        registry.observe_user_turn("turn-1", "fix the parser");
        let start_one = todo_round("turn-1", "start-one", "todo-1", "in_progress");
        registry.record_segment(&start_one, 0, 2, "turn-1");
        let work_one = round("turn-1", "work-one", false, false);
        let work_one_segment = registry.record_segment(&work_one, 0, 2, "turn-1").unwrap();
        let finish_one = todo_round("turn-1", "finish-one", "todo-1", "completed");
        registry.record_segment(&finish_one, 0, 2, "turn-1");
        let start_two = todo_round("turn-1", "start-two", "todo-2", "in_progress");
        registry.record_segment(&start_two, 0, 2, "turn-1");
        for suffix in ["work-two-a", "work-two-b"] {
            let messages = round("turn-1", suffix, false, false);
            registry.record_segment(&messages, 0, 2, "turn-1");
        }
        let first_snapshot = registry.schedule_snapshot("turn-1").unwrap().unwrap();
        let first_task_id = "task:turn-1:todo:todo-1";
        let first_revision = first_snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == first_task_id)
            .unwrap()
            .expected_revision;
        let completed = registry
            .reduce_estimator_delta(
                &first_snapshot,
                LifecycleEstimatorDelta {
                    snapshot_id: first_snapshot.snapshot_id.clone(),
                    task_updates: vec![update(
                        first_task_id,
                        first_revision,
                        LifecycleTaskState::Completed,
                        &["Todo 1 is marked completed and its inspection finished"],
                    )],
                },
            )
            .unwrap();
        assert_eq!(completed.applied_task_ids, vec![first_task_id]);

        for suffix in ["work-two-c", "work-two-d", "work-two-e"] {
            let messages = round("turn-1", suffix, false, false);
            registry.record_segment(&messages, 0, 2, "turn-1");
        }
        let second_snapshot = registry.schedule_snapshot("turn-1").unwrap().unwrap();
        let first_revision = second_snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == first_task_id)
            .unwrap()
            .expected_revision;
        let evictable = registry
            .reduce_estimator_delta(
                &second_snapshot,
                LifecycleEstimatorDelta {
                    snapshot_id: second_snapshot.snapshot_id.clone(),
                    task_updates: vec![update(
                        first_task_id,
                        first_revision,
                        LifecycleTaskState::Evictable,
                        &["Todo 1 is marked completed and its inspection finished"],
                    )],
                },
            )
            .unwrap();
        assert_eq!(evictable.applied_task_ids, vec![first_task_id]);
        assert_eq!(
            registry.shadow_candidate_segment_ids(),
            vec![work_one_segment]
        );
    }

    #[test]
    fn failed_todo_write_does_not_change_the_work_unit_ledger() {
        let mut registry = LifecycleRegistry::default();
        registry.observe_user_turn("turn-1", "fix the parser");
        let mut failed = todo_round("turn-1", "failed", "todo-1", "in_progress");
        if let MessageContent::ToolResult { is_error, .. } = &mut failed[1].content {
            *is_error = true;
        }
        registry.record_segment(&failed, 0, 2, "turn-1").unwrap();
        assert!(registry.todo_ledger.is_empty());
        assert_eq!(registry.active_work_unit_ids["turn-1"], "task:turn-1");
        assert_eq!(registry.tasks.len(), 1);
        registry.validate().unwrap();
    }

    #[test]
    fn failed_todo_write_stays_pinned_to_root_during_active_work() {
        let mut registry = LifecycleRegistry::default();
        registry.observe_user_turn("turn-1", "fix the parser");
        let started = todo_round("turn-1", "start", "todo-1", "in_progress");
        registry.record_segment(&started, 0, 2, "turn-1");
        let active_task_id = registry.active_work_unit_ids["turn-1"].clone();
        let mut failed = todo_round("turn-1", "failed", "todo-1", "completed");
        if let MessageContent::ToolResult { is_error, .. } = &mut failed[1].content {
            *is_error = true;
        }
        let failed_segment = registry.record_segment(&failed, 0, 2, "turn-1").unwrap();
        assert_eq!(
            registry.segments[&failed_segment].owner_task_id,
            "task:turn-1"
        );
        assert_eq!(registry.active_work_unit_ids["turn-1"], active_task_id);
        assert_eq!(
            registry.todo_ledger["task:turn-1:todo-1"].status,
            LifecycleTodoStatus::InProgress
        );
        registry.validate().unwrap();
    }

    #[test]
    fn duplicate_recording_is_idempotent() {
        let mut registry = LifecycleRegistry::default();
        let messages = round("turn-1", "one", false, false);
        let first = registry.record_segment(&messages, 0, 2, "turn-1").unwrap();
        let second = registry.record_segment(&messages, 0, 2, "turn-1").unwrap();
        assert_eq!(first, second);
        assert_eq!(registry.latest_tick_seq, 1);
        assert_eq!(registry.segments.len(), 1);
        registry.validate().unwrap();
    }

    #[test]
    fn legacy_segment_only_registry_becomes_active_and_safe() {
        let mut registry: LifecycleRegistry = serde_json::from_value(serde_json::json!({
            "version": 2,
            "latest_tick_seq": 3,
            "segments": {
                "legacy": {
                    "id": "legacy",
                    "turn_id": "turn-1",
                    "tick_seq": 3,
                    "tool_call_ids": ["call-1"],
                    "tool_names": ["Read"],
                    "has_error": false,
                    "has_pending_todo": false,
                    "token_estimate": 10,
                    "state": "evictable",
                    "completion_evidence": ["old data"],
                    "unresolved_questions": [],
                    "evicted": false
                }
            }
        }))
        .unwrap();
        registry.normalize_after_load();
        let segment = &registry.segments["legacy"];
        let task = &registry.tasks[&segment.owner_task_id];
        assert_eq!(task.lifecycle, LifecycleTaskState::Active);
        assert!(task.completion_evidence.is_empty());
        registry.validate().unwrap();
    }

    #[test]
    fn captures_error_and_pending_todo_as_deterministic_facts() {
        let mut registry = LifecycleRegistry::default();
        let messages = round("turn-1", "one", true, true);
        let id = registry
            .record_segment(&messages, 0, messages.len(), "turn-1")
            .unwrap();
        let segment = &registry.segments[&id];
        assert!(segment.has_error);
        assert!(segment.has_pending_todo);
        registry.validate().unwrap();
    }

    #[test]
    fn extracts_only_actual_user_input_for_the_requested_turn() {
        let messages = vec![
            Message::user("ignore this internal text".into())
                .with_turn_id("turn-1".into())
                .with_internal_reminder_kind(
                    crate::agentic::core::message::InternalReminderKind::Generic,
                ),
            Message::user("keep the complete user objective".into())
                .with_turn_id("turn-1".into())
                .with_semantic_kind(
                    crate::agentic::core::message::MessageSemanticKind::ActualUserInput,
                ),
        ];
        assert_eq!(
            actual_user_prompt(&messages, "turn-1").as_deref(),
            Some("keep the complete user objective")
        );
    }

    #[test]
    fn v2_registry_round_trip_preserves_task_ownership_and_full_prompt() {
        let mut registry = LifecycleRegistry::default();
        let prompt = format!("implement all requirements {}", "q".repeat(1_500));
        registry.observe_user_turn("turn-1", &prompt);
        let messages = round("turn-1", "one", false, false);
        registry.record_segment(&messages, 0, 2, "turn-1");

        let mut restored: LifecycleRegistry =
            serde_json::from_value(serde_json::to_value(&registry).unwrap()).unwrap();
        restored.normalize_after_load();
        restored.validate().unwrap();
        let task = restored.tasks.get("task:turn-1").unwrap();
        assert_eq!(task.original_user_prompt, prompt);
        assert_eq!(task.segment_ids.len(), 1);
    }

    fn update(
        task_id: &str,
        expected_revision: u64,
        lifecycle: LifecycleTaskState,
        evidence: &[&str],
    ) -> LifecycleTaskUpdate {
        LifecycleTaskUpdate {
            task_id: task_id.to_string(),
            expected_revision,
            lifecycle,
            completion_evidence: evidence.iter().map(|value| (*value).to_string()).collect(),
            unresolved_items: vec![],
        }
    }

    fn snapshot_with_two_tasks() -> (
        LifecycleRegistry,
        LifecycleEstimatorSnapshot,
        String,
        String,
    ) {
        let mut registry = LifecycleRegistry::default();
        let old_prompt = format!("old objective {}", "a".repeat(1_300));
        let current_prompt = format!("current objective {}", "b".repeat(1_500));
        registry.observe_user_turn("turn-1", &old_prompt);
        for suffix in ["one", "two"] {
            let messages = round("turn-1", suffix, false, false);
            registry.record_segment(&messages, 0, 2, "turn-1");
        }
        registry.observe_user_turn("turn-2", &current_prompt);
        let messages = round("turn-2", "three", false, false);
        registry.record_segment(&messages, 0, 2, "turn-2");
        // This fixture models explicit non-control work units so reducer and
        // candidate tests can exercise the legal transition independently of
        // the production no-Todo fallback rule.
        for task in registry.tasks.values_mut() {
            task.control_only = false;
        }
        let snapshot = registry.schedule_snapshot("turn-2").unwrap().unwrap();
        (registry, snapshot, old_prompt, current_prompt)
    }

    #[test]
    fn snapshot_runs_per_three_ticks_and_only_repeats_current_raw_prompt() {
        let (mut registry, snapshot, old_prompt, current_prompt) = snapshot_with_two_tasks();
        assert_eq!(snapshot.created_at_tick, 3);
        assert_eq!(snapshot.current_user_prompt, current_prompt);
        assert_eq!(snapshot.tasks.len(), 2);
        assert!(snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1")
            .unwrap()
            .normalized_objective
            .is_none());
        let serialized = serde_json::to_string(&snapshot).unwrap();
        assert!(!serialized.contains(&old_prompt));
        assert!(!serialized.contains("reasoning_content"));
        assert!(registry.schedule_snapshot("turn-2").unwrap().is_none());
    }

    #[test]
    fn unfinished_snapshot_is_recovered_once_after_restart() {
        let (mut registry, first_snapshot, _, _) = snapshot_with_two_tasks();
        assert_eq!(registry.last_snapshot_tick_seq, 3);
        assert_eq!(registry.last_finished_snapshot_tick_seq, 0);

        let recovery_snapshot = registry
            .schedule_recovery_snapshot("turn-2")
            .unwrap()
            .unwrap();
        assert_ne!(recovery_snapshot.snapshot_id, first_snapshot.snapshot_id);
        assert_eq!(
            recovery_snapshot.created_at_tick,
            first_snapshot.created_at_tick
        );
        registry.mark_snapshot_finished(&recovery_snapshot);
        assert_eq!(registry.last_finished_snapshot_tick_seq, 3);
        assert!(registry
            .schedule_recovery_snapshot("turn-2")
            .unwrap()
            .is_none());
    }

    #[test]
    fn stale_task_update_does_not_discard_fresh_update_for_another_task() {
        let (mut registry, snapshot, _, _) = snapshot_with_two_tasks();
        let old_revision = snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1")
            .unwrap()
            .expected_revision;
        let current_revision = snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-2")
            .unwrap()
            .expected_revision;

        let messages = round("turn-1", "after-snapshot", false, false);
        registry.record_segment(&messages, 0, 2, "turn-1");
        let outcome = registry
            .reduce_estimator_delta(
                &snapshot,
                LifecycleEstimatorDelta {
                    snapshot_id: snapshot.snapshot_id.clone(),
                    task_updates: vec![
                        update(
                            "task:turn-1",
                            old_revision,
                            LifecycleTaskState::Completed,
                            &["old task done"],
                        ),
                        update(
                            "task:turn-2",
                            current_revision,
                            LifecycleTaskState::Completed,
                            &["current task done"],
                        ),
                    ],
                },
            )
            .unwrap();
        assert_eq!(outcome.applied_task_ids, vec!["task:turn-2"]);
        assert_eq!(outcome.rejected_updates[0].reason, "stale_task_revision");
        assert_eq!(
            registry.tasks["task:turn-2"].lifecycle,
            LifecycleTaskState::Completed
        );
        assert_eq!(
            registry.tasks["task:turn-1"].lifecycle,
            LifecycleTaskState::Active
        );
    }

    #[test]
    fn reducer_rejects_direct_active_to_evictable_transition() {
        let (mut registry, snapshot, _, _) = snapshot_with_two_tasks();
        let revision = snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1")
            .unwrap()
            .expected_revision;
        let outcome = registry
            .reduce_estimator_delta(
                &snapshot,
                LifecycleEstimatorDelta {
                    snapshot_id: snapshot.snapshot_id.clone(),
                    task_updates: vec![update(
                        "task:turn-1",
                        revision,
                        LifecycleTaskState::Evictable,
                        &["not enough"],
                    )],
                },
            )
            .unwrap();
        assert!(outcome.applied_task_ids.is_empty());
        assert_eq!(
            outcome.rejected_updates[0].reason,
            "direct_active_to_evictable_transition"
        );
    }

    #[test]
    fn completed_old_task_becomes_a_shadow_candidate_only_after_guards_pass() {
        let (mut registry, first_snapshot, _, _) = snapshot_with_two_tasks();
        let first_revision = first_snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1")
            .unwrap()
            .expected_revision;
        registry
            .reduce_estimator_delta(
                &first_snapshot,
                LifecycleEstimatorDelta {
                    snapshot_id: first_snapshot.snapshot_id.clone(),
                    task_updates: vec![update(
                        "task:turn-1",
                        first_revision,
                        LifecycleTaskState::Completed,
                        &["inspection finished"],
                    )],
                },
            )
            .unwrap();

        for suffix in ["four", "five", "six"] {
            let messages = round("turn-2", suffix, false, false);
            registry.record_segment(&messages, 0, 2, "turn-2");
        }
        let second_snapshot = registry.schedule_snapshot("turn-2").unwrap().unwrap();
        let first_revision = second_snapshot
            .tasks
            .iter()
            .find(|task| task.task_id == "task:turn-1")
            .unwrap()
            .expected_revision;
        let outcome = registry
            .reduce_estimator_delta(
                &second_snapshot,
                LifecycleEstimatorDelta {
                    snapshot_id: second_snapshot.snapshot_id.clone(),
                    task_updates: vec![update(
                        "task:turn-1",
                        first_revision,
                        LifecycleTaskState::Evictable,
                        &["inspection finished"],
                    )],
                },
            )
            .unwrap();
        assert_eq!(outcome.applied_task_ids, vec!["task:turn-1"]);
        assert_eq!(
            registry.shadow_candidate_segment_ids(),
            vec!["turn-1:one-read".to_string(), "turn-1:two-read".to_string()]
        );
    }

    #[test]
    fn deterministic_error_or_pending_todo_prevents_evictable_transition() {
        for (error, pending) in [(true, false), (false, true)] {
            let mut registry = LifecycleRegistry::default();
            registry.observe_user_turn("turn-1", "risky task");
            let messages = round("turn-1", "one", error, pending);
            registry.record_segment(&messages, 0, 2, "turn-1");
            for suffix in ["two", "three", "four", "five", "six"] {
                registry.observe_user_turn("turn-2", "other task");
                let messages = round("turn-2", suffix, false, false);
                registry.record_segment(&messages, 0, 2, "turn-2");
            }
            let snapshot = registry.schedule_snapshot("turn-2").unwrap().unwrap();
            let task_one_revision = snapshot
                .tasks
                .iter()
                .find(|task| task.task_id == "task:turn-1")
                .unwrap()
                .expected_revision;
            registry
                .reduce_estimator_delta(
                    &snapshot,
                    LifecycleEstimatorDelta {
                        snapshot_id: snapshot.snapshot_id.clone(),
                        task_updates: vec![update(
                            "task:turn-1",
                            task_one_revision,
                            LifecycleTaskState::Completed,
                            &["finished despite bad intermediate result"],
                        )],
                    },
                )
                .unwrap();
            let next_snapshot = registry.schedule_snapshot("turn-2").unwrap();
            assert!(next_snapshot.is_none());
            let task_one_revision = registry.tasks["task:turn-1"].revision;
            let synthetic_snapshot = LifecycleEstimatorSnapshot {
                schema_version: REGISTRY_SCHEMA_VERSION,
                snapshot_id: "test-snapshot".to_string(),
                created_at_tick: registry.latest_tick_seq,
                current_turn_id: "turn-2".to_string(),
                current_task_id: "task:turn-2".to_string(),
                current_user_prompt: "other task".to_string(),
                todos: vec![],
                tasks: vec![LifecycleTaskSnapshot {
                    task_id: "task:turn-1".to_string(),
                    root_task_id: "task:turn-1".to_string(),
                    source: LifecycleWorkUnitSource::Todo,
                    control_only: false,
                    todo_ids: vec![],
                    expected_revision: task_one_revision,
                    title: None,
                    normalized_objective: None,
                    lifecycle: LifecycleTaskState::Completed,
                    acceptance_criteria: vec![],
                    completion_evidence: vec![
                        "finished despite bad intermediate result".to_string()
                    ],
                    unresolved_items: vec![],
                    dependencies: vec![],
                    last_activity_tick: 1,
                    segments: vec![],
                }],
            };
            let outcome = registry
                .reduce_estimator_delta(
                    &synthetic_snapshot,
                    LifecycleEstimatorDelta {
                        snapshot_id: "test-snapshot".to_string(),
                        task_updates: vec![update(
                            "task:turn-1",
                            task_one_revision,
                            LifecycleTaskState::Evictable,
                            &["finished despite bad intermediate result"],
                        )],
                    },
                )
                .unwrap();
            assert_eq!(
                outcome.rejected_updates[0].reason,
                "evictable_task_is_deterministically_protected"
            );
        }
    }

    #[tokio::test]
    async fn scheduler_poll_is_nonblocking_until_worker_has_completed() {
        let (_, snapshot, _, _) = snapshot_with_two_tasks();
        let (sender, receiver) = tokio::sync::oneshot::channel::<String>();
        let mut scheduler =
            LifecycleEstimatorScheduler::new(std::sync::Arc::new(TestWorker::new(receiver)));
        assert_eq!(
            scheduler.submit("session-1", snapshot.clone()),
            LifecycleScheduleDisposition::Started
        );
        assert!(scheduler.poll_ready("session-1").await.is_none());
        sender
            .send("{\"snapshotId\":\"ok\",\"taskUpdates\":[]}".to_string())
            .unwrap();

        let result = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if let Some(result) = scheduler.poll_ready("session-1").await {
                    return result;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed local worker should be observed without network");
        assert_eq!(result.snapshot.snapshot_id, snapshot.snapshot_id);
        assert!(result.response.is_ok());
        assert!(!scheduler.has_work("session-1"));
    }

    #[tokio::test]
    async fn scheduler_coalesces_newer_snapshot_while_one_is_inflight() {
        let (_, first_snapshot, _, _) = snapshot_with_two_tasks();
        let mut second_snapshot = first_snapshot.clone();
        second_snapshot.snapshot_id = "newer-snapshot".to_string();
        let (_sender, receiver) = tokio::sync::oneshot::channel::<String>();
        let mut scheduler =
            LifecycleEstimatorScheduler::new(std::sync::Arc::new(TestWorker::new(receiver)));
        scheduler.submit("session-1", first_snapshot);
        assert_eq!(
            scheduler.submit("session-1", second_snapshot),
            LifecycleScheduleDisposition::Coalesced
        );
        assert!(scheduler.has_work("session-1"));
        // Dropping the scheduler aborts the local test worker. The pending
        // snapshot must not start a real estimator request in this test.
    }

    #[tokio::test]
    async fn dropping_scheduler_aborts_unfinished_worker() {
        let (_, snapshot, _, _) = snapshot_with_two_tasks();
        let aborted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut scheduler = LifecycleEstimatorScheduler::new(std::sync::Arc::new(PendingWorker {
            aborted: aborted.clone(),
        }));
        scheduler.submit("session-1", snapshot);
        tokio::task::yield_now().await;
        drop(scheduler);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !aborted.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping a scheduler must abort its unfinished worker");
    }

    #[test]
    fn parser_accepts_fenced_json_and_rejects_non_json() {
        let parsed = parse_estimator_delta(
            "```json\n{\"snapshotId\":\"snapshot-1\",\"taskUpdates\":[]}\n```",
        )
        .unwrap();
        assert_eq!(parsed.snapshot_id, "snapshot-1");
        assert!(parse_estimator_delta("not a delta").is_err());
    }
}
