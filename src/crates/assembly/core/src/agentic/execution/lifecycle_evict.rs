//! Persistent lifecycle registry for the TokenPilot-inspired rollout.
//!
//! Step 1 records state across user turns. Step 2 asks an estimator for state
//! deltas and exposes only conservatively validated candidates. This module
//! never rewrites message history or invalidates the provider prompt cache.

use crate::agentic::core::message::{InternalReminderKind, Message, MessageContent, ToolCall};
use crate::util::errors::{BitFunError, BitFunResult};
use crate::util::types::Message as AIMessage;
use log::warn;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(crate) const LIFECYCLE_REGISTRY_METADATA_KEY: &str = "lifecycleEvictionRegistry";
/// Fixed during the first end-to-end rollout so CLI users need no session flag.
pub(crate) const LIFECYCLE_BATCH_SIZE: usize = 3;
pub(crate) const RECENT_SEGMENTS_TO_PROTECT: usize = 2;
const INTENT_PREVIEW_CHARS: usize = 800;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleRegistry {
    #[serde(default)]
    pub(crate) version: u64,
    #[serde(default)]
    pub(crate) last_processed_turn_seq: usize,
    #[serde(default)]
    pub(crate) tasks: BTreeMap<String, LifecycleTask>,
    #[serde(default)]
    pub(crate) segments: BTreeMap<String, LifecycleSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleTask {
    pub(crate) id: String,
    pub(crate) objective: String,
    pub(crate) state: LifecycleTaskState,
    #[serde(default)]
    pub(crate) covered_turn_ids: Vec<String>,
    #[serde(default)]
    pub(crate) completion_evidence: Vec<String>,
    #[serde(default)]
    pub(crate) unresolved_questions: Vec<String>,
    #[serde(default)]
    pub(crate) segment_ids: Vec<String>,
    pub(crate) last_touched_turn_seq: usize,
    #[serde(default)]
    pub(crate) evicted: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LifecycleTaskState {
    Active,
    Blocked,
    Completed,
    Evictable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleSegment {
    pub(crate) id: String,
    pub(crate) turn_id: String,
    pub(crate) turn_seq: usize,
    #[serde(default)]
    pub(crate) tool_call_ids: Vec<String>,
    #[serde(default)]
    pub(crate) tool_names: Vec<String>,
    #[serde(default)]
    pub(crate) task_ids: Vec<String>,
    #[serde(default)]
    pub(crate) artifact_handles: Vec<String>,
    pub(crate) has_error: bool,
    pub(crate) has_pending_todo: bool,
    pub(crate) token_estimate: usize,
    #[serde(default)]
    pub(crate) intent: String,
}

#[derive(Debug, Deserialize)]
struct LifecycleEstimatorDelta {
    #[serde(rename = "baseVersion")]
    base_version: u64,
    #[serde(rename = "taskUpdates")]
    task_updates: Vec<LifecycleTaskUpdate>,
}

#[derive(Debug, Deserialize)]
struct LifecycleTaskUpdate {
    #[serde(rename = "taskId")]
    task_id: String,
    objective: Option<String>,
    lifecycle: LifecycleTaskState,
    #[serde(rename = "coveredTurnAbsIds", default)]
    covered_turn_ids: Vec<String>,
    #[serde(rename = "completionEvidence", default)]
    completion_evidence: Vec<String>,
    #[serde(rename = "unresolvedQuestions", default)]
    unresolved_questions: Vec<String>,
}

impl LifecycleRegistry {
    /// Each turn starts provisionally. The estimator can later attach it to an
    /// earlier active task by returning both IDs in `coveredTurnAbsIds`.
    pub(crate) fn begin_turn(
        &mut self,
        turn_id: &str,
        turn_seq: usize,
        objective: String,
    ) -> String {
        let task_id = format!("{turn_id}-task");
        self.tasks
            .entry(task_id.clone())
            .or_insert_with(|| LifecycleTask {
                id: task_id.clone(),
                objective,
                state: LifecycleTaskState::Active,
                covered_turn_ids: vec![turn_id.to_string()],
                completion_evidence: vec![],
                unresolved_questions: vec![],
                segment_ids: vec![],
                last_touched_turn_seq: turn_seq,
                evicted: false,
            });
        task_id
    }

    /// `end_idx` is exclusive and captured immediately after this round's tool
    /// results. It prevents a future round from being attributed retroactively.
    pub(crate) fn record_segment(
        &mut self,
        messages: &[Message],
        assistant_msg_idx: usize,
        end_idx: usize,
        turn_id: &str,
        turn_seq: usize,
        task_id: &str,
    ) {
        let Some(MessageContent::Mixed {
            reasoning_content,
            text,
            tool_calls,
        }) = messages
            .get(assistant_msg_idx)
            .map(|message| &message.content)
        else {
            return;
        };
        if tool_calls.is_empty() {
            return;
        }
        let Some(results) = messages.get(assistant_msg_idx + 1..end_idx) else {
            warn!("lifecycle registry: invalid tool-result boundary");
            return;
        };
        let tool_call_ids: Vec<String> =
            tool_calls.iter().map(|call| call.tool_id.clone()).collect();
        let segment_id = format!("{}:{}", turn_id, tool_call_ids.join(","));
        if self.segments.contains_key(&segment_id) {
            return;
        }
        let intent = [reasoning_content.as_deref().unwrap_or(""), text.as_str()]
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let segment = LifecycleSegment {
            id: segment_id.clone(),
            turn_id: turn_id.to_string(),
            turn_seq,
            tool_call_ids,
            tool_names: tool_calls
                .iter()
                .map(|call| call.tool_name.clone())
                .collect(),
            task_ids: vec![task_id.to_string()],
            artifact_handles: results
                .iter()
                .filter_map(artifact_handle_from_message)
                .collect(),
            has_error: results.iter().any(|message| {
                matches!(
                    &message.content,
                    MessageContent::ToolResult { is_error: true, .. }
                )
            }),
            has_pending_todo: tool_calls.iter().any(todo_call_has_pending_items),
            token_estimate: results.iter().map(message_char_estimate).sum::<usize>() / 4,
            intent: truncate_chars(&intent, INTENT_PREVIEW_CHARS),
        };
        self.segments.insert(segment_id.clone(), segment);
        if let Some(task) = self.tasks.get_mut(task_id) {
            task.segment_ids.push(segment_id);
            task.last_touched_turn_seq = turn_seq;
        }
    }

    pub(crate) fn should_estimate(&self, turn_seq: usize, batch_size: usize) -> bool {
        batch_size > 0 && (turn_seq + 1) % batch_size == 0 && !self.segments.is_empty()
    }

    pub(crate) fn build_vi(
        &self,
        current_task_id: &str,
        current_turn_id: &str,
        current_turn_seq: usize,
    ) -> BitFunResult<String> {
        #[derive(Serialize)]
        struct Input<'a> {
            #[serde(rename = "baseVersion")]
            base_version: u64,
            #[serde(rename = "currentTaskId")]
            current_task_id: &'a str,
            #[serde(rename = "currentTurnId")]
            current_turn_id: &'a str,
            tasks: Vec<&'a LifecycleTask>,
            segments: Vec<&'a LifecycleSegment>,
            #[serde(rename = "evictableCandidateTaskIds")]
            candidates: Vec<&'a str>,
        }
        let candidate_ids = self.candidate_task_ids(current_task_id, current_turn_seq);
        serde_json::to_string(&Input {
            base_version: self.version,
            current_task_id,
            current_turn_id,
            tasks: self.tasks.values().collect(),
            segments: self.segments.values().collect(),
            candidates: candidate_ids.iter().map(String::as_str).collect(),
        })
        .map_err(|error| BitFunError::tool(format!("lifecycle registry Vi serialization: {error}")))
    }

    pub(crate) fn apply_delta(
        &mut self,
        text: &str,
        current_task_id: &str,
        current_turn_seq: usize,
    ) -> BitFunResult<()> {
        let delta: LifecycleEstimatorDelta =
            serde_json::from_str(extract_json_object(text).ok_or_else(|| {
                BitFunError::tool("lifecycle estimator: no JSON object in response".to_string())
            })?)
            .map_err(|error| {
                BitFunError::tool(format!("lifecycle estimator invalid delta: {error}"))
            })?;
        if delta.base_version != self.version {
            return Err(BitFunError::tool(format!(
                "lifecycle estimator stale baseVersion {} (expected {})",
                delta.base_version, self.version
            )));
        }
        for update in delta.task_updates {
            if !self.tasks.contains_key(&update.task_id)
                || !update
                    .covered_turn_ids
                    .iter()
                    .all(|turn_id| self.turn_is_owned(turn_id))
            {
                warn!(
                    "lifecycle estimator referenced unknown task or turn, skipping {}",
                    update.task_id
                );
                continue;
            }
            for turn_id in &update.covered_turn_ids {
                self.assign_turn_to_task(turn_id, &update.task_id);
            }
            let protected = self.task_has_protected_segment(&update.task_id, current_turn_seq);
            let Some(task) = self.tasks.get_mut(&update.task_id) else {
                continue;
            };
            let unsafe_completed = update.lifecycle == LifecycleTaskState::Completed
                && update.completion_evidence.is_empty();
            let unsafe_evictable = update.lifecycle == LifecycleTaskState::Evictable
                && (update.task_id == current_task_id
                    || update.completion_evidence.is_empty()
                    || !update.unresolved_questions.is_empty()
                    || protected
                    || task.state != LifecycleTaskState::Completed);
            if unsafe_completed || unsafe_evictable || update.lifecycle < task.state {
                warn!(
                    "lifecycle estimator proposed invalid transition for {}, skipping",
                    update.task_id
                );
                continue;
            }
            task.state = update.lifecycle;
            if let Some(objective) = update.objective.filter(|value| !value.trim().is_empty()) {
                task.objective = objective;
            }
            task.completion_evidence = update.completion_evidence;
            task.unresolved_questions = update.unresolved_questions;
            task.last_touched_turn_seq = current_turn_seq;
        }
        self.version += 1;
        self.last_processed_turn_seq = current_turn_seq;
        Ok(())
    }

    pub(crate) fn candidate_task_ids(
        &self,
        current_task_id: &str,
        current_turn_seq: usize,
    ) -> Vec<String> {
        self.tasks
            .values()
            .filter(|task| {
                task.id != current_task_id
                    && task.state == LifecycleTaskState::Evictable
                    && !task.evicted
                    && !task.completion_evidence.is_empty()
                    && task.unresolved_questions.is_empty()
                    && !self.task_has_protected_segment(&task.id, current_turn_seq)
            })
            .map(|task| task.id.clone())
            .collect()
    }

    fn turn_is_owned(&self, turn_id: &str) -> bool {
        self.tasks
            .values()
            .any(|task| task.covered_turn_ids.iter().any(|id| id == turn_id))
    }

    fn task_has_protected_segment(&self, task_id: &str, current_turn_seq: usize) -> bool {
        self.segments
            .values()
            .filter(|segment| segment.task_ids.iter().any(|id| id == task_id))
            .any(|segment| {
                segment.has_error
                    || segment.has_pending_todo
                    || segment.artifact_handles.is_empty()
                    || current_turn_seq.saturating_sub(segment.turn_seq)
                        < RECENT_SEGMENTS_TO_PROTECT
            })
    }

    fn assign_turn_to_task(&mut self, turn_id: &str, target_task_id: &str) {
        let sources: Vec<String> = self
            .tasks
            .values()
            .filter(|task| {
                task.id != target_task_id && task.covered_turn_ids.iter().any(|id| id == turn_id)
            })
            .map(|task| task.id.clone())
            .collect();
        if sources.is_empty() {
            return;
        }
        let moved: Vec<String> = self
            .segments
            .values()
            .filter(|segment| segment.turn_id == turn_id)
            .map(|segment| segment.id.clone())
            .collect();
        for segment_id in &moved {
            if let Some(segment) = self.segments.get_mut(segment_id) {
                segment.task_ids.retain(|id| !sources.contains(id));
                if !segment.task_ids.iter().any(|id| id == target_task_id) {
                    segment.task_ids.push(target_task_id.to_string());
                }
            }
        }
        for source_id in &sources {
            if let Some(source) = self.tasks.get_mut(source_id) {
                source.covered_turn_ids.retain(|id| id != turn_id);
                source.segment_ids.retain(|id| !moved.contains(id));
            }
        }
        if let Some(target) = self.tasks.get_mut(target_task_id) {
            if !target.covered_turn_ids.iter().any(|id| id == turn_id) {
                target.covered_turn_ids.push(turn_id.to_string());
            }
            for segment_id in moved {
                if !target.segment_ids.iter().any(|id| id == &segment_id) {
                    target.segment_ids.push(segment_id);
                }
            }
        }
        self.tasks.retain(|_, task| {
            !sources.contains(&task.id)
                || !task.covered_turn_ids.is_empty()
                || !task.segment_ids.is_empty()
        });
    }

    pub(crate) fn mark_tasks_evicted(&mut self, task_ids: &[String]) {
        for task_id in task_ids {
            if let Some(task) = self.tasks.get_mut(task_id) {
                task.evicted = true;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LifecycleEvictionPlan {
    pub(crate) task_ids: Vec<String>,
    pub(crate) artifact_handles: Vec<String>,
    replacements: Vec<LifecycleReplacement>,
}

#[derive(Debug, Clone)]
struct LifecycleReplacement {
    start: usize,
    end: usize,
    turn_id: String,
    task_id: String,
    artifact_handles: Vec<String>,
}

impl LifecycleEvictionPlan {
    /// Apply to a clone only after all target ranges and artifacts were verified.
    pub(crate) fn apply(&self, messages: &[Message]) -> BitFunResult<Vec<Message>> {
        let mut rewritten = messages.to_vec();
        let mut replacements = self.replacements.clone();
        replacements.sort_by(|left, right| right.start.cmp(&left.start));
        for replacement in replacements {
            if replacement.end > rewritten.len() || replacement.start >= replacement.end {
                return Err(BitFunError::tool(
                    "lifecycle eviction plan range became invalid",
                ));
            }
            let summary = format!(
                "[LIFECYCLE_EVICTION task_id={}]\nCompleted task output was evicted from context.\nArtifacts:\n{}",
                replacement.task_id,
                replacement.artifact_handles.join("\n"),
            );
            let reminder =
                Message::internal_reminder(InternalReminderKind::LifecycleEvictionSummary, summary)
                    .with_turn_id(replacement.turn_id);
            rewritten.splice(replacement.start..replacement.end, [reminder]);
        }
        Ok(rewritten)
    }
}

pub(crate) fn build_eviction_plan(
    registry: &LifecycleRegistry,
    messages: &[Message],
    task_ids: &[String],
) -> BitFunResult<LifecycleEvictionPlan> {
    let mut replacements = Vec::new();
    let mut artifact_handles = Vec::new();
    for task_id in task_ids {
        let Some(task) = registry.tasks.get(task_id) else {
            return Err(BitFunError::tool(format!(
                "lifecycle candidate task disappeared: {task_id}"
            )));
        };
        if task.evicted || task.state != LifecycleTaskState::Evictable {
            return Err(BitFunError::tool(format!(
                "lifecycle task is no longer evictable: {task_id}"
            )));
        }
        for segment_id in &task.segment_ids {
            let Some(segment) = registry.segments.get(segment_id) else {
                return Err(BitFunError::tool(format!(
                    "lifecycle segment disappeared: {segment_id}"
                )));
            };
            if segment.artifact_handles.is_empty() {
                return Err(BitFunError::tool(format!(
                    "lifecycle segment lacks artifact: {segment_id}"
                )));
            }
            let (start, end) = locate_segment(messages, segment)?;
            artifact_handles.extend(segment.artifact_handles.clone());
            replacements.push(LifecycleReplacement {
                start,
                end,
                turn_id: segment.turn_id.clone(),
                task_id: task_id.clone(),
                artifact_handles: segment.artifact_handles.clone(),
            });
        }
    }
    replacements.sort_by_key(|replacement| replacement.start);
    if replacements
        .windows(2)
        .any(|pair| pair[0].end > pair[1].start)
    {
        return Err(BitFunError::tool("lifecycle eviction targets overlap"));
    }
    if replacements.is_empty() {
        return Err(BitFunError::tool(
            "lifecycle eviction has no segments to replace",
        ));
    }
    Ok(LifecycleEvictionPlan {
        task_ids: task_ids.to_vec(),
        artifact_handles,
        replacements,
    })
}

fn locate_segment(
    messages: &[Message],
    segment: &LifecycleSegment,
) -> BitFunResult<(usize, usize)> {
    let start = messages
        .iter()
        .position(|message| {
            message.metadata.turn_id.as_deref() == Some(segment.turn_id.as_str())
                && matches!(&message.content, MessageContent::Mixed { tool_calls, .. }
                if tool_calls.iter().map(|call| &call.tool_id).eq(segment.tool_call_ids.iter()))
        })
        .ok_or_else(|| {
            BitFunError::tool(format!(
                "lifecycle segment cannot be relocated: {}",
                segment.id
            ))
        })?;
    let mut end = start + 1;
    let mut result_ids = Vec::new();
    while let Some(message) = messages.get(end) {
        let MessageContent::ToolResult { tool_id, .. } = &message.content else {
            break;
        };
        result_ids.push(tool_id.clone());
        end += 1;
    }
    if result_ids != segment.tool_call_ids {
        return Err(BitFunError::tool(format!(
            "lifecycle segment result IDs changed: {}",
            segment.id
        )));
    }
    Ok((start, end))
}

pub(crate) async fn estimate_registry_delta(vi_json: String) -> BitFunResult<String> {
    use crate::infrastructure::ai::get_global_ai_client_factory;
    const PROMPT: &str = r#"You are a lifecycle state estimator. Return ONLY JSON with {"baseVersion": number, "taskUpdates": [...]}. Each update must use taskId, lifecycle (active|blocked|completed|evictable), coveredTurnAbsIds, completionEvidence, and unresolvedQuestions. completed requires completionEvidence. evictable requires completionEvidence, no unresolvedQuestions, and the current task must never be evictable. Do not invent task or turn IDs."#;
    let factory = get_global_ai_client_factory()
        .await
        .map_err(|error| BitFunError::AIClient(format!("lifecycle estimator factory: {error}")))?;
    let client = factory
        .get_client_resolved("fast")
        .await
        .map_err(|error| BitFunError::AIClient(format!("lifecycle estimator client: {error}")))?;
    let response = client
        .send_message_with_trace(
            vec![
                AIMessage::system(PROMPT.to_string()),
                AIMessage::user(vi_json),
            ],
            None,
            None,
        )
        .await
        .map_err(|error| BitFunError::AIClient(format!("lifecycle estimator request: {error}")))?;
    Ok(response.text)
}

fn artifact_handle_from_message(message: &Message) -> Option<String> {
    let MessageContent::ToolResult {
        result_for_assistant: Some(text),
        ..
    } = &message.content
    else {
        return None;
    };
    text.lines()
        .find_map(|line| {
            line.strip_prefix("Full output saved to: ")
                .or_else(|| line.strip_prefix("Full output was saved to: "))
        })
        .map(str::to_string)
}

fn todo_call_has_pending_items(call: &ToolCall) -> bool {
    call.tool_name == "TodoWrite"
        && call
            .arguments
            .get("todos")
            .and_then(|value| value.as_array())
            .is_some_and(|todos| {
                todos.iter().any(|todo| {
                    matches!(
                        todo.get("status").and_then(|value| value.as_str()),
                        Some("pending" | "in_progress")
                    )
                })
            })
}

fn message_char_estimate(message: &Message) -> usize {
    match &message.content {
        MessageContent::ToolResult {
            result_for_assistant,
            ..
        } => result_for_assistant
            .as_deref()
            .unwrap_or("")
            .chars()
            .count(),
        _ => 0,
    }
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

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end >= start).then_some(&text[start..=end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::core::message::ToolResult;

    fn messages(turn: &str, persisted: bool, error: bool, pending: bool) -> Vec<Message> {
        let mut calls = vec![ToolCall {
            tool_id: format!("{turn}-read"),
            tool_name: "Read".into(),
            arguments: serde_json::json!({}),
            ..Default::default()
        }];
        if pending {
            calls.push(ToolCall {
                tool_id: format!("{turn}-todo"),
                tool_name: "TodoWrite".into(),
                arguments: serde_json::json!({"todos":[{"status":"in_progress"}]}),
                ..Default::default()
            });
        }
        vec![
            Message::assistant_with_tools("inspect".into(), calls),
            Message::tool_result(ToolResult {
                tool_id: format!("{turn}-read"),
                tool_name: "Read".into(),
                result: serde_json::json!({}),
                result_for_assistant: Some(if persisted {
                    "Full output saved to: /tmp/out.json".into()
                } else {
                    "short".into()
                }),
                is_error: error,
                duration_ms: None,
                image_attachments: None,
            }),
        ]
    }
    fn delta(version: u64, task: &str, state: &str, evidence: &[&str]) -> String {
        serde_json::json!({"baseVersion":version,"taskUpdates":[{"taskId":task,"lifecycle":state,"coveredTurnAbsIds":[],"completionEvidence":evidence,"unresolvedQuestions":[]}]}).to_string()
    }

    #[test]
    fn records_only_declared_result_boundary() {
        let mut ms = messages("one", true, false, false);
        ms.push(Message::tool_result(ToolResult {
            tool_id: "future".into(),
            tool_name: "Read".into(),
            result: serde_json::json!({}),
            result_for_assistant: Some("Full output saved to: /tmp/future".into()),
            is_error: true,
            duration_ms: None,
            image_attachments: None,
        }));
        let mut r = LifecycleRegistry::default();
        let task = r.begin_turn("one", 0, "x".into());
        r.record_segment(&ms, 0, 2, "one", 0, &task);
        let s = r.segments.values().next().unwrap();
        assert!(!s.has_error);
        assert_eq!(s.artifact_handles, vec!["/tmp/out.json"]);
    }
    #[test]
    fn estimator_can_merge_turns_into_one_task() {
        let mut r = LifecycleRegistry::default();
        let first = r.begin_turn("one", 0, "x".into());
        r.record_segment(&messages("one", true, false, false), 0, 2, "one", 0, &first);
        let provisional = r.begin_turn("two", 1, "y".into());
        r.record_segment(
            &messages("two", true, false, false),
            0,
            2,
            "two",
            1,
            &provisional,
        );
        let update = serde_json::json!({"baseVersion":0,"taskUpdates":[{"taskId":first,"lifecycle":"active","coveredTurnAbsIds":["one","two"],"completionEvidence":[],"unresolvedQuestions":[]}]}).to_string();
        r.apply_delta(&update, &provisional, 1).unwrap();
        assert!(!r.tasks.contains_key(&provisional));
        assert_eq!(r.tasks[&first].covered_turn_ids, vec!["one", "two"]);
        assert!(r
            .segments
            .values()
            .all(|s| s.task_ids == vec![first.clone()]));
    }
    #[test]
    fn rejects_stale_evidence_free_completion_and_direct_eviction() {
        let mut r = LifecycleRegistry::default();
        let task = r.begin_turn("one", 0, "x".into());
        assert!(r
            .apply_delta(&delta(1, &task, "completed", &["done"]), &task, 0)
            .is_err());
        r.apply_delta(&delta(0, &task, "completed", &[]), &task, 0)
            .unwrap();
        assert_eq!(r.tasks[&task].state, LifecycleTaskState::Active);
        r.apply_delta(&delta(1, &task, "evictable", &["done"]), "other-task", 3)
            .unwrap();
        assert_eq!(r.tasks[&task].state, LifecycleTaskState::Active);
    }
    #[test]
    fn candidate_requires_old_persisted_safe_segment_and_evidence() {
        let mut r = LifecycleRegistry::default();
        let old = r.begin_turn("one", 0, "x".into());
        r.record_segment(&messages("one", true, false, false), 0, 2, "one", 0, &old);
        let current = r.begin_turn("four", 3, "y".into());
        r.apply_delta(&delta(0, &old, "completed", &["done"]), &current, 3)
            .unwrap();
        r.apply_delta(&delta(1, &old, "evictable", &["done"]), &current, 3)
            .unwrap();
        assert_eq!(r.candidate_task_ids(&current, 3), vec![old]);
    }
    #[test]
    fn candidate_rejects_missing_artifact_errors_pending_and_recent() {
        for (persisted, error, pending, turn) in [
            (false, false, false, 3),
            (true, true, false, 3),
            (true, false, true, 3),
            (true, false, false, 1),
        ] {
            let mut r = LifecycleRegistry::default();
            let old = r.begin_turn("one", 0, "x".into());
            r.record_segment(
                &messages("one", persisted, error, pending),
                0,
                2,
                "one",
                0,
                &old,
            );
            let current = r.begin_turn("now", turn, "y".into());
            r.apply_delta(&delta(0, &old, "completed", &["done"]), &current, turn)
                .unwrap();
            r.apply_delta(&delta(1, &old, "evictable", &["done"]), &current, turn)
                .unwrap();
            assert!(r.candidate_task_ids(&current, turn).is_empty());
        }
    }
    #[test]
    fn batch_is_explicit_and_never_divides_by_zero() {
        let mut r = LifecycleRegistry::default();
        assert!(!r.should_estimate(2, 0));
        let task = r.begin_turn("one", 0, "x".into());
        r.record_segment(&messages("one", true, false, false), 0, 2, "one", 0, &task);
        assert!(r.should_estimate(2, 3));
        assert!(!r.should_estimate(1, 3));
    }

    #[test]
    fn plan_relocates_by_stable_ids_and_replaces_only_the_target_range() {
        let mut registry = LifecycleRegistry::default();
        let task = registry.begin_turn("turn-1", 0, "inspect".into());
        let messages = messages("turn-1", true, false, false)
            .into_iter()
            .map(|message| message.with_turn_id("turn-1".to_string()))
            .collect::<Vec<_>>();
        registry.record_segment(&messages, 0, 2, "turn-1", 0, &task);
        let record = registry.tasks.get_mut(&task).unwrap();
        record.state = LifecycleTaskState::Evictable;
        record.completion_evidence.push("done".into());

        let plan = build_eviction_plan(&registry, &messages, &[task.clone()]).unwrap();
        let rewritten = plan.apply(&messages).unwrap();
        assert_eq!(rewritten.len(), 1);
        assert!(matches!(
            rewritten[0].metadata.internal_reminder_kind,
            Some(InternalReminderKind::LifecycleEvictionSummary)
        ));
        assert_eq!(rewritten[0].metadata.turn_id.as_deref(), Some("turn-1"));
    }

    #[test]
    fn plan_rejects_changed_tool_result_ids_without_rewriting() {
        let mut registry = LifecycleRegistry::default();
        let task = registry.begin_turn("turn-1", 0, "inspect".into());
        let mut messages = messages("turn-1", true, false, false)
            .into_iter()
            .map(|message| message.with_turn_id("turn-1".to_string()))
            .collect::<Vec<_>>();
        registry.record_segment(&messages, 0, 2, "turn-1", 0, &task);
        registry.tasks.get_mut(&task).unwrap().state = LifecycleTaskState::Evictable;
        if let MessageContent::ToolResult { tool_id, .. } = &mut messages[1].content {
            *tool_id = "changed".into();
        }
        assert!(build_eviction_plan(&registry, &messages, &[task]).is_err());
    }
}
