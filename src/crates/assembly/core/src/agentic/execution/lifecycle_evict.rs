//! Segment-level lifecycle eviction for long-running agent tool loops.
//!
//! A lifecycle tick is one completed assistant/tool-result round. The model
//! estimates state every fixed batch of ticks; deterministic guards decide
//! whether an evictable segment may be physically replaced.

use crate::agentic::core::message::{InternalReminderKind, Message, MessageContent, ToolCall};
use crate::util::errors::{BitFunError, BitFunResult};
use crate::util::types::Message as AIMessage;
use log::warn;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub(crate) const LIFECYCLE_REGISTRY_METADATA_KEY: &str = "lifecycleEvictionRegistry";
pub(crate) const LIFECYCLE_BATCH_SIZE: usize = 3;
pub(crate) const RECENT_SEGMENTS_TO_PROTECT: u64 = 2;
const INTENT_PREVIEW_CHARS: usize = 800;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleRegistry {
    #[serde(default)]
    pub(crate) version: u64,
    #[serde(default)]
    pub(crate) latest_tick_seq: u64,
    #[serde(default)]
    pub(crate) last_estimated_tick_seq: u64,
    #[serde(default)]
    pub(crate) segments: BTreeMap<String, LifecycleSegment>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LifecycleSegmentState {
    Active,
    Completed,
    Evictable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LifecycleSegment {
    pub(crate) id: String,
    pub(crate) turn_id: String,
    pub(crate) tick_seq: u64,
    pub(crate) tool_call_ids: Vec<String>,
    pub(crate) tool_names: Vec<String>,
    pub(crate) has_error: bool,
    pub(crate) has_pending_todo: bool,
    pub(crate) token_estimate: usize,
    #[serde(default)]
    pub(crate) intent: String,
    pub(crate) state: LifecycleSegmentState,
    #[serde(default)]
    pub(crate) completion_evidence: Vec<String>,
    #[serde(default)]
    pub(crate) unresolved_questions: Vec<String>,
    #[serde(default)]
    pub(crate) evicted: bool,
}

#[derive(Debug, Deserialize)]
struct LifecycleEstimatorDelta {
    #[serde(rename = "baseVersion")]
    base_version: u64,
    #[serde(rename = "segmentUpdates")]
    segment_updates: Vec<LifecycleSegmentUpdate>,
}

#[derive(Debug, Deserialize)]
struct LifecycleSegmentUpdate {
    #[serde(rename = "segmentId")]
    segment_id: String,
    lifecycle: LifecycleSegmentState,
    #[serde(rename = "completionEvidence", default)]
    completion_evidence: Vec<String>,
    #[serde(rename = "unresolvedQuestions", default)]
    unresolved_questions: Vec<String>,
}

impl LifecycleRegistry {
    /// Records exactly one completed tool round and returns its stable segment ID.
    pub(crate) fn record_segment(
        &mut self,
        messages: &[Message],
        assistant_idx: usize,
        end_idx: usize,
        turn_id: &str,
    ) -> Option<String> {
        let MessageContent::Mixed {
            reasoning_content,
            text,
            tool_calls,
        } = &messages.get(assistant_idx)?.content
        else {
            return None;
        };
        if tool_calls.is_empty() {
            return None;
        }
        let results = messages.get(assistant_idx + 1..end_idx)?;
        let tool_call_ids: Vec<String> =
            tool_calls.iter().map(|call| call.tool_id.clone()).collect();
        let id = format!("{}:{}", turn_id, tool_call_ids.join(","));
        if self.segments.contains_key(&id) {
            return Some(id);
        }
        self.latest_tick_seq += 1;
        let intent = [reasoning_content.as_deref().unwrap_or(""), text.as_str()]
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        self.segments.insert(
            id.clone(),
            LifecycleSegment {
                id: id.clone(),
                turn_id: turn_id.to_string(),
                tick_seq: self.latest_tick_seq,
                tool_call_ids,
                tool_names: tool_calls
                    .iter()
                    .map(|call| call.tool_name.clone())
                    .collect(),
                has_error: results.iter().any(|message| {
                    matches!(
                        message.content,
                        MessageContent::ToolResult { is_error: true, .. }
                    )
                }),
                has_pending_todo: tool_calls.iter().any(todo_call_has_pending_items),
                token_estimate: results.iter().map(message_char_estimate).sum::<usize>() / 4,
                intent: truncate_chars(&intent, INTENT_PREVIEW_CHARS),
                state: LifecycleSegmentState::Active,
                completion_evidence: vec![],
                unresolved_questions: vec![],
                evicted: false,
            },
        );
        Some(id)
    }

    pub(crate) fn should_estimate(&self) -> bool {
        self.latest_tick_seq > self.last_estimated_tick_seq
            && self.latest_tick_seq % LIFECYCLE_BATCH_SIZE as u64 == 0
    }

    pub(crate) fn build_vi(&self, current_turn_id: &str) -> BitFunResult<String> {
        #[derive(Serialize)]
        struct Input<'a> {
            #[serde(rename = "baseVersion")]
            base_version: u64,
            #[serde(rename = "latestTickSeq")]
            latest_tick_seq: u64,
            #[serde(rename = "currentTurnId")]
            current_turn_id: &'a str,
            segments: Vec<&'a LifecycleSegment>,
        }
        serde_json::to_string(&Input {
            base_version: self.version,
            latest_tick_seq: self.latest_tick_seq,
            current_turn_id,
            segments: self
                .segments
                .values()
                .filter(|segment| !segment.evicted)
                .collect(),
        })
        .map_err(|error| BitFunError::tool(format!("lifecycle Vi serialization: {error}")))
    }

    pub(crate) fn apply_delta(&mut self, text: &str) -> BitFunResult<()> {
        let delta: LifecycleEstimatorDelta = serde_json::from_str(
            extract_json_object(text)
                .ok_or_else(|| BitFunError::tool("lifecycle estimator: no JSON object"))?,
        )
        .map_err(|error| {
            BitFunError::tool(format!("lifecycle estimator invalid delta: {error}"))
        })?;
        if delta.base_version != self.version {
            return Err(BitFunError::tool(format!(
                "lifecycle estimator stale baseVersion {} (expected {})",
                delta.base_version, self.version
            )));
        }
        for update in delta.segment_updates {
            let protected = self.segment_is_protected(&update.segment_id);
            let Some(segment) = self.segments.get_mut(&update.segment_id) else {
                warn!(
                    "lifecycle estimator referenced unknown segment {}, skipping",
                    update.segment_id
                );
                continue;
            };
            let invalid_completed = update.lifecycle == LifecycleSegmentState::Completed
                && update.completion_evidence.is_empty();
            let invalid_evictable = update.lifecycle == LifecycleSegmentState::Evictable
                && (segment.state != LifecycleSegmentState::Completed
                    || update.completion_evidence.is_empty()
                    || !update.unresolved_questions.is_empty()
                    || protected);
            if segment.evicted
                || update.lifecycle < segment.state
                || invalid_completed
                || invalid_evictable
            {
                warn!(
                    "lifecycle estimator proposed invalid transition for {}, skipping",
                    update.segment_id
                );
                continue;
            }
            segment.state = update.lifecycle;
            segment.completion_evidence = update.completion_evidence;
            segment.unresolved_questions = update.unresolved_questions;
        }
        self.version += 1;
        self.last_estimated_tick_seq = self.latest_tick_seq;
        Ok(())
    }

    pub(crate) fn candidate_segment_ids(&self) -> Vec<String> {
        self.segments
            .values()
            .filter(|segment| {
                segment.state == LifecycleSegmentState::Evictable
                    && !segment.evicted
                    && !segment.completion_evidence.is_empty()
                    && segment.unresolved_questions.is_empty()
                    && !self.segment_is_protected(&segment.id)
            })
            .map(|segment| segment.id.clone())
            .collect()
    }

    pub(crate) fn mark_segments_evicted(&mut self, ids: &[String]) {
        for id in ids {
            if let Some(segment) = self.segments.get_mut(id) {
                segment.evicted = true;
            }
        }
    }

    fn segment_is_protected(&self, id: &str) -> bool {
        let Some(segment) = self.segments.get(id) else {
            return true;
        };
        segment.has_error
            || segment.has_pending_todo
            || self.latest_tick_seq.saturating_sub(segment.tick_seq) < RECENT_SEGMENTS_TO_PROTECT
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LifecycleEvictionPlan {
    pub(crate) segment_ids: Vec<String>,
    replacements: Vec<LifecycleReplacement>,
}
#[derive(Debug, Clone)]
struct LifecycleReplacement {
    segment_id: String,
    start: usize,
    end: usize,
    turn_id: String,
    tool_names: Vec<String>,
    completion_evidence: Vec<String>,
}

impl LifecycleEvictionPlan {
    pub(crate) fn archive_payloads(
        &self,
        messages: &[Message],
    ) -> BitFunResult<Vec<(String, Vec<Message>)>> {
        self.replacements
            .iter()
            .map(|replacement| {
                let payload = messages
                    .get(replacement.start..replacement.end)
                    .ok_or_else(|| BitFunError::tool("lifecycle archive range became invalid"))?
                    .to_vec();
                Ok((replacement.segment_id.clone(), payload))
            })
            .collect()
    }
    pub(crate) fn apply(
        &self,
        messages: &[Message],
        archives: &HashMap<String, String>,
    ) -> BitFunResult<Vec<Message>> {
        let mut rewritten = messages.to_vec();
        let mut replacements = self.replacements.clone();
        replacements.sort_by(|a, b| b.start.cmp(&a.start));
        for replacement in replacements {
            let archive = archives
                .get(&replacement.segment_id)
                .ok_or_else(|| BitFunError::tool("lifecycle archive missing after validation"))?;
            if replacement.end > rewritten.len() || replacement.start >= replacement.end {
                return Err(BitFunError::tool(
                    "lifecycle replacement range became invalid",
                ));
            }
            let summary = format!("[LIFECYCLE_EVICTION segment_id={}]\nTools: {}\nCompletion evidence: {}\nFull segment archive: {}",
                replacement.segment_id, replacement.tool_names.join(", "), replacement.completion_evidence.join("; "), archive);
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
    ids: &[String],
) -> BitFunResult<LifecycleEvictionPlan> {
    let mut replacements = Vec::new();
    for id in ids {
        let Some(segment) = registry.segments.get(id) else {
            return Err(BitFunError::tool(format!(
                "lifecycle candidate disappeared: {id}"
            )));
        };
        if segment.state != LifecycleSegmentState::Evictable
            || segment.evicted
            || registry.segment_is_protected(id)
        {
            return Err(BitFunError::tool(format!(
                "lifecycle candidate is no longer safe: {id}"
            )));
        }
        let (start, end) = locate_segment(messages, segment)?;
        replacements.push(LifecycleReplacement {
            segment_id: id.clone(),
            start,
            end,
            turn_id: segment.turn_id.clone(),
            tool_names: segment.tool_names.clone(),
            completion_evidence: segment.completion_evidence.clone(),
        });
    }
    replacements.sort_by_key(|replacement| replacement.start);
    if replacements
        .windows(2)
        .any(|pair| pair[0].end > pair[1].start)
    {
        return Err(BitFunError::tool("lifecycle eviction targets overlap"));
    }
    (!replacements.is_empty())
        .then_some(LifecycleEvictionPlan {
            segment_ids: ids.to_vec(),
            replacements,
        })
        .ok_or_else(|| BitFunError::tool("lifecycle eviction has no targets"))
}

fn locate_segment(
    messages: &[Message],
    segment: &LifecycleSegment,
) -> BitFunResult<(usize, usize)> {
    let start = messages.iter().position(|message| message.metadata.turn_id.as_deref() == Some(segment.turn_id.as_str())
        && matches!(&message.content, MessageContent::Mixed { tool_calls, .. } if tool_calls.iter().map(|call| &call.tool_id).eq(segment.tool_call_ids.iter())))
        .ok_or_else(|| BitFunError::tool(format!("lifecycle segment cannot be relocated: {}", segment.id)))?;
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
            "lifecycle result IDs changed: {}",
            segment.id
        )));
    }
    Ok((start, end))
}

pub(crate) async fn estimate_registry_delta(vi_json: String) -> BitFunResult<String> {
    use crate::infrastructure::ai::get_global_ai_client_factory;
    const PROMPT: &str = r#"You are a conservative lifecycle estimator for an agent tool loop. Return ONLY JSON: {"baseVersion": number, "segmentUpdates": [{"segmentId": string, "lifecycle": "active"|"completed"|"evictable", "completionEvidence": [string], "unresolvedQuestions": [string]}]}. A segment is one assistant tool-call group plus results. Judge residual utility from later segments. completed needs concrete evidence. evictable requires completed evidence, no unresolved questions, and must never be one of the two newest segments. Do not invent segment IDs. Prefer active or completed when uncertain."#;
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
    fn round(turn: &str, suffix: &str, error: bool, pending: bool) -> Vec<Message> {
        let mut calls = vec![ToolCall {
            tool_id: format!("{suffix}-id"),
            tool_name: "Read".into(),
            arguments: serde_json::json!({}),
            ..Default::default()
        }];
        if pending {
            calls.push(ToolCall {
                tool_id: format!("{suffix}-todo"),
                tool_name: "TodoWrite".into(),
                arguments: serde_json::json!({"todos":[{"status":"pending"}]}),
                ..Default::default()
            });
        }
        vec![
            Message::assistant_with_tools("inspect".into(), calls).with_turn_id(turn.into()),
            Message::tool_result(ToolResult {
                tool_id: format!("{suffix}-id"),
                tool_name: "Read".into(),
                result: serde_json::json!({}),
                result_for_assistant: Some("full result".into()),
                is_error: error,
                duration_ms: None,
                image_attachments: None,
            })
            .with_turn_id(turn.into()),
        ]
    }
    fn delta(version: u64, id: &str, state: &str, evidence: &[&str]) -> String {
        serde_json::json!({"baseVersion":version,"segmentUpdates":[{"segmentId":id,"lifecycle":state,"completionEvidence":evidence,"unresolvedQuestions":[]}]}).to_string()
    }
    #[test]
    fn batch_is_counted_by_tool_ticks_not_outer_turn() {
        let mut r = LifecycleRegistry::default();
        for i in 0..3 {
            let m = round("same-user-turn", &i.to_string(), false, false);
            r.record_segment(&m, 0, 2, "same-user-turn");
        }
        assert_eq!(r.latest_tick_seq, 3);
        assert!(r.should_estimate());
    }
    #[test]
    fn old_segment_can_be_evicted_inside_current_user_turn() {
        let mut r = LifecycleRegistry::default();
        let mut all = Vec::new();
        let mut ids = Vec::new();
        for i in 0..4 {
            let m = round("same-user-turn", &i.to_string(), false, false);
            let start = all.len();
            all.extend(m);
            ids.push(
                r.record_segment(&all, start, start + 2, "same-user-turn")
                    .unwrap(),
            );
        }
        r.apply_delta(&delta(0, &ids[0], "completed", &["read complete"]))
            .unwrap();
        r.apply_delta(&delta(1, &ids[0], "evictable", &["read complete"]))
            .unwrap();
        assert_eq!(r.candidate_segment_ids(), vec![ids[0].clone()]);
        let plan = build_eviction_plan(&r, &all, &[ids[0].clone()]).unwrap();
        let archives = HashMap::from([(ids[0].clone(), "/tmp/archive.json".into())]);
        let rewritten = plan.apply(&all, &archives).unwrap();
        assert_eq!(rewritten.len(), 7);
        assert!(matches!(
            rewritten[0].metadata.internal_reminder_kind,
            Some(InternalReminderKind::LifecycleEvictionSummary)
        ));
    }
    #[test]
    fn rejects_direct_eviction_and_protected_error_or_todo() {
        let mut r = LifecycleRegistry::default();
        let m = round("turn", "one", false, false);
        let id = r.record_segment(&m, 0, 2, "turn").unwrap();
        r.apply_delta(&delta(0, &id, "evictable", &["done"]))
            .unwrap();
        assert_eq!(r.segments[&id].state, LifecycleSegmentState::Active);
        for (error, pending) in [(true, false), (false, true)] {
            let mut r = LifecycleRegistry::default();
            let m = round("turn", "x", error, pending);
            let id = r.record_segment(&m, 0, 2, "turn").unwrap();
            r.latest_tick_seq = 3;
            r.apply_delta(&delta(0, &id, "completed", &["done"]))
                .unwrap();
            r.apply_delta(&delta(1, &id, "evictable", &["done"]))
                .unwrap();
            assert!(r.candidate_segment_ids().is_empty());
        }
    }
    #[test]
    fn rejects_changed_result_id() {
        let mut r = LifecycleRegistry::default();
        let mut m = round("turn", "one", false, false);
        let id = r.record_segment(&m, 0, 2, "turn").unwrap();
        r.latest_tick_seq = 3;
        r.segments.get_mut(&id).unwrap().state = LifecycleSegmentState::Evictable;
        r.segments
            .get_mut(&id)
            .unwrap()
            .completion_evidence
            .push("done".into());
        if let MessageContent::ToolResult { tool_id, .. } = &mut m[1].content {
            *tool_id = "changed".into();
        }
        assert!(build_eviction_plan(&r, &m, &[id]).is_err());
    }
}
