//! Lifecycle-Aware Eviction (TokenPilot paper, Section 3.3)
//!
//! Maintains a registry of tool-call segments with monotone lifecycle states
//! (Active → Completed → Evictable). Every B rounds a Haiku estimator is
//! called to produce state updates ΔR; completed segments are physically
//! drained from the message vec and replaced with a one-line summary.

use crate::agentic::core::message::{
    InternalReminderKind, Message, MessageContent, ToolCall, ToolResult,
};
use crate::util::errors::{BitFunError, BitFunResult};
use crate::util::types::Message as AIMessage;
use log::{info, warn};
use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────────

pub(crate) const LIFECYCLE_BATCH_SIZE: usize = 3;

/// Char limit for regular (non-persisted) tool result previews in Vi.
const RESULT_PREVIEW_CHARS: usize = 600;

/// Partial match for the preview-section header inside a persisted tool result.
/// N is dynamic (TOOL_RESULT_PREVIEW_CHARS), so we match the prefix only —
/// same technique as observation_dedup.rs:160.
const PERSISTED_PREVIEW_SEARCH: &str = "---Preview";

/// Char limit for the task-context snippet fed to the estimator.
const TASK_CONTEXT_CHARS: usize = 500;

const ESTIMATOR_SYSTEM_PROMPT: &str = r#"You are a context lifecycle manager for a software engineering AI agent.
Analyze the historical tool call segments and determine which are still needed.

For each segment, assign ONE state:
- "active": outputs currently being built upon or referenced
- "completed": sub-task done, may still have downstream dependencies
- "evictable": sub-task fully done AND outputs no longer needed

Rules:
1. NEVER mark segments containing error results as evictable.
2. If a file was Read in round N but later Written/Edited to the same path, round N's Read is likely evictable.
3. The 2 most recent rounds should generally stay "active".
4. When uncertain, prefer "completed" over "evictable".
5. Only mark "evictable" when confident the content is no longer needed.
6. Some results may show "[Observation deduped: identical content already present at context position N...]"
   This means the same content appeared in an earlier round. Treat such segments as if their results
   were real outputs — the dedup marker just means the content was already seen.

Respond ONLY with valid JSON. Include only rounds whose state should change:
{"state_updates": {"0": "evictable", "3": "completed"}}"#;

// ── State machine ────────────────────────────────────────────────────────────

/// Monotone lifecycle state — values are ordered so `>` means "further along".
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SegmentState {
    Active = 0,
    Completed = 1,
    Evictable = 2,
}

// ── View types (serialised into Vi for the estimator) ────────────────────────

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SegmentView {
    round_index: usize,
    tool_calls: Vec<ToolCallView>,
    results: Vec<ResultView>,
}

#[derive(Debug, Clone, Serialize)]
struct ToolCallView {
    tool_name: String,
    /// Full args JSON — not truncated; cross-round dependency detection needs the full args.
    args_json: String,
}

#[derive(Debug, Clone, Serialize)]
struct ResultView {
    tool_name: String,
    /// For persisted results: text after the "---Preview…---\n" header line.
    /// For regular results: first RESULT_PREVIEW_CHARS chars.
    preview: String,
    is_error: bool,
    is_persisted: bool,
}

// ── Core segment record ───────────────────────────────────────────────────────

struct ContextSegment {
    round_index: usize,
    /// Index in the message vec of this round's Assistant message.
    assistant_msg_idx: usize,
    /// Exclusive upper bound: tool results occupy [assistant_msg_idx+1, end_idx).
    end_idx: usize,
    state: SegmentState,
    /// Pre-computed view used in Vi construction.
    view: SegmentView,
}

// ── Manager ───────────────────────────────────────────────────────────────────

pub(crate) struct LifecycleEvictionManager {
    /// Registry R from the paper — maintained incrementally across batches.
    segments: Vec<ContextSegment>,
    batch_size: usize,
}

impl LifecycleEvictionManager {
    pub(crate) fn new(batch_size: usize) -> Self {
        Self { segments: Vec::new(), batch_size }
    }

    /// Record the just-completed round as a ContextSegment.
    ///
    /// Call this after all tool results for the round have been pushed to
    /// `messages`.  `assistant_msg_idx` must be the index captured immediately
    /// after pushing the assistant message (i.e. `messages.len() - 1` at that
    /// point).  Pure-text rounds (no tool calls) are silently skipped.
    pub(crate) fn record_segment(
        &mut self,
        messages: &[Message],
        round_index: usize,
        assistant_msg_idx: usize,
    ) {
        let end_idx = messages.len();

        let has_tool_calls = matches!(
            &messages[assistant_msg_idx].content,
            MessageContent::Mixed { tool_calls, .. } if !tool_calls.is_empty()
        );
        if !has_tool_calls || end_idx <= assistant_msg_idx + 1 {
            return;
        }

        let view = build_segment_view(round_index, messages, assistant_msg_idx, end_idx);
        self.segments.push(ContextSegment {
            round_index,
            assistant_msg_idx,
            end_idx,
            state: SegmentState::Active,
            view,
        });
    }

    /// Returns true when `round_index` is a batch boundary and there is at
    /// least one segment to evaluate.
    pub(crate) fn should_run_estimator(&self, round_index: usize) -> bool {
        (round_index + 1) % self.batch_size == 0 && !self.segments.is_empty()
    }

    /// Run a full eviction cycle for this batch boundary.
    ///
    /// On estimator failure the method returns `Ok(0)` — eviction is skipped
    /// rather than propagating an error, so the agent continues uninterrupted.
    pub(crate) async fn run_batch_eviction(
        &mut self,
        messages: &mut Vec<Message>,
        round_index: usize,
        task_context: &str,
        tool_results_dir: Option<&std::path::Path>,
    ) -> BitFunResult<usize> {
        let vi_json = self.build_vi(round_index, task_context)?;

        let state_updates = match self.call_estimator(vi_json).await {
            Ok(updates) => updates,
            Err(e) => {
                warn!("lifecycle eviction: estimator failed, skipping batch: {}", e);
                return Ok(0);
            }
        };

        self.apply_state_updates(state_updates);
        let evicted = self.execute_eviction(messages, tool_results_dir).await?;

        if evicted > 0 {
            info!(
                "lifecycle eviction: round={}, evicted={} segments, messages_len={}",
                round_index,
                evicted,
                messages.len()
            );
        }
        Ok(evicted)
    }

    /// Clear all segment records after a full compression or emergency truncation —
    /// all stored message indices are stale after either event.
    pub(crate) fn reset_after_compression(&mut self) {
        self.segments.clear();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Build Vi: the compressed historical view sent to the estimator.
    fn build_vi(&self, round_index: usize, task_context: &str) -> BitFunResult<String> {
        #[derive(Serialize)]
        struct EstimatorInput<'a> {
            current_task: &'a str,
            current_round: usize,
            segments: Vec<&'a SegmentView>,
        }
        let input = EstimatorInput {
            current_task: task_context,
            current_round: round_index,
            segments: self
                .segments
                .iter()
                .filter(|s| s.state != SegmentState::Evictable)
                .map(|s| &s.view)
                .collect(),
        };
        serde_json::to_string(&input).map_err(|e| {
            BitFunError::tool(format!("lifecycle eviction: Vi serialization: {e}"))
        })
    }

    /// Send Vi to the Haiku estimator and return parsed state updates.
    async fn call_estimator(
        &self,
        vi_json: String,
    ) -> BitFunResult<std::collections::HashMap<usize, SegmentState>> {
        use crate::infrastructure::ai::get_global_ai_client_factory;

        let factory = get_global_ai_client_factory().await.map_err(|e| {
            BitFunError::AIClient(format!("lifecycle estimator factory: {e}"))
        })?;

        // get_client_resolved handles "fast" → primary fallback internally.
        let client = factory.get_client_resolved("fast").await.map_err(|e| {
            BitFunError::AIClient(format!("lifecycle estimator client: {e}"))
        })?;

        let req = vec![
            AIMessage::system(ESTIMATOR_SYSTEM_PROMPT.to_string()),
            AIMessage::user(vi_json),
        ];
        let response = client
            .send_message_with_trace(req, None, None)
            .await
            .map_err(|e| BitFunError::AIClient(format!("lifecycle estimator request: {e}")))?;

        parse_estimator_response(&response.text)
    }

    /// Apply ΔR updates to the registry (R_i ← R_{i-1} ⊕ ΔR_i).
    /// States are monotone: a segment can only advance, never regress.
    pub(crate) fn apply_state_updates(
        &mut self,
        updates: std::collections::HashMap<usize, SegmentState>,
    ) {
        for seg in self.segments.iter_mut() {
            let Some(new_state) = updates.get(&seg.round_index) else {
                continue;
            };
            // Double guard: the estimator system prompt prohibits this, but
            // we enforce it in code as well.
            if *new_state == SegmentState::Evictable
                && seg.view.results.iter().any(|r| r.is_error)
            {
                warn!(
                    "lifecycle: estimator marked error segment {} as evictable, ignoring",
                    seg.round_index
                );
                continue;
            }
            if *new_state > seg.state {
                seg.state = new_state.clone();
            }
        }
    }

    /// Physically drain evictable segments from `messages`, insert summary
    /// placeholders, and reindex surviving segments.
    async fn execute_eviction(
        &mut self,
        messages: &mut Vec<Message>,
        tool_results_dir: Option<&std::path::Path>,
    ) -> BitFunResult<usize> {
        struct EvictTarget {
            round_index: usize,
            asst_idx: usize,
            end_idx: usize,
        }

        let mut targets: Vec<EvictTarget> = self
            .segments
            .iter()
            .filter(|s| s.state == SegmentState::Evictable)
            .map(|s| EvictTarget {
                round_index: s.round_index,
                asst_idx: s.assistant_msg_idx,
                end_idx: s.end_idx,
            })
            .collect();

        if targets.is_empty() {
            return Ok(0);
        }

        // Process highest index first so earlier indices remain valid.
        targets.sort_by(|a, b| b.asst_idx.cmp(&a.asst_idx));

        for target in &targets {
            // Save a recovery file (best-effort; failure does not block eviction).
            let recovery_path = if let Some(dir) = tool_results_dir {
                match save_recovery_file(
                    target.round_index,
                    target.asst_idx,
                    target.end_idx,
                    messages,
                    dir,
                )
                .await
                {
                    Ok(p) => Some(p),
                    Err(e) => {
                        warn!(
                            "lifecycle: recovery file failed for round {}: {}",
                            target.round_index, e
                        );
                        None
                    }
                }
            } else {
                None
            };

            let tool_names: String = self
                .segments
                .iter()
                .find(|s| s.round_index == target.round_index)
                .map(|s| {
                    s.view
                        .tool_calls
                        .iter()
                        .map(|tc| tc.tool_name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();

            let recovery_hint = recovery_path
                .as_ref()
                .map(|p| {
                    format!(
                        "\nFull content saved to: {}\nUse the Read tool to recover if needed.",
                        p.display()
                    )
                })
                .unwrap_or_default();

            let summary_text = format!(
                "[LIFECYCLE_EVICTION_SUMMARY: round={}]\nSub-task completed and evicted. Tools: [{}].{}",
                target.round_index, tool_names, recovery_hint,
            );
            let summary_msg = Message::internal_reminder(
                InternalReminderKind::LifecycleEvictionSummary,
                summary_text,
            );

            messages.drain(target.asst_idx..target.end_idx);
            messages.insert(target.asst_idx, summary_msg);
        }

        // Reindex surviving segments using the original (pre-eviction) indices,
        // processed in ascending order so each shift accumulates correctly.
        let mut targets_asc = targets;
        targets_asc.sort_by_key(|t| t.asst_idx);

        for seg in self.segments.iter_mut() {
            if seg.state == SegmentState::Evictable {
                continue;
            }
            // Only targets fully before this segment contribute a shift.
            let shift: isize = targets_asc
                .iter()
                .filter(|t| t.end_idx <= seg.assistant_msg_idx)
                .map(|t| 1isize - (t.end_idx as isize - t.asst_idx as isize))
                .sum();

            seg.assistant_msg_idx = (seg.assistant_msg_idx as isize + shift) as usize;
            seg.end_idx = (seg.end_idx as isize + shift) as usize;
        }

        self.segments.retain(|s| s.state != SegmentState::Evictable);

        Ok(targets_asc.len())
    }
}

// ── Module-level helpers ──────────────────────────────────────────────────────

fn build_segment_view(
    round_index: usize,
    messages: &[Message],
    asst_idx: usize,
    end_idx: usize,
) -> SegmentView {
    let tool_calls = match &messages[asst_idx].content {
        MessageContent::Mixed { tool_calls, .. } => tool_calls
            .iter()
            .map(|tc| ToolCallView {
                tool_name: tc.tool_name.clone(),
                args_json: serde_json::to_string(&tc.arguments).unwrap_or_default(),
            })
            .collect(),
        _ => vec![],
    };

    let results = messages[asst_idx + 1..end_idx]
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::ToolResult {
                tool_name,
                result_for_assistant,
                is_error,
                ..
            } => {
                let text = result_for_assistant.as_deref().unwrap_or("");
                let is_persisted = bitfun_agent_tools::tool_result_is_persisted_output(text);
                Some(ResultView {
                    tool_name: tool_name.clone(),
                    preview: extract_result_preview(text, is_persisted),
                    is_error: *is_error,
                    is_persisted,
                })
            }
            _ => None,
        })
        .collect();

    SegmentView { round_index, tool_calls, results }
}

fn extract_result_preview(content: &str, is_persisted: bool) -> String {
    if is_persisted {
        // Partial match — N in "---Preview (first N chars)---" is dynamic.
        if let Some(pos) = content.find(PERSISTED_PREVIEW_SEARCH) {
            if let Some(newline) = content[pos..].find('\n') {
                return content[pos + newline + 1..].to_string();
            }
        }
    }
    truncate_chars(content, RESULT_PREVIEW_CHARS)
}

/// Truncate `s` to at most `max` Unicode scalar values, appending "…" if cut.
/// Exported so execution_engine.rs can use it for task_context extraction.
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let out: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() { format!("{out}…") } else { out }
}

fn parse_estimator_response(
    text: &str,
) -> BitFunResult<std::collections::HashMap<usize, SegmentState>> {
    #[derive(Deserialize)]
    struct EstimatorOutput {
        state_updates: std::collections::HashMap<String, String>,
    }

    let json_str = extract_json_object(text).ok_or_else(|| {
        BitFunError::tool(format!(
            "lifecycle estimator: no JSON object in response: {text}"
        ))
    })?;

    let output: EstimatorOutput = serde_json::from_str(json_str).map_err(|e| {
        BitFunError::tool(format!(
            "lifecycle estimator: invalid JSON ({e}): {json_str}"
        ))
    })?;

    let mut result = std::collections::HashMap::new();
    for (k, v) in output.state_updates {
        let round: usize = match k.parse() {
            Ok(n) => n,
            Err(_) => {
                warn!("lifecycle estimator: bad round key '{}', skipping", k);
                continue;
            }
        };
        let state = match v.as_str() {
            "active" => SegmentState::Active,
            "completed" => SegmentState::Completed,
            "evictable" => SegmentState::Evictable,
            other => {
                warn!("lifecycle estimator: unknown state '{}', skipping", other);
                continue;
            }
        };
        result.insert(round, state);
    }
    Ok(result)
}

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end >= start { Some(&text[start..=end]) } else { None }
}

async fn save_recovery_file(
    round_index: usize,
    asst_idx: usize,
    end_idx: usize,
    messages: &[Message],
    dir: &std::path::Path,
) -> BitFunResult<std::path::PathBuf> {
    let tool_calls: Vec<serde_json::Value> = match &messages[asst_idx].content {
        MessageContent::Mixed { tool_calls, .. } => tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "tool_id":   tc.tool_id,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                })
            })
            .collect(),
        _ => vec![],
    };

    let tool_results: Vec<serde_json::Value> = messages[asst_idx + 1..end_idx]
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::ToolResult {
                tool_id,
                tool_name,
                result_for_assistant,
                is_error,
                ..
            } => Some(serde_json::json!({
                "tool_id":   tool_id,
                "tool_name": tool_name,
                "is_error":  is_error,
                "content":   result_for_assistant.as_deref().unwrap_or(""),
            })),
            _ => None,
        })
        .collect();

    let record = serde_json::json!({
        "round_index":  round_index,
        "tool_calls":   tool_calls,
        "tool_results": tool_results,
    });

    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|e| BitFunError::io(format!("lifecycle: create dir: {e}")))?;

    let path = dir.join(format!(
        "evicted_round_{}_{}.json",
        round_index,
        uuid::Uuid::new_v4()
    ));
    tokio::fs::write(&path, serde_json::to_string_pretty(&record)?)
        .await
        .map_err(|e| BitFunError::io(format!("lifecycle: write recovery file: {e}")))?;
    Ok(path)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages(tool_count: usize) -> Vec<Message> {
        let mut msgs = vec![
            Message::system("sys".to_string()),
            Message::user("task".to_string()),
        ];
        let tool_calls: Vec<ToolCall> = (0..tool_count)
            .map(|i| ToolCall {
                tool_id: format!("id_{i}"),
                tool_name: format!("Tool{i}"),
                arguments: serde_json::json!({"file": format!("f{i}.rs")}),
                ..Default::default()
            })
            .collect();
        msgs.push(Message::assistant_with_tools(String::new(), tool_calls));
        for i in 0..tool_count {
            msgs.push(Message::tool_result(ToolResult {
                tool_id: format!("id_{i}"),
                tool_name: format!("Tool{i}"),
                result: serde_json::json!({}),
                result_for_assistant: Some(format!("result {i}")),
                is_error: false,
                duration_ms: None,
                image_attachments: None,
            }));
        }
        msgs
    }

    #[test]
    fn record_captures_correct_indices() {
        let msgs = make_messages(2);
        let asst_idx = 2;
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, asst_idx);

        assert_eq!(mgr.segments.len(), 1);
        let seg = &mgr.segments[0];
        assert_eq!(seg.assistant_msg_idx, 2);
        assert_eq!(seg.end_idx, 5); // sys + user + asst + 2 results
        assert_eq!(seg.state, SegmentState::Active);
        assert_eq!(seg.view.tool_calls.len(), 2);
        assert_eq!(seg.view.results.len(), 2);
    }

    #[test]
    fn text_only_round_not_recorded() {
        let msgs = vec![
            Message::system("sys".to_string()),
            Message::user("task".to_string()),
            Message::assistant("just text".to_string()),
        ];
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);
        assert!(mgr.segments.is_empty());
    }

    #[test]
    fn should_run_at_batch_boundaries() {
        let msgs = make_messages(1);
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);

        assert!(!mgr.should_run_estimator(0));
        assert!(!mgr.should_run_estimator(1));
        assert!(mgr.should_run_estimator(2));
        assert!(!mgr.should_run_estimator(3));
        assert!(mgr.should_run_estimator(5));
    }

    #[test]
    fn state_only_upgrades() {
        let msgs = make_messages(1);
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);
        mgr.segments[0].state = SegmentState::Completed;

        mgr.apply_state_updates([(0, SegmentState::Active)].into());
        assert_eq!(mgr.segments[0].state, SegmentState::Completed);

        mgr.apply_state_updates([(0, SegmentState::Evictable)].into());
        assert_eq!(mgr.segments[0].state, SegmentState::Evictable);
    }

    #[test]
    fn error_segment_protected_from_eviction() {
        let msgs = make_messages(1);
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);
        mgr.segments[0].view.results[0].is_error = true;

        mgr.apply_state_updates([(0, SegmentState::Evictable)].into());
        assert_eq!(mgr.segments[0].state, SegmentState::Active);
    }

    #[tokio::test]
    async fn reindex_correct_after_eviction() {
        // Layout: [sys(0), user(1), asst0(2), r00(3), r01(4), asst1(5), r10(6), r11(7), r12(8), asst2(9), r20(10)]
        // Seg0: asst=2 end=5 (2 results)   → Evictable
        // Seg1: asst=5 end=9 (3 results)   → Evictable
        // Seg2: asst=9 end=11 (1 result)   → Active
        let mut msgs = vec![
            Message::system("sys".into()),
            Message::user("task".into()),
        ];
        for round in 0..3usize {
            let count = match round {
                1 => 3,
                2 => 1,
                _ => 2,
            };
            let calls: Vec<ToolCall> = (0..count)
                .map(|i| ToolCall {
                    tool_id: format!("id_r{round}_{i}"),
                    tool_name: "Tool".into(),
                    arguments: serde_json::json!({}),
                    ..Default::default()
                })
                .collect();
            msgs.push(Message::assistant_with_tools(String::new(), calls));
            for i in 0..count {
                msgs.push(Message::tool_result(ToolResult {
                    tool_id: format!("id_r{round}_{i}"),
                    tool_name: "Tool".into(),
                    result: serde_json::json!({}),
                    result_for_assistant: Some("result".into()),
                    is_error: false,
                    duration_ms: None,
                    image_attachments: None,
                }));
            }
        }

        let empty_view =
            |idx| SegmentView { round_index: idx, tool_calls: vec![], results: vec![] };

        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.segments = vec![
            ContextSegment {
                round_index: 0,
                assistant_msg_idx: 2,
                end_idx: 5,
                state: SegmentState::Evictable,
                view: empty_view(0),
            },
            ContextSegment {
                round_index: 1,
                assistant_msg_idx: 5,
                end_idx: 9,
                state: SegmentState::Evictable,
                view: empty_view(1),
            },
            ContextSegment {
                round_index: 2,
                assistant_msg_idx: 9,
                end_idx: 11,
                state: SegmentState::Active,
                view: empty_view(2),
            },
        ];

        mgr.execute_eviction(&mut msgs, None).await.unwrap();

        assert_eq!(mgr.segments.len(), 1);
        let seg = &mgr.segments[0];
        assert_eq!(seg.round_index, 2);
        // shift from Seg0: 1-(5-2)=-2; from Seg1: 1-(9-5)=-3; total=-5
        assert_eq!(seg.assistant_msg_idx, 4); // 9 - 5
        assert_eq!(seg.end_idx, 6);           // 11 - 5
        // Evict Seg1 [5,9): drain 4 + insert 1 → 8 msgs
        // Evict Seg0 [2,5): drain 3 + insert 1 → 6 msgs
        assert_eq!(msgs.len(), 6);
    }
}
