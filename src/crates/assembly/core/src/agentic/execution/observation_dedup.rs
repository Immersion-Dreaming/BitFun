//! Cross-round observation deduplication
//!
//! When an agent reads the same file or executes the same command multiple times
//! within a dialog turn and the content is unchanged, storing the full observation
//! text on every round wastes context tokens. This module tracks a
//! content-addressable index of tool-result observations seen within one turn and
//! replaces exact duplicates with a lightweight reference message.
//!
//! Scope: one `TurnObservationDeduplicator` per `execute_turn` call; reset after
//! every context compression so references never point past a compacted boundary.

use crate::agentic::core::{Message, MessageContent};
use log::debug;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Minimum visible content length (chars) before we bother deduplicating.
/// Small results don't meaningfully bloat context; skip them to avoid overhead.
const MIN_DEDUP_CHARS: usize = 800;

/// Tool names whose results are never deduplicated regardless of content.
const EXCLUDED_TOOLS: &[&str] = &["GetToolSpec"];

// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct SeenObservation {
    /// Position of the original message in the `messages` Vec at the time it
    /// was stored.  Used only for the human-readable reference comment; we do
    /// not use it as an index into a live slice.
    msg_index: usize,
    round_index: usize,
    tool_name: String,
}

/// Per-turn tracker that detects duplicate tool-result observations and
/// replaces them with a compact reference.
#[derive(Debug, Default)]
pub(crate) struct TurnObservationDeduplicator {
    seen: HashMap<String, SeenObservation>,
}

impl TurnObservationDeduplicator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Examine a tool-result `Message` that is about to be appended to the
    /// conversation history.
    ///
    /// * If this is the **first** time we see this content, record it and
    ///   return the message unchanged.
    /// * If we have **seen the same content before**, return a copy of the
    ///   message with `result_for_assistant` replaced by a short reference.
    ///
    /// `current_messages_len` is `messages.len()` just before the push so
    /// that the reference can cite the original message's position.
    pub(crate) fn apply(
        &mut self,
        msg: &Message,
        current_messages_len: usize,
        round_index: usize,
    ) -> Message {
        let MessageContent::ToolResult {
            tool_name,
            result_for_assistant,
            is_error,
            image_attachments,
            ..
        } = &msg.content
        else {
            return msg.clone();
        };

        // Never dedup errors, image results, or excluded tool names.
        if *is_error {
            return msg.clone();
        }
        if EXCLUDED_TOOLS.contains(&tool_name.as_str()) {
            return msg.clone();
        }
        if image_attachments.as_ref().is_some_and(|a| !a.is_empty()) {
            return msg.clone();
        }

        let Some(content) = result_for_assistant.as_ref().filter(|s| !s.is_empty()) else {
            return msg.clone();
        };

        let Some(key) = compute_dedup_key(content) else {
            return msg.clone();
        };

        if let Some(prior) = self.seen.get(&key) {
            debug!(
                "Observation deduped: tool={}, round={}, original_msg_index={}",
                tool_name, round_index, prior.msg_index
            );
            let replacement = format!(
                "[Observation deduped: identical content already present at context \
                 position {} (round {}, {}). Omitted to reduce context size. \
                 Call Read/Bash again if the content may have changed.]",
                prior.msg_index, prior.round_index, prior.tool_name
            );
            return replace_result_for_assistant(msg, replacement);
        }

        // First occurrence — record and keep the message unchanged.
        self.seen.insert(
            key,
            SeenObservation {
                msg_index: current_messages_len,
                round_index,
                tool_name: tool_name.clone(),
            },
        );
        msg.clone()
    }

    /// Call this whenever context compression fires so that future dedup
    /// references do not point at positions that no longer exist.
    pub(crate) fn reset_after_compression(&mut self) {
        let count = self.seen.len();
        self.seen.clear();
        if count > 0 {
            debug!(
                "ObservationDeduplicator reset after compression: cleared {} entries",
                count
            );
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Key computation
// ────────────────────────────────────────────────────────────────────────────

/// Compute the dedup key for a `result_for_assistant` string.
///
/// For **persisted outputs** (the `[PERSISTED_OUTPUT: ...]` format produced by
/// `tool_result_storage`), the reference line contains a UUID-based file name
/// that differs on every invocation even when the underlying content is
/// identical. We strip everything before the `---Preview` marker and hash only
/// the preview text so that two reads of the same unchanged file produce the
/// same key.
///
/// Returns `None` when the normalised content is shorter than `MIN_DEDUP_CHARS`
/// (not worth tracking) or when the string is empty.
fn compute_dedup_key(result_for_assistant: &str) -> Option<String> {
    let canonical = if result_for_assistant.starts_with("[PERSISTED_OUTPUT") {
        // The persisted-output format is:
        //   [PERSISTED_OUTPUT: tool_name=..., chars=..., lines=...]
        //   Full output saved to: .../tool-results/<uuid>.{txt,json}
        //   ...metadata lines...
        //   ---Preview (first N chars)---
        //   <actual preview content>
        //   ...
        //
        // Everything from "---Preview" onward is stable for unchanged content.
        match result_for_assistant.find("---Preview") {
            Some(pos) => &result_for_assistant[pos..],
            // Malformed persisted output — fall back to full hash (UUID path
            // will prevent hits, but we won't crash).
            None => result_for_assistant,
        }
    } else {
        result_for_assistant
    };

    if canonical.chars().count() < MIN_DEDUP_CHARS {
        return None;
    }

    let hash = hex::encode(Sha256::digest(canonical.as_bytes()));
    Some(hash)
}

// ────────────────────────────────────────────────────────────────────────────
// Message mutation helper
// ────────────────────────────────────────────────────────────────────────────

fn replace_result_for_assistant(msg: &Message, new_content: String) -> Message {
    let MessageContent::ToolResult {
        tool_id,
        tool_name,
        result,
        is_error,
        image_attachments,
        ..
    } = &msg.content
    else {
        return msg.clone();
    };

    Message {
        content: MessageContent::ToolResult {
            tool_id: tool_id.clone(),
            tool_name: tool_name.clone(),
            result: result.clone(),
            result_for_assistant: Some(new_content),
            is_error: *is_error,
            image_attachments: image_attachments.clone(),
        },
        ..msg.clone()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::core::ToolResult;
    use serde_json::json;

    /// Extract the `result_for_assistant` text from a tool-result `Message`.
    fn result_text(msg: &Message) -> Option<&str> {
        match &msg.content {
            MessageContent::ToolResult {
                result_for_assistant,
                ..
            } => result_for_assistant.as_deref(),
            _ => None,
        }
    }

    fn make_tool_result_message(
        tool_name: &str,
        content: &str,
        is_error: bool,
    ) -> Message {
        let result = ToolResult {
            tool_id: format!("id-{}", uuid::Uuid::new_v4()),
            tool_name: tool_name.to_string(),
            result: json!({"content": content}),
            result_for_assistant: Some(content.to_string()),
            is_error,
            duration_ms: None,
            image_attachments: None,
        };
        Message::tool_result(result)
    }

    fn large(n: usize) -> String {
        "x".repeat(n)
    }

    #[test]
    fn first_occurrence_is_unchanged() {
        let mut dedup = TurnObservationDeduplicator::new();
        let msg = make_tool_result_message("Read", &large(1000), false);
        let out = dedup.apply(&msg, 10, 1);
        assert_eq!(
            result_text(&out),
            result_text(&msg),
            "first occurrence must be stored unchanged"
        );
    }

    #[test]
    fn second_occurrence_is_replaced() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = large(1000);
        let msg = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg, 10, 1);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 12, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("Observation deduped"),
            "duplicate should be replaced: got {:?}",
            text
        );
    }

    #[test]
    fn below_threshold_is_never_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = "short content";
        let msg = make_tool_result_message("Read", content, false);
        let _ = dedup.apply(&msg, 5, 1);
        let msg2 = make_tool_result_message("Read", content, false);
        let out = dedup.apply(&msg2, 6, 2);
        assert_eq!(
            result_text(&out),
            result_text(&msg2),
            "content below threshold must not be replaced"
        );
    }

    #[test]
    fn errors_are_never_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = large(2000);
        let msg = make_tool_result_message("Bash", &content, true);
        let _ = dedup.apply(&msg, 5, 1);
        let msg2 = make_tool_result_message("Bash", &content, true);
        let out = dedup.apply(&msg2, 6, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            !text.contains("Observation deduped"),
            "errors must never be deduped"
        );
    }

    #[test]
    fn reset_after_compression_clears_state() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = large(1000);
        let msg = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg, 5, 1);

        dedup.reset_after_compression();

        // After reset, the same content is treated as new.
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 0, 3);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            !text.contains("Observation deduped"),
            "after compression reset, same content should be treated as new"
        );
    }

    #[test]
    fn persisted_output_deduped_despite_different_uuid_path() {
        let mut dedup = TurnObservationDeduplicator::new();

        // Two persisted-output blobs with different UUID paths but same preview.
        let preview = large(2000);
        let content1 = format!(
            "[PERSISTED_OUTPUT: tool_name=Read, chars=100000, lines=1000]\n\
             Full output saved to: /sessions/s1/tool-results/read_aabbccdd.txt\n\
             Line count: 1000\n\
             ---Preview (first 2000 chars)---\n{}",
            preview
        );
        let content2 = format!(
            "[PERSISTED_OUTPUT: tool_name=Read, chars=100000, lines=1000]\n\
             Full output saved to: /sessions/s1/tool-results/read_11223344.txt\n\
             Line count: 1000\n\
             ---Preview (first 2000 chars)---\n{}",
            preview
        );

        let msg1 = make_tool_result_message("Read", &content1, false);
        let _ = dedup.apply(&msg1, 5, 1);

        let msg2 = make_tool_result_message("Read", &content2, false);
        let out = dedup.apply(&msg2, 8, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("Observation deduped"),
            "persisted outputs with same preview but different UUID path should dedup"
        );
    }

    #[test]
    fn different_content_is_not_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let msg1 = make_tool_result_message("Read", &large(1000), false);
        let _ = dedup.apply(&msg1, 5, 1);
        let msg2 = make_tool_result_message("Read", &large(1001), false);
        let out = dedup.apply(&msg2, 8, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            !text.contains("Observation deduped"),
            "different content must not be deduped"
        );
    }
}
