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
//!
//! Design notes:
//!
//! * Dedup keys are content addresses, never message positions. Oversized
//!   results are persisted by `tool_result_storage` with an explicit
//!   `Content sha256: <hash>` header line covering the full content plus
//!   metadata; that hash is the authoritative key. Legacy persisted blobs
//!   without the hash line are canonicalised to their preview section.
//! * Replacement markers are self-describing: they cite the round and a short
//!   descriptor of the original observation instead of a context position,
//!   because message positions shift when system reminders are injected
//!   before sending and when context compression prunes history.
//! * Reads of files that were edited after the original observation are never
//!   deduplicated, so the model always gets a fresh view after a mutation.

use crate::agentic::core::{Message, MessageContent};
use bitfun_agent_tools::PERSISTED_OUTPUT_TAG;
use log::debug;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Minimum visible content length (chars) before we bother deduplicating.
/// Small results don't meaningfully bloat context; skip them to avoid overhead.
const MIN_DEDUP_CHARS: usize = 800;

/// Tool names whose results are never deduplicated regardless of content.
const EXCLUDED_TOOLS: &[&str] = &["GetToolSpec"];

/// Tools whose success result mutates a file on disk. When such a file was
/// edited after an observation was recorded, later reads of the same path
/// must not be deduplicated against the pre-edit observation.
const FILE_MUTATION_TOOLS: &[&str] = &["Edit", "Write"];

/// Tool whose results are checked for file freshness after an edit.
const READ_TOOL_NAME: &str = "Read";

/// Prefix of marker messages injected in place of a duplicate observation.
const DEDUP_MARKER_PREFIX: &str = "[Observation deduped:";

/// Header line emitted by `build_persisted_tool_output_message` for oversized
/// results; carries the sha256 of the full persisted content plus metadata.
const CONTENT_SHA256_LINE_PREFIX: &str = "Content sha256: ";

/// Start of the preview section header inside a persisted-output message.
const PREVIEW_HEADER_PREFIX: &str = "Preview (first ";

/// Short excerpt used to make dedup markers self-describing.
const MAX_DESCRIPTOR_CHARS: usize = 120;

/// Appended to markers that survived a context compression, whose original
/// observation may no longer be present in history.
const COMPACTION_ANNOTATION: &str = "\n[Note: context was compacted since this observation; the original content may no longer be in history. Re-read if needed.]";

#[derive(Debug)]
struct SeenObservation {
    /// Round in which the original observation was stored. Rounds are stable
    /// chronological anchors (unlike message positions, which shift when
    /// system reminders are injected or history is compacted).
    round_index: usize,
    tool_name: String,
    /// Short, self-describing excerpt of the original content so the model can
    /// recognise what was omitted without a context position.
    descriptor: String,
}

/// Per-turn tracker that detects duplicate tool-result observations and
/// replaces them with a compact reference.
#[derive(Debug, Default)]
pub(crate) struct TurnObservationDeduplicator {
    seen: HashMap<String, SeenObservation>,
    /// Logical paths mutated by Edit/Write, mapped to the round of the last
    /// successful mutation. Kept across compression resets: edit rounds stay
    /// valid chronological anchors and keep the freshness guard working.
    edited_files: HashMap<String, usize>,
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
    ///   message with `result_for_assistant` replaced by a short reference —
    ///   unless the observation came from a file that was edited after the
    ///   original observation (freshness guard).
    ///
    /// `round_index` is the execution round the message belongs to; markers
    /// reference it instead of a message position so they stay valid across
    /// reminder injection and compression.
    pub(crate) fn apply(&mut self, msg: &Message, round_index: usize) -> Message {
        let MessageContent::ToolResult {
            tool_name,
            result,
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

        // Record successful file mutations so later reads of the same path
        // are kept fresh (see freshness guard below).
        if FILE_MUTATION_TOOLS.contains(&tool_name.as_str()) {
            let success = result
                .get("success")
                .and_then(|value| value.as_bool())
                .unwrap_or(true);
            if success {
                if let Some(path) = file_path_from_result(tool_name, result, content) {
                    self.edited_files.insert(path, round_index);
                }
            }
            return msg.clone();
        }

        let Some(key) = compute_dedup_key(content) else {
            return msg.clone();
        };

        if let Some(prior) = self.seen.get(&key) {
            let prior_round = prior.round_index;
            let prior_tool = prior.tool_name.clone();
            let prior_descriptor = prior.descriptor.clone();

            // Freshness guard: when a Read result matches a pre-edit
            // observation of the same path, the visible content can be
            // byte-identical while the file changed outside the read range.
            // Show the fresh result once and rebase the recorded round so
            // subsequent reads can deduplicate against the fresh view.
            //
            // `edited_round > prior_round` deliberately uses strict
            // inequality: a same-round edit + re-read (parallel tool calls in
            // one round) is ambiguous — the edit may have happened before or
            // after the original read — so suppressing dedup in that case
            // would hide legitimate dedup of unchanged content.
            if tool_name == READ_TOOL_NAME {
                if let Some(path) = file_path_from_result(tool_name, result, content) {
                    if let Some(edited_round) = self.edited_files.get(&path).copied() {
                        if edited_round > prior_round {
                            debug!(
                                "Observation dedup skipped after edit: tool={}, round={}, edited_round={}, descriptor={:?}",
                                tool_name, round_index, edited_round, prior_descriptor
                            );
                            self.seen.insert(
                                key,
                                SeenObservation {
                                    round_index,
                                    tool_name: tool_name.clone(),
                                    descriptor: build_descriptor(content),
                                },
                            );
                            return msg.clone();
                        }
                    }
                }
            }

            debug!(
                "Observation deduped: tool={}, round={}, original_round={}, descriptor={:?}",
                tool_name, round_index, prior_round, prior_descriptor
            );
            let replacement = format!(
                "[Observation deduped: identical content was already presented at round {} ({}: {}). Omitted to reduce context size. Call Read/Bash again if the content may have changed.]",
                prior_round, prior_tool, prior_descriptor
            );
            return replace_result_for_assistant(msg, replacement);
        }

        // First occurrence — record and keep the message unchanged.
        self.seen.insert(
            key,
            SeenObservation {
                round_index,
                tool_name: tool_name.clone(),
                descriptor: build_descriptor(content),
            },
        );
        msg.clone()
    }

    /// Call this whenever context compression fires so that future dedup
    /// references do not point at observations that no longer exist.
    ///
    /// Markers that were already written into history are annotated so the
    /// model knows their original observation may have been compacted away.
    pub(crate) fn reset_after_compression(&mut self, messages: &mut [Message]) {
        let count = self.seen.len();
        self.seen.clear();
        // `edited_files` is intentionally kept: edit rounds remain valid
        // chronological anchors after compression and keep the freshness
        // guard working for future reads.
        if count > 0 {
            debug!(
                "ObservationDeduplicator reset after compression: cleared {} entries",
                count
            );
        }

        for message in messages.iter_mut() {
            let MessageContent::ToolResult {
                result_for_assistant,
                ..
            } = &mut message.content
            else {
                continue;
            };
            let Some(text) = result_for_assistant.as_mut() else {
                continue;
            };
            if text.starts_with(DEDUP_MARKER_PREFIX) && !text.contains(COMPACTION_ANNOTATION) {
                text.push_str(COMPACTION_ANNOTATION);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Key computation
// ────────────────────────────────────────────────────────────────────────────

/// Compute the dedup key for a `result_for_assistant` string.
///
/// Key resolution order:
/// 1. **Persisted outputs with an explicit hash line** — `Content sha256:` is
///    emitted by `build_persisted_tool_output_message` and covers the full
///    persisted content plus metadata, so the key is exact and independent of
///    the UUID-based reference path.
/// 2. **Legacy persisted outputs** (no hash line) — the reference line
///    contains a UUID-based path that differs on every invocation, so we hash
///    only the preview section. This is best effort: distinct contents with
///    an identical preview would collide (new persisted outputs always carry
///    the hash line to avoid this).
/// 3. **Plain outputs** — the full text.
///
/// Returns `None` when the normalised content is shorter than `MIN_DEDUP_CHARS`
/// (not worth tracking) or when the string is empty.
fn compute_dedup_key(result_for_assistant: &str) -> Option<String> {
    if let Some(hash) = parse_content_sha256(result_for_assistant) {
        // Persisted outputs are normally large, but round-budget persistence
        // can also compress small results; keep the threshold consistent with
        // the plain-content path so tiny messages are never deduplicated.
        if result_for_assistant.chars().count() < MIN_DEDUP_CHARS {
            return None;
        }
        return Some(format!("persisted:{hash}"));
    }

    if result_for_assistant.starts_with(PERSISTED_OUTPUT_TAG) {
        let canonical = legacy_persisted_canonical(result_for_assistant)?;
        if canonical.chars().count() < MIN_DEDUP_CHARS {
            return None;
        }
        return Some(format!(
            "legacy-persisted:{}",
            hex::encode(Sha256::digest(canonical.as_bytes()))
        ));
    }

    if result_for_assistant.chars().count() < MIN_DEDUP_CHARS {
        return None;
    }
    Some(format!(
        "plain:{}",
        hex::encode(Sha256::digest(result_for_assistant.as_bytes()))
    ))
}

/// Extract the `Content sha256: <64 hex chars>` line from the header of a
/// persisted-output message.
///
/// Only the header lines (before the `Preview (first ...)` line) are
/// inspected, and the scan is bounded to the first few lines, so preview
/// content can never be mistaken for the marker — a preview may legitimately
/// contain lines that look like a content hash (e.g. checksum manifests).
fn parse_content_sha256(text: &str) -> Option<String> {
    if !text.starts_with(PERSISTED_OUTPUT_TAG) {
        return None;
    }
    for line in text.lines().take(16) {
        if line.starts_with(PREVIEW_HEADER_PREFIX) {
            break;
        }
        let Some(rest) = line.strip_prefix(CONTENT_SHA256_LINE_PREFIX) else {
            continue;
        };
        let hash = rest.trim();
        if hash.len() == 64 && hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
            return Some(hash.to_ascii_lowercase());
        }
    }
    None
}

/// Canonical tail of a legacy persisted-output message: everything after the
/// `Preview (first N chars):` header line, including any metadata section.
///
/// The search is bounded to the message header (first few lines) so a
/// malformed message whose preview content contains a `Preview (first ...`
/// line cannot yield a wrong canonical tail.
fn legacy_persisted_canonical(text: &str) -> Option<&str> {
    let mut offset = 0usize;
    for line in text.split_inclusive('\n').take(16) {
        if line.starts_with(PREVIEW_HEADER_PREFIX) {
            return Some(&text[offset + line.len()..]);
        }
        offset += line.len();
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// Descriptor / path helpers
// ────────────────────────────────────────────────────────────────────────────

/// First meaningful line of an observation, truncated to keep markers short.
fn build_descriptor(content: &str) -> String {
    // For persisted outputs the first line is just the tag; for any output,
    // skip leading blank lines so the descriptor is never empty (e.g. a Bash
    // result whose output starts with a newline).
    let skip = usize::from(content.starts_with(PERSISTED_OUTPUT_TAG));
    let first_line = content
        .lines()
        .skip(skip)
        .find(|line| !line.trim().is_empty())
        .unwrap_or(content);

    let truncated: String = first_line.chars().take(MAX_DESCRIPTOR_CHARS).collect();
    if truncated.chars().count() < first_line.chars().count() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

/// Logical path associated with a tool result, when determinable.
///
/// Prefers the structured `file_path` field (populated by Read/Edit/Write)
/// and falls back to parsing stable presentation prefixes for Edit and Read.
fn file_path_from_result(
    tool_name: &str,
    result: &serde_json::Value,
    result_for_assistant: &str,
) -> Option<String> {
    if let Some(path) = result.get("file_path").and_then(|value| value.as_str()) {
        let path = path.trim();
        if !path.is_empty() {
            return Some(path.to_string());
        }
    }

    if tool_name == "Edit" {
        // `Successfully edited <logical_path>`
        let path = result_for_assistant
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("Successfully edited "))
            .map(str::trim)
            .filter(|path| !path.is_empty())?;
        return Some(path.to_string());
    }

    if tool_name == READ_TOOL_NAME {
        // `Read lines X-Y from <path> (N total lines)`
        let first_line = result_for_assistant.lines().next()?;
        let rest = first_line.strip_prefix("Read lines ")?;
        let start = rest.find(" from ")? + " from ".len();
        let path = &rest[start..];
        let path = path.split(" (").next().unwrap_or(path).trim();
        if !path.is_empty() {
            return Some(path.to_string());
        }
    }

    None
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

    fn make_tool_result_message(tool_name: &str, content: &str, is_error: bool) -> Message {
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

    fn make_tool_result_with_path(
        tool_name: &str,
        content: &str,
        file_path: &str,
        success: Option<bool>,
    ) -> Message {
        let mut data = json!({
            "content": content,
            "file_path": file_path,
        });
        if let Some(success) = success {
            data["success"] = json!(success);
        }
        let result = ToolResult {
            tool_id: format!("id-{}", uuid::Uuid::new_v4()),
            tool_name: tool_name.to_string(),
            result: data,
            result_for_assistant: Some(content.to_string()),
            is_error: false,
            duration_ms: None,
            image_attachments: None,
        };
        Message::tool_result(result)
    }

    fn large(n: usize) -> String {
        "x".repeat(n)
    }

    fn sha256_hex(content: &str) -> String {
        hex::encode(Sha256::digest(content.as_bytes()))
    }

    /// Build a persisted-output message mirroring the production format
    /// emitted by `build_persisted_tool_output_message`.
    fn persisted_message(content: &str, uuid: &str, sha: Option<&str>) -> String {
        let preview: String = content.chars().take(2000).collect();
        let mut text = format!(
            "<persisted-output>\nOutput too large ({} chars). Full output saved to: bitfun-runtime://session/s1/tool-results/read_{}.txt\nLine count: {}\n",
            content.chars().count(),
            uuid,
            content.lines().count()
        );
        if let Some(hash) = sha {
            text.push_str(&format!("Content sha256: {hash}\n"));
        }
        text.push_str(&format!(
            "\nPreview (first 2000 chars):\n{}\n</persisted-output>",
            preview
        ));
        text
    }

    fn is_dedup_marker(text: &str) -> bool {
        text.starts_with(DEDUP_MARKER_PREFIX)
    }

    #[test]
    fn first_occurrence_is_unchanged() {
        let mut dedup = TurnObservationDeduplicator::new();
        let msg = make_tool_result_message("Read", &large(1000), false);
        let out = dedup.apply(&msg, 1);
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
        let _ = dedup.apply(&msg, 1);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("Observation deduped"),
            "duplicate should be replaced: got {:?}",
            text
        );
    }

    #[test]
    fn marker_is_self_describing_and_has_no_context_position() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = large(1000);
        let msg = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg, 3);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 4);
        let text = result_text(&out).unwrap_or_default();
        assert!(text.contains("round 3"), "marker must cite round: {text}");
        assert!(
            !text.contains("context position"),
            "marker must not cite a message position: {text}"
        );
    }

    #[test]
    fn below_threshold_is_never_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = "short content";
        let msg = make_tool_result_message("Read", content, false);
        let _ = dedup.apply(&msg, 1);
        let msg2 = make_tool_result_message("Read", content, false);
        let out = dedup.apply(&msg2, 2);
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
        let _ = dedup.apply(&msg, 1);
        let msg2 = make_tool_result_message("Bash", &content, true);
        let out = dedup.apply(&msg2, 2);
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
        let _ = dedup.apply(&msg, 1);

        let mut messages = vec![msg.clone()];
        dedup.reset_after_compression(&mut messages);

        // After reset, the same content is treated as new.
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 3);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            !is_dedup_marker(text),
            "after compression reset, same content should be treated as new"
        );
    }

    #[test]
    fn marker_is_annotated_after_compression() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = large(1000);
        let msg = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg, 1);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 2);

        let mut messages = vec![msg, out.clone()];
        dedup.reset_after_compression(&mut messages);

        let text = result_text(&messages[1]).unwrap_or_default();
        assert!(is_dedup_marker(text));
        assert!(
            text.contains("context was compacted"),
            "surviving marker should be annotated: {text}"
        );
    }

    #[test]
    fn persisted_output_deduped_despite_different_uuid_path() {
        let mut dedup = TurnObservationDeduplicator::new();

        // Two persisted blobs with different UUID paths but identical content
        // and hash must deduplicate.
        let content = large(3000);
        let sha = sha256_hex(&content);
        let msg1 = make_tool_result_message(
            "Read",
            &persisted_message(&content, "aabbccdd", Some(&sha)),
            false,
        );
        let _ = dedup.apply(&msg1, 1);

        let msg2 = make_tool_result_message(
            "Read",
            &persisted_message(&content, "11223344", Some(&sha)),
            false,
        );
        let out = dedup.apply(&msg2, 2);
        assert!(
            is_dedup_marker(result_text(&out).unwrap_or_default()),
            "persisted outputs with same content hash must dedup"
        );
    }

    #[test]
    fn persisted_output_with_different_content_is_not_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content1 = large(3000);
        let content2 = large(3001);
        let msg1 = make_tool_result_message(
            "Read",
            &persisted_message(&content1, "aabbccdd", Some(&sha256_hex(&content1))),
            false,
        );
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message(
            "Read",
            &persisted_message(&content2, "11223344", Some(&sha256_hex(&content2))),
            false,
        );
        let out = dedup.apply(&msg2, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "different content must not be deduped"
        );
    }

    #[test]
    fn legacy_persisted_output_without_hash_deduped_by_preview() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = large(3000);
        let msg1 = make_tool_result_message(
            "Read",
            &persisted_message(&content, "aabbccdd", None),
            false,
        );
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message(
            "Read",
            &persisted_message(&content, "11223344", None),
            false,
        );
        let out = dedup.apply(&msg2, 2);
        assert!(
            is_dedup_marker(result_text(&out).unwrap_or_default()),
            "legacy persisted outputs with identical preview must dedup"
        );
    }

    #[test]
    fn preview_hash_lookalike_is_not_used_as_dedup_key() {
        // A legacy persisted message has no header hash line. If the preview
        // content itself contains a `Content sha256: <64 hex>` line (e.g. a
        // checksum manifest), it must not be mistaken for the authoritative
        // hash: two different outputs sharing that lookalike line must not
        // deduplicate.
        let mut dedup = TurnObservationDeduplicator::new();
        let fake_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let body1 = format!("Content sha256: {fake_hash}\nline one\n{}", large(2000));
        let body2 = format!("Content sha256: {fake_hash}\nline two\n{}", large(2000));

        let msg1 =
            make_tool_result_message("Read", &persisted_message(&body1, "aabbccdd", None), false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 =
            make_tool_result_message("Read", &persisted_message(&body2, "11223344", None), false);
        let out = dedup.apply(&msg2, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "preview hash lookalike must not cause false dedup"
        );
    }

    #[test]
    fn descriptor_skips_blank_first_line() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = format!("\n{}", large(1000));
        let msg1 = make_tool_result_message("Bash", &content, false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message("Bash", &content, false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("(Bash: xxx"),
            "descriptor must use the first non-blank line: {text}"
        );
        assert!(
            !text.contains("(Bash: )"),
            "descriptor must never be empty: {text}"
        );
    }

    #[test]
    fn read_after_edit_is_kept_fresh_then_rebases() {
        let mut dedup = TurnObservationDeduplicator::new();
        let path = "src/lib.rs";
        let body = large(1000);
        let read_content = format!(
            "Read lines 1-50 from {} (100 total lines)\n<file_content>\n{}\n</file_content>",
            path, body
        );
        let read = make_tool_result_with_path("Read", &read_content, path, None);
        let edit =
            make_tool_result_with_path("Edit", "Successfully edited src/lib.rs", path, Some(true));

        // Round 1: first read is stored.
        let _ = dedup.apply(&read, 1);
        // Round 2: edit records the mutation.
        let out = dedup.apply(&edit, 2);
        assert_eq!(result_text(&out), result_text(&edit));

        // Round 3: identical read after the edit must NOT be deduped.
        let out = dedup.apply(&read, 3);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "read after edit must stay fresh"
        );

        // Round 4: subsequent identical read dedups against round 3.
        let out = dedup.apply(&read, 4);
        let text = result_text(&out).unwrap_or_default();
        assert!(is_dedup_marker(text));
        assert!(
            text.contains("round 3"),
            "marker must cite the fresh read: {text}"
        );
    }

    #[test]
    fn failed_edit_does_not_suppress_dedup() {
        let mut dedup = TurnObservationDeduplicator::new();
        let path = "src/lib.rs";
        let body = large(1000);
        let read_content = format!(
            "Read lines 1-50 from {} (100 total lines)\n<file_content>\n{}\n</file_content>",
            path, body
        );
        let read = make_tool_result_with_path("Read", &read_content, path, None);
        let failed_edit =
            make_tool_result_with_path("Edit", "old_string not found in file", path, Some(false));

        let _ = dedup.apply(&read, 1);
        let _ = dedup.apply(&failed_edit, 2);
        let out = dedup.apply(&read, 3);
        assert!(
            is_dedup_marker(result_text(&out).unwrap_or_default()),
            "failed edit must not block dedup"
        );
    }

    #[test]
    fn different_content_is_not_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let msg1 = make_tool_result_message("Read", &large(1000), false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message("Read", &large(1001), false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            !text.contains("Observation deduped"),
            "different content must not be deduped"
        );
    }
}
