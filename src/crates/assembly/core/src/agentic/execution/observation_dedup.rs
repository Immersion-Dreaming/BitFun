//! Cross-round observation deduplication
//!
//! When an agent reads the same large file range multiple times within a dialog
//! turn and the content is unchanged, storing the full observation text on every
//! round wastes context tokens. This module tracks a
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
//!   metadata; that hash is the authoritative key. Persisted blobs **without**
//!   the hash line are deliberately not deduplicated: preview-based hashing
//!   can collide when the truncated portion differs.
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

/// Minimum content length (chars) before a repeated Read is worth replacing.
///
/// A marker that makes the model immediately re-read a small source adds a
/// model/tool round while retaining little context. Keep the threshold high
/// enough that a skipped large Read can amortize the recovery risk.
const MIN_DEDUP_CHARS: usize = 8_000;

/// Only file Reads have a source identity and an unambiguous recovery action.
/// Other tools can yield identical text for semantically different invocations.
const DEDUPLICABLE_TOOLS: &[&str] = &["Read"];

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
/// Also the boundary of the message header: nothing after it may be parsed
/// as a header field.
const PREVIEW_HEADER_PREFIX: &str = "Preview (first ";

/// Key namespace for persisted outputs with an explicit content hash.
const PERSISTED_KEY_PREFIX: &str = "persisted:";

/// Short excerpt used to make dedup markers self-describing.
const MAX_DESCRIPTOR_CHARS: usize = 120;

/// Structured counters for observability. The execution engine drains them at
/// the end of each turn and reports them (see `take_stats`), so dedup impact
/// can be measured without scraping debug logs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DedupStats {
    /// Number of tool results replaced by a marker.
    pub replacements: usize,
    /// Replacements whose key came from the persisted full-content hash.
    pub persisted_hits: usize,
    /// Reads kept fresh because the file was edited after the original read.
    pub skipped_after_edit: usize,
    /// Context-compression resets performed.
    pub resets: usize,
    /// Explicit repeat reads that were recognized as recovery requests.
    pub recovery_requests: usize,
    /// Recovery requests that received the full tool result instead of another
    /// marker. This must equal `recovery_requests` in the current local
    /// recovery policy.
    pub recovery_successes: usize,
    /// Characters returned by explicit recovery reads. This is deliberately
    /// separate from `chars_saved`: recovery can offset the replacement saving.
    pub recovery_chars: usize,
    /// Characters saved (original content minus marker) across replacements.
    pub chars_saved: usize,
}

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
    /// The first duplicate is represented by a marker. The next explicit
    /// execution of the same observation restores full content.
    recovery_pending: bool,
    /// Once recovery has occurred, keep this exact source/version visible
    /// until an edit rebases it. Replacing it with more markers would repeat
    /// the same recovery loop that prompted the explicit re-read.
    full_delivery: bool,
}

/// The source of an observation matters as much as its bytes. In particular,
/// two files containing identical boilerplate must not be treated as a single
/// Read observation: their paths and line ranges carry different semantics to
/// the agent.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ObservationSource {
    Read {
        path: String,
        start_line: Option<u64>,
        end_line: Option<u64>,
    },
    Generic,
}

/// Content hash plus the tool and semantic source that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ObservationKey {
    tool_name: String,
    source: ObservationSource,
    content_key: String,
}

/// Per-turn tracker that detects duplicate tool-result observations and
/// replaces them with a compact reference.
#[derive(Debug, Default)]
pub(crate) struct TurnObservationDeduplicator {
    seen: HashMap<ObservationKey, SeenObservation>,
    /// Logical paths mutated by Edit/Write, mapped to the round of the last
    /// successful mutation. Kept across compression resets: edit rounds stay
    /// valid chronological anchors and keep the freshness guard working.
    edited_files: HashMap<String, usize>,
    stats: DedupStats,
}

impl TurnObservationDeduplicator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Drain the counters accumulated so far.
    pub(crate) fn take_stats(&mut self) -> DedupStats {
        std::mem::take(&mut self.stats)
    }

    /// Pre-scan a round's tool results for successful file mutations so that
    /// dedup decisions later in the same round see the post-edit state even
    /// when Edit/Write and Read results are processed in an arbitrary order
    /// (e.g. parallel tool calls).
    pub(crate) fn pre_scan_round_mutations(&mut self, messages: &[Message], round_index: usize) {
        for message in messages {
            self.record_mutation(message, round_index);
        }
    }

    /// Record a successful file mutation (Edit / Write created|overwritten)
    /// keyed by the normalized logical path.
    fn record_mutation(&mut self, msg: &Message, round_index: usize) {
        let MessageContent::ToolResult {
            tool_name,
            result,
            result_for_assistant,
            is_error,
            ..
        } = &msg.content
        else {
            return;
        };
        if *is_error || !FILE_MUTATION_TOOLS.contains(&tool_name.as_str()) {
            return;
        }
        let success = result
            .get("success")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        if !success {
            return;
        }
        // A Write whose target already holds identical content
        // (`status: already_exists_same_content`) does not change the file and
        // must not invalidate cached reads.
        if tool_name == "Write" {
            let status = result
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("created");
            if !matches!(status, "created" | "overwritten") {
                return;
            }
        }
        let Some(content) = result_for_assistant.as_ref().filter(|s| !s.is_empty()) else {
            return;
        };
        if let Some(path) = file_path_from_result(tool_name, result, content) {
            self.edited_files
                .insert(normalize_path_key(&path), round_index);
        }
    }

    /// Examine a tool-result `Message` that is about to be appended to the
    /// conversation history.
    ///
    /// * If this is the **first** time we see this content, record it and
    ///   return the message unchanged.
    /// * If we have **seen the same observation before**, replace its first
    ///   duplicate with a short reference. A further explicit read restores
    ///   the complete result and keeps that source visible until an edit.
    ///   A file edited after the original observation is always kept fresh.
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

        // Never dedup errors or images.
        if *is_error {
            return msg.clone();
        }
        if image_attachments.as_ref().is_some_and(|a| !a.is_empty()) {
            return msg.clone();
        }

        let Some(content) = result_for_assistant.as_ref().filter(|s| !s.is_empty()) else {
            return msg.clone();
        };

        // Record successful file mutations so later reads of the same path
        // are kept fresh. The execution engine additionally pre-scans the
        // whole round before dedup (see `pre_scan_round_mutations`); keeping
        // the recording here too makes standalone callers behave the same.
        if FILE_MUTATION_TOOLS.contains(&tool_name.as_str()) {
            self.record_mutation(msg, round_index);
            return msg.clone();
        }

        // Only Read has both a precise source identity and an unambiguous
        // recovery action. Other tools stay visible even when their text
        // happens to match a previous result.
        if !DEDUPLICABLE_TOOLS.contains(&tool_name.as_str()) {
            return msg.clone();
        }

        let Some(content_key) = compute_dedup_key(content) else {
            return msg.clone();
        };

        let key = ObservationKey {
            tool_name: tool_name.clone(),
            source: observation_source(tool_name, result, content),
            content_key,
        };

        if let Some(prior) = self.seen.get_mut(&key) {
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
                    let path_key = normalize_path_key(&path);
                    if let Some(edited_round) = self.edited_files.get(&path_key).copied() {
                        if edited_round > prior_round {
                            debug!(
                                "Observation dedup skipped after edit: tool={}, round={}, edited_round={}, descriptor={:?}",
                                tool_name, round_index, edited_round, prior_descriptor
                            );
                            self.stats.skipped_after_edit += 1;
                            *prior = SeenObservation {
                                round_index,
                                tool_name: tool_name.clone(),
                                descriptor: build_descriptor(tool_name, result, content),
                                recovery_pending: false,
                                full_delivery: false,
                            };
                            return msg.clone();
                        }
                    }
                }
            }

            if prior.recovery_pending {
                prior.recovery_pending = false;
                prior.full_delivery = true;
                self.stats.recovery_requests += 1;
                self.stats.recovery_successes += 1;
                self.stats.recovery_chars += content.chars().count();
                debug!(
                    "Observation recovery restored full result: tool={}, round={}, original_round={}, descriptor={:?}",
                    tool_name, round_index, prior_round, prior_descriptor
                );
                return msg.clone();
            }

            if prior.full_delivery {
                return msg.clone();
            }

            debug!(
                "Observation deduped: tool={}, round={}, original_round={}, descriptor={:?}",
                tool_name, round_index, prior_round, prior_descriptor
            );
            let replacement = format!(
                "{DEDUP_MARKER_PREFIX} identical content was already presented at round {} ({}: {}). Omitted to reduce context size. Re-read this exact source once if the full content is needed.]",
                prior_round, prior_tool, prior_descriptor
            );
            self.stats.replacements += 1;
            if key.content_key.starts_with(PERSISTED_KEY_PREFIX) {
                self.stats.persisted_hits += 1;
            }
            self.stats.chars_saved += content
                .chars()
                .count()
                .saturating_sub(replacement.chars().count());
            prior.recovery_pending = true;
            return replace_result_for_assistant(msg, replacement);
        }

        // First occurrence — record and keep the message unchanged.
        self.seen.insert(
            key,
            SeenObservation {
                round_index,
                tool_name: tool_name.clone(),
                descriptor: build_descriptor(tool_name, result, content),
                recovery_pending: false,
                full_delivery: false,
            },
        );
        msg.clone()
    }

    /// Call this whenever context compression fires so that future dedup
    /// references do not point at observations that no longer exist.
    ///
    /// Only the index is cleared; history is not rewritten. Markers already
    /// in history are self-describing (round + descriptor) and carry an
    /// explicit "re-read if the content may have changed" instruction, so
    /// they remain interpretable after compaction.
    pub(crate) fn reset_after_compression(&mut self) {
        self.stats.resets += 1;
        let count = self.seen.len();
        self.seen.clear();
        // Edit rounds stay valid chronological anchors. Recovery state is tied
        // to markers still visible in history, so clearing `seen` also drops
        // all pending/full-delivery state after compression.
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
/// Key resolution order:
/// 1. **Persisted outputs with an explicit hash line** — `Content sha256:` is
///    emitted by `build_persisted_tool_output_message` and covers the full
///    persisted content plus metadata, so the key is exact and independent of
///    the UUID-based reference path.
/// 2. **Legacy persisted outputs (no hash line)** — deliberately **not**
///    deduplicated. A preview-based key would collide whenever two outputs
///    share the same visible preview but differ in the truncated portion,
///    silently hiding a content difference. Such messages never deduplicated
///    before the hash line existed either, so skipping them loses nothing.
/// 3. **Plain outputs** — the full text.
///
/// Returns `None` when the original content is shorter than `MIN_DEDUP_CHARS`
/// (not worth tracking), when the string is empty, or for legacy persisted
/// outputs. Persisted outputs use their header's full-content length rather
/// than the short visible preview.
fn compute_dedup_key(result_for_assistant: &str) -> Option<String> {
    if let Some(hash) = parse_content_sha256(result_for_assistant) {
        let Some(content_chars) = parse_persisted_content_chars(result_for_assistant) else {
            return None;
        };
        if content_chars < MIN_DEDUP_CHARS {
            return None;
        }
        return Some(format!("{PERSISTED_KEY_PREFIX}{hash}"));
    }

    if result_for_assistant.starts_with(PERSISTED_OUTPUT_TAG) {
        return None;
    }

    if result_for_assistant.chars().count() < MIN_DEDUP_CHARS {
        return None;
    }
    Some(format!(
        "plain:{}",
        hex::encode(Sha256::digest(result_for_assistant.as_bytes()))
    ))
}

/// Build the semantic source portion of an observation key. Content equality
/// alone is not sufficient for Reads: identical text from different files or
/// line ranges has different meaning and must remain independently visible.
fn observation_source(
    tool_name: &str,
    result: &serde_json::Value,
    result_for_assistant: &str,
) -> ObservationSource {
    if tool_name != READ_TOOL_NAME {
        return ObservationSource::Generic;
    }

    let Some(path) = file_path_from_result(tool_name, result, result_for_assistant) else {
        return ObservationSource::Generic;
    };
    let start_line = result.get("start_line").and_then(|value| value.as_u64());
    let end_line = start_line.zip(
        result
            .get("lines_read")
            .and_then(|value| value.as_u64()),
    )
    .map(|(start, lines_read)| {
        if lines_read > 0 {
            start + lines_read - 1
        } else {
            start
        }
    });

    ObservationSource::Read {
        path: normalize_path_key(&path),
        start_line,
        end_line,
    }
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

/// Parse the original full-content length from the persisted-output header.
/// The preview is intentionally capped, so its visible length cannot decide
/// whether the underlying observation is large enough to deduplicate.
fn parse_persisted_content_chars(text: &str) -> Option<usize> {
    if !text.starts_with(PERSISTED_OUTPUT_TAG) {
        return None;
    }
    for line in text.lines().take(16) {
        if line.starts_with(PREVIEW_HEADER_PREFIX) {
            break;
        }
        let Some(rest) = line.strip_prefix("Output too large (") else {
            continue;
        };
        let (digits, _) = rest.split_once(" chars). Full output saved to:")?;
        return digits.parse().ok();
    }
    None
}

// ────────────────────────────────────────────────────────────────────────────
// Descriptor / path helpers
// ────────────────────────────────────────────────────────────────────────────

/// Short, self-describing excerpt used in markers: a single clean line of at
/// most `MAX_DESCRIPTOR_CHARS` characters.
fn build_descriptor(tool_name: &str, result: &serde_json::Value, content: &str) -> String {
    let raw = if tool_name == READ_TOOL_NAME {
        // Prefer the structured source description (path + line range). It is
        // available even for persisted results, whose visible text only
        // carries the artifact reference.
        source_read_descriptor(result, content)
    } else {
        first_meaningful_line(content).to_string()
    };
    sanitize_descriptor(&raw)
}

/// `Read lines {start}-{end} from {path} ({total} total lines)` built from
/// the structured result, falling back to the first meaningful line of the
/// visible content.
fn source_read_descriptor(result: &serde_json::Value, content: &str) -> String {
    if let (Some(path), Some(start), Some(lines_read)) = (
        result.get("file_path").and_then(|value| value.as_str()),
        result.get("start_line").and_then(|value| value.as_u64()),
        result.get("lines_read").and_then(|value| value.as_u64()),
    ) {
        let end = if lines_read > 0 {
            start + lines_read - 1
        } else {
            start
        };
        let total = result
            .get("total_lines")
            .and_then(|value| value.as_u64())
            .unwrap_or(end);
        return format!("Read lines {start}-{end} from {path} ({total} total lines)");
    }
    first_meaningful_line(content).to_string()
}

/// First non-blank line of an observation. For persisted outputs the first
/// line is just the tag, so it is skipped (the next meaningful line is the
/// `Output too large (N chars)...` header).
fn first_meaningful_line(content: &str) -> &str {
    let skip = usize::from(content.starts_with(PERSISTED_OUTPUT_TAG));
    content
        .lines()
        .skip(skip)
        .find(|line| !line.trim().is_empty())
        .unwrap_or(content)
}

/// Collapse a descriptor to a single clean line: strip control characters
/// (e.g. ANSI escapes in command output) and truncate to
/// `MAX_DESCRIPTOR_CHARS`.
fn sanitize_descriptor(text: &str) -> String {
    let cleaned: String = text.chars().filter(|ch| !ch.is_control()).collect();
    let trimmed = cleaned.trim();
    let truncated: String = trimmed.chars().take(MAX_DESCRIPTOR_CHARS).collect();
    if truncated.chars().count() < trimmed.chars().count() {
        format!("{truncated}…")
    } else {
        truncated
    }
}

/// Canonical key for a logical file path used by the edit-freshness map and
/// the recovery-read set.
///
/// Paths produced by the workspace resolver are already consistent
/// (workspace-relative logical paths), but defensive normalization keeps a
/// `./`-prefixed or whitespace-padded variant of the same path on the same
/// key. Absolute paths are intentionally not resolved: distinct files must
/// never be merged.
fn normalize_path_key(path: &str) -> String {
    let mut trimmed = path.trim();
    while let Some(stripped) = trimmed.strip_prefix("./") {
        trimmed = stripped;
    }
    trimmed.to_string()
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
            "start_line": 1,
            "lines_read": 50,
            "total_lines": 100,
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

    fn make_write_result(path: &str, status: &str) -> Message {
        let content = format!("Write result for {path}");
        let result = ToolResult {
            tool_id: format!("id-{}", uuid::Uuid::new_v4()),
            tool_name: "Write".to_string(),
            result: json!({
                "file_path": path,
                "success": true,
                "status": status,
                "bytes_written": if status == "created" || status == "overwritten" { 10 } else { 0 },
            }),
            result_for_assistant: Some(content),
            is_error: false,
            duration_ms: None,
            image_attachments: None,
        };
        Message::tool_result(result)
    }

    fn read_message(path: &str, body: &str, round_lines: bool) -> Message {
        read_message_with_range(path, 1, 50, body, round_lines)
    }

    fn read_message_with_range(
        path: &str,
        start: u64,
        end: u64,
        body: &str,
        round_lines: bool,
    ) -> Message {
        let total = 100;
        let header = format!("Read lines {start}-{end} from {path} ({total} total lines)");
        let content = if round_lines {
            format!("{header}\n<file_content>\n{body}\n</file_content>")
        } else {
            body.to_string()
        };
        let result = ToolResult {
            tool_id: format!("id-{}", uuid::Uuid::new_v4()),
            tool_name: "Read".to_string(),
            result: json!({
                "content": content,
                "file_path": path,
                "start_line": start,
                "lines_read": end - start + 1,
                "total_lines": total,
            }),
            result_for_assistant: Some(content),
            is_error: false,
            duration_ms: None,
            image_attachments: None,
        };
        Message::tool_result(result)
    }

    fn large(n: usize) -> String {
        "x".repeat(n)
    }

    fn dedup_candidate() -> String {
        large(MIN_DEDUP_CHARS)
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
        let content = dedup_candidate();
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
    fn marker_then_exact_reread_restores_and_keeps_full_content() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let read = read_message("src/lib.rs", &body, true);

        let _ = dedup.apply(&read, 1);
        let marker = dedup.apply(&read, 2);
        assert!(is_dedup_marker(result_text(&marker).unwrap_or_default()));

        let recovered = dedup.apply(&read, 3);
        assert_eq!(result_text(&recovered), result_text(&read));

        // A recovery upgrades this exact source until a real edit changes
        // the file version. It must not fall back into marker loops.
        let later_read = dedup.apply(&read, 4);
        assert_eq!(result_text(&later_read), result_text(&read));

        let stats = dedup.take_stats();
        assert_eq!(stats.replacements, 1);
        assert_eq!(stats.recovery_requests, 1);
        assert_eq!(stats.recovery_successes, 1);
    }

    #[test]
    fn edit_reenables_dedup_after_a_recovered_read() {
        let mut dedup = TurnObservationDeduplicator::new();
        let read = read_message("src/lib.rs", &dedup_candidate(), true);

        let _ = dedup.apply(&read, 1);
        let _ = dedup.apply(&read, 2);
        let recovered = dedup.apply(&read, 3);
        assert_eq!(result_text(&recovered), result_text(&read));

        let edit = make_tool_result_with_path(
            "Edit",
            "Successfully edited src/lib.rs",
            "src/lib.rs",
            Some(true),
        );
        let _ = dedup.apply(&edit, 4);

        // The first post-edit read is full, then the new file version can be
        // deduplicated again if the agent repeats it.
        let fresh = dedup.apply(&read, 5);
        assert_eq!(result_text(&fresh), result_text(&read));
        let marker = dedup.apply(&read, 6);
        assert!(is_dedup_marker(result_text(&marker).unwrap_or_default()));
    }

    #[test]
    fn persisted_reads_with_identical_payloads_but_different_paths_do_not_dedup() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let persisted = persisted_message(&body, "aabbccdd", Some(&sha256_hex(&body)));
        let first = read_message("src/a.rs", &persisted, false);
        let second = read_message("src/b.rs", &persisted, false);

        let _ = dedup.apply(&first, 1);
        let out = dedup.apply(&second, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "the Read path is part of the observation identity"
        );
    }

    #[test]
    fn identical_content_from_different_read_ranges_does_not_dedup() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let first = read_message_with_range("src/lib.rs", 1, 50, &body, false);
        let second = read_message_with_range("src/lib.rs", 51, 100, &body, false);

        let _ = dedup.apply(&first, 1);
        let out = dedup.apply(&second, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "the Read range is part of the observation identity"
        );
    }

    #[test]
    fn marker_is_self_describing_and_has_no_context_position() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = dedup_candidate();
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
        let content = large(MIN_DEDUP_CHARS - 1);
        let msg = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg, 1);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 2);
        assert_eq!(
            result_text(&out),
            result_text(&msg2),
            "content below threshold must not be replaced"
        );
    }

    #[test]
    fn non_read_tool_results_are_never_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = dedup_candidate();
        let first = make_tool_result_message("Glob", &content, false);
        let second = make_tool_result_message("Glob", &content, false);

        let _ = dedup.apply(&first, 1);
        let out = dedup.apply(&second, 2);
        assert_eq!(result_text(&out), result_text(&second));
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
        let content = dedup_candidate();
        let msg = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg, 1);

        dedup.reset_after_compression();

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
    fn compression_clears_pending_recovery_state() {
        let mut dedup = TurnObservationDeduplicator::new();
        let read = read_message("src/lib.rs", &dedup_candidate(), true);

        let _ = dedup.apply(&read, 1);
        let marker = dedup.apply(&read, 2);
        assert!(is_dedup_marker(result_text(&marker).unwrap_or_default()));

        dedup.reset_after_compression();

        // The old marker can be absent after compression, so the following
        // read becomes a new full observation rather than a stale recovery.
        let out = dedup.apply(&read, 3);
        assert_eq!(result_text(&out), result_text(&read));
        let stats = dedup.take_stats();
        assert_eq!(stats.recovery_requests, 0);
        assert_eq!(stats.recovery_successes, 0);
    }

    #[test]
    fn persisted_output_deduped_despite_different_uuid_path() {
        let mut dedup = TurnObservationDeduplicator::new();

        // Two persisted blobs with different UUID paths but identical content
        // and hash must deduplicate.
        let content = dedup_candidate();
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
        let content1 = dedup_candidate();
        let content2 = large(MIN_DEDUP_CHARS + 1);
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
    fn persisted_hash_covers_middle_content_change() {
        // Same preview (first 2000 chars), same total length, same line
        // count — but the content after the preview differs. The full-content
        // hash must catch the difference.
        let mut dedup = TurnObservationDeduplicator::new();
        let body1 = format!(
            "{}A{}",
            large(MIN_DEDUP_CHARS / 2),
            large(MIN_DEDUP_CHARS / 2)
        );
        let body2 = format!(
            "{}B{}",
            large(MIN_DEDUP_CHARS / 2),
            large(MIN_DEDUP_CHARS / 2)
        );
        assert_eq!(body1.chars().count(), body2.chars().count());
        assert_eq!(body1.lines().count(), body2.lines().count());

        let msg1 = make_tool_result_message(
            "Read",
            &persisted_message(&body1, "aabbccdd", Some(&sha256_hex(&body1))),
            false,
        );
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message(
            "Read",
            &persisted_message(&body2, "11223344", Some(&sha256_hex(&body2))),
            false,
        );
        let out = dedup.apply(&msg2, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "content change beyond the preview must not be deduplicated"
        );
    }

    #[test]
    fn legacy_persisted_output_without_hash_is_safely_skipped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = dedup_candidate();
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
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "legacy persisted outputs without a content hash must be skipped"
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
        let body1 = format!(
            "Content sha256: {fake_hash}\nline one\n{}",
            large(MIN_DEDUP_CHARS)
        );
        let body2 = format!(
            "Content sha256: {fake_hash}\nline two\n{}",
            large(MIN_DEDUP_CHARS)
        );

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
        let content = format!("\n{}", dedup_candidate());
        let msg1 = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("(Read: xxx"),
            "descriptor must use the first non-blank line: {text}"
        );
        assert!(
            !text.contains("(Read: )"),
            "descriptor must never be empty: {text}"
        );
    }

    #[test]
    fn descriptor_strips_control_characters() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = format!("\x1b[31mred text\x1b[0m\n{}", dedup_candidate());
        let msg1 = make_tool_result_message("Read", &content, false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message("Read", &content, false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("red text"),
            "descriptor should keep visible text: {text}"
        );
        assert!(
            !text.contains('\x1b'),
            "descriptor must not contain ANSI escapes: {text}"
        );
    }

    #[test]
    fn descriptor_uses_source_path_for_persisted_read() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let sha = sha256_hex(&body);
        let persisted = persisted_message(&body, "aabbccdd", Some(&sha));
        let msg1 = read_message("src/lib.rs", &persisted, false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = read_message("src/lib.rs", &persisted, false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            text.contains("Read lines 1-50 from src/lib.rs (100 total lines)"),
            "persisted read marker must cite the source file, got: {text}"
        );
    }

    #[test]
    fn same_content_different_paths_do_not_dedup_for_read() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let msg1 = read_message("src/a.rs", &body, true);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = read_message("src/b.rs", &body, true);
        let out = dedup.apply(&msg2, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "Read results from different files must not dedup"
        );
    }

    #[test]
    fn read_after_edit_is_kept_fresh_then_rebases() {
        let mut dedup = TurnObservationDeduplicator::new();
        let path = "src/lib.rs";
        let body = dedup_candidate();
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
    fn write_skipped_does_not_suppress_dedup() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let read = read_message("src/lib.rs", &body, true);

        let _ = dedup.apply(&read, 1);
        // A Write that found identical content must not invalidate the read.
        let skipped = make_write_result("src/lib.rs", "already_exists_same_content");
        let _ = dedup.apply(&skipped, 2);
        let out = dedup.apply(&read, 3);
        assert!(
            is_dedup_marker(result_text(&out).unwrap_or_default()),
            "write-skipped must not suppress dedup"
        );

        // A real Write (created) must invalidate the read.
        let mut dedup = TurnObservationDeduplicator::new();
        let _ = dedup.apply(&read, 1);
        let created = make_write_result("src/lib.rs", "created");
        let _ = dedup.apply(&created, 2);
        let out = dedup.apply(&read, 3);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "write-created must keep the read fresh"
        );
    }

    #[test]
    fn path_normalization_matches_dot_prefix() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let read = read_message("src/lib.rs", &body, true);
        let edit = make_tool_result_with_path(
            "Edit",
            "Successfully edited ./src/lib.rs",
            "./src/lib.rs",
            Some(true),
        );

        let _ = dedup.apply(&read, 1);
        let _ = dedup.apply(&edit, 2);
        let out = dedup.apply(&read, 3);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "normalized paths must match across ./ prefixes"
        );
    }

    #[test]
    fn pre_scan_round_mutations_makes_same_round_edits_visible() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let read = read_message("src/lib.rs", &body, true);
        let edit = make_tool_result_with_path(
            "Edit",
            "Successfully edited src/lib.rs",
            "src/lib.rs",
            Some(true),
        );

        // Round 1: first read is stored.
        let _ = dedup.apply(&read, 1);

        // Round 2: messages arrive as [read, edit]; without the pre-scan the
        // duplicate read would be deduped against round 1 even though the
        // edit happened in the same round.
        let round_messages = vec![read.clone(), edit.clone()];
        dedup.pre_scan_round_mutations(&round_messages, 2);
        let out = dedup.apply(&read, 2);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "same-round edit must keep the read fresh"
        );
    }

    #[test]
    fn failed_edit_does_not_suppress_dedup() {
        let mut dedup = TurnObservationDeduplicator::new();
        let path = "src/lib.rs";
        let body = dedup_candidate();
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
    fn dedup_stats_count_replacement_recovery_and_savings() {
        let mut dedup = TurnObservationDeduplicator::new();
        let body = dedup_candidate();
        let read = read_message("src/lib.rs", &body, true);

        let _ = dedup.apply(&read, 1);
        let out = dedup.apply(&read, 2);
        assert!(is_dedup_marker(result_text(&out).unwrap_or_default()));
        let out = dedup.apply(&read, 3);
        assert!(
            !is_dedup_marker(result_text(&out).unwrap_or_default()),
            "an explicit re-read after a marker must restore full content"
        );

        let edit = make_tool_result_with_path(
            "Edit",
            "Successfully edited src/lib.rs",
            "src/lib.rs",
            Some(true),
        );
        let _ = dedup.apply(&edit, 4);
        let out = dedup.apply(&read, 5);
        assert!(!is_dedup_marker(result_text(&out).unwrap_or_default()));

        dedup.reset_after_compression();
        let stats = dedup.take_stats();
        assert_eq!(stats.replacements, 1);
        assert_eq!(stats.skipped_after_edit, 1);
        assert_eq!(stats.resets, 1);
        assert_eq!(stats.recovery_requests, 1);
        assert_eq!(stats.recovery_successes, 1);
        assert_eq!(stats.recovery_chars, result_text(&read).unwrap_or_default().chars().count());
        assert!(stats.chars_saved > 0);
        assert_eq!(stats.persisted_hits, 0);
    }

    #[test]
    fn persisted_hits_are_counted() {
        let mut dedup = TurnObservationDeduplicator::new();
        let content = dedup_candidate();
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
        let _ = dedup.apply(&msg2, 2);

        let stats = dedup.take_stats();
        assert_eq!(stats.replacements, 1);
        assert_eq!(stats.persisted_hits, 1);
        assert!(stats.chars_saved > 0);
    }

    #[test]
    fn different_content_is_not_deduped() {
        let mut dedup = TurnObservationDeduplicator::new();
        let msg1 = make_tool_result_message("Read", &dedup_candidate(), false);
        let _ = dedup.apply(&msg1, 1);
        let msg2 = make_tool_result_message("Read", &large(MIN_DEDUP_CHARS + 1), false);
        let out = dedup.apply(&msg2, 2);
        let text = result_text(&out).unwrap_or_default();
        assert!(
            !text.contains("Observation deduped"),
            "different content must not be deduped"
        );
    }
}
