# Lifecycle-Aware Eviction — 完整改动方案

## 实施进度

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | `message.rs` +1 variant +1 match arm | ✅ 完成 | Section 一 |
| 2 | 新建 `lifecycle_evict.rs` | ✅ 完成 | Section 二（~350行） |
| 3 | `execution/mod.rs` +1行 mod 声明 | ✅ 完成 | Section 四 |
| 4 | `execution_engine.rs` Hook 1-5 | ✅ 完成 | Section 三 |
| 5 | `SessionConfig` 新增 `lifecycle_eviction_batch_size` | ✅ 完成 | `agent-runtime/src/session.rs` |
| 6 | `cargo build` 无错误 | ✅ 完成 | |
| 7 | `cargo test lifecycle` — 6个新测试通过 | ✅ 完成 | 11/11 通过 |

> **分支**：`feat/lifecycle-eviction`（从 `main` 创建）

---

> **修订日志（v2）**：修正了审核发现的问题：
> 1. `PERSISTED_PREVIEW_MARKER` → 改用 `find("---Preview")` 部分匹配（不依赖动态 N）
> 2. 测试中 `assistant_with_tool_calls` → `assistant_with_tools(String::new(), calls)`
> 3. 新增 **Hook 5**：`emergency_truncate_messages` 后重置 lifecycle manager
> 4. `should_drop_during_compaction` 已在 Section 一处理（原方案已有，非遗漏）
> 5. estimator prompt 新增 Rule 6：说明 deduped 内容的处理方式
> 6. `Message::internal_reminder` 创建的是 **user-role**，不是 system-role，无API格式问题
> 7. `bitfun_agent_tools` 已是 assembly/core 的依赖（Cargo.toml:89），无需额外引入
> 8. `lifecycle_task_context` 提取加了 user-role fallback
> 9. `batch_size` 改为从 `session.config.lifecycle_eviction_batch_size` 读取（默认3）

## 一、message.rs（2处，共3行新增）

文件：`src/crates/assembly/core/src/agentic/core/message.rs`

**改动1**：L110，`FinalizeCacheAnchor` 后追加1行：

```rust
    FinalizeCacheAnchor,
    LifecycleEvictionSummary,   // 新增
```

**改动2**：L140，`matches!` 末尾追加1个arm：

```rust
            | Self::FinalizeCacheAnchor
            | Self::LifecycleEvictionSummary  // 新增
```

理由：全量压缩后历史已被模型摘要替换，占位符无需保留，随压缩自动清除。

---

## 二、lifecycle_evict.rs（新文件）

路径：`src/crates/assembly/core/src/agentic/execution/lifecycle_evict.rs`

### 2.1 常量

```rust
pub(crate) const LIFECYCLE_BATCH_SIZE: usize = 3;

const ESTIMATOR_MODEL: &str = "claude-haiku-4-5-20251001";

/// 普通结果的 preview 截取长度
const RESULT_PREVIEW_CHARS: usize = 600;

/// persisted 结果的 preview 起始标记（N 是动态值，用部分匹配同 observation_dedup.rs:160）
/// 不要改为精确字符串，preview_chars 不一定是 2000。
const PERSISTED_PREVIEW_SEARCH: &str = "---Preview";

/// Vi 中任务上下文的截取长度
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
```


### 2.2 数据结构

```rust
/// 状态顺序 Active < Completed < Evictable
/// 显式判别值保证 derive(PartialOrd, Ord) 方向正确
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SegmentState {
    Active    = 0,
    Completed = 1,
    Evictable = 2,
}

/// 预构建视图：record_segment 时一次性计算，供 estimator Vi 使用
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct SegmentView {
    round_index: usize,
    tool_calls:  Vec<ToolCallView>,
    results:     Vec<ResultView>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ToolCallView {
    tool_name: String,
    /// 完整 JSON，不截断（args 比 result 更重要，用于检测跨 round 依赖）
    args_json: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ResultView {
    tool_name:    String,
    /// persisted → 提取 ---Preview--- 后的内容；普通 → 前 600 chars
    preview:      String,
    is_error:     bool,
    is_persisted: bool,
}

struct ContextSegment {
    round_index:       usize,
    /// messages[assistant_msg_idx] 是该 round 的 Assistant 消息
    assistant_msg_idx: usize,
    /// exclusive：tool results 占 [assistant_msg_idx+1, end_idx)
    end_idx:           usize,
    state:             SegmentState,
    /// 预计算视图，供 estimator 使用
    view:              SegmentView,
}
```

### 2.3 LifecycleEvictionManager

```rust
pub(crate) struct LifecycleEvictionManager {
    /// 论文中的 registry R：跨批次保留，apply_state_updates 做增量更新
    segments:   Vec<ContextSegment>,
    batch_size: usize,
}

impl LifecycleEvictionManager {
    pub(crate) fn new(batch_size: usize) -> Self {
        Self { segments: Vec::new(), batch_size }
    }
}
```

### 2.4 `record_segment`

```rust
/// 每 round 结束后调用（所有 tool results push 完毕之后）。
/// assistant_msg_idx：调用方在 push assistant message 那一行立即捕获的 index。
pub(crate) fn record_segment(
    &mut self,
    messages: &[Message],
    round_index: usize,
    assistant_msg_idx: usize,
) {
    let end_idx = messages.len();

    // 纯文字轮（无 tool calls）不记录，没有工具调用就没有可驱逐内容
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
```

模块级辅助函数 `build_segment_view`：

```rust
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
                    tool_name:    tool_name.clone(),
                    preview:      extract_result_preview(text, is_persisted),
                    is_error:     *is_error,
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
        // 用部分匹配（同 observation_dedup.rs:160），不依赖动态 N 的精确值
        if let Some(pos) = content.find(PERSISTED_PREVIEW_SEARCH) {
            // 跳过 "---Preview (first N chars)---\n" 整行，从下一行开始
            if let Some(newline) = content[pos..].find('\n') {
                return content[pos + newline + 1..].to_string();
            }
        }
    }
    truncate_chars(content, RESULT_PREVIEW_CHARS)
}

/// pub(crate) 供 execution_engine.rs Hook 1 中提取 task_context 使用
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    let mut chars = s.chars();
    let out: String = chars.by_ref().take(max).collect();
    if chars.next().is_some() { format!("{out}…") } else { out }
}
```

### 2.5 `should_run_estimator`

```rust
/// round_index：刚完成的轮次（0-indexed）。
/// batch_size=3 时在 round 2, 5, 8… 完成后触发（即每3轮一次）。
pub(crate) fn should_run_estimator(&self, round_index: usize) -> bool {
    (round_index + 1) % self.batch_size == 0 && !self.segments.is_empty()
}
```

### 2.6 `run_batch_eviction`

```rust
/// 返回实际驱逐的 segment 数量。
/// task_context：当前 turn 第一条 ActualUserInput 消息前 500 chars。
/// tool_results_dir：recovery 文件存放目录；None 时跳过 recovery 文件保存（best-effort）。
pub(crate) async fn run_batch_eviction(
    &mut self,
    messages: &mut Vec<Message>,
    round_index: usize,
    task_context: &str,
    tool_results_dir: Option<&std::path::Path>,
) -> BitFunResult<usize> {
    // 步骤1：构建 Vi（压缩历史视图，论文 Section 3.3）
    let vi_json = self.build_vi(round_index, task_context)?;

    // 步骤2：调用 Haiku estimator，失败时安全降级（不驱逐）
    let state_updates = match self.call_estimator(vi_json).await {
        Ok(updates) => updates,
        Err(e) => {
            warn!("lifecycle eviction: estimator failed, skipping: {}", e);
            return Ok(0);
        }
    };

    // 步骤3：应用状态更新（论文 R_i ← R_{i-1} ⊕ ΔR_i）
    self.apply_state_updates(state_updates);

    // 步骤4：物理驱逐
    let evicted = self.execute_eviction(messages, tool_results_dir).await?;

    if evicted > 0 {
        info!(
            "lifecycle eviction: round={}, evicted={} segments, messages_len={}",
            round_index, evicted, messages.len()
        );
    }
    Ok(evicted)
}
```

**步骤1 — build_vi：**

```rust
fn build_vi(&self, round_index: usize, task_context: &str) -> BitFunResult<String> {
    #[derive(serde::Serialize)]
    struct EstimatorInput<'a> {
        current_task:  &'a str,
        current_round: usize,
        /// 所有未驱逐段：让 estimator 看到完整历史来判断跨 round 依赖
        segments:      Vec<&'a SegmentView>,
    }
    let input = EstimatorInput {
        current_task:  task_context,
        current_round: round_index,
        segments: self.segments
            .iter()
            .filter(|s| s.state != SegmentState::Evictable)
            .map(|s| &s.view)
            .collect(),
    };
    serde_json::to_string(&input).map_err(|e| {
        BitFunError::tool(format!("lifecycle eviction: Vi serialization: {e}"))
    })
}
```

**步骤2 — call_estimator：**

与 [execution_engine.rs:1777](src/crates/assembly/core/src/agentic/execution/execution_engine.rs#L1777) 相同的 factory 模式；
调用与 [execution_engine.rs:1646](src/crates/assembly/core/src/agentic/execution/execution_engine.rs#L1646) 相同的 `send_message_with_trace`（`tools=None`，`trace_config=None`）。

```rust
async fn call_estimator(
    &self,
    vi_json: String,
) -> BitFunResult<std::collections::HashMap<usize, SegmentState>> {
    use crate::infrastructure::ai::get_global_ai_client_factory;

    let factory = get_global_ai_client_factory().await.map_err(|e| {
        BitFunError::AIClient(format!("lifecycle estimator factory: {e}"))
    })?;
    let client = factory.get_client_resolved(ESTIMATOR_MODEL).await.map_err(|e| {
        BitFunError::AIClient(format!("lifecycle estimator client: {e}"))
    })?;

    let req = vec![
        AIMessage::system(ESTIMATOR_SYSTEM_PROMPT.to_string()),
        AIMessage::user(vi_json),
    ];
    // 单轮，无工具，与 request_compression_summary_with_retry 同模式
    let response = client
        .send_message_with_trace(req, None, None)
        .await
        .map_err(|e| BitFunError::AIClient(format!("lifecycle estimator request: {e}")))?;

    parse_estimator_response(&response.text)
}
```

**parse_estimator_response + extract_json_object：**

```rust
fn parse_estimator_response(
    text: &str,
) -> BitFunResult<std::collections::HashMap<usize, SegmentState>> {
    #[derive(serde::Deserialize)]
    struct EstimatorOutput {
        state_updates: std::collections::HashMap<String, String>,
    }

    // 宽松解析：允许 estimator 在 JSON 前后有多余文字
    let json_str = extract_json_object(text).ok_or_else(|| {
        BitFunError::tool(format!("lifecycle estimator: no JSON object in response: {text}"))
    })?;

    let output: EstimatorOutput = serde_json::from_str(json_str).map_err(|e| {
        BitFunError::tool(format!("lifecycle estimator: invalid JSON ({e}): {json_str}"))
    })?;

    let mut result = std::collections::HashMap::new();
    for (k, v) in output.state_updates {
        let round: usize = match k.parse() {
            Ok(n)  => n,
            Err(_) => { warn!("lifecycle estimator: bad round key '{}', skipping", k); continue; }
        };
        let state = match v.as_str() {
            "active"    => SegmentState::Active,
            "completed" => SegmentState::Completed,
            "evictable" => SegmentState::Evictable,
            other => { warn!("lifecycle estimator: unknown state '{}', skipping", other); continue; }
        };
        result.insert(round, state);
    }
    Ok(result)
}

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end   = text.rfind('}')?;
    if end >= start { Some(&text[start..=end]) } else { None }
}
```

**步骤3 — apply_state_updates（论文 ΔR 增量合并）：**

```rust
fn apply_state_updates(
    &mut self,
    updates: std::collections::HashMap<usize, SegmentState>,
) {
    for seg in self.segments.iter_mut() {
        let Some(new_state) = updates.get(&seg.round_index) else { continue; };

        // 代码层双重保护：错误结果不允许变为 evictable（estimator system prompt 已有约束）
        if *new_state == SegmentState::Evictable
            && seg.view.results.iter().any(|r| r.is_error)
        {
            warn!(
                "lifecycle: estimator marked error segment {} as evictable, ignoring",
                seg.round_index
            );
            continue;
        }
        // 状态只升不降（Active → Completed → Evictable），防止 estimator 回退
        if *new_state > seg.state {
            seg.state = new_state.clone();
        }
    }
}
```

**步骤4 — execute_eviction（完整逻辑）：**

关键设计：先收集所有驱逐目标的**原始索引**，按 asst_idx **降序**执行 drain+insert（后面的先处理，不影响前面的索引），最后用原始索引的**升序**统一 reindex。

```rust
async fn execute_eviction(
    &mut self,
    messages: &mut Vec<Message>,
    tool_results_dir: Option<&std::path::Path>,
) -> BitFunResult<usize> {
    struct EvictTarget {
        round_index: usize,
        asst_idx:    usize,  // 原始索引（操作前捕获）
        end_idx:     usize,  // 原始索引，exclusive
    }

    // 操作前收集原始范围，后续不再依赖 segment.assistant_msg_idx
    let mut targets: Vec<EvictTarget> = self.segments
        .iter()
        .filter(|s| s.state == SegmentState::Evictable)
        .map(|s| EvictTarget {
            round_index: s.round_index,
            asst_idx:    s.assistant_msg_idx,
            end_idx:     s.end_idx,
        })
        .collect();

    if targets.is_empty() {
        return Ok(0);
    }

    // 从后往前处理：高 asst_idx 先处理，保证前面的原始索引不受影响
    targets.sort_by(|a, b| b.asst_idx.cmp(&a.asst_idx));

    for target in &targets {
        // 1. 保存 recovery 文件（best-effort，失败不阻断驱逐）
        let recovery_path = if let Some(dir) = tool_results_dir {
            match save_recovery_file(
                target.round_index, target.asst_idx, target.end_idx, messages, dir,
            ).await {
                Ok(p)  => Some(p),
                Err(e) => {
                    warn!("lifecycle: recovery file failed for round {}: {}", target.round_index, e);
                    None
                }
            }
        } else {
            None
        };

        // 2. 构建 summary placeholder
        let tool_names: String = self.segments
            .iter()
            .find(|s| s.round_index == target.round_index)
            .map(|s| {
                s.view.tool_calls.iter()
                    .map(|tc| tc.tool_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        let recovery_hint = recovery_path
            .as_ref()
            .map(|p| format!("\nFull content saved to: {}\nUse the Read tool to recover if needed.", p.display()))
            .unwrap_or_default();

        let summary_text = format!(
            "[LIFECYCLE_EVICTION_SUMMARY: round={}]\nSub-task completed and evicted. Tools: [{}].{}",
            target.round_index, tool_names, recovery_hint,
        );
        let summary_msg = Message::internal_reminder(
            InternalReminderKind::LifecycleEvictionSummary,
            summary_text,
        );

        // 3. drain [asst_idx, end_idx) 并插入 1 条 summary（net 变化 = 1 - len）
        messages.drain(target.asst_idx..target.end_idx);
        messages.insert(target.asst_idx, summary_msg);
    }

    // 4. Reindex 剩余 segments
    // 使用原始索引升序，每个 target 的净变化 = 1 - (end_idx - asst_idx)
    let mut targets_asc = targets;
    targets_asc.sort_by_key(|t| t.asst_idx);

    for seg in self.segments.iter_mut() {
        if seg.state == SegmentState::Evictable { continue; }

        // 只有完全位于 seg 之前的 target（t.end_idx <= seg.assistant_msg_idx）才产生偏移
        let shift: isize = targets_asc.iter()
            .filter(|t| t.end_idx <= seg.assistant_msg_idx)
            .map(|t| 1isize - (t.end_idx as isize - t.asst_idx as isize))
            .sum();

        seg.assistant_msg_idx = (seg.assistant_msg_idx as isize + shift) as usize;
        seg.end_idx           = (seg.end_idx           as isize + shift) as usize;
    }

    // 5. 移除已驱逐条目
    self.segments.retain(|s| s.state != SegmentState::Evictable);

    Ok(targets_asc.len())
}
```

Reindex 公式验证（示例）：
- Segment A 原始 [2,4)，Segment B 原始 [4,7)，Segment C 原始 [7,9)
- A 和 B 标记为 evictable，C 存活
- targets_asc = [A(asst=2,end=4), B(asst=4,end=7)]
- C 计算 shift：
  - A: end_idx=4 ≤ 7 → shift += 1-(4-2) = -1
  - B: end_idx=7 ≤ 7 → shift += 1-(7-4) = -2
  - total shift = -3
- C.new_asst = 7+(-3) = 4，C.new_end = 9+(-3) = 6 ✓

**save_recovery_file（模块级辅助函数）：**

```rust
async fn save_recovery_file(
    round_index: usize,
    asst_idx: usize,
    end_idx: usize,
    messages: &[Message],
    dir: &std::path::Path,
) -> BitFunResult<std::path::PathBuf> {
    let tool_calls: Vec<serde_json::Value> = match &messages[asst_idx].content {
        MessageContent::Mixed { tool_calls, .. } => tool_calls.iter().map(|tc| {
            serde_json::json!({
                "tool_id":   tc.tool_id,
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
            })
        }).collect(),
        _ => vec![],
    };

    let tool_results: Vec<serde_json::Value> = messages[asst_idx + 1..end_idx]
        .iter()
        .filter_map(|m| match &m.content {
            MessageContent::ToolResult {
                tool_id, tool_name, result_for_assistant, is_error, ..
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

    tokio::fs::create_dir_all(dir).await.map_err(|e| {
        BitFunError::io(format!("lifecycle: create dir: {e}"))
    })?;

    let path = dir.join(format!(
        "evicted_round_{}_{}.json",
        round_index,
        uuid::Uuid::new_v4()
    ));
    tokio::fs::write(&path, serde_json::to_string_pretty(&record)?).await.map_err(|e| {
        BitFunError::io(format!("lifecycle: write recovery file: {e}"))
    })?;
    Ok(path)
}
```

### 2.7 `reset_after_compression`

```rust
/// 全量压缩后调用：所有 message index 已失效，清空 registry。
pub(crate) fn reset_after_compression(&mut self) {
    self.segments.clear();
}
```

---

## 三、execution_engine.rs（4个 Hook）

文件：`src/crates/assembly/core/src/agentic/execution/execution_engine.rs`

### Hook 1 — 初始化

位置：紧接 **L2754** `let mut observation_deduplicator = TurnObservationDeduplicator::new();` 之后。

```rust
// L2754（现有）
let mut observation_deduplicator = TurnObservationDeduplicator::new();

// ── Hook 1：lifecycle eviction 初始化 ───────────────────────────────────────
// batch_size：优先从 session.config 读（留给后续调优），默认 LIFECYCLE_BATCH_SIZE=3
let lifecycle_batch_size = session
    .config
    .lifecycle_eviction_batch_size
    .unwrap_or(lifecycle_evict::LIFECYCLE_BATCH_SIZE);
let mut lifecycle_eviction_manager =
    LifecycleEvictionManager::new(lifecycle_batch_size);

// 提取任务上下文一次，后续 Hook 3 复用（第一条 ActualUserInput 消息前 500 chars）
// messages 在 round loop 前已包含本 turn 的用户消息；fallback 到最后一条 user 消息，
// 防止极端情况下 ActualUserInput 不存在导致 task_context 为空串。
let lifecycle_task_context: String = messages
    .iter()
    .find(|m| matches!(
        m.metadata.semantic_kind,
        Some(MessageSemanticKind::ActualUserInput)
    ))
    .or_else(|| {
        messages.iter().rev().find(|m| m.role == MessageRole::User)
    })
    .map(|m| match &m.content {
        MessageContent::Text(t)                 => lifecycle_evict::truncate_chars(t, 500),
        MessageContent::Multimodal { text, .. } => lifecycle_evict::truncate_chars(text, 500),
        _ => String::new(),
    })
    .unwrap_or_default();

// recovery 文件目录：与 tool_result_storage.rs 中 ToolUseContext 的约定相同
let lifecycle_tool_results_dir: Option<std::path::PathBuf> = context
    .workspace
    .as_ref()
    .and_then(|ws| ws.current_workspace_session_tool_results_dir(&context.session_id).ok());
// ────────────────────────────────────────────────────────────────────────────
```

> **注意**：`current_workspace_session_tool_results_dir` 在 `ToolUseContext` 上已确认存在（tool_result_storage.rs:433），实现时需确认 `WorkspaceBinding` 上是否有同名方法；如没有，用同路径约定手动构造 PathBuf。

### Hook 2 — 记录 segment（两处）

**Hook 2a**：在 **L3207** `messages.push(round_result.assistant_message.clone())` **之后立即**（同行代码块内）：

```rust
// L3207（现有）
messages.push(round_result.assistant_message.clone());
let lifecycle_assistant_msg_idx = messages.len() - 1;  // Hook 2a：push 后立即捕获 index
```

**Hook 2b**：在 **L3236** tool results 循环结束后（`debug!("Updated round messages…")` 之前）：

```rust
// L3236（tool results 循环末尾，现有）
// ...（所有 tool results 已 push）...

// ── Hook 2b：记录本 round 为 ContextSegment ──────────────────────────────────
lifecycle_eviction_manager.record_segment(
    &messages,
    round_index,
    lifecycle_assistant_msg_idx,
);
// ────────────────────────────────────────────────────────────────────────────
```

### Hook 3 — 批次驱逐

位置：Hook 2b 之后，**L3244** `total_tools += round_result.tool_calls.len();` 之前。

```rust
// ── Hook 3：Lifecycle batch eviction ────────────────────────────────────────
if lifecycle_eviction_manager.should_run_estimator(round_index) {
    let evicted = lifecycle_eviction_manager
        .run_batch_eviction(
            &mut messages,
            round_index,
            &lifecycle_task_context,
            lifecycle_tool_results_dir.as_deref(),
        )
        .await
        .unwrap_or_else(|e| {
            warn!("lifecycle eviction error: {}", e);
            0
        });

    if evicted > 0 {
        // 同步 session manager（与全量压缩后保持一致）
        self.session_manager
            .replace_context_messages(&context.session_id, messages.clone())
            .await;
        self.session_manager
            .invalidate_prompt_cache(
                &context.session_id,
                crate::agentic::session::PromptCacheScope::All,
                "lifecycle_eviction_applied",
            )
            .await;
        // message index 已变，重置 deduplicator
        observation_deduplicator.reset_after_compression();
        debug!(
            "Lifecycle eviction sync complete: session={}, evicted={}, messages_len={}",
            context.session_id, evicted, messages.len()
        );
    }
}
// ────────────────────────────────────────────────────────────────────────────
```

### Hook 4 — 全量压缩后重置

位置：**L3011** `observation_deduplicator.reset_after_compression();` 之后立即：

```rust
// L3011（现有）
observation_deduplicator.reset_after_compression();
// Hook 4（新增，紧跟其后）
lifecycle_eviction_manager.reset_after_compression();
```

### Hook 5 — Emergency truncation 后重置（原方案遗漏）

位置：**L3064** `prune_token_anchors_to_messages` 调用之后：

```rust
// L3057（现有）
messages = Self::emergency_truncate_messages(
    messages,
    context_window,
    tool_definitions.as_deref(),
    send_prepended_reminder_tokens,
);
self.session_manager
    .prune_token_anchors_to_messages(&context.session_id, &messages)
    .await;
// Hook 5（新增，紧跟其后）
// emergency_truncate_messages 物理删除了旧消息，
// segment 里的 assistant_msg_idx/end_idx 已失效，必须重置。
lifecycle_eviction_manager.reset_after_compression();
```

> **背景**：`emergency_truncate_messages`（L2: 压缩层）是独立于全量压缩（L1）的另一条路径，发生在 L3057。
> 如果不重置，下次 `execute_eviction` 执行 `messages.drain(target.asst_idx..target.end_idx)` 
> 可能 panic（index out of range）或静默删错消息。

---

## 四、mod.rs（+1行）

文件：`src/crates/assembly/core/src/agentic/execution/mod.rs`

在 `pub(crate) mod observation_dedup;` 之后追加：

```rust
pub(crate) mod lifecycle_evict;
```

---

## 五、单元测试（6个，lifecycle_evict.rs 底部）

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ── 辅助：构造 Message vec（system + user + assistant + N tool results）──
    fn make_messages(tool_count: usize) -> Vec<Message> {
        let mut msgs = vec![
            Message::system("sys".to_string()),
            Message::user("task".to_string()),
        ];
        let tool_calls: Vec<_> = (0..tool_count)
            .map(|i| ToolCall {
                tool_id:   format!("id_{i}"),
                tool_name: format!("Tool{i}"),
                arguments: serde_json::json!({"file": format!("f{i}.rs")}),
            })
            .collect();
        msgs.push(Message::assistant_with_tools(String::new(), tool_calls));
        for i in 0..tool_count {
            msgs.push(Message::tool_result(ToolResult {
                tool_id:             format!("id_{i}"),
                tool_name:           format!("Tool{i}"),
                result:              serde_json::json!({}),
                result_for_assistant: Some(format!("result {i}")),
                is_error:            false,
                duration_ms:         None,
                image_attachments:   None,
            }));
        }
        msgs
    }

    // ── Test 1：record_segment 正确捕获 assistant_msg_idx 和 end_idx ──────────
    #[test]
    fn record_captures_correct_indices() {
        let msgs = make_messages(2); // 4 messages: sys, user, asst, r0, r1
        let asst_idx = 2;
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, asst_idx);

        assert_eq!(mgr.segments.len(), 1);
        let seg = &mgr.segments[0];
        assert_eq!(seg.assistant_msg_idx, 2);
        assert_eq!(seg.end_idx, 5);     // msgs.len() = 5
        assert_eq!(seg.state, SegmentState::Active);
        assert_eq!(seg.view.tool_calls.len(), 2);
        assert_eq!(seg.view.results.len(), 2);
    }

    // ── Test 2：纯文字轮不被记录 ────────────────────────────────────────────
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

    // ── Test 3：should_run_estimator 在正确边界触发（batch=3）────────────────
    #[test]
    fn should_run_at_batch_boundaries() {
        let msgs = make_messages(1);
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);

        assert!(!mgr.should_run_estimator(0)); // round 0：(0+1)%3=1 ≠ 0
        assert!(!mgr.should_run_estimator(1)); // round 1：(1+1)%3=2 ≠ 0
        assert!( mgr.should_run_estimator(2)); // round 2：(2+1)%3=0 → 触发
        assert!(!mgr.should_run_estimator(3)); // round 3：(3+1)%3=1 ≠ 0
        assert!( mgr.should_run_estimator(5)); // round 5：(5+1)%3=0 → 触发
    }

    // ── Test 4：状态只升不降 ─────────────────────────────────────────────────
    #[test]
    fn state_only_upgrades() {
        let msgs = make_messages(1);
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);
        mgr.segments[0].state = SegmentState::Completed;

        // 尝试降级到 Active → 不应生效
        mgr.apply_state_updates([(0, SegmentState::Active)].into());
        assert_eq!(mgr.segments[0].state, SegmentState::Completed);

        // 升级到 Evictable → 生效
        mgr.apply_state_updates([(0, SegmentState::Evictable)].into());
        assert_eq!(mgr.segments[0].state, SegmentState::Evictable);
    }

    // ── Test 5：error result 的 segment 不允许被 evict ───────────────────────
    #[test]
    fn error_segment_protected_from_eviction() {
        let msgs = make_messages(1);
        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.record_segment(&msgs, 0, 2);
        // 标记 result 为 error
        mgr.segments[0].view.results[0].is_error = true;

        // 尝试标记为 evictable → 应被保护
        mgr.apply_state_updates([(0, SegmentState::Evictable)].into());
        assert_eq!(mgr.segments[0].state, SegmentState::Active); // 保持不变
    }

    // ── Test 6：execute_eviction 驱逐后 reindex 正确 ─────────────────────────
    #[tokio::test]
    async fn reindex_correct_after_eviction() {
        // 手动构造3个 segment，分别在 round 0, 1, 2
        // 布局：[sys(0), user(1), asst0(2), r0(3), r1(4), asst1(5), r2(6), r3(7), r4(8), asst2(9), r5(10)]
        // Segment 0: asst_idx=2, end_idx=5 (2 results)
        // Segment 1: asst_idx=5, end_idx=9 (3 results)
        // Segment 2: asst_idx=9, end_idx=11 (1 result)
        let mut msgs = vec![
            Message::system("sys".into()),
            Message::user("task".into()),
        ];
        for round in 0..3usize {
            let count = if round == 1 { 3 } else { if round == 2 { 1 } else { 2 } };
            let calls: Vec<_> = (0..count).map(|i| ToolCall {
                tool_id:   format!("id_r{round}_{i}"),
                tool_name: format!("Tool"),
                arguments: serde_json::json!({}),
            }).collect();
            msgs.push(Message::assistant_with_tools(String::new(), calls));
            for i in 0..count {
                msgs.push(Message::tool_result(ToolResult {
                    tool_id:             format!("id_r{round}_{i}"),
                    tool_name:           "Tool".into(),
                    result:              serde_json::json!({}),
                    result_for_assistant: Some(format!("result")),
                    is_error:            false,
                    duration_ms:         None,
                    image_attachments:   None,
                }));
            }
        }

        let mut mgr = LifecycleEvictionManager::new(3);
        mgr.segments = vec![
            ContextSegment { round_index: 0, assistant_msg_idx: 2,  end_idx: 5,  state: SegmentState::Evictable, view: SegmentView { round_index: 0, tool_calls: vec![], results: vec![] } },
            ContextSegment { round_index: 1, assistant_msg_idx: 5,  end_idx: 9,  state: SegmentState::Evictable, view: SegmentView { round_index: 1, tool_calls: vec![], results: vec![] } },
            ContextSegment { round_index: 2, assistant_msg_idx: 9,  end_idx: 11, state: SegmentState::Active,    view: SegmentView { round_index: 2, tool_calls: vec![], results: vec![] } },
        ];

        mgr.execute_eviction(&mut msgs, None).await.unwrap();

        // Round 2 应该剩余，indices 要更新
        assert_eq!(mgr.segments.len(), 1);
        let seg = &mgr.segments[0];
        assert_eq!(seg.round_index, 2);
        // 原始 asst=9：
        //   Round0 end=5 ≤ 9 → shift += 1-(5-2) = -2
        //   Round1 end=9 ≤ 9 → shift += 1-(9-5) = -3
        //   total = -5 → new asst = 9-5 = 4
        assert_eq!(seg.assistant_msg_idx, 4);
        // 原始 end=11 → 11-5 = 6
        assert_eq!(seg.end_idx, 6);
        // messages 长度：原始 11 - 2 - 3 + 1 + 1 = 8
        assert_eq!(msgs.len(), 8);
    }
}
```

---

## 六、边界情况处理

**1. Estimator 响应解析失败**
`parse_estimator_response` 返回 `Err` → `call_estimator` 返回 `Err` → `run_batch_eviction` warn + return Ok(0)，本批次不驱逐，下一批次正常继续。

**2. Estimator 将所有 segment 标记为 evictable**
合法情况。`execute_eviction` 驱逐所有段，`segments` 清空。后续 `should_run_estimator` 因 `segments.is_empty()` 返回 false，直到下一个 `record_segment` 调用。

**3. Recovery 文件写入失败**
`save_recovery_file` 返回 `Err` → warn + `recovery_path = None` → summary placeholder 不含文件路径 → 驱逐继续。Agent 无法恢复该内容，但不崩溃。

**4. 全量压缩与 lifecycle eviction 的顺序关系**
- 全量压缩发生在**每 round 开始**（L2925 `should_compress` 检查），lifecycle eviction 发生在**每 round 结束**（Hook 3）。
- 二者不可能在同一 round 内交叉执行（压缩 → AI 调用 → 结果 → eviction）。
- 若当前 round 发生了全量压缩（Hook 4 被触发），`segments` 已清空，本 round 结束时 Hook 3 的 `should_run_estimator` 即使条件满足也因 `segments.is_empty()` 返回 false。

**5. should_run_estimator 边界：round_index = 0**
(0+1) % 3 = 1 ≠ 0 → false，第一轮不触发。最早在 round_index=2 时触发，此时至少有1个 segment（且在全量压缩前一直累积）。

**6. 驱逐后 KV-cache 一次性失效**
`invalidate_prompt_cache` 在 Hook 3 中主动调用，使下一次 AI 请求产生一次 cache miss。这符合论文设计：`freed_tokens × (B-1) > miss_penalty_once`。对于 B=3，驱逐节省的 token 在后续 2 轮中摊销。

---

## 七、改动文件汇总

| 文件 | 类型 | 改动 |
|------|------|------|
| `src/.../core/message.rs` | 修改 | +1 variant，+1 match arm |
| `src/.../execution/lifecycle_evict.rs` | 新建 | 完整实现（~350行） |
| `src/.../execution/execution_engine.rs` | 修改 | 5个 hook（~45行，新增 emergency truncation 重置） |
| `src/.../execution/mod.rs` | 修改 | +1行 pub(crate) mod 声明 |
