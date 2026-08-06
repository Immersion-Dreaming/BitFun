# Observation Dedup 修复方案与执行记录

> 分支：`feat/observation-dedup`　日期：2026-08-06
> 本文档记录跨 round 观察去重功能的缺陷分析、修复方案、代码改动与验证结果，便于后续复盘。

---

## 1. 背景

`feat/observation-dedup` 引入的 `TurnObservationDeduplicator`
（`src/crates/assembly/core/src/agentic/execution/observation_dedup.rs`）用于在单次
`execute_turn` 内对内容完全相同的工具结果（Read / Bash 等）做去重：首次出现时完整存储，
后续重复出现时替换为一行 `[Observation deduped: ...]` marker，节省上下文 token。

通过代码审查 + 真实 benchmark trace（`observation-dedup-Deduplicate—rerun`，21 个任务、
每个任务多个 agent 运行目录）分析，发现以下问题。

## 2. 问题根因分析

### P0-1（死代码）：持久化输出永远不会命中去重

- 生产格式由 `build_persisted_tool_output_message`
  （`src/crates/execution/tool-contracts/src/tool_result_storage.rs`）生成，实际为：

  ```
  <persisted-output>
  Output too large (N chars). Full output saved to: .../tool-results/<uuid>.txt
  Line count: N
  Content sha256: ...（本次新增）
  Preview (first 2000 chars):
  <preview>
  </persisted-output>
  ```

- `compute_dedup_key` 却按 `"[PERSISTED_OUTPUT"` 前缀判断、并查找 `"---Preview"` 标记：
  - 前缀判断与生产 tag `<persisted-output>` 不一致；
  - `find("---Preview")` 恒为 `None`（生产格式是 `Preview (first 2000 chars):`）。
  - 结果：持久化输出永远走"整串 hash"回退路径，hash 包含 UUID 路径 → **永不命中去重**。
- 内嵌测试使用了假格式（`[PERSISTED_OUTPUT: ...] ---Preview (first 2000 chars)---`），
  测试通过但生产失效，属于"测试与生产格式脱节"的典型问题。

### P0-2（位置偏移）：marker 引用的 context position 全部漂移

- 记录索引用的是执行期 `messages.len()`（不含发送前注入的提醒），
  而实际发请求时 `build_ai_messages_for_send` 会在 system 之后临时注入 5 条提醒
  （collapsed tool/skill/agent listing、runtime context、user context，
  见 `agent-runtime/src/prompt.rs::PrependedPromptReminders::ordered_reminders`）。
- 实测：所有去重事件中 marker 引用位置比真实原始内容位置 **+5**（9/9 精确命中偏移后的位置），
  即模型看到的"context position N"是错的；cattrs 任务里 marker 写 `position 8`、
  实际原始内容在 `position 13`。
- 影响：模型可能根据错误位置找不到原始内容，或误读其他消息。

### P1-1：marker 无内容描述

旧 marker 只写 `(round N, Read)`，模型无法判断被省略的内容是什么，只能盲目重读。

### P1-2：编辑后重读无新鲜度保护

若某文件在原始观察之后被 Edit/Write 修改，随后对同一路径的 Read 只要内容 hash 相同
（例如编辑发生在读取范围之外、或内容被改回原样）就会被去重，模型拿不到"编辑后仍一致"
的新鲜确认。

### P1-3：压缩后 marker 悬空

上下文压缩后 `reset_after_compression` 只清空 `seen` map，历史里已写入的 marker 仍引用
可能已被压缩掉的原始内容。

### P2：preview 哈希碰撞

旧逻辑只对 preview 部分 hash，两个内容不同但 preview 相同（前 2000 字符一致）的输出会被
误判为相同。

## 3. 修复方案（已实施）

### Fix 1：持久化输出头部注入全量内容哈希（解决 P0-1 + P2）

- `PersistedToolOutput` 新增 `content_sha256: String` 字段；
  `persist_tool_result` 在持久化时对**完整序列化内容 + metadata** 计算 sha256 写入。
- `build_persisted_tool_output_message` 在头部渲染 `Content sha256: <hex>` 行
  （hash 为空时省略该行，保持旧格式兼容）。
- `compute_dedup_key` 解析优先级：
  1. 命中 `Content sha256:` 行（仅限 `<persisted-output>` 头部 16 行内，校验 64 位 hex）
     → key = `persisted:<hash>`，精确且与 UUID 路径无关；
  2. 旧格式（无 hash 行）→ 取 `Preview (first N chars):` 之后的尾部（含 metadata）hash，
     key = `legacy-persisted:<hash>`，作为兼容回退；
  3. 普通输出 → 全量 hash，key = `plain:<hash>`。
- 不同 key 命名空间隔离，避免跨格式误命中。
- 说明：metadata（`exit_code`、`working_directory` 等）也参与 hash——它们是可见内容的一部分，
  相同的输出但不同的执行上下文不应去重。

### Fix 2：marker 自描述化、彻底去掉 position（解决 P0-2 + P1-1）

- `SeenObservation` 移除 `msg_index`，改为记录 `round_index` + `descriptor`。
- `descriptor` = 原始内容首个有意义行（持久化输出取头部第一行描述），截断到 120 字符。
- 新 marker 格式：

  ```
  [Observation deduped: identical content was already presented at round N (Read: Read lines 1-37 from /app/src/schema/types/schemaProps.ts (37 total lines)). Omitted to reduce context size. Call Read/Bash again if the content may have changed.]
  ```

- round 是稳定时间锚点，不随提醒注入 / 压缩而漂移。
- `apply()` 签名同步简化：删除 `current_messages_len` 参数（该参数正是 position 漂移的来源）。

### Fix 3：编辑感知（解决 P1-2）

- 新增 `edited_files: HashMap<path, round>`。
- Edit / Write 成功结果（结构化 `file_path`，回退解析 `Successfully edited <path>` /
  `Successfully created <path>`）记录 `path -> round`；`success=false` 不记录。
- Read 命中去重时，若该路径在原始观察 round 之后被编辑过 → **不去重**，完整展示本次读取，
  并把 `seen` 记录 rebase 到当前 round（后续读取可继续针对"新鲜的那次"去重）。
- 局限：通过 Bash（`sed -i`、`git checkout` 等）的文件变更无法感知，文档注明。

### Fix 4：压缩时标注幸存 marker（解决 P1-3）

- `reset_after_compression` 改为接收 `&mut [Message]`：清空 `seen`（保留 `edited_files`，
  编辑 round 在压缩后仍是有效时间锚点），并对历史中已存在的 marker 追加：

  ```
  [Note: context was compacted since this observation; the original content may no longer be in history. Re-read if needed.]
  ```

### Fix 5/6（本次未实施，见第 6 节）

- 可配置开关 / 阈值（接入 AIConfig）本次未做，保持默认行为；改动面大、收益不确定，列为后续项。

## 4. 改动明细

| 文件 | 改动 |
| --- | --- |
| `src/crates/execution/tool-contracts/src/tool_result_storage.rs` | `PersistedToolOutput` 新增 `content_sha256`；`build_persisted_tool_output_message` 渲染 hash 行 |
| `src/crates/assembly/core/src/agentic/tools/tool_result_storage.rs` | 持久化时计算 `compute_persisted_content_sha256`（内容+metadata）；测试断言 hash 行 |
| `src/crates/assembly/core/src/agentic/execution/observation_dedup.rs` | 重写 key 计算、marker、编辑感知、压缩标注；13 个单元测试 |
| `src/crates/assembly/core/src/agentic/execution/execution_engine.rs` | 适配 `apply(msg, round)` 与 `reset_after_compression(&mut messages)` |
| `src/crates/execution/tool-contracts/tests/tool_contracts.rs` | 持久化消息测试补 hash 行断言 + 空 hash 省略行测试 |

## 5. 测试与验证

### 单元测试（全部通过）

- `observation_dedup`：15 个测试（首现不变、重复替换、marker 自描述无 position、
  阈值、错误不参与、压缩重置、压缩标注、持久化同内容去重 / 不同内容不去重、
  旧格式 preview 回退、preview 内 hash 伪装行不误用、descriptor 跳过空首行、
  编辑后保持新鲜并 rebase、失败编辑不抑制去重、不同内容不去重）。
- `tool_contracts`：2 个持久化消息格式测试（hash 行渲染 / 空 hash 省略）。
- `bitfun-core::agentic::tools::tool_result_storage`：7 个持久化测试（含 hash 行断言，
  以及"两次超大 Read → 持久化 → 去重 marker"的端到端管道测试）。

### 全量回归

- `cargo test -p bitfun-agent-tools`：44 + 100 全部通过。
- `cargo test -p bitfun-core --lib`：996 通过 / 8 失败 —— 8 个失败均为存量失败
  （`canvas_tools` 7 个 + `git_adapter` 1 个），已在干净基线（stash 后）复现，与本次改动无关。

### 真实 trace 重分析（20 次 benchmark 运行）

- 旧逻辑共 15 个去重 marker（9 个唯一内容事件，原始内容 895 ~ 14,140 字符，
  全部为 Read 结果，此前已逐条验证与原始内容完全一致）。
- **新逻辑下全部 15 个去重决策保留**，无回退、无新增误去重
  （模拟发现的唯一"新事件"是 `[TOOL ERROR]` 错误消息，生产按 `is_error` 排除，不会去重）。
- **新鲜度保护 0 次触发**：这批运行中不存在"文件编辑后内容 hash 仍相同"的重读场景，
  说明 Fix 3 只增加安全性、不减少现有去重收益。
- 这批运行中持久化输出没有出现重复（P0-1 是潜在缺陷、未被激活）；Fix 1 的正确性由单元测试覆盖，
  并保证未来出现重复大输出（如反复跑超长测试/命令）时能正确去重。
- 新 marker 实测示例（dynamodb-toolbox）：

  ```
  [Observation deduped: identical content was already presented at round 2 (Read: Read lines 1-37 from /app/src/schema/types/schemaProps.ts (37 total lines)). Omitted to reduce context size. Call Read/Bash again if the content may have changed.]
  ```

## 6. 遗留事项与后续建议

1. **可配置化（Fix 5/6）**：将 `MIN_DEDUP_CHARS`、启用开关接入 `AIConfig`，
   需要把配置从 service 层透传到 execution engine，改动面较大，建议独立评审后再做。
2. **命令类文件变更感知**：Bash 内 `sed -i` / `git checkout` 等对文件的修改无法被
   `edited_files` 记录；如需覆盖，可考虑对 Bash 结果做变更路径启发式提取（风险较高，暂缓）。
3. **压缩边界观测**：目前压缩时统一 reset + 标注；若后续希望压缩后保留部分 seen 状态，
   需要压缩逻辑返回"被保留的原始消息"清单。
4. **跨 turn 去重**：去重器生命周期为单 `execute_turn`，跨 turn 复用需要持久化 key 索引，
   收益与风险需单独评估。

## 7. 复盘记录

- **决策 1**：content hash 覆盖"序列化内容 + metadata"，而不是仅内容——metadata 是可见消息的一部分
  （如 `exit_code`），不纳入会导致"相同输出、不同退出码"被错误去重。
- **决策 2**：`edited_files` 在压缩 reset 时**保留**——编辑 round 是稳定时间锚点，
  保留可让压缩后的新鲜度保护继续生效。
- **决策 3**：`apply()` 删除 `current_messages_len` 参数——它正是 position 漂移的根源，
  保留只会诱导未来再次引用消息位置。
- **决策 4**：旧格式持久化输出仍按 preview 尾部 hash 回退匹配（best effort，
  存在 preview 碰撞风险但仅影响旧历史消息；新消息一律带全量 hash）。
- **执行情况**：方案经专家评审后实施；代码、测试、文档一次提交（见 git log）；
  未做配置化与跨 turn 功能，符合"最小正确改动"原则。

## 8. 第二轮审核与修复（2026-08-06）

针对第一版提交做了逐行挑刺式复审，修复以下问题并补测：

- **H1（真实 bug，触发面窄）**：`parse_content_sha256` 原先扫描前 16 行时不检查
  `Preview (first ...` 边界。旧格式消息的 preview 若恰好含一行
  `Content sha256: <64位hex>`（如校验和清单），会被误认为权威 hash，导致两个内容不同
  但共享该行的输出被错误去重。修复：解析到 `Preview (` 行即停止（头部边界）。
- **H2（真实 bug）**：`build_descriptor` 对普通内容取 `lines().next()` 不跳过空行，
  Bash 输出以空行开头时 marker 出现 `(round 3, Bash: )` 空描述。修复：与持久化分支一致，
  取第一个非空行。
- **L1（健壮性）**：`legacy_persisted_canonical` 原先全文 `find("Preview (first ")`；
  与 H1 一起改为仅在前 16 行头部内查找，避免畸形消息的 preview 内容误匹配。
- **L2（一致性）**：新格式 persisted 去重路径补上 `MIN_DEDUP_CHARS` 阈值检查
  （round-budget 可能持久化小结果），与 plain 路径对齐。
- **L3（边界说明）**：新鲜度守卫保持严格 `>`（同 round 并行 Edit+Read 语义模糊，
  抑制会误伤未变内容），已加注释说明。
- **H3（测试缺口）**：新增端到端管道测试
  `duplicate_large_read_is_deduplicated_with_persisted_hash`——两次超大 Read 走真实
  `maybe_persist_large_tool_result` → `Message::tool_result` → `apply`，验证第二次被替换为
  marker，且两侧 hash 行一致；防止持久化格式与去重解析之间再次漂移（P0-1 的教训）。
- 新增回归测试：`preview_hash_lookalike_is_not_used_as_dedup_key`（H1）、
  `descriptor_skips_blank_first_line`（H2）。

复核结论：修复后 H1/H2 的旧行为均被新测试捕获（新代码下通过）；全量测试与真实 trace
重分析结论不变（15 个旧去重决策保留、0 次新鲜度抑制）。

仍未处理（设计取舍，建议单独评估）：
- 重复大输出在去重前会被持久化两次（磁盘/IO 浪费，M1）。
- `PersistedToolOutput` 公共结构体新增必填字段为 semver 破坏（仓库内仅 2 处构造点，M2）。
- `working_directory`/`terminal_session_id` 进 hash 会压低跨上下文 Bash 去重命中率（M3）。
- 历史会话中的旧 marker 仍引用错位 position，不做迁移（turn 级影响有限，L4）。
- 持久化去重后 marker 不携带原始 reference，压缩后模型只能重跑（L5）。

## 9. 第三轮审核与修复（2026-08-06，按外部专家 Fix 1–6 清单执行）

第三轮针对外部专家对前两轮提交的逐项审核意见（Fix 1–6）落地修复，核心思路：
**去掉基于消息位置的引用、去掉有碰撞风险的 preview fallback、让 marker 自描述、让编辑感知更准确，
并补上结构化 telemetry 与针对性测试。**

### Fix 1（采纳方案 A）：删除 legacy persisted fallback，缺失全量 hash 时不去重

- `PersistedToolOutput` 已增加 `content_sha256`（第二轮落地），
  `build_persisted_tool_output_message()` 在头部输出 `Content sha256: ...` 行，
  该行位于 `Preview (first N chars):` 边界之前。
- `compute_dedup_key` 现在：
  - 以 `<persisted-output>` 常量解析（不再匹配旧的 `[PERSISTED_OUTPUT` 假格式）；
  - 优先从头部解析 `Content sha256` → key 为 `persisted:<hash>`；
  - 旧格式 `persisted-output` 消息**缺少 hash 时直接返回 `None` 不去重**
    （删除原先 `preview + original_chars + line_count` 的 best-effort 回退，
    该回退在 preview/长度/行数相同但中间内容不同时仍会误去重，不能称为零碰撞）；
  - 普通输出仍按全量 sha256 去重；
  - persisted 路径同样要求内容 ≥ `MIN_DEDUP_CHARS`，与 plain 路径对齐。
- 测试：`legacy_persisted_output_without_hash_is_safely_skipped`、
  `persisted_hash_covers_middle_content_change`（preview、长度、行数相同但中间内容不同 → 不去重）、
  `preview_hash_lookalike_is_not_used_as_dedup_key` 继续有效。

### Fix 2：marker 完全自描述，彻底移除位置引用

- `apply()` 不接收任何消息位置参数（第二轮已删除 `current_messages_len`），
  本轮确认引擎侧无残留位置写入。
- `build_descriptor` 按工具类型构造结构化描述：
  - Read：从结构化结果提取 `file_path / start_line / lines_read / total_lines`，
    生成 `Read lines {start}-{end} from {path} ({total} total lines)`；
  - **持久化 Read 也从原始结果提取源文件描述**，而不是写 artifact 路径
    （`descriptor_uses_source_path_for_persisted_read` 覆盖）；
  - 其他工具取第一个非空行；
  - `sanitize_descriptor` 统一做单行化、控制字符清理（ANSI 等）、trim 与 120 字符截断。
- marker 保留 `round`（稳定时间锚点），不再保留任何 context position。

### Fix 3：编辑感知细化

- `record_mutation`：
  - Write 仅 `created / overwritten` 视为文件变更；
    **`already_exists_same_content`（Write skipped, identical content）不记录**
    （已核实 `write_file.rs` 语义），避免跳过内容相同的写入却抑制去重；
  - Edit 仍按 `success` 记录，失败 Edit 不抑制去重（`failed_edit_does_not_suppress_dedup`）。
- 路径 key 统一规范化：`normalize_path_key` trim + 去除 `./` 前缀，相对/带前缀路径映射到同一 key
  （`path_normalization_matches_dot_prefix`）。
- 新增 `pre_scan_round_mutations(&messages, round_index)`：引擎在每轮 apply 循环前先扫描
  成功编辑结果，保证同一 round 内（如并行工具调用）先 Edit 再 Read 时，
  Read 看到的是"已编辑"状态（`pre_scan_round_mutations_makes_same_round_edits_visible`）。
- 明确未覆盖场景：Bash 内部修改、外部进程、远程工作区修改无法被该 map 覆盖（见遗留事项）。

### Fix 4：压缩后不再改写历史 marker

- `reset_after_compression()` 改为**无参**，只清空 `seen` 索引；
  `edited_files` 与 `deduped_read_paths` 保留（编辑 round 是稳定锚点，新鲜度保护继续生效）。
- 不再对压缩后幸存的历史 marker 做任何改写：Fix 2 的自描述 marker 无需位置迁移即可解释，
  同时避免上下文重写带来的 cache miss。
- 已核实：压缩后 `messages = compressed_messages` 本来就不回写 session cache，
  原"标注幸存 marker"逻辑收益极低，删除安全。

### Fix 5：结构化 telemetry（不含配置开关）

- 新增 `DedupStats`：`replacements / persisted_hits / skipped_after_edit / resets /
  recovery_reads / chars_saved` 六个计数器。
- 引擎在每次 turn 循环结束后调用 `take_stats()` 并通过 `info!` 上报
  （`Turn observation dedup stats: ...`），不再依赖 debug 日志计数。
- 可配置开关（开关、阈值进 `AIConfig`）暂未做：需要把配置从 service 层透传到
  execution engine，改动面较大，继续列入遗留事项。
- `recovery_reads` 语义：本 turn 内对"曾被子去重路径"的 Read 计数（含去重读本身与编辑后的新鲜读），
  用于评估去重是否扰动 agent 执行路径。

### Fix 6：测试补充

`observation_dedup` 模块测试 23 个全部通过，本轮新增/改写覆盖：
- 同样内容、不同文件路径不互相去重（`same_content_different_paths_do_not_dedup_for_read`）；
- 同一文件被去重后再次显式 Read 返回完整内容并 rebase（`read_after_edit_is_kept_fresh_then_rebases`）；
- 旧格式无 hash 安全跳过去重；`Content sha256` 覆盖中间内容变化；
- 真实 `<persisted-output>` / `Preview (first N chars):` 格式（不再用旧假格式）；
- Write skipped 不去抑制去重、路径规范化、同 round 预扫描、失败 Edit、stats 计数、persisted hits 计数。

### 验证结果

- `cargo test -p bitfun-core --lib observation_dedup`：23 passed。
- `cargo test -p bitfun-core --lib agentic::tools::tool_result_storage`：7 passed
  （含 H3 端到端 persisted 管道测试）。
- `cargo test -p bitfun-agent-tools`：100 passed。
- `cargo test -p bitfun-core --lib`：1007 passed / 8 failed —— 8 个失败仍为存量失败
  （`canvas_tools` 7 + `git_adapter` 1，干净基线已复现，与本次改动无关）。
- 真实 trace 重分析（`/tmp/dedup_full_sim.py`，21 任务）：15 个旧 marker 的去重决策全部保留，
  0 次新鲜度抑制；无新增误去重（模拟中唯一"新事件"为 `[TOOL ERROR]` 消息，生产按 `is_error` 排除）。
- `cargo fmt -p bitfun-core` 后已还原被误格式化的无关文件 `infrastructure/storage/cleanup.rs`。

### 提交说明

- 分支：`feat/observation-dedup`
- 本轮提交：`fix(execution): third-round observation dedup hardening per expert review`
  （含 engine 调用点、去重器重写、文档；详见 git log）。
