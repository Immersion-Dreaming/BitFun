# Lifecycle Estimator Refactor Progress

Updated: 2026-08-06

## Scope and terminology

This branch adapts the lifecycle portion of TokenPilot to BitFun. It implements
only `active -> completed -> evictable` as **shadow analysis**. It does not
delete, replace, archive, or summarize context messages.

- A **dialog turn** is one actual user input with a `dialog_turn_id`.
- A **tick** is one completed assistant tool-call group plus its tool results.
- A **segment** is the message range that could eventually be replaced.
- A **task** is a user objective spanning one or more segments. Lifecycle state
  belongs to the task, not directly to a segment.

The scheduler cadence is fixed at three ticks. A candidate may refer to an
older task, not only to the three ticks that caused a new snapshot.

## Phase 1: deterministic task and segment facts

Status: complete and audited.

Implemented:

- Every actual dialog turn gets a provisional `task:<turn_id>` record.
- `original_user_prompt` and initial `objective` retain the complete source
  prompt; no 1,200-character cap or other prompt truncation is used.
- Every tool tick is recorded under exactly one owner task.
- Tool facts contain tool ID, name, and serialized arguments. Result facts
  contain error status, length, SHA-256, and a 600-character preview. Raw
  assistant reasoning is neither persisted nor sent to the estimator.
- v1 segment-only metadata migrates to conservative active legacy tasks. Its
  previous `evictable` state is discarded rather than trusted.
- Registry validation checks one-to-one turn/task and task/segment ownership
  before the registry is used.

Safety result:

- The old synchronous estimator call, archive creation, message replacement,
  and prompt-cache invalidation path were removed from lifecycle handling.
- Lifecycle records use stable turn and tool IDs, so ordinary context
  compression does not invalidate their ownership mapping.

## Phase 2: snapshot and reducer

Status: complete and audited.

Implemented:

- A snapshot is eligible after three newly recorded ticks. It has a stable ID,
  capture tick, current task ID, and expected revision for every task.
- The full raw prompt is included only for the current dialog turn. Older raw
  prompts remain local persistence and are not repeated in snapshot JSON.
- The reducer validates each task update against both its snapshot revision and
  live revision. A stale update for one task does not discard a fresh update
  for another task in the same response.
- The reducer rejects direct `active -> evictable`, model-only reopen or
  downgrade, missing completion evidence, unresolved items, recent tasks,
  error segments, pending Todos, and no-op updates.
- Deterministic protection wins over the model. A model cannot make a protected
  task evictable merely by asserting that it is done.

Known limitation, deliberately retained:

- A later dialog turn is a new provisional task. This branch does not yet merge
  a follow-up user request into an earlier task. That may lower future candidate
  rate, but it cannot cause a wrong physical deletion in this shadow phase.

## Phase 3: asynchronous scheduler and recovery

Status: complete and audited.

Implemented:

- One in-flight estimator job and one newest pending snapshot are kept per
  session. Newer work coalesces instead of creating unbounded requests.
- Destroying a scheduler explicitly aborts an unfinished worker. Tokio would
  otherwise detach the request, which could cause an unnecessary model charge
  and a duplicate recovery request after engine restart.
- The execution loop polls only completed worker handles. It never awaits the
  estimator network request.
- A 30-second timeout converts a slow provider into a traced failure rather
  than blocking the agent.
- `last_snapshot_tick_seq` and `last_finished_snapshot_tick_seq` are persisted.
  On a later dialog turn, an unfinished snapshot is submitted once for recovery
  after restart; it is not silently lost.
- Trace serialization and filesystem writes are spawned in a background task,
  so a slow disk does not extend the agent round. A trace-write failure is
  logged but does not change registry state.

Important trade-off:

- Background trace writes are best-effort. A process crash immediately after a
  reducer result may lose that trace file, while the registry itself is
  persisted synchronously before the trace is queued. Correct state is favored
  over trace durability on the critical path.

## Phase 4: configured fast-model shadow estimator

Status: code complete; end-to-end CLI/provider verification pending.

Authorization:

- On 2026-08-06 the user approved sending the lifecycle snapshot to BitFun's
  configured `fast` model slot.
- The outbound payload is the full current user prompt plus task lifecycle
  facts, tool IDs/names/serialized arguments, result previews and hashes,
  errors, and Todo status. It does not include raw assistant reasoning or full
  historical tool-result payloads.

Implemented behavior:

- The worker obtains the configured `fast` client through BitFun's normal AI
  client factory and asks it for strict JSON only.
- The accepted response schema is `snapshotId` plus revision-checked task
  updates containing lifecycle, completion evidence, and unresolved items.
- Every completion writes a shadow trace with snapshot, response or failure,
  reducer outcome, candidate segment IDs, duration, and
  `physicalEvictionApplied: false`.
- Candidate IDs are diagnostic only. No lifecycle path calls
  `replace_context_messages`, `invalidate_prompt_cache`, archive creation, or
  a context deletion API.
- `LifecycleEvictionSummary` remains as a legacy message enum value so old
  persisted sessions can deserialize. There is no current lifecycle producer
  for it, and it does not apply or hide an eviction.

## Verification record

Passed:

- `cargo test --offline -p bitfun-core lifecycle_evict --lib`: 17 passed,
  0 failed. Coverage includes full-prompt preservation, ownership, idempotence,
  v1 migration, per-task stale rejection, legal transitions, deterministic
  guards, scheduler non-blocking/coalescing/cancellation, parser handling, and
  restart recovery.
- `cargo test --offline -p bitfun-core --lib`: 1008 passed, 0 failed,
  1 ignored, after allowing Canvas tests to create their normal temporary
  directories under `~/.bitfun`.

Initially failed, then resolved as environment permission rather than code:

- The first sandboxed full-suite run had 7 Canvas test failures because it was
  forbidden to create `~/.bitfun/projects/.../canvases`. Those Canvas tests
  passed after the narrowly scoped test-directory permission was granted. This
  is recorded because the first run was not a code failure, but it did block a
  complete verification result until rerun.

Still required before claiming runtime effectiveness:

1. Build the CLI and run a tool-heavy task with a valid `fast` model
   configuration. The test suite uses local workers and does not make a paid
   provider call.
2. Inspect the emitted `lifecycle-estimator-*.json` files. Verify cadence,
   input size, timeout/failure rate, lifecycle transitions, and whether the
   deterministic guard rejects unsafe model proposals.
3. Measure estimator latency and cost separately from the primary model. The
   three-tick cadence reduces request frequency; it does not prove that the
   estimator is economical for every benchmark.
4. Do not interpret a shadow candidate as token savings. Physical apply needs a
   separate design for immutable artifact recovery, message reconstruction,
   cache-impact measurement, and rollback.

## Explicit non-goals of this branch

- Physical message deletion, summary insertion, archival, or prompt-cache
  invalidation.
- Treating internal tool ticks as identical to the paper's benchmark user turns.
- Sending raw chain-of-thought to the estimator.
- Claiming benchmark quality, token savings, or cache benefit before a
  controlled apply experiment.
