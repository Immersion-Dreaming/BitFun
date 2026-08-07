# Lifecycle Estimator Refactor Progress

Updated: 2026-08-07

## Phase 6: estimator eligibility and event-driven transition repair

Status: implementation complete; targeted unit verification complete; live
SWE-bench shadow replay pending.

Trace analysis of the 2026-08-06 batch found that 16/16 verifier results
passed, but the lifecycle estimator was ineffective: it received whole-history
snapshots, emitted mostly root `active` no-ops, reached the 30-second timeout,
and never proposed an `evictable` transition. The fix deliberately does not
relax deterministic protection and does not enable physical eviction.

Implemented:

- Estimation now has an explicit mode: `completion` accepts only
  `active -> completed`; `eviction` accepts only `completed -> evictable`.
  The model cannot emit `active`, skip a state, or update a context-only task.
- The registry computes `eligibleTaskIds`. Eligibility requires a Todo-derived,
  non-control work unit, at least one ordinary tool segment, completed Todo
  state, and applicable deterministic protections. Eviction additionally
  requires later ordinary work owned by another Todo unit in the same root
  task, proving a reuse horizon.
- A successful Todo completion with recorded ordinary work triggers an
  asynchronous completion assessment immediately. Later work and a completed
  reducer transition can trigger an eviction assessment. The three-tick cadence
  remains only as a bounded recovery path.
- Snapshots contain only eligible targets plus at most two segments from the
  current active Todo as dependency context. Root/control tasks never become
  estimator targets; unrelated historical tasks and Todo records are omitted.
- The reducer now rejects task IDs outside `eligibleTaskIds` and lifecycle
  values that do not match the snapshot mode before applying revision checks.
- Traces retain the snapshot mode, trigger, eligible IDs, input size, reducer
  result, and protections for eligible targets only. `physicalEvictionApplied`
  remains `false`.

Test requirements for the next CLI run:

1. A no-Todo trajectory must show no `lifecycle-estimator-*-scheduled.json`
   files after the initial recovery attempt.
2. A sequential Todo trajectory must show a `todo_completed` completion
   snapshot, followed after later same-root work by an `eviction` snapshot.
3. A valid model response may produce a shadow candidate, but no context
   messages, archives, or prompt-cache entries may be modified.
4. Report estimator request count, input characters, timeout rate, and latency
   separately from primary-agent token and cache statistics.

## SWE-bench adaptation plan

The original model treated one dialog turn as one lifecycle task. That is safe
but ineffective for SWE-bench: one user prompt commonly contains dozens of
tool rounds, so the task remains recent for the whole trajectory.

1. Keep a root objective for the complete user prompt and add work units below
   it. A work unit is initially created from one Todo ID; a no-Todo fallback
   remains attached to the root and is never evictable.
2. Maintain a latest Todo ledger. Historical `pending` values are facts only;
   protection follows the latest status. TodoWrite/control segments stay pinned
   to the root because they contain the plan for later work.
3. Assign ordinary tool segments to the unique current `in_progress` Todo. If
   assignment is ambiguous, keep the segment on the non-evictable root. The
   model cannot arbitrarily reassign segments in this phase.
4. Update snapshots, reducer revisions, protection reasons, and traces to name
   work units and expose the Todo/control decision. Physical eviction remains
   disabled.
5. Replay a real SWE-bench snapshot and require an old completed work unit to
   become a candidate only when its Todo is currently complete, it is outside
   the recent window, and no dependency/error guard applies.
6. Only after this produces explainable candidates design artifact-backed
   physical apply and recovery. Shadow mode does not measure token/cache or
   quality effects.

Trajectories without reliable Todo structure remain shadow-only and produce
fewer candidates until a separately validated model-assisted segmentation phase
is added.

## Scope and terminology

This branch adapts the lifecycle portion of TokenPilot to BitFun. It implements
only `active -> completed -> evictable` as **shadow analysis**. It does not
delete, replace, archive, or summarize context messages.

- A **dialog turn** is one actual user input with a `dialog_turn_id`.
- A **tick** is one completed assistant tool-call group plus its tool results.
- A **segment** is the message range that could eventually be replaced.
- A **root objective** is one complete user input and is retained.
- A **work unit** is an intra-turn Todo-derived task spanning one or more
  segments. Lifecycle state belongs to the work unit, not directly to a segment.

The scheduler cadence is fixed at three ticks. A candidate may refer to an
older task, not only to the three ticks that caused a new snapshot.

## Phase 5: SWE-bench work-unit adaptation

Status: implementation complete and unit/regression tested; live benchmark
shadow trace pending.

Implemented so far:

- Registry schema v3 retains one root work unit per dialog turn and creates a
  Todo-derived work unit only for a unique current `in_progress` Todo.
- TodoWrite segments are control context and remain owned by the root, even
  when the write fails. Ordinary later tool rounds follow the active Todo.
- The persisted Todo ledger follows only successful TodoWrite results. A later
  `completed` status removes the Todo protection; an earlier `pending` status
  remains audit data rather than a permanent eviction ban.
- A successful whole-list TodoWrite retains omitted historical items for
  ownership validation. Omitted unfinished items become conservatively
  `cancelled`; omitted completed items stay `completed`.
- No-Todo and ambiguous-Todo trajectories stay on the root fallback and cannot
  become candidates. This is intentionally conservative.
- Snapshot task records now expose root ID, work-unit source, control flag, and
  linked Todo IDs. Completed traces include protection reasons for every work
  unit.

Errors found and fixed during this phase:

- Initial schema additions left six constructors incomplete; `cargo check`
  reported each missing field before behavior tests were run.
- A TodoWrite request was initially admitted from request arguments even when
  its tool result failed. The ledger now requires a matching successful result.
- Failed TodoWrite was initially not treated as a retained control segment.
  It is now pinned to the root and cannot contaminate an active Todo unit.
- Whole-list replacement initially removed an omitted Todo that an older work
  unit still referenced. A targeted test caught the resulting registry
  validation failure. The ledger now retains the audit item and blocks its
  work unit by marking an omitted unfinished item `cancelled`.

Remaining limitations:

- Tool-error protection remains task-wide and conservative. A resolved earlier
  error still prevents eviction until a separate, evidence-backed resolution
  mechanism is designed.
- Work-unit dependencies are only consumed if supplied by the estimator; this
  phase does not infer cross-Todo dependencies deterministically.
- This phase does not read raw assistant reasoning or enable physical eviction.

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

Earlier checkpoint, before Phase 5:

- `cargo test --offline -p bitfun-core lifecycle_evict --lib`: 17 passed,
  0 failed. Coverage includes full-prompt preservation, ownership, idempotence,
  v1 migration, per-task stale rejection, legal transitions, deterministic
  guards, scheduler non-blocking/coalescing/cancellation, parser handling, and
  restart recovery.
Current Phase 5 verification:

- `cargo test --offline -p bitfun-core lifecycle_evict --lib`: 25 passed,
  0 failed. Added coverage proves Todo control ownership, result-gated ledger
  updates, generated Todo IDs, whole-list omitted-item handling, ambiguous/no-Todo
  fallback protection, completed Todo candidate eligibility during later
  intra-turn work, and estimator snapshot input for the active Todo work unit.
- `cargo test --offline -p bitfun-core --lib`: 1016 passed, 0 failed,
  1 ignored after the final ledger fix.
- `cargo check --offline -p bitfun-cli -q` after Phase 5: passed.

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
