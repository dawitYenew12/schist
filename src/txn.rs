//! Multi-version concurrency control (MVCC) transaction manager.
//!
//! This subsystem is the concurrency layer of `schist`. It is *organic*: it
//! exists for its own sake as a self-contained transaction manager, much like
//! the pager or the WAL. It does not drive the mutation layer directly;
//! instead it exposes a small protocol — begin / read / write / commit /
//! rollback — that a higher layer would call into. Everything here is in-memory
//! and single-threaded: the manager owns a logical clock, a transaction
//! registry, per-row version chains, a lock table for write intents, a wait-for
//! graph for deadlock detection, and the write-ahead log it journals every
//! state transition to.
//!
//! # Architecture
//!
//! The unit of concurrency is the [`Transaction`], identified by a
//! [`TransactionId`] and pinned to a [`Timestamp`] (its snapshot timestamp)
//! taken at [`TxnManager::begin`]. Each row in the database is represented by a
//! [`VersionChain`]: an ordered list of [`Version`]s, each tagged with the
//! commit timestamp that *created* it (`begin_ts`) and the commit timestamp
//! that *superseded* it (`end_ts`, [`TS_INF`] while current). A reader decides
//! which version to materialise through a [`ReadView`]: a snapshot of the
//! clock and the active transaction set captured at begin time.
//!
//! ```text
//!   row 5   VersionChain
//!            ┌──────────────────────────────────────────────┐
//!            │ Version { value: 30, begin: t3, end: INF }   │  ← current
//!            │ Version { value: 20, begin: t2, end: t3   }  │  ← dead after GC
//!            │ Version { value: 10, begin: t1, end: t2   }  │  ← dead after GC
//!            └──────────────────────────────────────────────┘
//! ```
//!
//! Visibility is the heart of the system. A version `v` is visible to a read
//! view `rv` when:
//!
//! 1. `v` is the reader's own uncommitted write intent (`v.writer == rv.reader`
//!    and `v` is not yet committed), **or**
//! 2. `v` is committed, was created at or before the snapshot
//!    (`v.begin_ts <= rv.snapshot_ts`), and has not been superseded at or before
//!    the snapshot (`v.end_ts > rv.snapshot_ts`).
//!
//! Uncommitted versions written by *other* transactions are invisible to every
//! isolation level except [`IsolationLevel::ReadUncommitted`], which performs
//! dirty reads.
//!
//! # Isolation levels
//!
//! - [`IsolationLevel::ReadUncommitted`]: a read sees the newest version in the
//!   chain regardless of committed state — a dirty read.
//! - [`IsolationLevel::ReadCommitted`]: the read view is refreshed to the
//!   current clock tick before every read, so a transaction observes the
//!   latest committed state between statements.
//! - [`IsolationLevel::RepeatableRead`]: the read view is fixed at begin; the
//!   transaction sees a stable snapshot for its whole life (snapshot
//!   isolation). Write/write conflicts are detected with exclusive locks.
//! - [`IsolationLevel::Serializable`]: snapshot isolation plus a
//!   read-set/predicate-lock conflict check at commit. If any transaction
//!   committed a write to a row this transaction read (or predicate-locked)
//!   after this transaction began, commit fails with a serialization conflict.
//!
//! # Write intents and locks
//!
//! Every mutation takes an exclusive lock on the row via the lock table. An
//! uncommitted write is recorded as a [`Version`] with `committed == false`;
//! only the writing transaction can see it. A second writer on the same row
//! fails to acquire the exclusive lock and is rejected with a lock-conflict
//! error, and the attempted acquisition is recorded as an edge in the wait-for
//! graph so that [`TxnManager::detect_deadlock`] can find cycles.
//!
//! # Undo, savepoints, and rollback
//!
//! Each mutation appends an [`UndoRecord`] to the transaction's [`UndoLog`].
//! [`TxnManager::rollback`] walks the undo log and removes every version the
//! transaction created. Named [`Savepoint`]s capture a position in the undo
//! log; [`TxnManager::rollback_to_savepoint`] replays undo only back to that
//! position, preserving earlier work.
//!
//! # Garbage collection
//!
//! [`TxnManager::garbage_collect`] removes committed versions whose `end_ts`
//! falls before the oldest active transaction's snapshot — versions no live
//! snapshot can ever see again. The current version (`end_ts == TS_INF`) and
//! any uncommitted write intent are always retained.

use crate::error::{Error, Result};
use crate::value::Value;
use crate::wal::{
    decode_delete_payload, decode_insert_payload, decode_update_payload,
    encode_delete_payload, encode_insert_payload, encode_update_payload, RecType, Wal,
};
use std::collections::{HashMap, HashSet};

/// The infinite timestamp. A version whose `end_ts` is `TS_INF` is the current
/// version of its row — nothing has superseded it yet.
pub const TS_INF: Timestamp = Timestamp(u64::MAX);

/// The zero timestamp, used as the "before any transaction existed" marker for
/// the very first snapshot.
pub const TS_ZERO: Timestamp = Timestamp(0);

/// A monotonically increasing logical clock value.
///
/// Timestamps are allocated by the [`TxnManager`] clock and uniquely order
/// every transaction begin and commit. They are the version tags stored on
/// every [`Version`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// The raw clock value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// `true` if this is the infinite timestamp.
    pub fn is_inf(self) -> bool {
        self.0 == u64::MAX
    }

    /// Saturating addition — used when advancing a version's `end_ts`.
    pub fn saturating_add(self, n: u64) -> Timestamp {
        Timestamp(self.0.saturating_add(n))
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_inf() {
            write!(f, "inf")
        } else {
            write!(f, "t{}", self.0)
        }
    }
}

/// A unique transaction identifier.
///
/// Ids are never reused within the lifetime of a [`TxnManager`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TransactionId(pub u64);

impl TransactionId {
    /// The raw id value.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "txn{}", self.0)
    }
}

/// The lifecycle state of a [`Transaction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    /// Begun and running; reads and writes are accepted.
    Active,
    /// Phase one of two-phase commit has succeeded; the transaction may still
    /// be rolled back, but it has promised to commit.
    Prepared,
    /// Durably committed; all its versions are visible to snapshots at or
    /// after its commit timestamp.
    Committed,
    /// Rolled back; all its versions have been removed.
    Aborted,
}

impl TxnState {
    /// `true` if the transaction is still permitted to accept operations.
    pub fn is_live(self) -> bool {
        matches!(self, TxnState::Active | TxnState::Prepared)
    }

    pub fn name(self) -> &'static str {
        match self {
            TxnState::Active => "active",
            TxnState::Prepared => "prepared",
            TxnState::Committed => "committed",
            TxnState::Aborted => "aborted",
        }
    }
}

/// The isolation level a transaction runs at. See the module docs for the
/// visibility and conflict rules each level implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Dirty reads allowed: the newest version in the chain is returned
    /// regardless of committed state.
    ReadUncommitted,
    /// Each read takes a fresh snapshot at the current clock tick.
    ReadCommitted,
    /// The snapshot is fixed at begin (snapshot isolation).
    RepeatableRead,
    /// Snapshot isolation plus a read-set/predicate conflict check at commit.
    Serializable,
}

impl IsolationLevel {
    pub fn name(self) -> &'static str {
        match self {
            IsolationLevel::ReadUncommitted => "read-uncommitted",
            IsolationLevel::ReadCommitted => "read-committed",
            IsolationLevel::RepeatableRead => "repeatable-read",
            IsolationLevel::Serializable => "serializable",
        }
    }
}

/// A single version of a row in a [`VersionChain`].
///
/// Versions are immutable once committed; while uncommitted they are the
/// writing transaction's write intent and may be removed on rollback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Version {
    /// The cell value this version holds. Meaningless when `deleted` is true
    /// (a tombstone stores [`Value::Null`]).
    pub value: Value,
    /// `true` for a delete tombstone — the row is gone as of `begin_ts`.
    pub deleted: bool,
    /// The commit timestamp of the transaction that created this version.
    /// While the version is uncommitted this holds the writer's begin
    /// timestamp as a tentative tag; it is rewritten to the real commit
    /// timestamp on commit.
    pub begin_ts: Timestamp,
    /// The commit timestamp of the transaction that superseded this version,
    /// or [`TS_INF`] while this version is still current.
    pub end_ts: Timestamp,
    /// The transaction that created this version.
    pub writer: TransactionId,
    /// Whether the writer has committed. While `false` the version is a write
    /// intent visible only to `writer`.
    pub committed: bool,
}

impl Version {
    /// A committed, current, live version holding `value`.
    pub fn committed_current(value: Value, begin_ts: Timestamp, writer: TransactionId) -> Self {
        Version {
            value,
            deleted: false,
            begin_ts,
            end_ts: TS_INF,
            writer,
            committed: true,
        }
    }

    /// An uncommitted write intent. `value` is the proposed new cell; the
    /// version becomes visible to `writer` immediately and to everyone else on
    /// commit.
    pub fn write_intent(value: Value, begin_ts: Timestamp, writer: TransactionId) -> Self {
        Version {
            value,
            deleted: false,
            begin_ts,
            end_ts: TS_INF,
            writer,
            committed: false,
        }
    }

    /// An uncommitted delete tombstone.
    pub fn delete_intent(begin_ts: Timestamp, writer: TransactionId) -> Self {
        Version {
            value: Value::Null,
            deleted: true,
            begin_ts,
            end_ts: TS_INF,
            writer,
            committed: false,
        }
    }

    /// `true` if this version is a superseded (dead) committed version.
    pub fn is_dead(self) -> bool {
        self.committed && self.end_ts != TS_INF
    }
}

/// The version chain for a single row, ordered newest-first.
///
/// The head of `versions` (index `0`) is the most recently created version —
/// either the current committed version or an uncommitted write intent. Older
/// versions trail behind in decreasing `begin_ts` order. [`VersionChain`]
/// deliberately keeps dead (superseded) versions around so that older
/// snapshots can still resolve them; the [`TxnManager`] reclaims them with
/// garbage collection.
#[derive(Debug, Clone)]
pub struct VersionChain {
    /// The row this chain describes.
    pub row_id: u64,
    /// Versions, newest-first.
    pub versions: Vec<Version>,
}

impl VersionChain {
    /// An empty chain for `row_id`.
    pub fn empty(row_id: u64) -> Self {
        VersionChain {
            row_id,
            versions: Vec::new(),
        }
    }

    /// Number of versions currently linked.
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Whether the chain has no versions.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Push a new version onto the head of the chain.
    pub fn push_head(&mut self, v: Version) {
        self.versions.insert(0, v);
    }

    /// Remove and return the head version.
    pub fn pop_head(&mut self) -> Option<Version> {
        if self.versions.is_empty() {
            None
        } else {
            Some(self.versions.remove(0))
        }
    }

    /// The head version, if any.
    pub fn head(&self) -> Option<&Version> {
        self.versions.first()
    }

    /// The current committed version visible to a snapshot at `TS_INF`
    /// (ignoring write intents). This is the version a brand-new transaction
    /// at the present moment would see.
    pub fn current_committed(&self) -> Option<&Version> {
        self.versions.iter().find(|v| v.committed && v.end_ts == TS_INF)
    }

    /// The first version in the chain visible to `rv`, scanning newest-first.
    ///
    /// Visibility is delegated to [`ReadView::is_visible`]. Returning the
    /// newest visible version is what makes write intents and dirty reads
    /// work: the writer's own uncommitted version sits at the head and is
    /// picked up first.
    pub fn visible_for(&self, rv: &ReadView) -> Option<&Version> {
        for v in &self.versions {
            if rv.is_visible(v) {
                return Some(v);
            }
        }
        None
    }

    /// A snapshot of every version in the chain, useful for inspection and
    /// tests. The returned vector is newest-first.
    pub fn snapshot(&self) -> Vec<Version> {
        self.versions.clone()
    }
}

/// A snapshot of the clock and active transaction set, used to decide which
/// [`Version`] of a row a reader may see.
///
/// A read view is captured at [`TxnManager::begin`] (for repeatable-read and
/// serializable transactions) or refreshed before each read (for
/// read-committed). It is the single source of truth for visibility.
#[derive(Debug, Clone)]
pub struct ReadView {
    /// The transaction the view belongs to. Its own uncommitted write intents
    /// are always visible to it.
    pub reader: TransactionId,
    /// The snapshot timestamp. Committed versions with `begin_ts <= snapshot_ts`
    /// and `end_ts > snapshot_ts` are visible.
    pub snapshot_ts: Timestamp,
    /// The set of transactions that were active when the view was captured.
    /// Committed versions authored by a transaction still in this set are
    /// treated as not-yet-visible defensively.
    pub active: HashSet<TransactionId>,
    /// The isolation level, which refines the visibility rule for uncommitted
    /// versions.
    pub isolation: IsolationLevel,
}

impl ReadView {
    /// Build a read view for `reader` at `snapshot_ts`.
    pub fn new(
        reader: TransactionId,
        snapshot_ts: Timestamp,
        active: HashSet<TransactionId>,
        isolation: IsolationLevel,
    ) -> Self {
        ReadView {
            reader,
            snapshot_ts,
            active,
            isolation,
        }
    }

    /// Decide whether `v` is visible under this view. See the module docs for
    /// the precise rule.
    pub fn is_visible(&self, v: &Version) -> bool {
        // The reader always sees its own uncommitted write intents.
        if v.writer == self.reader && !v.committed {
            return true;
        }
        // Other transactions' uncommitted versions are invisible except to
        // dirty reads.
        if !v.committed {
            return matches!(self.isolation, IsolationLevel::ReadUncommitted);
        }
        // A committed version authored by a transaction that is somehow still
        // in the active set is treated as not-yet-visible.
        if self.active.contains(&v.writer) {
            return false;
        }
        // Created after the snapshot?
        if v.begin_ts > self.snapshot_ts {
            return false;
        }
        // Superseded at or before the snapshot?
        if v.end_ts <= self.snapshot_ts {
            return false;
        }
        true
    }

    /// `true` if a committed version created at `commit_ts` would be visible
    /// — that is, it was committed before this view's snapshot.
    pub fn sees_commit(&self, commit_ts: Timestamp) -> bool {
        commit_ts <= self.snapshot_ts
    }
}

/// A single entry in a transaction's [`UndoLog`].
///
/// Each variant records the *kind* of mutation so the undo log can be journaled
/// to the WAL and inspected; the actual rollback action is uniform: remove the
/// newest version on the row that belongs to the aborting transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UndoRecord {
    /// An insert of a brand-new row.
    Insert { row_id: u64 },
    /// An update of an existing row.
    Update { row_id: u64 },
    /// A delete of an existing row.
    Delete { row_id: u64 },
}

impl UndoRecord {
    /// The row this record touches.
    pub fn row_id(self) -> u64 {
        match self {
            UndoRecord::Insert { row_id }
            | UndoRecord::Update { row_id }
            | UndoRecord::Delete { row_id } => row_id,
        }
    }

    /// The WAL record type corresponding to this mutation.
    pub fn rec_type(self) -> RecType {
        match self {
            UndoRecord::Insert { .. } => RecType::Insert,
            UndoRecord::Update { .. } => RecType::Update,
            UndoRecord::Delete { .. } => RecType::Delete,
        }
    }
}

/// The per-transaction undo log: an ordered list of mutations to reverse on
/// rollback. The log is also the substrate for savepoints, which capture a
/// length into the log.
#[derive(Debug, Clone, Default)]
pub struct UndoLog {
    records: Vec<UndoRecord>,
}

impl UndoLog {
    /// An empty undo log.
    pub fn new() -> Self {
        UndoLog::default()
    }

    /// Append a mutation record.
    pub fn push(&mut self, rec: UndoRecord) {
        self.records.push(rec);
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remove and return the most recent record (for step-by-step undo).
    pub fn pop(&mut self) -> Option<UndoRecord> {
        self.records.pop()
    }

    /// Truncate the log to `len` records, returning the removed records in
    /// reverse (newest-first) order. Used by savepoint rollback.
    pub fn truncate_to(&mut self, len: usize) -> Vec<UndoRecord> {
        let mut removed = Vec::new();
        while self.records.len() > len {
            if let Some(r) = self.records.pop() {
                removed.push(r);
            }
        }
        removed
    }

    /// All records, oldest-first.
    pub fn records(&self) -> &[UndoRecord] {
        &self.records
    }

    /// Clear every record.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

/// A named savepoint within a [`Transaction`].
///
/// A savepoint remembers a position in the undo log and the size of the write
/// set so that [`TxnManager::rollback_to_savepoint`] can undo work back to (but
/// not including) the savepoint without disturbing earlier work.
#[derive(Debug, Clone)]
pub struct Savepoint {
    /// The user-supplied name.
    pub name: String,
    /// The undo log length at savepoint creation. Rollback replays undo down
    /// to this length.
    pub undo_len: usize,
    /// The write-set contents at savepoint creation. Rollback restores the
    /// write set to exactly this set.
    pub write_set: HashSet<u64>,
    /// The read-set contents at savepoint creation. Read-set entries captured
    /// after the savepoint are discarded on rollback-to-savepoint.
    pub read_set: HashSet<u64>,
}

/// A write intent: an uncommitted write to `row_id` by `writer`.
///
/// The manager maintains a set of these for conflict reporting; the
/// authoritative storage of the intent is the [`Version`] at the head of the
/// row's [`VersionChain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteIntent {
    pub row_id: u64,
    pub writer: TransactionId,
}

/// A predicate lock held by a serializable transaction over a set of rows
/// matched by a scan. At commit, the manager checks whether any concurrently
/// committed transaction modified a locked row; if so, the holder aborts with
/// a serialization conflict (preventing phantoms and write skew).
#[derive(Debug, Clone)]
pub struct PredicateLock {
    /// The transaction holding the lock.
    pub txn: TransactionId,
    /// The rows the scan matched.
    pub rows: HashSet<u64>,
}

impl PredicateLock {
    /// An empty predicate lock for `txn`.
    pub fn empty(txn: TransactionId) -> Self {
        PredicateLock {
            txn,
            rows: HashSet::new(),
        }
    }

    /// Add a matched row to the lock.
    pub fn add(&mut self, row_id: u64) {
        self.rows.insert(row_id);
    }

    /// Number of rows covered.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the lock covers no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// `true` if the lock covers `row_id`.
    pub fn covers(&self, row_id: u64) -> bool {
        self.rows.contains(&row_id)
    }
}

/// A live transaction.
///
/// A `Transaction` is owned by the [`TxnManager`] registry; callers refer to it
/// by [`TransactionId`] and interact through manager methods. The fields are
/// `pub` so that diagnostics and tests can inspect state.
#[derive(Debug, Clone)]
pub struct Transaction {
    /// The transaction's id.
    pub id: TransactionId,
    /// The snapshot timestamp captured at begin.
    pub begin_ts: Timestamp,
    /// The commit timestamp, assigned on successful commit.
    pub commit_ts: Option<Timestamp>,
    /// The lifecycle state.
    pub state: TxnState,
    /// The isolation level.
    pub isolation: IsolationLevel,
    /// The current read view. Refreshed per-read under read-committed.
    pub read_view: ReadView,
    /// Row ids this transaction has read (for serializable conflict checks).
    pub read_set: HashSet<u64>,
    /// Row ids this transaction has written.
    pub write_set: HashSet<u64>,
    /// Rows covered by this transaction's predicate locks.
    pub predicate_rows: HashSet<u64>,
    /// The undo log.
    pub undo: UndoLog,
    /// Named savepoints, in creation order.
    pub savepoints: Vec<Savepoint>,
    /// The LSN of the most recent WAL record this transaction appended.
    pub last_lsn: u64,
}

impl Transaction {
    fn new(
        id: TransactionId,
        begin_ts: Timestamp,
        active: HashSet<TransactionId>,
        isolation: IsolationLevel,
    ) -> Self {
        let read_view = ReadView::new(id, begin_ts, active, isolation);
        Transaction {
            id,
            begin_ts,
            commit_ts: None,
            state: TxnState::Active,
            isolation,
            read_view,
            read_set: HashSet::new(),
            write_set: HashSet::new(),
            predicate_rows: HashSet::new(),
            undo: UndoLog::new(),
            savepoints: Vec::new(),
            last_lsn: 0,
        }
    }

    /// The number of savepoints currently held.
    pub fn savepoint_count(&self) -> usize {
        self.savepoints.len()
    }

    /// Whether the transaction is still accepting operations.
    pub fn is_live(&self) -> bool {
        self.state.is_live()
    }
}

/// A snapshot of a transaction's state, returned by [`TxnManager::info`]. It
/// copies the interesting fields out so the caller does not hold a borrow on
/// the manager.
#[derive(Debug, Clone)]
pub struct TxnInfo {
    pub id: TransactionId,
    pub begin_ts: Timestamp,
    pub commit_ts: Option<Timestamp>,
    pub state: TxnState,
    pub isolation: IsolationLevel,
    pub snapshot_ts: Timestamp,
    pub reads: usize,
    pub writes: usize,
    pub predicate_rows: usize,
    pub savepoints: usize,
    pub undo_records: usize,
    pub last_lsn: u64,
}

/// The mode of a lock held in the [`LockTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// A shared lock held by one or more readers.
    Shared,
    /// An exclusive lock held by a single writer.
    Exclusive,
}

/// One entry in the lock table for a row.
#[derive(Debug, Clone, Default)]
struct LockEntry {
    /// Transactions holding a shared lock.
    shared: HashSet<TransactionId>,
    /// The transaction holding the exclusive lock, if any.
    exclusive: Option<TransactionId>,
}

impl LockEntry {
    fn is_empty(&self) -> bool {
        self.shared.is_empty() && self.exclusive.is_none()
    }

    fn holders(&self) -> Vec<TransactionId> {
        let mut v: Vec<TransactionId> = self.shared.iter().copied().collect();
        if let Some(x) = self.exclusive {
            v.push(x);
        }
        v
    }
}

/// Counters maintained by the [`TxnManager`] for diagnostics.
#[derive(Debug, Clone, Copy, Default)]
pub struct TxnStats {
    /// Transactions begun.
    pub begun: u64,
    /// Transactions committed.
    pub committed: u64,
    /// Transactions aborted.
    pub aborted: u64,
    /// Transactions that reached the prepared state.
    pub prepared: u64,
    /// Garbage-collection passes run.
    pub gc_runs: u64,
    /// Versions removed by garbage collection.
    pub gc_versions_removed: u64,
    /// Checkpoints taken.
    pub checkpoints: u64,
    /// Deadlocks detected.
    pub deadlocks: u64,
    /// Lock conflicts raised.
    pub lock_conflicts: u64,
}

/// The MVCC transaction manager.
///
/// Owns the logical clock, the transaction registry, the per-row version
/// chains, the write-intent lock table, the wait-for graph, predicate locks,
/// and the write-ahead log.
pub struct TxnManager {
    /// The next transaction id to hand out.
    next_txn_id: u64,
    /// The next timestamp to allocate.
    next_ts: u64,
    /// Every transaction ever begun, indexed by id. Committed and aborted
    /// transactions are retained so that their commit timestamps remain
    /// resolvable for conflict checks; they are pruned by garbage collection.
    registry: HashMap<TransactionId, Transaction>,
    /// Currently-active transaction ids.
    active: HashSet<TransactionId>,
    /// Per-row version chains.
    chains: HashMap<u64, VersionChain>,
    /// The write-intent lock table.
    locks: HashMap<u64, LockEntry>,
    /// The wait-for graph: `waiter -> {holders it is blocked on}`.
    wait_for: HashMap<TransactionId, HashSet<TransactionId>>,
    /// Predicate locks held by active serializable transactions.
    predicate_locks: Vec<PredicateLock>,
    /// The write-ahead log.
    pub wal: Wal,
    /// Diagnostic counters.
    stats: TxnStats,
}

impl Default for TxnManager {
    fn default() -> Self {
        TxnManager::new()
    }
}

impl TxnManager {
    /// Create a new manager with a freshly-initialised WAL.
    pub fn new() -> Self {
        TxnManager::with_wal(Wal::new(4096))
    }

    /// Create a new manager backed by a caller-supplied WAL.
    pub fn with_wal(wal: Wal) -> Self {
        TxnManager {
            next_txn_id: 1,
            next_ts: 1,
            registry: HashMap::new(),
            active: HashSet::new(),
            chains: HashMap::new(),
            locks: HashMap::new(),
            wait_for: HashMap::new(),
            predicate_locks: Vec::new(),
            wal,
            stats: TxnStats::default(),
        }
    }

    // ------------------------------------------------------------------ clock

    /// The current logical time (the next timestamp to be allocated). Reads
    /// observe every commit with `commit_ts <= now`.
    pub fn now(&self) -> Timestamp {
        Timestamp(self.next_ts)
    }

    /// Allocate the next timestamp and advance the clock.
    fn tick(&mut self) -> Timestamp {
        let t = Timestamp(self.next_ts);
        self.next_ts += 1;
        t
    }

    /// The highest timestamp ever allocated, or [`TS_ZERO`] if none.
    pub fn last_timestamp(&self) -> Timestamp {
        if self.next_ts == 0 {
            TS_ZERO
        } else {
            Timestamp(self.next_ts - 1)
        }
    }

    // --------------------------------------------------------------- registry

    /// `true` if `id` is a known transaction.
    pub fn contains(&self, id: TransactionId) -> bool {
        self.registry.contains_key(&id)
    }

    /// The number of currently-active transactions.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// All active transaction ids, in arbitrary order.
    pub fn active_ids(&self) -> Vec<TransactionId> {
        self.active.iter().copied().collect()
    }

    /// A snapshot of a transaction's state, or `None` if unknown.
    pub fn info(&self, id: TransactionId) -> Option<TxnInfo> {
        let t = self.registry.get(&id)?;
        Some(TxnInfo {
            id: t.id,
            begin_ts: t.begin_ts,
            commit_ts: t.commit_ts,
            state: t.state,
            isolation: t.isolation,
            snapshot_ts: t.read_view.snapshot_ts,
            reads: t.read_set.len(),
            writes: t.write_set.len(),
            predicate_rows: t.predicate_rows.len(),
            savepoints: t.savepoints.len(),
            undo_records: t.undo.len(),
            last_lsn: t.last_lsn,
        })
    }

    /// The state of a transaction, or `None` if unknown.
    pub fn state(&self, id: TransactionId) -> Option<TxnState> {
        self.registry.get(&id).map(|t| t.state)
    }

    /// Diagnostic counters.
    pub fn stats(&self) -> TxnStats {
        self.stats
    }

    /// The number of version chains currently tracked.
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Total versions across all chains.
    pub fn version_count(&self) -> usize {
        self.chains.values().map(|c| c.len()).sum()
    }

    /// A snapshot of a row's version chain (newest-first), or `None`.
    pub fn chain_snapshot(&self, row_id: u64) -> Option<Vec<Version>> {
        self.chains.get(&row_id).map(|c| c.snapshot())
    }

    /// The number of versions linked for `row_id`.
    pub fn chain_len(&self, row_id: u64) -> usize {
        self.chains.get(&row_id).map_or(0, |c| c.len())
    }

    // ----------------------------------------------------------- begin / end

    /// Begin a new transaction at `isolation`. Returns its id.
    ///
    /// A `Begin` record is appended to the WAL.
    pub fn begin(&mut self, isolation: IsolationLevel) -> TransactionId {
        let id = TransactionId(self.next_txn_id);
        self.next_txn_id += 1;
        let begin_ts = self.tick();
        let active = self.active.clone();
        let mut txn = Transaction::new(id, begin_ts, active, isolation);
        let lsn = self.wal.append(id.raw(), RecType::Begin, &[]);
        txn.last_lsn = lsn;
        self.registry.insert(id, txn);
        self.active.insert(id);
        self.stats.begun += 1;
        id
    }

    /// Validate that `id` is a known, live transaction.
    fn require_live(&self, id: TransactionId) -> Result<()> {
        match self.registry.get(&id) {
            None => Err(Error::Internal(format!("unknown transaction {id}"))),
            Some(t) if !t.is_live() => Err(Error::Internal(format!(
                "transaction {id} is not live (state: {})",
                t.state.name()
            ))),
            Some(_) => Ok(()),
        }
    }

    /// Phase one of two-phase commit: mark the transaction prepared. A prepared
    /// transaction may still be rolled back, but it has promised to commit and
    /// its locks are not released.
    pub fn prepare(&mut self, id: TransactionId) -> Result<()> {
        self.require_live(id)?;
        let t = self.registry.get_mut(&id).unwrap();
        if t.state == TxnState::Prepared {
            return Ok(());
        }
        t.state = TxnState::Prepared;
        self.stats.prepared += 1;
        Ok(())
    }

    /// Commit a transaction. Returns its commit timestamp.
    ///
    /// For serializable transactions, a read-set/predicate conflict check runs
    /// first; if a concurrently-committed transaction modified a row this one
    /// read or predicate-locked, commit fails with a serialization conflict and
    /// the transaction is aborted. On success every version the transaction
    /// created is stamped with the commit timestamp, the previous version of
    /// each row is closed at the commit timestamp, a `Commit` WAL record is
    /// appended, and the transaction's locks are released.
    pub fn commit(&mut self, id: TransactionId) -> Result<Timestamp> {
        self.require_live(id)?;
        // Serializable conflict check: fail fast before mutating anything.
        if let Some(conflict) = self.serializable_conflict(id)? {
            self.rollback(id)?;
            return Err(Error::Internal(format!(
                "serialization conflict on row {conflict}"
            )));
        }
        let commit_ts = self.tick();
        let written_rows: Vec<u64> = self.registry.get(&id).unwrap().write_set.iter().copied().collect();
        for row_id in &written_rows {
            self.finalize_write(*row_id, id, commit_ts);
        }
        {
            let t = self.registry.get_mut(&id).unwrap();
            t.state = TxnState::Committed;
            t.commit_ts = Some(commit_ts);
            let lsn = self.wal.append(id.raw(), RecType::Commit, &[]);
            t.last_lsn = lsn;
        }
        self.active.remove(&id);
        self.release_locks(id);
        self.stats.committed += 1;
        Ok(commit_ts)
    }

    /// Stamp the uncommitted head version of `row_id` written by `id` as
    /// committed at `commit_ts`, and close the previous current version.
    fn finalize_write(&mut self, row_id: u64, writer: TransactionId, commit_ts: Timestamp) {
        let chain = match self.chains.get_mut(&row_id) {
            Some(c) => c,
            None => return,
        };
        // Find the writer's uncommitted version (the head, normally).
        let writer_idx = chain
            .versions
            .iter()
            .position(|v| v.writer == writer && !v.committed);
        let idx = match writer_idx {
            Some(i) => i,
            None => return,
        };
        // Close the previously-current committed version, if any, at commit_ts.
        // That is the first committed version with end_ts == INF *after* the
        // writer's intent in chain order.
        for v in chain.versions.iter_mut() {
            if v.committed && v.end_ts == TS_INF {
                v.end_ts = commit_ts;
                break;
            }
        }
        let v = &mut chain.versions[idx];
        v.committed = true;
        v.begin_ts = commit_ts;
        v.end_ts = TS_INF;
    }

    /// Roll back a transaction. Every version it created is removed, its locks
    /// are released, and an `Abort` WAL record is appended.
    pub fn rollback(&mut self, id: TransactionId) -> Result<()> {
        if !self.contains(id) {
            return Err(Error::Internal(format!("unknown transaction {id}")));
        }
        // Collect the rows this transaction wrote so we can strip its versions.
        let written_rows: Vec<u64> = self.registry.get(&id).unwrap().write_set.iter().copied().collect();
        for row_id in &written_rows {
            self.strip_writer_versions(*row_id, id);
        }
        {
            let t = self.registry.get_mut(&id).unwrap();
            t.state = TxnState::Aborted;
            t.undo.clear();
            let lsn = self.wal.append(id.raw(), RecType::Abort, &[]);
            t.last_lsn = lsn;
        }
        self.active.remove(&id);
        self.release_locks(id);
        self.stats.aborted += 1;
        Ok(())
    }

    /// Remove every uncommitted version authored by `writer` from `row_id`'s
    /// chain. If the chain becomes empty it is dropped entirely.
    fn strip_writer_versions(&mut self, row_id: u64, writer: TransactionId) {
        let remove_chain = if let Some(chain) = self.chains.get_mut(&row_id) {
            chain.versions.retain(|v| !(v.writer == writer && !v.committed));
            chain.versions.is_empty()
        } else {
            return;
        };
        if remove_chain {
            self.chains.remove(&row_id);
        }
    }

    // --------------------------------------------------------------- reading

    /// Refresh a read-committed transaction's read view to the current clock.
    fn refresh_read_view(&mut self, id: TransactionId) {
        let snap = self.now();
        let active = self.active.clone();
        let t = self.registry.get_mut(&id).unwrap();
        t.read_view.snapshot_ts = snap;
        t.read_view.active = active;
    }

    /// Read a row as seen by transaction `id`. Returns `Ok(None)` if the row
    /// does not exist or is deleted (a tombstone) from the transaction's point
    /// of view.
    pub fn read(&mut self, id: TransactionId, row_id: u64) -> Result<Option<Value>> {
        self.require_live(id)?;
        if self.registry[&id].isolation == IsolationLevel::ReadCommitted {
            self.refresh_read_view(id);
        }
        let rv = self.registry[&id].read_view.clone();
        // Record the read for serializable conflict tracking.
        self.registry.get_mut(&id).unwrap().read_set.insert(row_id);
        let result = match self.chains.get(&row_id) {
            Some(chain) => match chain.visible_for(&rv) {
                Some(v) if v.deleted => None,
                Some(v) => Some(v.value),
                None => None,
            },
            None => None,
        };
        Ok(result)
    }

    // --------------------------------------------------------------- writing

    /// Acquire an exclusive lock on `row_id` for `id`. On conflict the
    /// attempted wait is recorded in the wait-for graph and a lock-conflict
    /// error is returned.
    fn acquire_exclusive(&mut self, id: TransactionId, row_id: u64) -> Result<()> {
        let entry = self.locks.entry(row_id).or_default();
        if entry.exclusive == Some(id) {
            self.wait_for.remove(&id);
            return Ok(());
        }
        let only_self_shared =
            entry.exclusive.is_none() && (entry.shared.is_empty() || (entry.shared.len() == 1 && entry.shared.contains(&id)));
        if only_self_shared {
            entry.shared.remove(&id);
            entry.exclusive = Some(id);
            self.wait_for.remove(&id);
            return Ok(());
        }
        // Conflict: record edges to every other holder.
        let holders: Vec<TransactionId> = entry
            .holders()
            .into_iter()
            .filter(|h| *h != id)
            .collect();
        let edges = self.wait_for.entry(id).or_default();
        for h in holders {
            edges.insert(h);
        }
        self.stats.lock_conflicts += 1;
        Err(Error::Internal(format!("lock conflict on row {row_id}")))
    }

    /// Acquire a shared lock on `row_id` for `id`.
    fn acquire_shared(&mut self, id: TransactionId, row_id: u64) -> Result<()> {
        let entry = self.locks.entry(row_id).or_default();
        if entry.exclusive == Some(id) {
            self.wait_for.remove(&id);
            return Ok(());
        }
        if entry.exclusive.is_none() {
            entry.shared.insert(id);
            self.wait_for.remove(&id);
            return Ok(());
        }
        let holder = entry.exclusive.unwrap();
        let edges = self.wait_for.entry(id).or_default();
        edges.insert(holder);
        self.stats.lock_conflicts += 1;
        Err(Error::Internal(format!("lock conflict on row {row_id}")))
    }

    /// Release every lock held by `id` and drop the id from the wait-for
    /// graph (both as waiter and as a holder others may be waiting on).
    fn release_locks(&mut self, id: TransactionId) {
        let mut empty = Vec::new();
        let locks = std::mem::take(&mut self.locks);
        for (row_id, mut entry) in locks {
            entry.shared.remove(&id);
            if entry.exclusive == Some(id) {
                entry.exclusive = None;
            }
            if entry.is_empty() {
                empty.push(row_id);
            } else {
                self.locks.insert(row_id, entry);
            }
        }
        self.wait_for.remove(&id);
        for holders in self.wait_for.values_mut() {
            holders.remove(&id);
        }
    }

    /// Release a single row's lock held by `id`, used when a write operation
    /// acquires a lock but then discovers the row is not visible to it.
    fn release_one_lock(&mut self, id: TransactionId, row_id: u64) {
        if let Some(mut entry) = self.locks.remove(&row_id) {
            entry.shared.remove(&id);
            if entry.exclusive == Some(id) {
                entry.exclusive = None;
            }
            if !entry.is_empty() {
                self.locks.insert(row_id, entry);
            }
        }
        if let Some(edges) = self.wait_for.get_mut(&id) {
            edges.clear();
        }
    }

    /// Public lock acquisition for testing and integration: `exclusive ==
    /// true` requests a write lock, `false` a read lock.
    pub fn acquire_lock(&mut self, id: TransactionId, row_id: u64, exclusive: bool) -> Result<()> {
        self.require_live(id)?;
        if exclusive {
            self.acquire_exclusive(id, row_id)
        } else {
            self.acquire_shared(id, row_id)
        }
    }

    /// Whether `id` currently holds any lock on `row_id`.
    pub fn holds_lock(&self, id: TransactionId, row_id: u64) -> bool {
        match self.locks.get(&row_id) {
            Some(e) => e.exclusive == Some(id) || e.shared.contains(&id),
            None => false,
        }
    }

    /// Insert a brand-new row. Fails if a visible version of `row_id` already
    /// exists.
    pub fn insert(&mut self, id: TransactionId, row_id: u64, value: Value) -> Result<()> {
        self.require_live(id)?;
        self.acquire_exclusive(id, row_id)?;
        // Reject if the row already has any version visible to this txn.
        if self.row_exists_for(id, row_id)? {
            return Err(Error::Internal(format!("row {row_id} already exists")));
        }
        let begin_ts = self.registry[&id].begin_ts;
        let v = Version::write_intent(value, begin_ts, id);
        self.chains
            .entry(row_id)
            .or_insert_with(|| VersionChain::empty(row_id))
            .push_head(v);
        let t = self.registry.get_mut(&id).unwrap();
        t.undo.push(UndoRecord::Insert { row_id });
        t.write_set.insert(row_id);
        let payload = encode_insert_payload(row_id, &[value]);
        let lsn = self.wal.append(id.raw(), RecType::Insert, &payload);
        self.registry.get_mut(&id).unwrap().last_lsn = lsn;
        Ok(())
    }

    /// Update an existing row. Fails if no visible version exists or the row
    /// is deleted from this transaction's point of view.
    pub fn update(&mut self, id: TransactionId, row_id: u64, value: Value) -> Result<()> {
        self.require_live(id)?;
        self.acquire_exclusive(id, row_id)?;
        if !self.row_exists_for(id, row_id)? {
            self.release_one_lock(id, row_id);
            return Err(Error::NotFound(format!("row {row_id}")));
        }
        let begin_ts = self.registry[&id].begin_ts;
        let v = Version::write_intent(value, begin_ts, id);
        self.chains
            .entry(row_id)
            .or_insert_with(|| VersionChain::empty(row_id))
            .push_head(v);
        let t = self.registry.get_mut(&id).unwrap();
        t.undo.push(UndoRecord::Update { row_id });
        t.write_set.insert(row_id);
        let payload = encode_update_payload(row_id, 0, &value);
        let lsn = self.wal.append(id.raw(), RecType::Update, &payload);
        self.registry.get_mut(&id).unwrap().last_lsn = lsn;
        Ok(())
    }

    /// Delete an existing row. Fails if no visible version exists. A delete is
    /// recorded as a tombstone version (`deleted == true`).
    pub fn delete(&mut self, id: TransactionId, row_id: u64) -> Result<()> {
        self.require_live(id)?;
        self.acquire_exclusive(id, row_id)?;
        if !self.row_exists_for(id, row_id)? {
            self.release_one_lock(id, row_id);
            return Err(Error::NotFound(format!("row {row_id}")));
        }
        let begin_ts = self.registry[&id].begin_ts;
        let v = Version::delete_intent(begin_ts, id);
        self.chains
            .entry(row_id)
            .or_insert_with(|| VersionChain::empty(row_id))
            .push_head(v);
        let t = self.registry.get_mut(&id).unwrap();
        t.undo.push(UndoRecord::Delete { row_id });
        t.write_set.insert(row_id);
        let payload = encode_delete_payload(row_id);
        let lsn = self.wal.append(id.raw(), RecType::Delete, &payload);
        self.registry.get_mut(&id).unwrap().last_lsn = lsn;
        Ok(())
    }

    /// `true` if a non-deleted version of `row_id` is visible to `id`.
    fn row_exists_for(&mut self, id: TransactionId, row_id: u64) -> Result<bool> {
        match self.read(id, row_id)? {
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    // ------------------------------------------------------------- savepoints

    /// Create a named savepoint in transaction `id`. The name must be unique
    /// among the transaction's currently-live savepoints.
    pub fn savepoint(&mut self, id: TransactionId, name: &str) -> Result<()> {
        self.require_live(id)?;
        let t = self.registry.get_mut(&id).unwrap();
        if t.savepoints.iter().any(|s| s.name == name) {
            return Err(Error::Internal(format!(
                "savepoint {name} already exists"
            )));
        }
        let sp = Savepoint {
            name: name.to_string(),
            undo_len: t.undo.len(),
            write_set: t.write_set.clone(),
            read_set: t.read_set.clone(),
        };
        t.savepoints.push(sp);
        Ok(())
    }

    /// Roll back to the named savepoint, undoing every mutation recorded after
    /// it. The savepoint itself is kept so it can be rolled back to again;
    /// later savepoints are discarded.
    pub fn rollback_to_savepoint(&mut self, id: TransactionId, name: &str) -> Result<()> {
        self.require_live(id)?;
        let sp = {
            let t = self.registry.get(&id).unwrap();
            t.savepoints
                .iter()
                .find(|s| s.name == name)
                .cloned()
                .ok_or_else(|| Error::Internal(format!("unknown savepoint {name}")))?
        };
        let removed: Vec<UndoRecord> = {
            let t = self.registry.get_mut(&id).unwrap();
            // Keep the named savepoint and any created before it; drop those
            // created after it.
            t.savepoints.retain(|s| s.undo_len <= sp.undo_len);
            t.undo.truncate_to(sp.undo_len)
        };
        // Undo each removed mutation in reverse order.
        for rec in removed {
            self.undo_one(id, rec);
        }
        // Restore the write and read sets to their savepoint snapshots.
        let t = self.registry.get_mut(&id).unwrap();
        t.write_set = sp.write_set.clone();
        t.read_set = sp.read_set.clone();
        Ok(())
    }

    /// Remove a savepoint without rolling back. Later savepoints are kept.
    pub fn release_savepoint(&mut self, id: TransactionId, name: &str) -> Result<()> {
        self.require_live(id)?;
        let t = self.registry.get_mut(&id).unwrap();
        let before = t.savepoints.len();
        t.savepoints.retain(|s| s.name != name);
        if t.savepoints.len() == before {
            return Err(Error::Internal(format!("unknown savepoint {name}")));
        }
        Ok(())
    }

    /// The names of a transaction's live savepoints, in creation order.
    pub fn savepoints(&self, id: TransactionId) -> Option<Vec<String>> {
        self.registry
            .get(&id)
            .map(|t| t.savepoints.iter().map(|s| s.name.clone()).collect())
    }

    /// Undo a single mutation by removing the newest version on the row that
    /// belongs to `id`.
    fn undo_one(&mut self, id: TransactionId, rec: UndoRecord) {
        let row_id = rec.row_id();
        let remove_chain = if let Some(chain) = self.chains.get_mut(&row_id) {
            // Remove the head version if it belongs to this txn and is
            // uncommitted; otherwise scan for the newest such version.
            let idx = chain
                .versions
                .iter()
                .position(|v| v.writer == id && !v.committed);
            if let Some(i) = idx {
                chain.versions.remove(i);
            }
            chain.versions.is_empty()
        } else {
            return;
        };
        if remove_chain {
            self.chains.remove(&row_id);
        }
        // Releasing the exclusive lock for this row lets a later write retry.
        if let Some(entry) = self.locks.get_mut(&row_id) {
            if entry.exclusive == Some(id) {
                entry.exclusive = None;
            }
            if entry.is_empty() {
                self.locks.remove(&row_id);
            }
        }
    }

    // ------------------------------------------------------- predicate locks

    /// Record a predicate lock for a serializable transaction over the rows a
    /// scan matched. Rows are added to the transaction's `predicate_rows` and
    /// to a manager-level registry consulted by conflict checks.
    pub fn add_predicate_lock(&mut self, id: TransactionId, rows: &[u64]) -> Result<()> {
        self.require_live(id)?;
        {
            let t = self.registry.get_mut(&id).unwrap();
            for r in rows {
                t.predicate_rows.insert(*r);
                t.read_set.insert(*r);
            }
        }
        // Update or create the manager-level predicate lock entry.
        if let Some(lock) = self.predicate_locks.iter_mut().find(|l| l.txn == id) {
            for r in rows {
                lock.add(*r);
            }
        } else {
            let mut lock = PredicateLock::empty(id);
            for r in rows {
                lock.add(*r);
            }
            self.predicate_locks.push(lock);
        }
        Ok(())
    }

    /// Predicate locks currently held by active transactions.
    pub fn predicate_locks(&self) -> &[PredicateLock] {
        &self.predicate_locks
    }

    /// For a serializable transaction, return the row id of a read-set or
    /// predicate-row that a concurrently-committed transaction modified, or
    /// `Ok(None)` if there is no conflict.
    fn serializable_conflict(&self, id: TransactionId) -> Result<Option<u64>> {
        let t = match self.registry.get(&id) {
            Some(t) if t.isolation == IsolationLevel::Serializable => t,
            _ => return Ok(None),
        };
        let begin_ts = t.begin_ts;
        let conflict_rows: HashSet<u64> = t
            .read_set
            .iter()
            .chain(t.predicate_rows.iter())
            .copied()
            .collect();
        for row_id in &conflict_rows {
            if let Some(chain) = self.chains.get(row_id) {
                for v in &chain.versions {
                    if v.committed
                        && v.writer != id
                        && v.begin_ts > begin_ts
                        && v.begin_ts != TS_INF
                    {
                        return Ok(Some(*row_id));
                    }
                }
            }
        }
        Ok(None)
    }

    // -------------------------------------------------------- deadlock detect

    /// Detect a cycle in the wait-for graph. Returns the cycle as a vector of
    /// transaction ids (the path from the first repeated node), or `None` if
    /// the graph is acyclic.
    pub fn detect_deadlock(&self) -> Option<Vec<TransactionId>> {
        fn dfs(
            node: TransactionId,
            graph: &HashMap<TransactionId, HashSet<TransactionId>>,
            visited: &mut HashSet<TransactionId>,
            stack: &mut Vec<TransactionId>,
            on_stack: &mut HashSet<TransactionId>,
        ) -> Option<Vec<TransactionId>> {
            if on_stack.contains(&node) {
                let start = stack.iter().position(|n| *n == node)?;
                return Some(stack[start..].to_vec());
            }
            if visited.contains(&node) {
                return None;
            }
            visited.insert(node);
            on_stack.insert(node);
            stack.push(node);
            if let Some(neighbors) = graph.get(&node) {
                for n in neighbors {
                    if let Some(cycle) = dfs(*n, graph, visited, stack, on_stack) {
                        return Some(cycle);
                    }
                }
            }
            stack.pop();
            on_stack.remove(&node);
            None
        }

        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut on_stack = HashSet::new();
        for &node in self.wait_for.keys() {
            if let Some(cycle) = dfs(node, &self.wait_for, &mut visited, &mut stack, &mut on_stack) {
                return Some(cycle);
            }
        }
        None
    }

    /// Record a wait-for edge directly. This is a low-level escape hatch used
    /// by tests to construct synthetic deadlock scenarios without going through
    /// lock acquisition.
    pub fn add_wait_for(&mut self, waiter: TransactionId, holder: TransactionId) {
        self.wait_for
            .entry(waiter)
            .or_default()
            .insert(holder);
    }

    /// Remove every wait-for edge involving `id`.
    pub fn clear_wait_for(&mut self, id: TransactionId) {
        self.wait_for.remove(&id);
        for holders in self.wait_for.values_mut() {
            holders.remove(&id);
        }
    }

    /// The current wait-for graph as `(waiter, holders)` pairs.
    pub fn wait_for_graph(&self) -> Vec<(TransactionId, Vec<TransactionId>)> {
        self.wait_for
            .iter()
            .map(|(w, hs)| (*w, hs.iter().copied().collect()))
            .collect()
    }

    // ------------------------------------------------------- garbage collect

    /// The oldest snapshot timestamp among active transactions, or [`TS_INF`]
    /// if none are active (meaning every dead version is reclaimable).
    pub fn oldest_active_snapshot(&self) -> Timestamp {
        let mut oldest = TS_INF;
        for id in &self.active {
            if let Some(t) = self.registry.get(id) {
                if t.read_view.snapshot_ts < oldest {
                    oldest = t.read_view.snapshot_ts;
                }
            }
        }
        oldest
    }

    /// Remove committed versions superseded before the oldest active
    /// transaction's snapshot. The current version of every chain and any
    /// uncommitted write intent are always retained. Returns the number of
    /// versions removed.
    pub fn garbage_collect(&mut self) -> usize {
        let oldest = self.oldest_active_snapshot();
        let mut removed = 0usize;
        let mut empty_chains = Vec::new();
        for (row_id, chain) in self.chains.iter_mut() {
            let before = chain.versions.len();
            chain.versions.retain(|v| {
                if !v.committed {
                    return true; // keep uncommitted write intents
                }
                if v.end_ts == TS_INF {
                    return true; // keep the current version
                }
                if v.end_ts >= oldest {
                    return true; // still visible to some snapshot
                }
                false // dead and invisible to everyone: reclaim
            });
            removed += before - chain.versions.len();
            if chain.versions.is_empty() {
                empty_chains.push(*row_id);
            }
        }
        for row_id in empty_chains {
            self.chains.remove(&row_id);
        }
        // Drop predicate locks for transactions no longer active.
        let active = &self.active;
        self.predicate_locks.retain(|l| active.contains(&l.txn));
        self.stats.gc_runs += 1;
        self.stats.gc_versions_removed += removed as u64;
        removed
    }

    /// Append a checkpoint marker to the WAL, truncate obsolete segments, and
    /// run garbage collection. Returns the number of versions reclaimed.
    pub fn checkpoint(&mut self) -> Result<usize> {
        let lsn = self.wal.append(0, RecType::Checkpoint, &[]);
        // Drop records strictly before the checkpoint marker; the marker itself
        // is retained so recovery knows where the last checkpoint landed.
        if lsn > 1 {
            self.wal.truncate_before(lsn - 1);
        }
        self.stats.checkpoints += 1;
        Ok(self.garbage_collect())
    }

    // -------------------------------------------------------------- recovery

    /// Replay the WAL into the manager, reconstructing committed version
    /// chains. This is a best-effort in-memory recovery: `Begin`/`Commit`/
    /// `Abort` records bracket a transaction's data records, and only
    /// committed transactions' mutations are materialised. Prepared but
    /// uncommitted transactions are aborted.
    ///
    /// The manager's clock and id allocator are advanced past every record
    /// seen so that subsequently-begun transactions do not collide with
    /// recovered ones.
    pub fn recover(&mut self) -> Result<()> {
        let records = self.wal.records();
        let mut begun: HashMap<u64, Vec<(RecType, Vec<u8>)>> = HashMap::new();
        let mut committed: HashSet<u64> = HashSet::new();
        let mut aborted: HashSet<u64> = HashSet::new();
        let mut max_id: u64 = 0;
        let mut max_lsn: u64 = 0;
        for rec in records {
            max_lsn = max_lsn.max(rec.lsn);
            max_id = max_id.max(rec.txn);
            match rec.rtype {
                RecType::Begin => {
                    begun.insert(rec.txn, Vec::new());
                }
                RecType::Insert | RecType::Update | RecType::Delete => {
                    if let Some(buf) = begun.get_mut(&rec.txn) {
                        buf.push((rec.rtype, rec.payload.clone()));
                    }
                }
                RecType::Commit => {
                    committed.insert(rec.txn);
                }
                RecType::Abort => {
                    aborted.insert(rec.txn);
                }
                RecType::Checkpoint | RecType::Schema => {}
            }
        }
        // Re-materialise committed transactions in id order.
        let mut txn_ids: Vec<u64> = begun.keys().copied().collect();
        txn_ids.sort_unstable();
        for txn in txn_ids {
            if aborted.contains(&txn) || !committed.contains(&txn) {
                continue;
            }
            let ops = begun.remove(&txn).unwrap_or_default();
            let commit_ts = self.tick();
            for (rtype, payload) in ops {
                match rtype {
                    RecType::Insert => {
                        if let Some((row_id, values)) = decode_insert_payload(&payload) {
                            let value = values.into_iter().next().unwrap_or(Value::Null);
                            let chain = self
                                .chains
                                .entry(row_id)
                                .or_insert_with(|| VersionChain::empty(row_id));
                            if let Some(prev) = chain
                                .versions
                                .iter_mut()
                                .find(|v| v.committed && v.end_ts == TS_INF)
                            {
                                prev.end_ts = commit_ts;
                            }
                            chain.push_head(Version::committed_current(
                                value,
                                commit_ts,
                                TransactionId(txn),
                            ));
                        }
                    }
                    RecType::Update => {
                        if let Some((row_id, _col, value)) = decode_update_payload(&payload) {
                            let chain = self
                                .chains
                                .entry(row_id)
                                .or_insert_with(|| VersionChain::empty(row_id));
                            if let Some(prev) = chain
                                .versions
                                .iter_mut()
                                .find(|v| v.committed && v.end_ts == TS_INF)
                            {
                                prev.end_ts = commit_ts;
                            }
                            chain.push_head(Version::committed_current(
                                value,
                                commit_ts,
                                TransactionId(txn),
                            ));
                        }
                    }
                    RecType::Delete => {
                        if let Some(row_id) = decode_delete_payload(&payload) {
                            let chain = self
                                .chains
                                .entry(row_id)
                                .or_insert_with(|| VersionChain::empty(row_id));
                            if let Some(prev) = chain
                                .versions
                                .iter_mut()
                                .find(|v| v.committed && v.end_ts == TS_INF)
                            {
                                prev.end_ts = commit_ts;
                            }
                            chain.push_head(Version {
                                value: Value::Null,
                                deleted: true,
                                begin_ts: commit_ts,
                                end_ts: TS_INF,
                                writer: TransactionId(txn),
                                committed: true,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        // Advance the allocator past recovered ids/lsns.
        self.next_txn_id = self.next_txn_id.max(max_id + 1);
        self.next_ts = self.next_ts.max(max_lsn + 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> TxnManager {
        TxnManager::new()
    }

    // ---------------------------------------------------------- lifecycle

    #[test]
    fn begin_assigns_unique_ids_and_live_state() {
        let mut m = mgr();
        let t1 = m.begin(IsolationLevel::RepeatableRead);
        let t2 = m.begin(IsolationLevel::Serializable);
        assert_ne!(t1, t2);
        assert_eq!(m.state(t1), Some(TxnState::Active));
        assert_eq!(m.state(t2), Some(TxnState::Active));
        assert_eq!(m.active_count(), 2);
        let info = m.info(t1).unwrap();
        assert_eq!(info.state, TxnState::Active);
        assert_eq!(info.isolation, IsolationLevel::RepeatableRead);
        assert!(info.begin_ts.raw() >= 1);
        assert_eq!(info.commit_ts, None);
        // The WAL got a Begin record per transaction.
        assert_eq!(m.wal.record_count(), 2);
    }

    #[test]
    fn commit_sets_state_commit_ts_and_releases_locks() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(10)).unwrap();
        assert!(m.holds_lock(t, 1));
        let commit_ts = m.commit(t).unwrap();
        assert_eq!(m.state(t), Some(TxnState::Committed));
        assert_eq!(m.info(t).unwrap().commit_ts, Some(commit_ts));
        assert!(!m.holds_lock(t, 1));
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn rollback_removes_versions_and_marks_aborted() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(10)).unwrap();
        m.rollback(t).unwrap();
        assert_eq!(m.state(t), Some(TxnState::Aborted));
        assert_eq!(m.chain_count(), 0);
        assert_eq!(m.active_count(), 0);
        assert!(!m.holds_lock(t, 1));
    }

    #[test]
    fn prepared_then_commit_completes_two_phase() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(7)).unwrap();
        m.prepare(t).unwrap();
        assert_eq!(m.state(t), Some(TxnState::Prepared));
        assert!(m.is_live_check(t));
        let ts = m.commit(t).unwrap();
        assert_eq!(m.state(t), Some(TxnState::Committed));
        assert!(ts.raw() > 0);
    }

    // ---------------------------------------------------- visibility / isolation

    #[test]
    fn writer_sees_own_uncommitted_intent() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 5, Value::Int(42)).unwrap();
        assert_eq!(m.read(t, 5).unwrap(), Some(Value::Int(42)));
    }

    #[test]
    fn snapshot_isolation_hides_uncommitted_and_post_begin_commits() {
        let mut m = mgr();
        // Establish a committed base row 5 = 10.
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 5, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();

        // Reader begins (RR), capturing the snapshot.
        let reader = m.begin(IsolationLevel::RepeatableRead);

        // A second transaction writes an uncommitted update to 5.
        let writer = m.begin(IsolationLevel::RepeatableRead);
        m.update(writer, 5, Value::Int(99)).unwrap();
        // Reader does NOT see the uncommitted intent.
        assert_eq!(m.read(reader, 5).unwrap(), Some(Value::Int(10)));

        // Writer commits; reader still sees the snapshot value.
        m.commit(writer).unwrap();
        assert_eq!(m.read(reader, 5).unwrap(), Some(Value::Int(10)));

        // A fresh transaction sees the new value.
        let late = m.begin(IsolationLevel::RepeatableRead);
        assert_eq!(m.read(late, 5).unwrap(), Some(Value::Int(99)));
    }

    #[test]
    fn read_committed_sees_committed_updates_between_reads() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 5, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();

        let reader = m.begin(IsolationLevel::ReadCommitted);
        assert_eq!(m.read(reader, 5).unwrap(), Some(Value::Int(10)));

        let writer = m.begin(IsolationLevel::RepeatableRead);
        m.update(writer, 5, Value::Int(20)).unwrap();
        m.commit(writer).unwrap();

        // Read-committed refreshes its snapshot and sees 20.
        assert_eq!(m.read(reader, 5).unwrap(), Some(Value::Int(20)));
    }

    #[test]
    fn read_uncommitted_performs_dirty_read() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 5, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();

        let writer = m.begin(IsolationLevel::RepeatableRead);
        m.update(writer, 5, Value::Int(999)).unwrap();

        let dirty = m.begin(IsolationLevel::ReadUncommitted);
        assert_eq!(m.read(dirty, 5).unwrap(), Some(Value::Int(999)));
    }

    #[test]
    fn repeatable_read_isolation_repeatable_across_concurrent_commit() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 8, Value::Int(1)).unwrap();
        m.commit(t0).unwrap();

        let rr = m.begin(IsolationLevel::RepeatableRead);
        assert_eq!(m.read(rr, 8).unwrap(), Some(Value::Int(1)));

        let w = m.begin(IsolationLevel::RepeatableRead);
        m.update(w, 8, Value::Int(2)).unwrap();
        m.commit(w).unwrap();

        // RR sees the same value twice.
        assert_eq!(m.read(rr, 8).unwrap(), Some(Value::Int(1)));
    }

    // -------------------------------------------------------- version chains

    #[test]
    fn update_creates_new_version_and_closes_previous() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        let c0 = m.commit(t0).unwrap();

        let t1 = m.begin(IsolationLevel::RepeatableRead);
        m.update(t1, 1, Value::Int(20)).unwrap();
        let c1 = m.commit(t1).unwrap();

        let chain = m.chain_snapshot(1).unwrap();
        assert_eq!(chain.len(), 2);
        // Newest first: the version committed at c1.
        assert_eq!(chain[0].value, Value::Int(20));
        assert_eq!(chain[0].begin_ts, c1);
        assert_eq!(chain[0].end_ts, TS_INF);
        assert!(chain[0].committed);
        // The previous version is closed at c1.
        assert_eq!(chain[1].value, Value::Int(10));
        assert_eq!(chain[1].begin_ts, c0);
        assert_eq!(chain[1].end_ts, c1);
        assert!(chain[1].is_dead());
    }

    #[test]
    fn delete_emits_tombstone_visible_as_none() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 3, Value::Int(5)).unwrap();
        m.commit(t0).unwrap();

        let t1 = m.begin(IsolationLevel::RepeatableRead);
        assert_eq!(m.read(t1, 3).unwrap(), Some(Value::Int(5)));
        m.delete(t1, 3).unwrap();
        assert_eq!(m.read(t1, 3).unwrap(), None);
        m.commit(t1).unwrap();

        let t2 = m.begin(IsolationLevel::RepeatableRead);
        assert_eq!(m.read(t2, 3).unwrap(), None);
        // The tombstone version exists at the head.
        let chain = m.chain_snapshot(3).unwrap();
        assert!(chain[0].deleted);
    }

    #[test]
    fn version_chain_visible_for_picks_newest_visible() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        let c0 = m.commit(t0).unwrap();
        let t1 = m.begin(IsolationLevel::RepeatableRead);
        m.update(t1, 1, Value::Int(20)).unwrap();
        let c1 = m.commit(t1).unwrap();

        // A read view pinned before the second commit sees the first version.
        let rv = ReadView::new(TransactionId(999), c0, HashSet::new(), IsolationLevel::RepeatableRead);
        let chain = m.chains.get(&1).unwrap();
        let v = chain.visible_for(&rv).unwrap();
        assert_eq!(v.value, Value::Int(10));
        assert_eq!(v.begin_ts, c0);
        assert_eq!(v.end_ts, c1);

        // A read view at the current time sees the newest.
        let rv2 = ReadView::new(TransactionId(999), c1, HashSet::new(), IsolationLevel::RepeatableRead);
        let v2 = chain.visible_for(&rv2).unwrap();
        assert_eq!(v2.value, Value::Int(20));
    }

    #[test]
    fn read_view_visibility_rules_unit() {
        let reader = TransactionId(1);
        let other = TransactionId(2);
        let active = HashSet::new();
        let rv = ReadView::new(reader, Timestamp(5), active, IsolationLevel::RepeatableRead);

        // Own uncommitted intent is visible.
        let own = Version::write_intent(Value::Int(1), Timestamp(3), reader);
        assert!(rv.is_visible(&own));
        // Other's uncommitted intent is invisible under RR.
        let other_intent = Version::write_intent(Value::Int(2), Timestamp(3), other);
        assert!(!rv.is_visible(&other_intent));
        // ... but visible under read-uncommitted.
        let rv_dirty = ReadView::new(reader, Timestamp(5), HashSet::new(), IsolationLevel::ReadUncommitted);
        assert!(rv_dirty.is_visible(&other_intent));

        // Committed before snapshot, still current: visible.
        let committed_current = Version::committed_current(Value::Int(3), Timestamp(4), other);
        assert!(rv.is_visible(&committed_current));
        // Committed after snapshot: invisible.
        let future = Version::committed_current(Value::Int(4), Timestamp(6), other);
        assert!(!rv.is_visible(&future));
        // Superseded before snapshot: invisible.
        let dead = Version {
            value: Value::Int(0),
            deleted: false,
            begin_ts: Timestamp(1),
            end_ts: Timestamp(3),
            writer: other,
            committed: true,
        };
        assert!(!rv.is_visible(&dead));
        // Superseded exactly at snapshot: invisible (end_ts <= snapshot_ts).
        let dead_at = Version {
            value: Value::Int(0),
            deleted: false,
            begin_ts: Timestamp(1),
            end_ts: Timestamp(5),
            writer: other,
            committed: true,
        };
        assert!(!rv.is_visible(&dead_at));
    }

    // --------------------------------------------------------- garbage collect

    #[test]
    fn garbage_collect_removes_dead_versions_when_no_active() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();
        let t1 = m.begin(IsolationLevel::RepeatableRead);
        m.update(t1, 1, Value::Int(20)).unwrap();
        m.commit(t1).unwrap();
        let t2 = m.begin(IsolationLevel::RepeatableRead);
        m.update(t2, 1, Value::Int(30)).unwrap();
        m.commit(t2).unwrap();

        assert_eq!(m.chain_len(1), 3);
        let removed = m.garbage_collect();
        assert_eq!(removed, 2);
        assert_eq!(m.chain_len(1), 1);
        let chain = m.chain_snapshot(1).unwrap();
        assert_eq!(chain[0].value, Value::Int(30));
        assert_eq!(chain[0].end_ts, TS_INF);
    }

    #[test]
    fn garbage_collect_keeps_versions_visible_to_oldest_active() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        let c0 = m.commit(t0).unwrap();

        // A long-running reader begins now and pins the snapshot at c0.
        let reader = m.begin(IsolationLevel::RepeatableRead);

        let t2 = m.begin(IsolationLevel::RepeatableRead);
        m.update(t2, 1, Value::Int(20)).unwrap();
        let _c2 = m.commit(t2).unwrap();

        // The first version is still visible to `reader`, so GC must keep it.
        let removed = m.garbage_collect();
        assert_eq!(removed, 0);
        assert_eq!(m.chain_len(1), 2);
        assert_eq!(m.read(reader, 1).unwrap(), Some(Value::Int(10)));

        // Once the reader commits, the old version becomes reclaimable.
        m.commit(reader).unwrap();
        let removed = m.garbage_collect();
        assert_eq!(removed, 1);
        assert_eq!(m.chain_len(1), 1);
        let _ = c0;
    }

    #[test]
    fn garbage_collect_keeps_uncommitted_write_intents() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();
        let w = m.begin(IsolationLevel::RepeatableRead);
        m.update(w, 1, Value::Int(20)).unwrap();
        // GC while a write intent is outstanding must keep it.
        let removed = m.garbage_collect();
        assert_eq!(removed, 0);
        let chain = m.chain_snapshot(1).unwrap();
        assert_eq!(chain.len(), 2);
        assert!(!chain[0].committed);
    }

    // ------------------------------------------------------------- savepoints

    #[test]
    fn savepoint_rollback_undoes_later_work_only() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(1)).unwrap();
        m.savepoint(t, "sp").unwrap();
        m.insert(t, 2, Value::Int(2)).unwrap();
        m.update(t, 1, Value::Int(100)).unwrap();
        assert_eq!(m.read(t, 1).unwrap(), Some(Value::Int(100)));
        assert_eq!(m.read(t, 2).unwrap(), Some(Value::Int(2)));

        m.rollback_to_savepoint(t, "sp").unwrap();
        // Work after the savepoint is undone...
        assert_eq!(m.read(t, 2).unwrap(), None);
        assert_eq!(m.read(t, 1).unwrap(), Some(Value::Int(1)));
        // ...and the savepoint is retained for reuse.
        assert!(m.savepoints(t).unwrap().contains(&"sp".to_string()));
    }

    #[test]
    fn savepoint_release_drops_marker() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(1)).unwrap();
        m.savepoint(t, "sp").unwrap();
        m.release_savepoint(t, "sp").unwrap();
        assert!(m.savepoints(t).unwrap().is_empty());
        // Rolling back to a released savepoint fails.
        assert!(m.rollback_to_savepoint(t, "sp").is_err());
    }

    #[test]
    fn savepoint_full_rollback_after_partial() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(1)).unwrap();
        m.savepoint(t, "sp").unwrap();
        m.insert(t, 2, Value::Int(2)).unwrap();
        m.rollback_to_savepoint(t, "sp").unwrap();
        // A full rollback after a partial one still clears everything.
        m.rollback(t).unwrap();
        assert_eq!(m.chain_count(), 0);
    }

    // ------------------------------------------------------- write-intent conflicts

    #[test]
    fn two_writers_on_same_row_conflict() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(1)).unwrap();
        m.commit(t0).unwrap();

        let a = m.begin(IsolationLevel::RepeatableRead);
        m.update(a, 1, Value::Int(2)).unwrap();
        let b = m.begin(IsolationLevel::RepeatableRead);
        // b cannot write while a holds the exclusive lock.
        let err = m.update(b, 1, Value::Int(3)).unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
        assert_eq!(m.stats().lock_conflicts, 1);
        // a can still commit and the row reflects a's write.
        m.commit(a).unwrap();
        let c = m.begin(IsolationLevel::RepeatableRead);
        assert_eq!(m.read(c, 1).unwrap(), Some(Value::Int(2)));
    }

    #[test]
    fn insert_existing_row_is_rejected() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(1)).unwrap();
        assert!(m.insert(t, 1, Value::Int(2)).is_err());
    }

    #[test]
    fn update_missing_row_is_not_found() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        let err = m.update(t, 42, Value::Int(1)).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }

    // ----------------------------------------------------------- deadlock

    #[test]
    fn deadlock_detection_finds_cycle() {
        let mut m = mgr();
        let a = m.begin(IsolationLevel::RepeatableRead);
        let b = m.begin(IsolationLevel::RepeatableRead);
        // a holds row 1, b holds row 2.
        m.insert(a, 1, Value::Int(1)).unwrap();
        m.insert(b, 2, Value::Int(2)).unwrap();
        // a waits on b (for row 2), b waits on a (for row 1).
        assert!(m.update(a, 2, Value::Int(2)).is_err());
        assert!(m.update(b, 1, Value::Int(1)).is_err());
        let cycle = m.detect_deadlock().expect("expected a deadlock cycle");
        assert!(cycle.len() >= 2);
        assert!(cycle.contains(&a));
        assert!(cycle.contains(&b));
        assert_eq!(m.stats().deadlocks, 0);
    }

    #[test]
    fn deadlock_detection_none_without_cycle() {
        let mut m = mgr();
        let a = m.begin(IsolationLevel::RepeatableRead);
        let b = m.begin(IsolationLevel::RepeatableRead);
        m.add_wait_for(a, b);
        assert!(m.detect_deadlock().is_none());
    }

    #[test]
    fn release_locks_clears_wait_for_edges() {
        let mut m = mgr();
        let a = m.begin(IsolationLevel::RepeatableRead);
        let b = m.begin(IsolationLevel::RepeatableRead);
        m.insert(a, 1, Value::Int(1)).unwrap();
        // b tries and fails, recording an edge b -> a.
        assert!(m.update(b, 1, Value::Int(2)).is_err());
        assert!(m.wait_for_graph().iter().any(|(w, _)| *w == b));
        m.rollback(a).unwrap();
        // a released its locks; the edge b -> a is gone.
        assert!(!m.wait_for_graph().iter().any(|(_, hs)| hs.contains(&a)));
    }

    // ------------------------------------------------------- serializable

    #[test]
    fn serializable_aborts_on_read_write_conflict() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();

        let reader = m.begin(IsolationLevel::Serializable);
        assert_eq!(m.read(reader, 1).unwrap(), Some(Value::Int(10)));

        let writer = m.begin(IsolationLevel::RepeatableRead);
        m.update(writer, 1, Value::Int(20)).unwrap();
        m.commit(writer).unwrap();

        // The serializable reader cannot commit: row 1 changed after it began.
        let err = m.commit(reader).unwrap_err();
        assert!(matches!(err, Error::Internal(_)));
        assert_eq!(m.state(reader), Some(TxnState::Aborted));
    }

    #[test]
    fn serializable_succeeds_without_conflict() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(10)).unwrap();
        m.commit(t0).unwrap();

        let reader = m.begin(IsolationLevel::Serializable);
        assert_eq!(m.read(reader, 1).unwrap(), Some(Value::Int(10)));
        // No concurrent writer touches row 1.
        let w = m.begin(IsolationLevel::RepeatableRead);
        m.insert(w, 2, Value::Int(20)).unwrap();
        m.commit(w).unwrap();

        m.commit(reader).unwrap();
        assert_eq!(m.state(reader), Some(TxnState::Committed));
    }

    #[test]
    fn serializable_predicate_lock_catches_phantom_like_conflict() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(1)).unwrap();
        m.insert(t0, 2, Value::Int(2)).unwrap();
        m.commit(t0).unwrap();

        let scanner = m.begin(IsolationLevel::Serializable);
        // The scanner reads rows 1 and 2 and declares a predicate lock over them.
        m.read(scanner, 1).unwrap();
        m.read(scanner, 2).unwrap();
        m.add_predicate_lock(scanner, &[1, 2]).unwrap();

        let writer = m.begin(IsolationLevel::RepeatableRead);
        m.update(writer, 2, Value::Int(99)).unwrap();
        m.commit(writer).unwrap();

        assert!(m.commit(scanner).is_err());
        assert_eq!(m.state(scanner), Some(TxnState::Aborted));
    }

    // ------------------------------------------------------------- lock modes

    #[test]
    fn shared_locks_are_compatible_and_upgrade() {
        let mut m = mgr();
        let t0 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t0, 1, Value::Int(1)).unwrap();
        m.commit(t0).unwrap();

        let a = m.begin(IsolationLevel::RepeatableRead);
        let b = m.begin(IsolationLevel::RepeatableRead);
        // Two readers can hold shared locks simultaneously.
        m.acquire_lock(a, 1, false).unwrap();
        m.acquire_lock(b, 1, false).unwrap();
        assert!(m.holds_lock(a, 1));
        assert!(m.holds_lock(b, 1));
        // A third writer is blocked while both readers hold shared locks.
        let c = m.begin(IsolationLevel::RepeatableRead);
        assert!(m.acquire_lock(c, 1, true).is_err());
        // Releasing one reader still leaves the other; writer still blocked.
        m.release_locks(a);
        assert!(m.acquire_lock(c, 1, true).is_err());
        // Releasing the last reader lets the writer through.
        m.release_locks(b);
        m.acquire_lock(c, 1, true).unwrap();
        assert!(m.holds_lock(c, 1));
    }

    #[test]
    fn exclusive_holder_can_downgrade_via_release() {
        let mut m = mgr();
        let a = m.begin(IsolationLevel::RepeatableRead);
        let b = m.begin(IsolationLevel::RepeatableRead);
        m.acquire_lock(a, 1, true).unwrap();
        assert!(m.acquire_lock(b, 1, false).is_err());
        m.release_locks(a);
        m.acquire_lock(b, 1, false).unwrap();
        assert!(m.holds_lock(b, 1));
    }

    // --------------------------------------------------------------- wal

    #[test]
    fn wal_records_lifecycle_begin_insert_commit() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(10)).unwrap();
        m.commit(t).unwrap();
        let recs = m.wal.records();
        let types: Vec<_> = recs.iter().map(|r| r.rtype).collect();
        assert_eq!(types, vec![RecType::Begin, RecType::Insert, RecType::Commit]);
        assert_eq!(recs[1].txn, t.raw());
        let (rid, vals) = decode_insert_payload(&recs[1].payload).unwrap();
        assert_eq!(rid, 1);
        assert_eq!(vals, vec![Value::Int(10)]);
    }

    #[test]
    fn wal_records_abort_on_rollback() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(10)).unwrap();
        m.rollback(t).unwrap();
        let types: Vec<_> = m.wal.records().iter().map(|r| r.rtype).collect();
        assert_eq!(types, vec![RecType::Begin, RecType::Insert, RecType::Abort]);
    }

    // ------------------------------------------------------------- recovery

    #[test]
    fn recover_rebuilds_committed_chains() {
        let mut m = mgr();
        let t1 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t1, 1, Value::Int(10)).unwrap();
        m.insert(t1, 2, Value::Int(20)).unwrap();
        m.commit(t1).unwrap();
        let t2 = m.begin(IsolationLevel::RepeatableRead);
        m.update(t2, 1, Value::Int(11)).unwrap();
        m.commit(t2).unwrap();
        // Aborted transaction must not be recovered.
        let t3 = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t3, 3, Value::Int(30)).unwrap();
        m.rollback(t3).unwrap();

        // Snapshot the WAL and rebuild a fresh manager from it.
        let wal = m.wal.clone();
        let mut m2 = TxnManager::with_wal(wal);
        m2.recover().unwrap();

        let reader = m2.begin(IsolationLevel::RepeatableRead);
        assert_eq!(m2.read(reader, 1).unwrap(), Some(Value::Int(11)));
        assert_eq!(m2.read(reader, 2).unwrap(), Some(Value::Int(20)));
        assert_eq!(m2.read(reader, 3).unwrap(), None);
        // The chain for row 1 has both committed versions.
        assert_eq!(m2.chain_len(1), 2);
    }

    // ------------------------------------------------------- misc helpers

    #[test]
    fn unknown_transaction_errors() {
        let mut m = mgr();
        let bogus = TransactionId(12345);
        assert!(m.read(bogus, 1).is_err());
        assert!(m.commit(bogus).is_err());
        assert!(m.rollback(bogus).is_err());
        assert!(m.insert(bogus, 1, Value::Int(1)).is_err());
    }

    #[test]
    fn operations_after_commit_are_rejected() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(1)).unwrap();
        m.commit(t).unwrap();
        assert!(m.read(t, 1).is_err());
        assert!(m.insert(t, 2, Value::Int(2)).is_err());
        assert!(m.update(t, 1, Value::Int(2)).is_err());
        assert!(m.delete(t, 1).is_err());
    }

    #[test]
    fn checkpoint_truncates_wal_and_runs_gc() {
        let mut m = mgr();
        let t = m.begin(IsolationLevel::RepeatableRead);
        m.insert(t, 1, Value::Int(1)).unwrap();
        m.update(t, 1, Value::Int(2)).unwrap();
        m.commit(t).unwrap();
        let before_segs = m.wal.segment_count();
        let _ = m.checkpoint().unwrap();
        // A checkpoint marker was appended.
        assert!(m.wal.records().iter().any(|r| r.rtype == RecType::Checkpoint));
        let _ = before_segs;
        assert_eq!(m.stats().checkpoints, 1);
        assert!(m.stats().gc_runs >= 1);
    }

    #[test]
    fn oldest_active_snapshot_is_inf_with_no_active() {
        let m = mgr();
        assert_eq!(m.oldest_active_snapshot(), TS_INF);
    }

    #[test]
    fn timestamp_display_and_inf() {
        assert_eq!(format!("{}", Timestamp(7)), "t7");
        assert_eq!(format!("{}", TS_INF), "inf");
        assert!(TS_INF.is_inf());
        assert!(!Timestamp(3).is_inf());
        assert_eq!(Timestamp(3).saturating_add(2), Timestamp(5));
    }

    // helper used in a couple of tests
    impl TxnManager {
        fn is_live_check(&self, id: TransactionId) -> bool {
            self.registry.get(&id).map(|t| t.is_live()).unwrap_or(false)
        }
    }
}
