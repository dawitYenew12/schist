//! The system catalog.
//!
//! While a single `.sht` file holds exactly one table's worth of columnar data,
//! a running engine tracks metadata about many objects: table definitions,
//! their columns and constraints, secondary indexes, sequences that hand out
//! surrogate keys, and named views (stored query text). The catalog is the
//! in-memory registry for that metadata. It assigns stable object ids, enforces
//! name uniqueness within a namespace, and answers the lookups the planner and
//! the DDL statements need.

use crate::schema::{ColKind, Encoding};
use std::collections::HashMap;

/// A stable identifier for a catalog object.
pub type ObjectId = u32;

/// A column definition within a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub kind: ColKind,
    pub encoding: Encoding,
    pub nullable: bool,
    pub primary_key: bool,
    pub default: Option<String>,
}

impl ColumnDef {
    /// A nullable column with plain encoding.
    pub fn new(name: &str, kind: ColKind) -> ColumnDef {
        ColumnDef {
            name: name.to_string(),
            kind,
            encoding: Encoding::Plain,
            nullable: true,
            primary_key: false,
            default: None,
        }
    }

    /// Builder: set the encoding.
    pub fn with_encoding(mut self, enc: Encoding) -> ColumnDef {
        self.encoding = enc;
        self
    }

    /// Builder: mark not-null.
    pub fn not_null(mut self) -> ColumnDef {
        self.nullable = false;
        self
    }

    /// Builder: mark as primary key (implies not-null).
    pub fn primary_key(mut self) -> ColumnDef {
        self.primary_key = true;
        self.nullable = false;
        self
    }
}

/// A table's definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDef {
    pub id: ObjectId,
    pub name: String,
    pub columns: Vec<ColumnDef>,
    /// Data page ids that hold this table's rows (opaque to the catalog).
    pub page_ids: Vec<u32>,
    pub row_estimate: u64,
}

impl TableDef {
    /// Column position by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Borrow a column by name.
    pub fn column(&self, name: &str) -> Option<&ColumnDef> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The primary-key column positions.
    pub fn primary_key_columns(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.primary_key)
            .map(|(i, _)| i)
            .collect()
    }
}

/// The kind of a secondary index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    BTree,
    Hash,
    Bitmap,
}

/// A secondary index definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDef {
    pub id: ObjectId,
    pub name: String,
    pub table: ObjectId,
    pub columns: Vec<usize>,
    pub kind: IndexKind,
    pub unique: bool,
}

/// A sequence generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub id: ObjectId,
    pub name: String,
    pub next: i64,
    pub step: i64,
}

impl Sequence {
    /// Hand out the next value and advance.
    pub fn advance(&mut self) -> i64 {
        let v = self.next;
        self.next = self.next.saturating_add(self.step);
        v
    }
}

/// A named view (stored query text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewDef {
    pub id: ObjectId,
    pub name: String,
    pub query: String,
}

/// An error from a catalog operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogError {
    DuplicateName(String),
    UnknownTable(String),
    UnknownIndex(String),
    UnknownColumn { table: String, column: String },
    DependencyExists { object: String, dependent: String },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::DuplicateName(n) => write!(f, "object '{n}' already exists"),
            CatalogError::UnknownTable(n) => write!(f, "unknown table '{n}'"),
            CatalogError::UnknownIndex(n) => write!(f, "unknown index '{n}'"),
            CatalogError::UnknownColumn { table, column } => {
                write!(f, "unknown column '{column}' on table '{table}'")
            }
            CatalogError::DependencyExists { object, dependent } => {
                write!(f, "cannot drop '{object}': '{dependent}' depends on it")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

/// The system catalog.
#[derive(Debug, Default)]
pub struct Catalog {
    next_id: ObjectId,
    tables: HashMap<ObjectId, TableDef>,
    indexes: HashMap<ObjectId, IndexDef>,
    sequences: HashMap<ObjectId, Sequence>,
    views: HashMap<ObjectId, ViewDef>,
    table_by_name: HashMap<String, ObjectId>,
    index_by_name: HashMap<String, ObjectId>,
    sequence_by_name: HashMap<String, ObjectId>,
    view_by_name: HashMap<String, ObjectId>,
}

impl Catalog {
    /// A fresh, empty catalog.
    pub fn new() -> Catalog {
        Catalog::default()
    }

    fn alloc_id(&mut self) -> ObjectId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Register a new table. Fails if the name is taken.
    pub fn create_table(
        &mut self,
        name: &str,
        columns: Vec<ColumnDef>,
    ) -> Result<ObjectId, CatalogError> {
        if self.table_by_name.contains_key(name) {
            return Err(CatalogError::DuplicateName(name.to_string()));
        }
        let id = self.alloc_id();
        let def = TableDef {
            id,
            name: name.to_string(),
            columns,
            page_ids: Vec::new(),
            row_estimate: 0,
        };
        self.tables.insert(id, def);
        self.table_by_name.insert(name.to_string(), id);
        Ok(id)
    }

    /// Borrow a table by name.
    pub fn table(&self, name: &str) -> Option<&TableDef> {
        self.table_by_name.get(name).and_then(|id| self.tables.get(id))
    }

    /// Borrow a table by id.
    pub fn table_by_id(&self, id: ObjectId) -> Option<&TableDef> {
        self.tables.get(&id)
    }

    /// Mutable table borrow by name.
    pub fn table_mut(&mut self, name: &str) -> Option<&mut TableDef> {
        let id = *self.table_by_name.get(name)?;
        self.tables.get_mut(&id)
    }

    /// All table names, sorted.
    pub fn table_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.table_by_name.keys().cloned().collect();
        names.sort();
        names
    }

    /// Drop a table. Fails if an index still references it.
    pub fn drop_table(&mut self, name: &str) -> Result<(), CatalogError> {
        let id = *self
            .table_by_name
            .get(name)
            .ok_or_else(|| CatalogError::UnknownTable(name.to_string()))?;
        if let Some(idx) = self.indexes.values().find(|i| i.table == id) {
            return Err(CatalogError::DependencyExists {
                object: name.to_string(),
                dependent: idx.name.clone(),
            });
        }
        self.tables.remove(&id);
        self.table_by_name.remove(name);
        Ok(())
    }

    /// Create a secondary index on named columns of a table.
    pub fn create_index(
        &mut self,
        name: &str,
        table: &str,
        columns: &[&str],
        kind: IndexKind,
        unique: bool,
    ) -> Result<ObjectId, CatalogError> {
        if self.index_by_name.contains_key(name) {
            return Err(CatalogError::DuplicateName(name.to_string()));
        }
        let tid = *self
            .table_by_name
            .get(table)
            .ok_or_else(|| CatalogError::UnknownTable(table.to_string()))?;
        let tdef = &self.tables[&tid];
        let mut col_ids = Vec::with_capacity(columns.len());
        for c in columns {
            let ci = tdef.column_index(c).ok_or_else(|| CatalogError::UnknownColumn {
                table: table.to_string(),
                column: c.to_string(),
            })?;
            col_ids.push(ci);
        }
        let id = self.alloc_id();
        let def = IndexDef {
            id,
            name: name.to_string(),
            table: tid,
            columns: col_ids,
            kind,
            unique,
        };
        self.indexes.insert(id, def);
        self.index_by_name.insert(name.to_string(), id);
        Ok(id)
    }

    /// Borrow an index by name.
    pub fn index(&self, name: &str) -> Option<&IndexDef> {
        self.index_by_name.get(name).and_then(|id| self.indexes.get(id))
    }

    /// All indexes on a table.
    pub fn indexes_on(&self, table: &str) -> Vec<&IndexDef> {
        let tid = match self.table_by_name.get(table) {
            Some(id) => *id,
            None => return Vec::new(),
        };
        let mut out: Vec<&IndexDef> = self.indexes.values().filter(|i| i.table == tid).collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Drop an index.
    pub fn drop_index(&mut self, name: &str) -> Result<(), CatalogError> {
        let id = *self
            .index_by_name
            .get(name)
            .ok_or_else(|| CatalogError::UnknownIndex(name.to_string()))?;
        self.indexes.remove(&id);
        self.index_by_name.remove(name);
        Ok(())
    }

    /// Create a sequence.
    pub fn create_sequence(
        &mut self,
        name: &str,
        start: i64,
        step: i64,
    ) -> Result<ObjectId, CatalogError> {
        if self.sequence_by_name.contains_key(name) {
            return Err(CatalogError::DuplicateName(name.to_string()));
        }
        let id = self.alloc_id();
        self.sequences.insert(
            id,
            Sequence {
                id,
                name: name.to_string(),
                next: start,
                step,
            },
        );
        self.sequence_by_name.insert(name.to_string(), id);
        Ok(id)
    }

    /// Advance a sequence and return the value handed out.
    pub fn next_sequence_value(&mut self, name: &str) -> Option<i64> {
        let id = *self.sequence_by_name.get(name)?;
        self.sequences.get_mut(&id).map(|s| s.advance())
    }

    /// Create a view.
    pub fn create_view(&mut self, name: &str, query: &str) -> Result<ObjectId, CatalogError> {
        if self.view_by_name.contains_key(name) {
            return Err(CatalogError::DuplicateName(name.to_string()));
        }
        let id = self.alloc_id();
        self.views.insert(
            id,
            ViewDef {
                id,
                name: name.to_string(),
                query: query.to_string(),
            },
        );
        self.view_by_name.insert(name.to_string(), id);
        Ok(id)
    }

    /// Borrow a view by name.
    pub fn view(&self, name: &str) -> Option<&ViewDef> {
        self.view_by_name.get(name).and_then(|id| self.views.get(id))
    }

    /// Count of objects of each kind: (tables, indexes, sequences, views).
    pub fn object_counts(&self) -> (usize, usize, usize, usize) {
        (
            self.tables.len(),
            self.indexes.len(),
            self.sequences.len(),
            self.views.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Catalog {
        let mut c = Catalog::new();
        c.create_table(
            "users",
            vec![
                ColumnDef::new("id", ColKind::Int).primary_key(),
                ColumnDef::new("name", ColKind::Text),
                ColumnDef::new("age", ColKind::Int),
            ],
        )
        .unwrap();
        c
    }

    #[test]
    fn create_and_lookup_table() {
        let c = sample();
        let t = c.table("users").unwrap();
        assert_eq!(t.columns.len(), 3);
        assert_eq!(t.primary_key_columns(), vec![0]);
        assert_eq!(t.column_index("age"), Some(2));
    }

    #[test]
    fn duplicate_table_rejected() {
        let mut c = sample();
        assert_eq!(
            c.create_table("users", vec![]),
            Err(CatalogError::DuplicateName("users".to_string()))
        );
    }

    #[test]
    fn index_lifecycle() {
        let mut c = sample();
        c.create_index("idx_age", "users", &["age"], IndexKind::BTree, false)
            .unwrap();
        assert_eq!(c.indexes_on("users").len(), 1);
        // Cannot drop a table with a dependent index.
        assert!(matches!(
            c.drop_table("users"),
            Err(CatalogError::DependencyExists { .. })
        ));
        c.drop_index("idx_age").unwrap();
        assert!(c.drop_table("users").is_ok());
    }

    #[test]
    fn index_unknown_column() {
        let mut c = sample();
        assert!(matches!(
            c.create_index("bad", "users", &["nope"], IndexKind::Hash, false),
            Err(CatalogError::UnknownColumn { .. })
        ));
    }

    #[test]
    fn sequences_advance() {
        let mut c = Catalog::new();
        c.create_sequence("s", 100, 5).unwrap();
        assert_eq!(c.next_sequence_value("s"), Some(100));
        assert_eq!(c.next_sequence_value("s"), Some(105));
        assert_eq!(c.next_sequence_value("missing"), None);
    }

    #[test]
    fn views_and_counts() {
        let mut c = sample();
        c.create_view("adults", "select * from users where age >= 18")
            .unwrap();
        assert!(c.view("adults").is_some());
        assert_eq!(c.object_counts(), (1, 0, 0, 1));
    }
}
