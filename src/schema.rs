//! Schema model: columns, types, encodings, and the on-disk schema record.
//!
//! A `schist` database has exactly one schema (a fixed, ordered set of
//! columns). The first column is always the row-id column and is implicitly
//! `Int` with `Plain` encoding and `nullable = false`; the row-id column is
//! what the row-id map and the secondary indexes key off.

use crate::error::{DecodeError, Result};

/// The storage class of a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColKind {
    Bool,
    Int,
    Real,
    Text,
}

impl ColKind {
    pub fn as_u8(self) -> u8 {
        match self {
            ColKind::Bool => 0,
            ColKind::Int => 1,
            ColKind::Real => 2,
            ColKind::Text => 3,
        }
    }

    pub fn from_u8(b: u8) -> Option<ColKind> {
        match b {
            0 => Some(ColKind::Bool),
            1 => Some(ColKind::Int),
            2 => Some(ColKind::Real),
            3 => Some(ColKind::Text),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ColKind::Bool => "bool",
            ColKind::Int => "int",
            ColKind::Real => "real",
            ColKind::Text => "text",
        }
    }

    /// The fixed width of a value of this kind when stored in `Plain` encoding.
    /// Text under `Plain` is stored as a 4-byte dictionary id; under
    /// `Dictionary` encoding the column references dictionary pages directly.
    pub fn plain_width(self) -> usize {
        match self {
            ColKind::Bool => 1,
            ColKind::Int => 8,
            ColKind::Real => 8,
            ColKind::Text => 4,
        }
    }
}

/// How a column's values are laid out inside its data pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// One fixed-width cell per slot, back to back.
    Plain,
    /// Values are dictionary ids; the ids index into the column's dictionary
    /// pages. Two ids are equal iff their dictionary entries are equal.
    Dictionary,
    /// Run-length encoding: a column page is a sequence of `(value, count)`
    /// runs. Text RLE runs store dictionary ids.
    Rle,
}

impl Encoding {
    pub fn as_u8(self) -> u8 {
        match self {
            Encoding::Plain => 0,
            Encoding::Dictionary => 1,
            Encoding::Rle => 2,
        }
    }

    pub fn from_u8(b: u8) -> Option<Encoding> {
        match b {
            0 => Some(Encoding::Plain),
            1 => Some(Encoding::Dictionary),
            2 => Some(Encoding::Rle),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Encoding::Plain => "plain",
            Encoding::Dictionary => "dictionary",
            Encoding::Rle => "rle",
        }
    }
}

/// A single column definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub kind: ColKind,
    pub encoding: Encoding,
    pub nullable: bool,
    /// For `Dictionary`/`Rle` text columns, the dictionary page id this column
    /// resolves ids against. `None` for non-text or `Plain` columns.
    pub dict_page: Option<u32>,
    /// For indexed columns, the index page id. `None` if unindexed.
    pub index_page: Option<u32>,
}

impl Column {
    pub fn new(name: &str, kind: ColKind, encoding: Encoding) -> Self {
        Column {
            name: name.to_string(),
            kind,
            encoding,
            nullable: true,
            dict_page: None,
            index_page: None,
        }
    }

    /// The row-id column every database has.
    pub fn row_id() -> Self {
        Column {
            name: "id".to_string(),
            kind: ColKind::Int,
            encoding: Encoding::Plain,
            nullable: false,
            dict_page: None,
            index_page: None,
        }
    }
}

/// The full schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn new(columns: Vec<Column>) -> Self {
        let mut s = Schema { columns };
        if s.columns.is_empty() {
            s.columns.push(Column::row_id());
        }
        s
    }

    /// Find a column by name (case-sensitive).
    pub fn find(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// The index of the row-id column (always 0).
    pub fn row_id_index(&self) -> usize {
        0
    }

    /// Iterate over non-row-id columns.
    pub fn data_columns(&self) -> impl Iterator<Item = (usize, &Column)> {
        self.columns.iter().enumerate().skip(1)
    }

    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    /// Encode the schema into bytes for the `.sht` container.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.columns.len() as u8);
        for col in &self.columns {
            let name_bytes = col.name.as_bytes();
            out.push(name_bytes.len() as u8);
            out.extend_from_slice(name_bytes);
            out.push(col.kind.as_u8());
            out.push(col.encoding.as_u8());
            out.push(col.nullable as u8);
            // dict_page and index_page, each u32::MAX when absent.
            let dp = col.dict_page.unwrap_or(u32::MAX);
            out.extend_from_slice(&dp.to_le_bytes());
            let ip = col.index_page.unwrap_or(u32::MAX);
            out.extend_from_slice(&ip.to_le_bytes());
        }
        out
    }

    /// Decode a schema from its encoded form.
    pub fn decode(buf: &[u8]) -> Result<Schema> {
        if buf.is_empty() {
            return Err(DecodeError::BadSchema.into());
        }
        let n = buf[0] as usize;
        let mut pos = 1usize;
        let mut columns = Vec::with_capacity(n);
        for _ in 0..n {
            if pos >= buf.len() {
                return Err(DecodeError::BadSchema.into());
            }
            let nlen = buf[pos] as usize;
            pos += 1;
            if pos + nlen > buf.len() {
                return Err(DecodeError::BadSchema.into());
            }
            let name = std::str::from_utf8(&buf[pos..pos + nlen])
                .map_err(|_| DecodeError::BadSchema)?
                .to_string();
            pos += nlen;
            if pos + 4 > buf.len() {
                return Err(DecodeError::BadSchema.into());
            }
            let kind = ColKind::from_u8(buf[pos]).ok_or(DecodeError::BadSchema)?;
            let encoding =
                Encoding::from_u8(buf[pos + 1]).ok_or(DecodeError::BadSchema)?;
            let nullable = buf[pos + 2] != 0;
            pos += 3;
            let dp = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let ip = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;
            let dict_page = (dp != u32::MAX).then_some(dp);
            let index_page = (ip != u32::MAX).then_some(ip);
            columns.push(Column {
                name,
                kind,
                encoding,
                nullable,
                dict_page,
                index_page,
            });
        }
        if columns.is_empty() {
            return Err(DecodeError::BadSchema.into());
        }
        Ok(Schema { columns })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_round_trips() {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("a", ColKind::Int, Encoding::Plain));
        let mut c = Column::new("name", ColKind::Text, Encoding::Dictionary);
        c.dict_page = Some(7);
        c.index_page = Some(9);
        cols.push(c);
        let s = Schema::new(cols);
        let bytes = s.encode();
        let s2 = Schema::decode(&bytes).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn row_id_always_present() {
        let s = Schema::new(vec![]);
        assert_eq!(s.arity(), 1);
        assert_eq!(s.columns[0].name, "id");
    }
}
