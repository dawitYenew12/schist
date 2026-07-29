//! Error type used across every subsystem.

use std::fmt;

/// A fault raised by some `schist` subsystem.
///
/// These are all *expected* control-flow errors (malformed input, a violated
/// decode-time invariant, an unknown identifier). They never represent a
/// memory-safety problem; the engine is structured so that a bad input is
/// rejected before any unsafe accessor is reached with out-of-range parameters.
#[derive(Debug)]
pub enum Error {
    /// The `.sht` blob could not be decoded.
    Decode(DecodeError),
    /// A cross-page invariant failed verification.
    Verify(VerifyError),
    /// A script statement could not be parsed or executed.
    Script(ScriptError),
    /// A request referenced something that does not exist.
    NotFound(String),
    /// A value did not match the expected column type.
    TypeMismatch { col: String, expected: String },
    /// A generic internal error with a message.
    Internal(String),
}

#[derive(Debug)]
pub enum DecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u32),
    BadChecksum { page: u32 },
    UnknownPageType(u8),
    BadDirectory,
    BadSchema,
    BadEncoding,
    Oversized,
}

#[derive(Debug)]
pub enum VerifyError {
    PageCountMismatch,
    SchemaDataMismatch,
    DictRange { page: u32 },
    IndexCoverage { page: u32 },
    FsmTotal,
    ZoneMapLiveness { page: u32 },
    RowIdBounds { row: u64 },
    OrphanPage { page: u32 },
    Other(String),
}

#[derive(Debug)]
pub enum ScriptError {
    Lex(String),
    Parse(String),
    UnknownStatement(String),
    Arity { stmt: String, got: usize },
    BadValue(String),
    EndOfInput,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Decode(e) => write!(f, "decode error: {e:?}"),
            Error::Verify(e) => write!(f, "verify error: {e:?}"),
            Error::Script(e) => write!(f, "script error: {e:?}"),
            Error::NotFound(s) => write!(f, "not found: {s}"),
            Error::TypeMismatch { col, expected } => {
                write!(f, "type mismatch on {col}: expected {expected}")
            }
            Error::Internal(s) => write!(f, "internal error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Self {
        Error::Decode(e)
    }
}
impl From<VerifyError> for Error {
    fn from(e: VerifyError) -> Self {
        Error::Verify(e)
    }
}
impl From<ScriptError> for Error {
    fn from(e: ScriptError) -> Self {
        Error::Script(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
