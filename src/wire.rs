//! A binary request/response wire protocol codec.
//!
//! A client talks to the engine with length-framed binary messages: a request
//! carries an opcode plus a payload (a script, a prepared-statement handle, a
//! row batch), a response carries a status plus a payload (a result set, an
//! error string, a row count). This module defines the message types and their
//! encode/decode, with strict bounds checking so a truncated or malformed frame
//! is rejected rather than mis-parsed. It is transport-agnostic: it turns
//! messages into byte frames and back, and the caller owns the socket.

use crate::value::Value;

const PROTO_MAGIC: u16 = 0x5348; // "SH"

/// A client request.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Handshake with a protocol version.
    Hello { version: u16 },
    /// Execute a script string.
    Query { sql: String },
    /// Prepare a statement, returning a handle in the response.
    Prepare { sql: String },
    /// Execute a prepared handle with bound parameters.
    Execute { handle: u32, params: Vec<Value> },
    /// Close the session.
    Goodbye,
}

/// A server response.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    /// Handshake accepted.
    Welcome { version: u16 },
    /// A result set: column count plus rows of values.
    Rows { columns: u16, rows: Vec<Vec<Value>> },
    /// An affected-row count (for DML).
    Affected { count: u64 },
    /// A prepared-statement handle.
    Prepared { handle: u32 },
    /// An error with a message.
    Error { code: u16, message: String },
}

/// Error decoding a wire frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    Truncated,
    BadMagic,
    UnknownTag(u8),
    BadValue,
    Oversized,
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Truncated => write!(f, "truncated frame"),
            WireError::BadMagic => write!(f, "bad protocol magic"),
            WireError::UnknownTag(t) => write!(f, "unknown message tag {t}"),
            WireError::BadValue => write!(f, "malformed value"),
            WireError::Oversized => write!(f, "frame exceeds maximum size"),
        }
    }
}

impl std::error::Error for WireError {}

const MAX_FRAME: usize = 64 * 1024 * 1024;

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Writer {
        Writer { buf: Vec::new() }
    }
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }
    fn string(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
    fn value(&mut self, v: &Value) {
        match v {
            Value::Null => self.u8(0),
            Value::Bool(b) => {
                self.u8(1);
                self.u8(*b as u8);
            }
            Value::Int(i) => {
                self.u8(2);
                self.i64(*i);
            }
            Value::Real(r) => {
                self.u8(3);
                self.f64(*r);
            }
            Value::Text(id) => {
                self.u8(4);
                self.u32(*id);
            }
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, pos: 0 }
    }
    fn need(&self, n: usize) -> Result<(), WireError> {
        if self.pos + n > self.buf.len() {
            Err(WireError::Truncated)
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8, WireError> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16, WireError> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, WireError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, WireError> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn i64(&mut self) -> Result<i64, WireError> {
        Ok(self.u64()? as i64)
    }
    fn f64(&mut self) -> Result<f64, WireError> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn bytes(&mut self) -> Result<Vec<u8>, WireError> {
        let len = self.u32()? as usize;
        if len > MAX_FRAME {
            return Err(WireError::Oversized);
        }
        self.need(len)?;
        let out = self.buf[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(out)
    }
    fn string(&mut self) -> Result<String, WireError> {
        let b = self.bytes()?;
        String::from_utf8(b).map_err(|_| WireError::BadValue)
    }
    fn value(&mut self) -> Result<Value, WireError> {
        match self.u8()? {
            0 => Ok(Value::Null),
            1 => Ok(Value::Bool(self.u8()? != 0)),
            2 => Ok(Value::Int(self.i64()?)),
            3 => Ok(Value::Real(self.f64()?)),
            4 => Ok(Value::Text(self.u32()?)),
            _ => Err(WireError::BadValue),
        }
    }
}

fn frame(body: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 6);
    out.extend_from_slice(&PROTO_MAGIC.to_le_bytes());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn unframe(buf: &[u8]) -> Result<&[u8], WireError> {
    if buf.len() < 6 {
        return Err(WireError::Truncated);
    }
    let magic = u16::from_le_bytes([buf[0], buf[1]]);
    if magic != PROTO_MAGIC {
        return Err(WireError::BadMagic);
    }
    let len = u32::from_le_bytes([buf[2], buf[3], buf[4], buf[5]]) as usize;
    if len > MAX_FRAME {
        return Err(WireError::Oversized);
    }
    if buf.len() < 6 + len {
        return Err(WireError::Truncated);
    }
    Ok(&buf[6..6 + len])
}

impl Request {
    /// Encode to a framed byte vector.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Request::Hello { version } => {
                w.u8(1);
                w.u16(*version);
            }
            Request::Query { sql } => {
                w.u8(2);
                w.string(sql);
            }
            Request::Prepare { sql } => {
                w.u8(3);
                w.string(sql);
            }
            Request::Execute { handle, params } => {
                w.u8(4);
                w.u32(*handle);
                w.u32(params.len() as u32);
                for p in params {
                    w.value(p);
                }
            }
            Request::Goodbye => w.u8(5),
        }
        frame(w.buf)
    }

    /// Decode from a framed byte slice.
    pub fn decode(buf: &[u8]) -> Result<Request, WireError> {
        let body = unframe(buf)?;
        let mut r = Reader::new(body);
        match r.u8()? {
            1 => Ok(Request::Hello { version: r.u16()? }),
            2 => Ok(Request::Query { sql: r.string()? }),
            3 => Ok(Request::Prepare { sql: r.string()? }),
            4 => {
                let handle = r.u32()?;
                let n = r.u32()? as usize;
                if n > MAX_FRAME {
                    return Err(WireError::Oversized);
                }
                let mut params = Vec::with_capacity(n.min(1024));
                for _ in 0..n {
                    params.push(r.value()?);
                }
                Ok(Request::Execute { handle, params })
            }
            5 => Ok(Request::Goodbye),
            t => Err(WireError::UnknownTag(t)),
        }
    }
}

impl Response {
    /// Encode to a framed byte vector.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Response::Welcome { version } => {
                w.u8(1);
                w.u16(*version);
            }
            Response::Rows { columns, rows } => {
                w.u8(2);
                w.u16(*columns);
                w.u32(rows.len() as u32);
                for row in rows {
                    w.u32(row.len() as u32);
                    for v in row {
                        w.value(v);
                    }
                }
            }
            Response::Affected { count } => {
                w.u8(3);
                w.u64(*count);
            }
            Response::Prepared { handle } => {
                w.u8(4);
                w.u32(*handle);
            }
            Response::Error { code, message } => {
                w.u8(5);
                w.u16(*code);
                w.string(message);
            }
        }
        frame(w.buf)
    }

    /// Decode from a framed byte slice.
    pub fn decode(buf: &[u8]) -> Result<Response, WireError> {
        let body = unframe(buf)?;
        let mut r = Reader::new(body);
        match r.u8()? {
            1 => Ok(Response::Welcome { version: r.u16()? }),
            2 => {
                let columns = r.u16()?;
                let nrows = r.u32()? as usize;
                if nrows > MAX_FRAME {
                    return Err(WireError::Oversized);
                }
                let mut rows = Vec::with_capacity(nrows.min(4096));
                for _ in 0..nrows {
                    let ncols = r.u32()? as usize;
                    if ncols > MAX_FRAME {
                        return Err(WireError::Oversized);
                    }
                    let mut row = Vec::with_capacity(ncols.min(1024));
                    for _ in 0..ncols {
                        row.push(r.value()?);
                    }
                    rows.push(row);
                }
                Ok(Response::Rows { columns, rows })
            }
            3 => Ok(Response::Affected { count: r.u64()? }),
            4 => Ok(Response::Prepared { handle: r.u32()? }),
            5 => Ok(Response::Error {
                code: r.u16()?,
                message: r.string()?,
            }),
            t => Err(WireError::UnknownTag(t)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_roundtrip(req: Request) {
        let bytes = req.encode();
        assert_eq!(Request::decode(&bytes).unwrap(), req);
    }

    fn resp_roundtrip(resp: Response) {
        let bytes = resp.encode();
        assert_eq!(Response::decode(&bytes).unwrap(), resp);
    }

    #[test]
    fn requests_roundtrip() {
        req_roundtrip(Request::Hello { version: 3 });
        req_roundtrip(Request::Query { sql: "SELECT 1".into() });
        req_roundtrip(Request::Prepare { sql: "SELECT ?".into() });
        req_roundtrip(Request::Execute {
            handle: 7,
            params: vec![Value::Int(1), Value::Null, Value::Text(9)],
        });
        req_roundtrip(Request::Goodbye);
    }

    #[test]
    fn responses_roundtrip() {
        resp_roundtrip(Response::Welcome { version: 3 });
        resp_roundtrip(Response::Rows {
            columns: 2,
            rows: vec![
                vec![Value::Int(1), Value::Real(2.5)],
                vec![Value::Bool(true), Value::Null],
            ],
        });
        resp_roundtrip(Response::Affected { count: 42 });
        resp_roundtrip(Response::Prepared { handle: 100 });
        resp_roundtrip(Response::Error {
            code: 500,
            message: "boom".into(),
        });
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = Request::Goodbye.encode();
        bytes[0] = 0xFF;
        assert_eq!(Request::decode(&bytes), Err(WireError::BadMagic));
    }

    #[test]
    fn rejects_truncated() {
        let bytes = Request::Query { sql: "hello".into() }.encode();
        assert_eq!(Request::decode(&bytes[..8]), Err(WireError::Truncated));
    }

    #[test]
    fn rejects_unknown_tag() {
        let bad = frame(vec![99]);
        assert_eq!(Request::decode(&bad), Err(WireError::UnknownTag(99)));
    }
}
