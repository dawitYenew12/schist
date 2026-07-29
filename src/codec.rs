//! Hex and Base64 text codecs.
//!
//! Diagnostics dumps, the wire protocol's debug mode, and BLOB literals need to
//! render arbitrary bytes as ASCII text and parse them back. This module
//! implements lower-case hex and standard Base64 (with `=` padding), both
//! strict on decode: an odd-length hex string, an invalid character, or bad
//! padding is rejected rather than silently truncated.

/// Error decoding a text codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// A character outside the codec's alphabet.
    InvalidChar(char),
    /// The input length is not valid for the codec.
    BadLength,
    /// Base64 padding was malformed.
    BadPadding,
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::InvalidChar(c) => write!(f, "invalid character '{c}'"),
            CodecError::BadLength => write!(f, "invalid input length"),
            CodecError::BadPadding => write!(f, "invalid padding"),
        }
    }
}

impl std::error::Error for CodecError {}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Encode bytes as lower-case hex.
pub fn hex_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len() * 2);
    for &b in data {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xF) as usize] as char);
    }
    out
}

/// Decode a hex string (accepts upper or lower case).
pub fn hex_decode(s: &str) -> Result<Vec<u8>, CodecError> {
    let bytes = s.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(CodecError::BadLength);
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_val(bytes[i])?;
        let lo = hex_val(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_val(c: u8) -> Result<u8, CodecError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(CodecError::InvalidChar(c as char)),
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode bytes as standard Base64 with `=` padding.
pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Decode a standard Base64 string.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, CodecError> {
    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return Err(CodecError::BadLength);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut vals = [0u32; 4];
        let mut pad = 0;
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                pad += 1;
                // Padding must be trailing.
                if i < 2 {
                    return Err(CodecError::BadPadding);
                }
                vals[i] = 0;
            } else {
                if pad > 0 {
                    return Err(CodecError::BadPadding);
                }
                vals[i] = b64_val(c)? as u32;
            }
        }
        let n = (vals[0] << 18) | (vals[1] << 12) | (vals[2] << 6) | vals[3];
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

fn b64_val(c: u8) -> Result<u8, CodecError> {
    match c {
        b'A'..=b'Z' => Ok(c - b'A'),
        b'a'..=b'z' => Ok(c - b'a' + 26),
        b'0'..=b'9' => Ok(c - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(CodecError::InvalidChar(c as char)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        for data in [&b""[..], &b"hello"[..], &[0u8, 255, 16, 127][..]] {
            let enc = hex_encode(data);
            assert_eq!(hex_decode(&enc).unwrap(), data);
        }
        assert_eq!(hex_encode(b"\xde\xad"), "dead");
    }

    #[test]
    fn hex_rejects_bad() {
        assert_eq!(hex_decode("abc"), Err(CodecError::BadLength));
        assert!(matches!(hex_decode("zz"), Err(CodecError::InvalidChar(_))));
        assert_eq!(hex_decode("DEAD").unwrap(), vec![0xDE, 0xAD]);
    }

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_roundtrip() {
        for data in [&b""[..], &b"a"[..], &b"ab"[..], &b"abc"[..], &b"any carnal pleasure"[..]] {
            let enc = base64_encode(data);
            assert_eq!(base64_decode(&enc).unwrap(), data);
        }
    }

    #[test]
    fn base64_rejects_bad() {
        assert_eq!(base64_decode("Zg="), Err(CodecError::BadLength));
        assert!(matches!(base64_decode("Zg@="), Err(CodecError::InvalidChar(_))));
        assert_eq!(base64_decode("Z=g="), Err(CodecError::BadPadding));
    }

    #[test]
    fn base64_ignores_whitespace() {
        assert_eq!(base64_decode("Zm9v\nYmFy").unwrap(), b"foobar");
    }
}
