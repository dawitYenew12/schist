//! A SQL front end: lexer, parser, and logical-plan AST.
//!
//! This is a self-contained parser for a practical subset of SQL — `SELECT`
//! with projection, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, joins,
//! and the DDL/DML statements `CREATE TABLE`, `INSERT`, `UPDATE`, `DELETE`, and
//! `DROP`. It produces a [`Statement`] tree that the planner lowers to physical
//! operators. The lexer and the Pratt (precedence-climbing) expression parser
//! are hand-written so the crate stays dependency-free.

use std::fmt;

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    Keyword(Keyword),
    Int(i64),
    Real(f64),
    Str(String),
    // Punctuation / operators.
    LParen,
    RParen,
    Comma,
    Star,
    Dot,
    Semicolon,
    Plus,
    Minus,
    Slash,
    Percent,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Eof,
}

/// Reserved keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Select,
    From,
    Where,
    Group,
    By,
    Having,
    Order,
    Asc,
    Desc,
    Limit,
    Offset,
    Insert,
    Into,
    Values,
    Update,
    Set,
    Delete,
    Create,
    Table,
    Drop,
    Join,
    Inner,
    Left,
    On,
    As,
    And,
    Or,
    Not,
    Null,
    Is,
    In,
    Like,
    Between,
    True,
    False,
    Distinct,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Int,
    Real,
    Text,
    Bool,
}

impl Keyword {
    fn from_word(w: &str) -> Option<Keyword> {
        use Keyword::*;
        Some(match w.to_ascii_uppercase().as_str() {
            "SELECT" => Select,
            "FROM" => From,
            "WHERE" => Where,
            "GROUP" => Group,
            "BY" => By,
            "HAVING" => Having,
            "ORDER" => Order,
            "ASC" => Asc,
            "DESC" => Desc,
            "LIMIT" => Limit,
            "OFFSET" => Offset,
            "INSERT" => Insert,
            "INTO" => Into,
            "VALUES" => Values,
            "UPDATE" => Update,
            "SET" => Set,
            "DELETE" => Delete,
            "CREATE" => Create,
            "TABLE" => Table,
            "DROP" => Drop,
            "JOIN" => Join,
            "INNER" => Inner,
            "LEFT" => Left,
            "ON" => On,
            "AS" => As,
            "AND" => And,
            "OR" => Or,
            "NOT" => Not,
            "NULL" => Null,
            "IS" => Is,
            "IN" => In,
            "LIKE" => Like,
            "BETWEEN" => Between,
            "TRUE" => True,
            "FALSE" => False,
            "DISTINCT" => Distinct,
            "COUNT" => Count,
            "SUM" => Sum,
            "AVG" => Avg,
            "MIN" => Min,
            "MAX" => Max,
            "INT" | "INTEGER" | "BIGINT" => Int,
            "REAL" | "FLOAT" | "DOUBLE" => Real,
            "TEXT" | "VARCHAR" | "STRING" => Text,
            "BOOL" | "BOOLEAN" => Bool,
            _ => return None,
        })
    }
}

/// A parse error with the offending position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqlError {
    pub message: String,
    pub pos: usize,
}

impl fmt::Display for SqlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at offset {}", self.message, self.pos)
    }
}

impl std::error::Error for SqlError {}

/// The lexer.
pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// A lexer over `src`.
    pub fn new(src: &'a str) -> Lexer<'a> {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn err(&self, msg: &str) -> SqlError {
        SqlError {
            message: msg.to_string(),
            pos: self.pos,
        }
    }

    /// Tokenize the whole input.
    pub fn tokenize(&mut self) -> Result<Vec<(Token, usize)>, SqlError> {
        let mut out = Vec::new();
        loop {
            self.skip_ws_and_comments();
            let start = self.pos;
            let tok = self.next_token()?;
            let done = tok == Token::Eof;
            out.push((tok, start));
            if done {
                break;
            }
        }
        Ok(out)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b) if b.is_ascii_whitespace() => self.pos += 1,
                Some(b'-') if self.src.get(self.pos + 1) == Some(&b'-') => {
                    while let Some(b) = self.peek() {
                        self.pos += 1;
                        if b == b'\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, SqlError> {
        let b = match self.peek() {
            None => return Ok(Token::Eof),
            Some(b) => b,
        };
        match b {
            b'(' => self.single(Token::LParen),
            b')' => self.single(Token::RParen),
            b',' => self.single(Token::Comma),
            b'*' => self.single(Token::Star),
            b'.' => self.single(Token::Dot),
            b';' => self.single(Token::Semicolon),
            b'+' => self.single(Token::Plus),
            b'-' => self.single(Token::Minus),
            b'/' => self.single(Token::Slash),
            b'%' => self.single(Token::Percent),
            b'=' => self.single(Token::Eq),
            b'<' => {
                self.pos += 1;
                match self.peek() {
                    Some(b'=') => self.single(Token::LtEq),
                    Some(b'>') => self.single(Token::NotEq),
                    _ => Ok(Token::Lt),
                }
            }
            b'>' => {
                self.pos += 1;
                match self.peek() {
                    Some(b'=') => self.single(Token::GtEq),
                    _ => Ok(Token::Gt),
                }
            }
            b'!' => {
                self.pos += 1;
                match self.peek() {
                    Some(b'=') => self.single(Token::NotEq),
                    _ => Err(self.err("unexpected '!'")),
                }
            }
            b'\'' => self.lex_string(),
            b'"' => self.lex_quoted_ident(),
            b if b.is_ascii_digit() => self.lex_number(),
            b if b == b'_' || b.is_ascii_alphabetic() => Ok(self.lex_word()),
            _ => Err(self.err("unexpected character")),
        }
    }

    fn single(&mut self, tok: Token) -> Result<Token, SqlError> {
        self.pos += 1;
        Ok(tok)
    }

    fn lex_string(&mut self) -> Result<Token, SqlError> {
        self.pos += 1; // opening quote
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err(self.err("unterminated string")),
                Some(b'\'') => {
                    self.pos += 1;
                    if self.peek() == Some(b'\'') {
                        s.push('\'');
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                Some(b) => {
                    s.push(b as char);
                    self.pos += 1;
                }
            }
        }
        Ok(Token::Str(s))
    }

    fn lex_quoted_ident(&mut self) -> Result<Token, SqlError> {
        self.pos += 1;
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err(self.err("unterminated identifier")),
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b) => {
                    s.push(b as char);
                    self.pos += 1;
                }
            }
        }
        Ok(Token::Ident(s))
    }

    fn lex_number(&mut self) -> Result<Token, SqlError> {
        let start = self.pos;
        let mut is_real = false;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else if b == b'.' && !is_real {
                is_real = true;
                self.pos += 1;
            } else if b == b'e' || b == b'E' {
                is_real = true;
                self.pos += 1;
                if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        if is_real {
            text.parse::<f64>()
                .map(Token::Real)
                .map_err(|_| self.err("invalid number"))
        } else {
            text.parse::<i64>()
                .map(Token::Int)
                .map_err(|_| self.err("integer overflow"))
        }
    }

    fn lex_word(&mut self) -> Token {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'_' || b.is_ascii_alphanumeric() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        match Keyword::from_word(text) {
            Some(kw) => Token::Keyword(kw),
            None => Token::Ident(text.to_string()),
        }
    }
}

/// An expression in the parsed AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    Qualified(String, String),
    LitInt(i64),
    LitReal(f64),
    LitStr(String),
    LitBool(bool),
    Null,
    Star,
    Unary(UnaryOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    IsNull(Box<Expr>, bool),
    Like(Box<Expr>, Box<Expr>),
    Between(Box<Expr>, Box<Expr>, Box<Expr>),
    InList(Box<Expr>, Vec<Expr>),
    Aggregate(AggFunc, bool, Box<Expr>),
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

impl BinOp {
    fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => 3,
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Mul | BinOp::Div | BinOp::Mod => 5,
        }
    }
}

/// Aggregate functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// A single projected item.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// A join clause.
#[derive(Debug, Clone, PartialEq)]
pub struct Join {
    pub kind: JoinKind,
    pub table: String,
    pub alias: Option<String>,
    pub on: Expr,
}

/// The kind of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
}

/// An ORDER BY item.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderItem {
    pub expr: Expr,
    pub ascending: bool,
}

/// A parsed statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Select(SelectStmt),
    Insert(InsertStmt),
    Update(UpdateStmt),
    Delete(DeleteStmt),
    CreateTable(CreateTableStmt),
    DropTable(String),
}

/// A `SELECT` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStmt {
    pub distinct: bool,
    pub items: Vec<SelectItem>,
    pub from: Option<String>,
    pub from_alias: Option<String>,
    pub joins: Vec<Join>,
    pub filter: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderItem>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// An `INSERT` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStmt {
    pub table: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Expr>>,
}

/// An `UPDATE` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStmt {
    pub table: String,
    pub assignments: Vec<(String, Expr)>,
    pub filter: Option<Expr>,
}

/// A `DELETE` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStmt {
    pub table: String,
    pub filter: Option<Expr>,
}

/// A column declaration in `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSpec {
    pub name: String,
    pub kind: Keyword,
    pub not_null: bool,
    pub primary_key: bool,
}

/// A `CREATE TABLE` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStmt {
    pub name: String,
    pub columns: Vec<ColumnSpec>,
}

/// The recursive-descent + Pratt parser.
pub struct Parser {
    tokens: Vec<(Token, usize)>,
    pos: usize,
}

impl Parser {
    /// Build a parser by tokenizing `src`.
    pub fn new(src: &str) -> Result<Parser, SqlError> {
        let tokens = Lexer::new(src).tokenize()?;
        Ok(Parser { tokens, pos: 0 })
    }

    /// Parse a single statement (optionally terminated by `;`).
    pub fn parse_statement(&mut self) -> Result<Statement, SqlError> {
        let stmt = match self.peek() {
            Token::Keyword(Keyword::Select) => Statement::Select(self.parse_select()?),
            Token::Keyword(Keyword::Insert) => Statement::Insert(self.parse_insert()?),
            Token::Keyword(Keyword::Update) => Statement::Update(self.parse_update()?),
            Token::Keyword(Keyword::Delete) => Statement::Delete(self.parse_delete()?),
            Token::Keyword(Keyword::Create) => self.parse_create()?,
            Token::Keyword(Keyword::Drop) => self.parse_drop()?,
            _ => return Err(self.err("expected a statement")),
        };
        if self.peek() == Token::Semicolon {
            self.advance();
        }
        Ok(stmt)
    }

    /// Parse a whole script of `;`-separated statements.
    pub fn parse_script(src: &str) -> Result<Vec<Statement>, SqlError> {
        let mut p = Parser::new(src)?;
        let mut out = Vec::new();
        while p.peek() != Token::Eof {
            out.push(p.parse_statement()?);
        }
        Ok(out)
    }

    fn peek(&self) -> Token {
        self.tokens.get(self.pos).map(|(t, _)| t.clone()).unwrap_or(Token::Eof)
    }

    fn peek2(&self) -> Token {
        self.tokens.get(self.pos + 1).map(|(t, _)| t.clone()).unwrap_or(Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let t = self.peek();
        self.pos += 1;
        t
    }

    fn cur_pos(&self) -> usize {
        self.tokens.get(self.pos).map(|(_, p)| *p).unwrap_or(0)
    }

    fn err(&self, msg: &str) -> SqlError {
        SqlError {
            message: msg.to_string(),
            pos: self.cur_pos(),
        }
    }

    fn expect(&mut self, tok: Token, what: &str) -> Result<(), SqlError> {
        if self.peek() == tok {
            self.advance();
            Ok(())
        } else {
            Err(self.err(what))
        }
    }

    fn expect_keyword(&mut self, kw: Keyword) -> Result<(), SqlError> {
        if self.peek() == Token::Keyword(kw) {
            self.advance();
            Ok(())
        } else {
            Err(self.err(&format!("expected keyword {kw:?}")))
        }
    }

    fn parse_ident(&mut self) -> Result<String, SqlError> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            _ => Err(self.err("expected an identifier")),
        }
    }

    fn parse_select(&mut self) -> Result<SelectStmt, SqlError> {
        self.expect_keyword(Keyword::Select)?;
        let distinct = if self.peek() == Token::Keyword(Keyword::Distinct) {
            self.advance();
            true
        } else {
            false
        };
        let mut items = Vec::new();
        loop {
            let item = self.parse_select_item()?;
            items.push(item);
            if self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        let mut from = None;
        let mut from_alias = None;
        let mut joins = Vec::new();
        if self.peek() == Token::Keyword(Keyword::From) {
            self.advance();
            from = Some(self.parse_ident()?);
            from_alias = self.parse_optional_alias()?;
            loop {
                match self.peek() {
                    Token::Keyword(Keyword::Join)
                    | Token::Keyword(Keyword::Inner)
                    | Token::Keyword(Keyword::Left) => {
                        joins.push(self.parse_join()?);
                    }
                    _ => break,
                }
            }
        }
        let filter = if self.peek() == Token::Keyword(Keyword::Where) {
            self.advance();
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        let mut group_by = Vec::new();
        if self.peek() == Token::Keyword(Keyword::Group) {
            self.advance();
            self.expect_keyword(Keyword::By)?;
            loop {
                group_by.push(self.parse_expr(0)?);
                if self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        let having = if self.peek() == Token::Keyword(Keyword::Having) {
            self.advance();
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        let mut order_by = Vec::new();
        if self.peek() == Token::Keyword(Keyword::Order) {
            self.advance();
            self.expect_keyword(Keyword::By)?;
            loop {
                let expr = self.parse_expr(0)?;
                let ascending = match self.peek() {
                    Token::Keyword(Keyword::Asc) => {
                        self.advance();
                        true
                    }
                    Token::Keyword(Keyword::Desc) => {
                        self.advance();
                        false
                    }
                    _ => true,
                };
                order_by.push(OrderItem { expr, ascending });
                if self.peek() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        let mut limit = None;
        let mut offset = None;
        if self.peek() == Token::Keyword(Keyword::Limit) {
            self.advance();
            limit = Some(self.parse_int_literal()?);
        }
        if self.peek() == Token::Keyword(Keyword::Offset) {
            self.advance();
            offset = Some(self.parse_int_literal()?);
        }
        Ok(SelectStmt {
            distinct,
            items,
            from,
            from_alias,
            joins,
            filter,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_optional_alias(&mut self) -> Result<Option<String>, SqlError> {
        if self.peek() == Token::Keyword(Keyword::As) {
            self.advance();
            Ok(Some(self.parse_ident()?))
        } else if let Token::Ident(_) = self.peek() {
            Ok(Some(self.parse_ident()?))
        } else {
            Ok(None)
        }
    }

    fn parse_join(&mut self) -> Result<Join, SqlError> {
        let kind = match self.advance() {
            Token::Keyword(Keyword::Inner) => {
                self.expect_keyword(Keyword::Join)?;
                JoinKind::Inner
            }
            Token::Keyword(Keyword::Left) => {
                self.expect_keyword(Keyword::Join)?;
                JoinKind::Left
            }
            Token::Keyword(Keyword::Join) => JoinKind::Inner,
            _ => return Err(self.err("expected JOIN")),
        };
        let table = self.parse_ident()?;
        let alias = self.parse_optional_alias()?;
        self.expect_keyword(Keyword::On)?;
        let on = self.parse_expr(0)?;
        Ok(Join {
            kind,
            table,
            alias,
            on,
        })
    }

    fn parse_select_item(&mut self) -> Result<SelectItem, SqlError> {
        if self.peek() == Token::Star {
            self.advance();
            return Ok(SelectItem {
                expr: Expr::Star,
                alias: None,
            });
        }
        let expr = self.parse_expr(0)?;
        let alias = if self.peek() == Token::Keyword(Keyword::As) {
            self.advance();
            Some(self.parse_ident()?)
        } else if let Token::Ident(_) = self.peek() {
            // Bare alias.
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(SelectItem { expr, alias })
    }

    fn parse_int_literal(&mut self) -> Result<i64, SqlError> {
        match self.advance() {
            Token::Int(i) => Ok(i),
            _ => Err(self.err("expected an integer literal")),
        }
    }

    fn parse_insert(&mut self) -> Result<InsertStmt, SqlError> {
        self.expect_keyword(Keyword::Insert)?;
        self.expect_keyword(Keyword::Into)?;
        let table = self.parse_ident()?;
        let mut columns = Vec::new();
        if self.peek() == Token::LParen {
            self.advance();
            loop {
                columns.push(self.parse_ident()?);
                match self.advance() {
                    Token::Comma => continue,
                    Token::RParen => break,
                    _ => return Err(self.err("expected ',' or ')'")),
                }
            }
        }
        self.expect_keyword(Keyword::Values)?;
        let mut rows = Vec::new();
        loop {
            self.expect(Token::LParen, "expected '('")?;
            let mut row = Vec::new();
            loop {
                row.push(self.parse_expr(0)?);
                match self.advance() {
                    Token::Comma => continue,
                    Token::RParen => break,
                    _ => return Err(self.err("expected ',' or ')'")),
                }
            }
            rows.push(row);
            if self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(InsertStmt {
            table,
            columns,
            rows,
        })
    }

    fn parse_update(&mut self) -> Result<UpdateStmt, SqlError> {
        self.expect_keyword(Keyword::Update)?;
        let table = self.parse_ident()?;
        self.expect_keyword(Keyword::Set)?;
        let mut assignments = Vec::new();
        loop {
            let col = self.parse_ident()?;
            self.expect(Token::Eq, "expected '='")?;
            let val = self.parse_expr(0)?;
            assignments.push((col, val));
            if self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        let filter = if self.peek() == Token::Keyword(Keyword::Where) {
            self.advance();
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        Ok(UpdateStmt {
            table,
            assignments,
            filter,
        })
    }

    fn parse_delete(&mut self) -> Result<DeleteStmt, SqlError> {
        self.expect_keyword(Keyword::Delete)?;
        self.expect_keyword(Keyword::From)?;
        let table = self.parse_ident()?;
        let filter = if self.peek() == Token::Keyword(Keyword::Where) {
            self.advance();
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        Ok(DeleteStmt { table, filter })
    }

    fn parse_create(&mut self) -> Result<Statement, SqlError> {
        self.expect_keyword(Keyword::Create)?;
        self.expect_keyword(Keyword::Table)?;
        let name = self.parse_ident()?;
        self.expect(Token::LParen, "expected '('")?;
        let mut columns = Vec::new();
        loop {
            let col_name = self.parse_ident()?;
            let kind = match self.advance() {
                Token::Keyword(k @ (Keyword::Int | Keyword::Real | Keyword::Text | Keyword::Bool)) => k,
                _ => return Err(self.err("expected a column type")),
            };
            let mut not_null = false;
            let mut primary_key = false;
            loop {
                match self.peek() {
                    Token::Keyword(Keyword::Not) => {
                        self.advance();
                        self.expect_keyword(Keyword::Null)?;
                        not_null = true;
                    }
                    Token::Ident(ref s) if s.eq_ignore_ascii_case("primary") => {
                        self.advance();
                        // expect KEY
                        if let Token::Ident(k) = self.peek() {
                            if k.eq_ignore_ascii_case("key") {
                                self.advance();
                            }
                        }
                        primary_key = true;
                        not_null = true;
                    }
                    _ => break,
                }
            }
            columns.push(ColumnSpec {
                name: col_name,
                kind,
                not_null,
                primary_key,
            });
            match self.advance() {
                Token::Comma => continue,
                Token::RParen => break,
                _ => return Err(self.err("expected ',' or ')'")),
            }
        }
        Ok(Statement::CreateTable(CreateTableStmt { name, columns }))
    }

    fn parse_drop(&mut self) -> Result<Statement, SqlError> {
        self.expect_keyword(Keyword::Drop)?;
        self.expect_keyword(Keyword::Table)?;
        let name = self.parse_ident()?;
        Ok(Statement::DropTable(name))
    }

    /// Pratt expression parser with precedence `min_bp`.
    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, SqlError> {
        let mut lhs = self.parse_prefix()?;
        loop {
            // Postfix: IS [NOT] NULL, LIKE, BETWEEN, IN.
            match self.peek() {
                Token::Keyword(Keyword::Is) => {
                    self.advance();
                    let negated = if self.peek() == Token::Keyword(Keyword::Not) {
                        self.advance();
                        true
                    } else {
                        false
                    };
                    self.expect_keyword(Keyword::Null)?;
                    lhs = Expr::IsNull(Box::new(lhs), negated);
                    continue;
                }
                Token::Keyword(Keyword::Like) => {
                    self.advance();
                    let rhs = self.parse_prefix()?;
                    lhs = Expr::Like(Box::new(lhs), Box::new(rhs));
                    continue;
                }
                Token::Keyword(Keyword::Between) => {
                    self.advance();
                    let lo = self.parse_expr(3)?;
                    self.expect_keyword(Keyword::And)?;
                    let hi = self.parse_expr(3)?;
                    lhs = Expr::Between(Box::new(lhs), Box::new(lo), Box::new(hi));
                    continue;
                }
                Token::Keyword(Keyword::In) => {
                    self.advance();
                    self.expect(Token::LParen, "expected '(' after IN")?;
                    let mut list = Vec::new();
                    loop {
                        list.push(self.parse_expr(0)?);
                        match self.advance() {
                            Token::Comma => continue,
                            Token::RParen => break,
                            _ => return Err(self.err("expected ',' or ')'")),
                        }
                    }
                    lhs = Expr::InList(Box::new(lhs), list);
                    continue;
                }
                _ => {}
            }
            let op = match self.peek_binop() {
                Some(op) => op,
                None => break,
            };
            let bp = op.precedence();
            if bp < min_bp {
                break;
            }
            self.advance();
            let rhs = self.parse_expr(bp + 1)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn peek_binop(&self) -> Option<BinOp> {
        Some(match self.peek() {
            Token::Plus => BinOp::Add,
            Token::Minus => BinOp::Sub,
            Token::Star => BinOp::Mul,
            Token::Slash => BinOp::Div,
            Token::Percent => BinOp::Mod,
            Token::Eq => BinOp::Eq,
            Token::NotEq => BinOp::NotEq,
            Token::Lt => BinOp::Lt,
            Token::LtEq => BinOp::LtEq,
            Token::Gt => BinOp::Gt,
            Token::GtEq => BinOp::GtEq,
            Token::Keyword(Keyword::And) => BinOp::And,
            Token::Keyword(Keyword::Or) => BinOp::Or,
            _ => return None,
        })
    }

    fn parse_prefix(&mut self) -> Result<Expr, SqlError> {
        match self.peek() {
            Token::Minus => {
                self.advance();
                let e = self.parse_prefix()?;
                Ok(Expr::Unary(UnaryOp::Neg, Box::new(e)))
            }
            Token::Keyword(Keyword::Not) => {
                self.advance();
                let e = self.parse_expr(2)?;
                Ok(Expr::Unary(UnaryOp::Not, Box::new(e)))
            }
            Token::LParen => {
                self.advance();
                let e = self.parse_expr(0)?;
                self.expect(Token::RParen, "expected ')'")?;
                Ok(e)
            }
            Token::Int(i) => {
                self.advance();
                Ok(Expr::LitInt(i))
            }
            Token::Real(r) => {
                self.advance();
                Ok(Expr::LitReal(r))
            }
            Token::Str(s) => {
                self.advance();
                Ok(Expr::LitStr(s))
            }
            Token::Keyword(Keyword::True) => {
                self.advance();
                Ok(Expr::LitBool(true))
            }
            Token::Keyword(Keyword::False) => {
                self.advance();
                Ok(Expr::LitBool(false))
            }
            Token::Keyword(Keyword::Null) => {
                self.advance();
                Ok(Expr::Null)
            }
            Token::Keyword(kw @ (Keyword::Count | Keyword::Sum | Keyword::Avg | Keyword::Min | Keyword::Max)) => {
                self.advance();
                self.parse_aggregate(kw)
            }
            Token::Ident(_) => self.parse_column_or_call(),
            _ => Err(self.err("expected an expression")),
        }
    }

    fn parse_aggregate(&mut self, kw: Keyword) -> Result<Expr, SqlError> {
        let func = match kw {
            Keyword::Count => AggFunc::Count,
            Keyword::Sum => AggFunc::Sum,
            Keyword::Avg => AggFunc::Avg,
            Keyword::Min => AggFunc::Min,
            Keyword::Max => AggFunc::Max,
            _ => unreachable!(),
        };
        self.expect(Token::LParen, "expected '(' after aggregate")?;
        let distinct = if self.peek() == Token::Keyword(Keyword::Distinct) {
            self.advance();
            true
        } else {
            false
        };
        let arg = if self.peek() == Token::Star {
            self.advance();
            Expr::Star
        } else {
            self.parse_expr(0)?
        };
        self.expect(Token::RParen, "expected ')'")?;
        Ok(Expr::Aggregate(func, distinct, Box::new(arg)))
    }

    fn parse_column_or_call(&mut self) -> Result<Expr, SqlError> {
        let name = self.parse_ident()?;
        if self.peek() == Token::Dot {
            self.advance();
            if self.peek() == Token::Star {
                self.advance();
                return Ok(Expr::Qualified(name, "*".to_string()));
            }
            let col = self.parse_ident()?;
            return Ok(Expr::Qualified(name, col));
        }
        let _ = self.peek2();
        Ok(Expr::Column(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(src: &str) -> Statement {
        Parser::new(src).unwrap().parse_statement().unwrap()
    }

    #[test]
    fn lex_basic() {
        let toks = Lexer::new("SELECT a, b FROM t WHERE a >= 3")
            .tokenize()
            .unwrap();
        assert_eq!(toks[0].0, Token::Keyword(Keyword::Select));
        assert!(toks.iter().any(|(t, _)| *t == Token::GtEq));
    }

    #[test]
    fn parse_select_all() {
        let s = parse_one("SELECT * FROM users");
        if let Statement::Select(sel) = s {
            assert_eq!(sel.from.as_deref(), Some("users"));
            assert_eq!(sel.items[0].expr, Expr::Star);
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_where_precedence() {
        let s = parse_one("SELECT a FROM t WHERE a = 1 AND b > 2 OR c < 3");
        if let Statement::Select(sel) = s {
            // OR is the lowest precedence → root.
            match sel.filter.unwrap() {
                Expr::Binary(BinOp::Or, _, _) => {}
                other => panic!("unexpected root: {other:?}"),
            }
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_group_having_order_limit() {
        let s = parse_one(
            "SELECT dept, COUNT(*) AS n FROM emp GROUP BY dept HAVING COUNT(*) > 5 ORDER BY n DESC LIMIT 10 OFFSET 2",
        );
        if let Statement::Select(sel) = s {
            assert_eq!(sel.group_by.len(), 1);
            assert!(sel.having.is_some());
            assert_eq!(sel.limit, Some(10));
            assert_eq!(sel.offset, Some(2));
            assert!(!sel.order_by[0].ascending);
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_join() {
        let s = parse_one("SELECT * FROM a INNER JOIN b ON a.id = b.aid");
        if let Statement::Select(sel) = s {
            assert_eq!(sel.joins.len(), 1);
            assert_eq!(sel.joins[0].kind, JoinKind::Inner);
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_insert_multi_row() {
        let s = parse_one("INSERT INTO t (a, b) VALUES (1, 2), (3, 4)");
        if let Statement::Insert(ins) = s {
            assert_eq!(ins.columns, vec!["a", "b"]);
            assert_eq!(ins.rows.len(), 2);
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_update_delete() {
        let u = parse_one("UPDATE t SET a = a + 1 WHERE b = 2");
        assert!(matches!(u, Statement::Update(_)));
        let d = parse_one("DELETE FROM t WHERE a < 0");
        assert!(matches!(d, Statement::Delete(_)));
    }

    #[test]
    fn parse_create_table() {
        let c = parse_one("CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL, v REAL)");
        if let Statement::CreateTable(ct) = c {
            assert_eq!(ct.columns.len(), 3);
            assert!(ct.columns[0].primary_key);
            assert!(ct.columns[1].not_null);
        } else {
            panic!();
        }
    }

    #[test]
    fn parse_special_predicates() {
        let s = parse_one("SELECT * FROM t WHERE a IS NOT NULL AND b LIKE 'x%' AND c BETWEEN 1 AND 9 AND d IN (1, 2, 3)");
        assert!(matches!(s, Statement::Select(_)));
    }

    #[test]
    fn script_of_statements() {
        let stmts = Parser::parse_script("SELECT 1; SELECT 2; DROP TABLE t;").unwrap();
        assert_eq!(stmts.len(), 3);
    }
}
