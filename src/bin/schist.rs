//! The `schist` command-line tool.
//!
//! Usage:
//!
//! ```text
//!   schist dump <file.sht>            decode + verify + print a full report
//!   schist run  <file.sht> <script>   decode + verify + run a script file
//!   schist new  <file.sht>            write a fresh empty database file
//! ```

use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: schist <dump|run|new> <file.sht> [script]");
        return ExitCode::from(2);
    }
    match args[1].as_str() {
        "dump" if args.len() >= 3 => match load(&args[2]) {
            Ok(mut db) => {
                print!("{}", schist::diag::full_report(&mut db));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(1)
            }
        },
        "run" if args.len() >= 4 => match load(&args[2]) {
            Ok(mut db) => {
                let script = fs::read_to_string(&args[3]).unwrap_or_default();
                match schist::script::run_script(&mut db, &script) {
                    Ok(report) => {
                        println!("{}", schist::diag::summary(&db));
                        println!(
                            "statements={} executed={} errors={} rows_scanned={}",
                            report.statements, report.executed, report.errors, report.rows_scanned
                        );
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("run error: {e}");
                        ExitCode::from(1)
                    }
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(1)
            }
        },
        "new" if args.len() >= 3 => {
            let db = schist::Database::new(schist::schema::Schema::new(vec![
                schist::schema::Column::row_id(),
            ]));
            let bytes = schist::format::encode_database(&db);
            match fs::write(&args[2], &bytes) {
                Ok(_) => {
                    println!("wrote {}", args[2]);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("write error: {e}");
                    ExitCode::from(1)
                }
            }
        }
        _ => {
            eprintln!("usage: schist <dump|run|new> <file.sht> [script]");
            ExitCode::from(2)
        }
    }
}

fn load(path: &str) -> Result<schist::Database, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let db = schist::format::decode(&bytes).map_err(|e| e.to_string())?;
    schist::verify::verify(&db).map_err(|e| e.to_string())?;
    Ok(db)
}
