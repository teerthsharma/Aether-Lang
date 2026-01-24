//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS CLI - Cross-Platform Command Line Interface
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Run AEGIS programs natively on Windows, Linux, and macOS.
//!
//! Usage:
//!   aegis repl              - Interactive REPL
//!   aegis run <file.aegis>  - Execute a script
//!   aegis check <file.aegis> - Validate syntax
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use clap::{Parser as ClapParser, Subcommand};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::PathBuf;

use aegis_lang::{Interpreter, Parser};

/// AEGIS - The Universal Programming Language
#[derive(ClapParser)]
#[command(name = "aegis")]
#[command(author = "AEGIS Research Team")]
#[command(version = "0.1.0")]
#[command(about = "Cross-platform AEGIS language runtime", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive REPL
    Repl,

    /// Run an AEGIS script
    Run {
        /// Path to .aegis or .ag file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Check syntax of an AEGIS script
    Check {
        /// Path to .aegis file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
}

fn main() {
    // Windows often has a small default stack (1MB). AEGIS involves deep recursion/large structs.
    // Spawn a thread with 8MB stack to prevent overflow.
    let builder = std::thread::Builder::new().stack_size(8 * 1024 * 1024);

    let handler = builder.spawn(|| {
        let cli = Cli::parse();

        match cli.command {
            Some(Commands::Repl) | None => run_repl(),
            Some(Commands::Run { file }) => run_file(&file),
            Some(Commands::Check { file }) => check_file(&file),
        }
    }).unwrap();

    handler.join().unwrap();
}

/// Interactive REPL for AEGIS
fn run_repl() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  🛡️ AEGIS v0.1.0 - The Universal Programming Language");
    println!("  Cross-Platform REPL (Windows/Linux/macOS)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Type 'exit' or Ctrl+C to quit. End statements with ~");
    println!();

    let mut rl = match DefaultEditor::new() {
        Ok(editor) => editor,
        Err(e) => {
            eprintln!("Error initializing readline: {}", e);
            return;
        }
    };

    let mut interpreter = Interpreter::new();

    loop {
        let readline = rl.readline("aegis> ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                if trimmed == "exit" || trimmed == "quit" {
                    println!("Goodbye! 🦭");
                    break;
                }

                if trimmed.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(&line);

                // Parse and execute
                match execute_line(&mut interpreter, trimmed) {
                    Ok(result) => {
                        println!("{}", result);
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nInterrupted. Type 'exit' to quit.");
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye! 🦭");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }
}

/// Execute a single line in the REPL
fn execute_line(interpreter: &mut Interpreter, source: &str) -> Result<String, String> {
    // Parser internally creates a lexer and tokenizes
    let mut parser = Parser::new(source);
    let ast = parser
        .parse()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    // Execute and format result
    let value = interpreter
        .execute(&ast)
        .map_err(|e| format!("Runtime error: {}", e))?;
    Ok(format!("{:?}", value))
}

/// Run an AEGIS script file
fn run_file(path: &PathBuf) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  🛡️ AEGIS - Running: {}", path.display());
    println!("═══════════════════════════════════════════════════════════════");

    let source = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        }
    };

    // Extension check
    if let Some(ext) = path.extension() {
        let s = ext.to_string_lossy();
        if s != "aegis" && s != "ag" {
            println!("Warning: File extension '.{}' is not standard (.aegis or .ag)", s);
        }
    }

    let mut interpreter = Interpreter::new();

    // Parse (parser handles tokenization internally)
    let mut parser = Parser::new(&source);
    let ast = match parser.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Parse error: {:?}", e);
            std::process::exit(1);
        }
    };

    // Execute
    match interpreter.execute(&ast) {
        Ok(result) => {
            println!("{:?}", result);
            println!();
            println!("Execution complete. 🦭");
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Check syntax of an AEGIS script
fn check_file(path: &PathBuf) {
    println!("Checking: {}", path.display());

    let source = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        }
    };

    // Parse (parser handles tokenization internally)
    let mut parser = Parser::new(&source);
    match parser.parse() {
        Ok(_) => {
            println!("✓ Syntax OK");
        }
        Err(e) => {
            eprintln!("❌ Parse error: {:?}", e);
            std::process::exit(1);
        }
    }
}
