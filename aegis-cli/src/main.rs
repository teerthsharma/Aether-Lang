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

use clap::{Parser, Subcommand};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::PathBuf;

use aegis_lang::{Lexer, Parser as AegisParser, Interpreter};

/// AEGIS - The Universal Programming Language
#[derive(Parser)]
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
        /// Path to .aegis file
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
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Repl) | None => run_repl(),
        Some(Commands::Run { file }) => run_file(&file),
        Some(Commands::Check { file }) => check_file(&file),
    }
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
                        if !result.is_empty() {
                            println!("{}", result);
                        }
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
    // Tokenize
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().map_err(|e| format!("Lexer error: {:?}", e))?;
    
    // Parse
    let mut parser = AegisParser::new(tokens);
    let ast = parser.parse().map_err(|e| format!("Parse error: {:?}", e))?;
    
    // Execute
    interpreter.execute(&ast).map_err(|e| format!("Runtime error: {:?}", e))
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
    
    let mut interpreter = Interpreter::new();
    
    // Tokenize
    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {:?}", e);
            std::process::exit(1);
        }
    };
    
    // Parse
    let mut parser = AegisParser::new(tokens);
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
            if !result.is_empty() {
                println!("{}", result);
            }
            println!();
            println!("Execution complete. 🦭");
        }
        Err(e) => {
            eprintln!("Runtime error: {:?}", e);
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
    
    // Tokenize
    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Lexer error at {:?}", e);
            std::process::exit(1);
        }
    };
    
    // Parse
    let mut parser = AegisParser::new(tokens);
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
