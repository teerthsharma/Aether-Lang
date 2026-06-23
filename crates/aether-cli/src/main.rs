//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS CLI - Cross-Platform Command Line Interface
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Run AEGIS programs natively on Windows, Linux, and macOS.
//!
//! Usage:
//!   aether repl              - Interactive REPL
//!   aether run <file.aether>  - Execute a script
//!   aether check <file.aether> - Validate syntax
//!
//! ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//


use clap::{Parser as ClapParser, Subcommand};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::path::PathBuf;

use aether_lang::{Interpreter, Parser};
use aether_lang::parser::ParseError;

/// AEGIS - The Universal Programming Language
#[derive(ClapParser)]
#[command(name = "aether")]
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
        /// Path to .aether or .ae file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Execution Mode: "bio" (default) or "titan"
        #[arg(long, default_value = "bio")]
        mode: String,
    },

    /// Check syntax of an AEGIS script
    Check {
        /// Path to .aether file
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
            Some(Commands::Run { file, mode }) => run_file(&file, &mode),
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
        let readline = rl.readline("aether> ");
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

fn format_parse_error(error: &ParseError) -> String {
    format!(
        "Parse error at line {}, column {}: {}",
        error.line, error.column, error.message
    )
}

/// Execute a single line in the REPL
fn execute_line(interpreter: &mut Interpreter, source: &str) -> Result<String, String> {
    // Parser internally creates a lexer and tokenizes
    let mut parser = Parser::new(source);
    let ast = parser
        .parse()
        .map_err(|e| format_parse_error(&e))?;

    // Execute and format result
    let value = interpreter
        .execute(&ast)
        .map_err(|e| format!("Runtime error: {}", e))?;
    Ok(format!("{:?}", value))
}

/// Run an AEGIS script file
fn run_file(path: &PathBuf, mode: &str) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  🛡️ AEGIS - Running: {}", path.display());
    println!("  Mode: {}", mode);
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
        if s != "aether" && s != "ae" {
            println!("Warning: File extension '.{}' is not standard (.aether or .ae)", s);
        }
    }

    // Parse (parser handles tokenization internally)
    let mut parser = Parser::new(&source);
    let ast = match parser.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{}", format_parse_error(&e));
            std::process::exit(1);
        }
    };

    if mode == "titan" {
        use aether_lang::vm::{TitanVM, Compiler};
        // Compile to Bytecode
        let compiler = Compiler::new();
        let code = compiler.compile(&ast);
        
        let mut vm = TitanVM::new();
        vm.load_code(code);
        
        match vm.run() {
            Ok(result) => {
                println!("{:?}", result);
                println!();
                println!("Titan Execution complete. ⚡");
            }
            Err(e) => {
                eprintln!("Titan Runtime error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Bio-Script (Standard Interpreter)
        let mut interpreter = Interpreter::new();
        match interpreter.execute(&ast) {
            Ok(result) => {
                println!("{:?}", result);
                println!();
                println!("Bio-Script Execution complete. 🦭");
            }
            Err(e) => {
                eprintln!("Runtime error: {}", e);
                std::process::exit(1);
            }
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
            eprintln!("❌ {}", format_parse_error(&e));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_lang::parser::ParseError;

    #[test]
    fn formats_parse_error_with_line_and_column() {
        let error = ParseError::new("expected identifier, found =", 2, 7);

        assert_eq!(
            format_parse_error(&error),
            "Parse error at line 2, column 7: expected identifier, found ="
        );
    }
}
