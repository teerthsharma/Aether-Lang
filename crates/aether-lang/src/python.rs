use pyo3::prelude::*;
use crate::Interpreter;
use crate::parser::Parser;
use crate::interpreter::Value;

/// The Aether Interpreter exposed to Python.
#[pyclass(unsendable)]
struct AetherInterpreter {
    inner: Interpreter,
}

#[pymethods]
impl AetherInterpreter {
    #[new]
    fn new() -> Self {
        AetherInterpreter {
            inner: Interpreter::new(),
        }
    }

    /// Execute an Aether script and return the result as a string.
    fn run(&mut self, source: String) -> PyResult<String> {
        let mut parser = Parser::new(&source);
        let program = match parser.parse() {
            Ok(p) => p,
            Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Parse error at {}:{}: {}", e.line, e.column, e.message))),
        };

        match self.inner.execute(&program) {
            Ok(val) => Ok(format!("{:?}", val)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
    
    /// Reset the interpreter state.
    fn reset(&mut self) {
        self.inner = Interpreter::new();
    }
}

/// A module for executing Aether code directly.
#[pymodule]
fn aether_lang(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AetherInterpreter>()?;
    
    /// Convenience function to run a script once
    #[pyfn(m)]
    fn run(source: String) -> PyResult<String> {
        let mut interpreter = Interpreter::new();
         let mut parser = Parser::new(&source);
        let program = match parser.parse() {
            Ok(p) => p,
            Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(format!("Parse error at {}:{}: {}", e.line, e.column, e.message))),
        };

        match interpreter.execute(&program) {
            Ok(val) => Ok(format!("{:?}", val)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    Ok(())
}
