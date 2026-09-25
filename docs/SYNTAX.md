# Syntax Compatibility Page

The old syntax tutorial has been replaced by neutral reference documentation:

- [Syntax](language/syntax.md)
- [Execution Model](language/execution-model.md)
- [Module Contracts](language/modules.md)

This repository keeps support for `seal` and its Unicode alias at the lexer
level. Documentation should prefer `seal` in examples for portability.

<div class="ts-viz" data-viz="pipe-lexer" data-preset="seal" data-title="Source to tokens to AST" data-caption="Both spellings lex to the same TokenKind::Seal, so the parser builds the same Loop node."></div>

