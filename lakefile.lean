import Lake
open Lake DSL

package «aether-formal» where
  version := v!"0.1.0"

@[default_target]
lean_lib Aether where
  roots := #[`Aether]
