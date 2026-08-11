#![no_std]

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

// `extern crate alloc` used to sit here and was removed as genuinely unused.
//
// A previous revision of this file claimed it was required by a no_std build,
// citing an import in `ml/autograd.rs`. That file is not part of this crate:
// `lib.rs` declares `memory` and nothing else, so Cargo never compiles it, and
// the import cited as evidence is in a source file the compiler has never seen.
// Deleting the declaration changes no build, which is how the claim was found to
// be wrong.
//
// The same reasoning is correct for aether-lang, where the equivalent line is
// load-bearing: removing it there fails the kernel build with
// `cannot find module or crate alloc`. The difference is that its modules are
// declared and compiled.
#[cfg(feature = "std")]
extern crate std;

pub mod memory;
