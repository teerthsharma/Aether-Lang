#![no_std]

// ═══════════════════════════════════════════════════════════════════════════════
// Aether-Lang — invented by Teerth Sharma
// https://github.com/teerthsharma/Aether-Lang
// Copyright (c) 2026 Teerth Sharma. All Rights Reserved.
// ═══════════════════════════════════════════════════════════════════════════════
//

// Looks unused under `std` and is required without it. See the same note in
// aether-lang/src/lib.rs: this crate's import from alloc is behind a `std`
// cfg, so a host build reports this line as dead while a no_std build needs
// it.
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod memory;
