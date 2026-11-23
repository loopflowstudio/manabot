//! Rust allocator selection and optional process-wide allocation counters.
//! The benchmark feature wraps the backing allocator to count requests,
//! including alloc_zeroed and realloc. Timing builds use the backing allocator
//! directly. Python and Torch manage their own allocations.

use std::{
    alloc::{GlobalAlloc, Layout},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

#[cfg(all(feature = "mimalloc", target_family = "wasm"))]
compile_error!("the mimalloc feature requires a native target");

#[cfg(all(feature = "mimalloc", not(target_family = "wasm")))]
use mimalloc::MiMalloc as Backing;
#[cfg(any(not(feature = "mimalloc"), target_family = "wasm"))]
use std::alloc::System as Backing;

pub const NAME: &str = if cfg!(feature = "mimalloc") {
    "mimalloc"
} else {
    "system"
};
/// Whether the library installed the counting wrapper, even outside a measurement.
pub const COUNTING: bool = cfg!(feature = "benchmark");

/// Counts allocation requests passed to the selected backing allocator.
///
/// Each instance owns its counters, so a binary that installs its own counting
/// wrapper reads the instance it installed; nothing else can report a zero
/// count on its behalf.
pub struct CountingAllocator {
    recording: AtomicBool,
    allocations: AtomicU64,
    bytes: AtomicU64,
}

impl CountingAllocator {
    pub const fn new() -> Self {
        Self {
            recording: AtomicBool::new(false),
            allocations: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    /// Clears the counters and starts recording requests from every thread.
    pub fn start(&self) {
        self.allocations.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.recording.store(true, Ordering::Relaxed);
    }

    /// Stops recording and returns `(allocations, requested bytes)`.
    pub fn stop(&self) -> (u64, u64) {
        self.recording.store(false, Ordering::Relaxed);
        self.snapshot()
    }

    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.allocations.load(Ordering::Relaxed),
            self.bytes.load(Ordering::Relaxed),
        )
    }

    fn record(&self, size: usize) {
        if self.recording.load(Ordering::Relaxed) {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(size as u64, Ordering::Relaxed);
        }
    }
}

impl Default for CountingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.record(layout.size());
        unsafe { Backing.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.record(layout.size());
        unsafe { Backing.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { Backing.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        self.record(size);
        unsafe { Backing.realloc(ptr, layout, size) }
    }
}

#[cfg(all(feature = "mimalloc", not(feature = "benchmark")))]
#[global_allocator]
static ALLOCATOR: Backing = Backing;

#[cfg(feature = "benchmark")]
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator::new();

// The process-wide counters exist only where the library installed them.

#[cfg(feature = "benchmark")]
pub fn start() {
    ALLOCATOR.start();
}

#[cfg(feature = "benchmark")]
pub fn stop() -> (u64, u64) {
    ALLOCATOR.stop()
}

#[cfg(feature = "benchmark")]
pub fn snapshot() -> (u64, u64) {
    ALLOCATOR.snapshot()
}
