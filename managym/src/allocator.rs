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

/// Records process-wide allocation requests on the selected backing allocator.
pub struct CountingAllocator;
static RECORDING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

fn record(size: usize) {
    if RECORDING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { Backing.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { Backing.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { Backing.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { Backing.realloc(ptr, layout, size) }
    }
}

#[cfg(all(feature = "mimalloc", not(feature = "benchmark")))]
#[global_allocator]
static ALLOCATOR: Backing = Backing;

#[cfg(feature = "benchmark")]
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub fn start() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    RECORDING.store(true, Ordering::Relaxed);
}

pub fn stop() -> (u64, u64) {
    RECORDING.store(false, Ordering::Relaxed);
    snapshot()
}

pub fn snapshot() -> (u64, u64) {
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}
