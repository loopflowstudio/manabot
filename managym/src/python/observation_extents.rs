//! Read occupied rows and referenced segments from strided observation buffers.
//! The caller supplies the row axis, reference columns and coordinate system.
#![allow(clippy::useless_conversion)] // PyO3 function wrappers
use pyo3::{
    buffer::{PyBuffer, ReadOnlyCell},
    exceptions::PyValueError,
    prelude::*,
};

/// A float32 buffer viewed as [batch, seat, row, column]. Missing seat and
/// column axes are singletons. The PyBuffer pins the storage for the scan.
struct Rows<'a> {
    buffer: &'a PyBuffer<f32>,
    batch: usize,
    seats: usize,
    rows: usize,
    columns: usize,
    strides: [isize; 4],
}

impl<'a> Rows<'a> {
    fn new(buffer: &'a PyBuffer<f32>, axis: usize) -> PyResult<Self> {
        let shape = buffer.shape();
        let strides = buffer.strides();
        // PyO3 0.22's float buffer check also admits big-endian formats on
        // little-endian hosts. Scalar reads below use native byte order.
        if cfg!(target_endian = "little") && buffer.format().to_bytes().starts_with(b">") {
            return Err(PyValueError::new_err("expected native-endian float32"));
        }
        if buffer.suboffsets().is_some()
            || !(1..=2).contains(&axis)
            || !(axis + 1..=axis + 2).contains(&shape.len())
        {
            return Err(PyValueError::new_err(
                "expected a strided [batch, (seat), row, (column)] buffer",
            ));
        }
        Ok(Self {
            buffer,
            batch: shape[0],
            seats: if axis == 2 { shape[1] } else { 1 },
            rows: shape[axis],
            columns: shape.get(axis + 1).copied().unwrap_or(1),
            strides: [
                strides[0],
                if axis == 2 { strides[1] } else { 0 },
                strides[axis],
                strides.get(axis + 1).copied().unwrap_or(0),
            ],
        })
    }

    #[inline]
    fn value(&self, batch: usize, seat: usize, row: usize, column: usize) -> f32 {
        debug_assert!(batch < self.batch && seat < self.seats);
        debug_assert!(row < self.rows && column < self.columns);
        let [b, s, r, c] = self.strides;
        let offset =
            batch as isize * b + seat as isize * s + row as isize * r + column as isize * c;
        // SAFETY: callers pass indices within the exported shape (checked in
        // debug), strides are in bytes, the buffer is direct and alive, and we
        // hold the GIL throughout.
        unsafe {
            self.buffer
                .buf_ptr()
                .byte_offset(offset)
                .cast::<f32>()
                .read_unaligned()
        }
    }
}

/// Index of the last element whose float32 bits are not a signed zero. NaN
/// counts as occupied, matching `value != 0.0`.
fn last_occupied(values: &[ReadOnlyCell<f32>]) -> Option<usize> {
    const CHUNK: usize = 16;
    const MAGNITUDE: u32 = 0x7fff_ffff;
    let occupied = |value: &ReadOnlyCell<f32>| value.get() != 0.0;
    let (chunks, tail) = values.as_chunks::<CHUNK>();
    if let Some(index) = tail.iter().rposition(occupied) {
        return Some(chunks.len() * CHUNK + index);
    }
    // OR each chunk's magnitude bits to skip padding without branching per element.
    let chunk = chunks.iter().rposition(|chunk| {
        chunk
            .iter()
            .fold(0, |bits, value| bits | value.get().to_bits())
            & MAGNITUDE
            != 0
    })?;
    Some(chunk * CHUNK + chunks[chunk].iter().rposition(occupied)?)
}

/// One past the last nonzero row for each batch item, including NaN padding.
#[pyfunction]
fn observation_row_extents(
    py: Python<'_>,
    buffer: PyBuffer<f32>,
    axis: usize,
) -> PyResult<Vec<usize>> {
    let view = Rows::new(&buffer, axis)?;
    let mut extents = vec![0; view.batch];
    if buffer.item_count() == 0 {
        return Ok(extents);
    }
    let contiguous = buffer.as_slice(py);
    for (batch, extent) in extents.iter_mut().enumerate() {
        for seat in 0..view.seats {
            let row = if let Some(values) = contiguous {
                let start = (batch * view.seats + seat) * view.rows;
                let tail =
                    &values[(start + *extent) * view.columns..(start + view.rows) * view.columns];
                last_occupied(tail).map(|index| *extent + index / view.columns)
            } else {
                (*extent..view.rows).rev().find(|&row| {
                    (0..view.columns).any(|column| view.value(batch, seat, row, column) != 0.0)
                })
            };
            if let Some(row) = row {
                *extent = row + 1;
            }
        }
    }
    Ok(extents)
}

/// For each (start, length) segment, return the per-batch extent named by
/// any finite reference. Each supplied column view carries its coordinate base.
#[pyfunction]
fn observation_reference_extents(
    references: Vec<(PyBuffer<f32>, usize)>,
    segments: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<usize>>> {
    let Some((first, _)) = references.first() else {
        return Err(PyValueError::new_err(
            "expected at least one reference buffer",
        ));
    };
    let count = Rows::new(first, 1)?.batch;
    let mut extents = vec![vec![0; count]; segments.len()];
    for (buffer, base) in &references {
        let view = Rows::new(buffer, 1)?;
        if view.batch != count || *base > 1 {
            return Err(PyValueError::new_err(
                "expected matching batch sizes and base 0 or 1",
            ));
        }
        for batch in 0..count {
            for row in 0..view.rows {
                for column in 0..view.columns {
                    let index = view.value(batch, 0, row, column) - *base as f32;
                    if !(0.0..usize::MAX as f32).contains(&index) {
                        continue;
                    }
                    let index = index as usize; // floor of a nonnegative coordinate
                    for (extent, &(start, length)) in extents.iter_mut().zip(&segments) {
                        if let Some(local) = index.checked_sub(start).filter(|&i| i < length) {
                            extent[batch] = extent[batch].max(local + 1);
                        }
                    }
                }
            }
        }
    }
    Ok(extents)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(observation_row_extents, module)?)?;
    module.add_function(wrap_pyfunction!(observation_reference_extents, module)?)?;
    Ok(())
}
