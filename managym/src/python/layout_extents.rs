//! Pin the layout's batched NumPy buffers, then scan their occupied and referenced rows.
use super::layout_bindings::ObservationLayout;
use crate::agent::observation_extents::{joins, row_axis, Join};
use pyo3::{buffer::PyBuffer, exceptions::PyValueError, prelude::*, types::PyDict};
use std::collections::BTreeMap;

fn direct_f32(array: &Bound<'_, PyAny>) -> Option<PyBuffer<f32>> {
    let buffer = PyBuffer::get_bound(array).ok()?;
    // PyO3 0.22 accepts a big-endian format on little-endian hosts.
    let foreign = if cfg!(target_endian = "little") {
        b'>'
    } else {
        b'<'
    };
    (buffer.suboffsets().is_none() && buffer.format().to_bytes().first() != Some(&foreign))
        .then_some(buffer)
}

struct Rows {
    buffer: PyBuffer<f32>,
    seats: usize,
    rows: usize,
    columns: usize,
    strides: [isize; 4],
    contiguous: bool,
}

impl Rows {
    fn new(buffer: PyBuffer<f32>, shape: &[usize], axis: usize) -> PyResult<Self> {
        if buffer.shape() != shape || !(axis + 1..=axis + 2).contains(&shape.len()) {
            return Err(PyValueError::new_err("array shape differs from its buffer"));
        }
        let strides = buffer.strides();
        let strides = [
            strides[0],
            if axis == 2 { strides[1] } else { 0 },
            strides[axis],
            strides.get(axis + 1).copied().unwrap_or(0),
        ];
        Ok(Self {
            contiguous: buffer.is_c_contiguous(),
            buffer,
            seats: if axis == 2 { shape[1] } else { 1 },
            rows: shape[axis],
            columns: shape.get(axis + 1).copied().unwrap_or(1),
            strides,
        })
    }

    #[inline]
    fn value(&self, batch: usize, seat: usize, row: usize, column: usize) -> f32 {
        debug_assert!(seat < self.seats && row < self.rows && column < self.columns);
        let [b, s, r, c] = self.strides;
        let offset =
            b * batch as isize + s * seat as isize + r * row as isize + c * column as isize;
        // SAFETY: each direct export pins its allocation and strides. The scan
        // visits only its checked shape. Callers keep the input stable until return.
        unsafe {
            self.buffer
                .buf_ptr()
                .byte_offset(offset)
                .cast::<f32>()
                .read_unaligned()
        }
    }

    fn occupied(&self, batch: usize) -> usize {
        let mut extent = 0;
        for seat in 0..self.seats {
            if self.columns > 0 && self.contiguous {
                let start = (batch * self.seats + seat) * self.rows + extent;
                // SAFETY: an aligned contiguous export covers exactly these
                // rows. No Python calls or writes occur during this borrow.
                let tail = unsafe {
                    std::slice::from_raw_parts(
                        self.buffer
                            .buf_ptr()
                            .cast::<f32>()
                            .add(start * self.columns),
                        (self.rows - extent) * self.columns,
                    )
                };
                if let Some(last) = last_occupied(tail) {
                    extent += last / self.columns + 1;
                }
                continue;
            }
            if let Some(last) = (extent..self.rows)
                .rev()
                .find(|&row| (0..self.columns).any(|c| self.value(batch, seat, row, c) != 0.0))
            {
                extent = last + 1;
            }
        }
        extent
    }
}

fn last_occupied(values: &[f32]) -> Option<usize> {
    let (chunks, tail) = values.as_chunks::<16>();
    let nonzero = |v: &f32| *v != 0.0;
    if let Some(last) = tail.iter().rposition(nonzero) {
        return Some(chunks.len() * 16 + last);
    }
    let chunk = chunks
        .iter()
        .rposition(|chunk| chunk.iter().fold(0, |bits, v| bits | v.to_bits()) & 0x7fff_ffff != 0)?;
    Some(chunk * 16 + chunks[chunk].iter().rposition(nonzero)?)
}

struct Table {
    axis: &'static str,
    rows: Rows,
    // Non-f32 exports (including native int32 action_focus) use source arithmetic.
    joins: Vec<(Join, Option<Rows>)>,
}

impl Table {
    fn new(
        array: &Bound<'_, PyAny>,
        shape: &[usize],
        axis: &'static str,
        dimension: usize,
        joins: &[Join],
        numpy: &Bound<'_, PyModule>,
    ) -> PyResult<Self> {
        if let Some(buffer) = direct_f32(array) {
            return Ok(Self {
                axis,
                rows: Rows::new(buffer, shape, dimension)?,
                joins: joins.iter().map(|&join| (join, None)).collect(),
            });
        }
        // Focus is bounded by the action mask and holds only zero-based
        // coordinates. For other dtypes test occupancy before narrowing:
        // even a float64 too small for float32 keeps its row alive.
        let values = if matches!(joins, [Join::Objects]) {
            array.clone()
        } else {
            numpy.getattr("not_equal")?.call1((array, 0))?
        };
        let buffer = values.call_method1("astype", ("float32",))?.extract()?;
        let rows = Rows::new(buffer, shape, dimension)?;
        let column = |i: usize| array.get_item((array.py().Ellipsis(), i));
        let mut coordinates = Vec::new();
        for &join in joins {
            let (values, projected) = match join {
                Join::Objects => {
                    coordinates.push((join, None));
                    continue;
                }
                Join::Object(i, base) => {
                    let v = column(i)?;
                    (if base == 0 { v } else { v.sub(base)? }, Join::Object(0, 0))
                }
                Join::Extent(axis, i) => (column(i)?, Join::Extent(axis, 0)),
                Join::Span(axis, offset, count) => {
                    let dtype = numpy
                        .getattr("result_type")?
                        .call1((array.getattr("dtype")?, "float32"))?;
                    (
                        column(offset)?
                            .call_method1("astype", (dtype,))?
                            .add(column(count)?)?,
                        Join::Extent(axis, 0),
                    )
                }
            };
            let values = values.call_method1("astype", ("float32",))?;
            let shape = &shape[..shape.len() - 1];
            coordinates.push((
                projected,
                Some(Rows::new(values.extract()?, shape, dimension)?),
            ));
        }
        Ok(Self {
            axis,
            rows,
            joins: coordinates,
        })
    }
}

type Extents = BTreeMap<&'static str, Vec<i64>>;

fn measure(
    tables: &[Table],
    mask: &Rows,
    count: usize,
    capacities: &BTreeMap<&'static str, usize>,
) -> Extents {
    let mut extents: Extents = capacities
        .keys()
        .map(|&axis| (axis, vec![0; count]))
        .collect();
    let mut start = 2;
    let segments = ["cards", "cards", "permanents", "permanents", "stack"].map(|axis| {
        let end = start + capacities.get(axis).copied().unwrap_or(0);
        let segment = (axis, start, end);
        start = end;
        segment
    });
    for b in 0..count {
        let valid_actions = mask.occupied(b);
        extents.get_mut("actions").unwrap()[b] = valid_actions as i64;
        for table in tables {
            let mut keep = |axis, rows: usize| {
                let extent = &mut extents.get_mut(axis).unwrap()[b];
                *extent = (*extent).max(rows as i64);
            };
            let limit = if matches!(table.joins.as_slice(), [(Join::Objects, _)]) {
                valid_actions
            } else {
                let limit = table.rows.occupied(b);
                keep(table.axis, limit);
                limit
            };
            for (join, projected) in &table.joins {
                let rows = projected.as_ref().unwrap_or(&table.rows);
                let value = |row, column| rows.value(b, 0, row, column);
                let mut keep_object = |index: f32| {
                    if index >= 2.0 && index.is_finite() {
                        let index = index as usize;
                        if let Some(&(axis, start, _)) =
                            segments.iter().find(|&&(_, _, end)| index < end)
                        {
                            keep(axis, index - start + 1);
                        }
                    }
                };
                match *join {
                    Join::Objects => {
                        for row in 0..limit {
                            for column in 0..rows.columns {
                                keep_object(value(row, column));
                            }
                        }
                    }
                    Join::Object(column, base) => {
                        for row in 0..limit {
                            keep_object(value(row, column) - base as f32);
                        }
                    }
                    Join::Extent(axis, column) | Join::Span(axis, column, _) => {
                        let end = (0..limit)
                            .map(|row| match *join {
                                Join::Span(_, _, count) => value(row, column) + value(row, count),
                                _ => value(row, column),
                            })
                            .filter(|n| n.is_finite())
                            .fold(0.0, f32::max);
                        keep(axis, (end as usize).min(capacities[axis]));
                    }
                }
            }
        }
    }
    extents
}

pub(super) fn axis_extents(
    py: Python<'_>,
    layout: &ObservationLayout,
    obs: &Bound<'_, PyAny>,
    omit_memory: bool,
) -> PyResult<Option<PyObject>> {
    let numpy = py.import_bound("numpy")?;
    let mut tables = Vec::new();
    let mut capacities = BTreeMap::new();
    let mut batch = None;
    let mut mask = None;
    let mut present = 0;
    let mut inputs = Vec::new();
    for (key, array) in &layout.arrays {
        if !obs.contains(key)? {
            if omit_memory && key.starts_with("decision_") {
                continue;
            }
            return Ok(None);
        }
        present += 1;
        let value = obs.get_item(key)?;
        let Ok(shape) = value
            .getattr("shape")
            .and_then(|s| s.extract::<Vec<usize>>())
        else {
            return Ok(None);
        };
        if shape.len() != array.shape.len() + 1
            || !shape[1..]
                .iter()
                .map(|&x| Some(x))
                .eq(array.shape.iter().copied())
            || *batch.get_or_insert(shape[0]) != shape[0]
        {
            return Ok(None);
        }
        if let Some((axis, dim)) = row_axis(key, array) {
            capacities.insert(axis, shape[dim]);
            inputs.push((key, value, shape, axis, dim));
        }
    }
    if obs.len()? != present {
        return Ok(None);
    }
    for (key, value, shape, axis, dim) in inputs {
        let table = Table::new(
            &value,
            &shape,
            axis,
            dim,
            joins(key, layout.config.policy_history),
            &numpy,
        )?;
        if key == "actions_valid" {
            mask = Some(table.rows);
        } else {
            tables.push(table);
        }
    }
    let mask = mask.expect("layout includes actions_valid");
    let extents = py.allow_threads(|| measure(&tables, &mask, batch.unwrap(), &capacities));
    let out = PyDict::new_bound(py);
    let asarray = numpy.getattr("asarray")?;
    for (axis, values) in extents {
        out.set_item(axis, asarray.call1((values, "int64"))?)?;
    }
    Ok(Some(out.into()))
}
