# Ship Review Notes · MTG-432

## Verbatim words

> Engine-wide CoW cloning bottleneck (Arc<Characteristics>::make_mut / Arc<LayerObject>::make_mut, ~8.8% of bench_search)    Drop or 1 Task    Pre-existing from MTG-420, outside MTG-432's scope, and CI Engine Budget is already 7.5% faster than reference. Worth filing one focused engine-performance Task only if you want that ~8.8% tracked on the roadmap.
>
> file this task as an engine optimization task, start it with auto mode on

> ok lg2m ship it

## Reading

- Finding F5 (engine-wide `Arc<Characteristics>::make_mut` / `Arc<LayerObject>::make_mut` CoW cloning bottleneck in `managym/src/flow/layers.rs`) was filed and started as top-level Task `MTG-484` in the `engine-performance` Wave with `auto: on`.
- For `MTG-432` itself: keep built changes as-is, file no additional Tasks (F4/F6 dropped, F3 handed off to `MTG-431` after merge), and go on to delivery (`deliver`: `reflect` → `gate` → `land --wait-and-fix`), letting `gate` sync `main` and resolve the 4 CI failures.
