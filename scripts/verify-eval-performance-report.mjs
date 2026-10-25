// Headless contract proof of the existing report with uneven real-format timers.
import { chromium } from '../academy/ui/node_modules/@playwright/test/index.mjs';
import { writeFile, mkdtemp, cp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';

const directory = await mkdtemp(join(tmpdir(), 'eval-perf-'));
const tree = (native = true) => ({name: 'workers', seconds: 10, status: 'recorded', category: 'run', source: 'fixture receipt', children: [
  ...(native ? [{name: 'native_and_encoding', seconds: 4, status: 'recorded', category: 'eval', source: 'fixture receipt'}] : []),
  {name: 'inference_round_trip', seconds: 3, status: 'recorded', category: 'eval', source: 'fixture receipt'},
  {name: 'command_durable_acknowledgement', seconds: 1, status: 'recorded', category: 'eval', source: 'fixture receipt'},
  {name: 'unattributed', seconds: native ? 2 : 6, status: 'derived', category: 'unattributed', source: 'fixture receipt'},
]});
const span = {name: 'forward_host', start: 1, end: 2, process: 42, thread: 1, request: 'request-1', batch: 'batch-1'};
const experiment = {
  id: 'evaluation-performance', title: 'Evaluation performance', question: 'Equal-work evaluation', source: 'Test fixtures, not performance evidence', synthetic: false,
  decision_unit: 'evaluation', decks: [], deck_names: [], anchors: [], cells: [], direct: [], baseline: 'scalar',
  runs: [{id: 'a', job_name: 'scalar', seed: 1, evals: [{updates: 1, decisions: 100, hours: .01}]}, {id: 'b', job_name: 'scalar', seed: 2, evals: [{updates: 1, decisions: 200, hours: .02}]}, {id: 'c', job_name: 'batched', seed: 1, evals: [{updates: 1, decisions: 100, hours: .01}]}],
  timing: [{run: 0, resource: 'Worker elapsed (sum, overlaps)', decisions: 100, tree: tree()}, {run: 1, resource: 'Worker elapsed (sum, overlaps)', decisions: 200, tree: tree(false)}, {run: 2, resource: 'Worker elapsed (sum, overlaps)', decisions: 100, tree: tree()}],
  training: {series: []}, layout: {clock: 'updates', evals: [], ml: [], perf: [{kind: 'flame', title: 'Evaluation time', scale: 'total'}]},
  context: {status: 'incomplete', world: 'fixture', device: 'cpu', captured_at: null, heartbeat_at: null},
  eval_performance: {runs: [{run: 0, evidence: 'receipt.json', allocation_ids: ['one'], valid_games: 4, valid_blocks: 1, games_per_wall_hour: 100, blocks_per_wall_hour: 25, failed_games: 1, interrupted_games: 1, unattempted_games: 2, coverage: {native: 1}, game_population: 2, spans: [span], optimization: {status: 'hypothesis', observation: 'Measured intervals overlap', mechanism: 'Batching', validation: 'Matched run', coverage: 'one of two', evidence: 'receipt.json'}}], allocations: []}
};
experiment.eval_performance.allocations.push({identity: 'one', elapsed_seconds: 20, included_cells: 1, valid_games: 4, valid_blocks: 1, games_per_allocation_hour: 720, cost: {allocation_method: 'unallocated', measured_usd: null, estimated_usd: null}, tree: tree(), evidence: 'receipt.json', serving: {request_p50_seconds: .01, request_p95_seconds: .02, latency_sample_count: 2, sample_coverage: 'bounded first observations', counters: {decisions: 100, model_loads: 1, catalog_builds: 1}, traces: ['receipt.json'], unavailable: ['pure FFI overhead'], spans: [{...span, thread: 2}], measurements: {batch_queue: {maximum_seconds: .01}}}});
let browser;
try {
  for (const name of ['report.html', 'model.js', 'charts.js']) await cp(resolve('experiments/report_prototype', name), join(directory, name));
  await writeFile(join(directory, 'data.js'), `window.REPORT=${JSON.stringify({experiments: [experiment]})};`);
  await writeFile(join(directory, 'receipt.json'), '{}');
  browser = await chromium.launch({headless: true});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`file://${join(directory, 'report.html')}`);
  await page.waitForSelector('#perf-body .flame');
  assert.deepEqual(errors, []);
  assert.match(await page.locator('#perf-body').innerText(), /1\/2 seeds measured/);
  assert.match(await page.locator('#perf-body').innerText(), /never zero/);
  assert.match(await page.locator('#perf-body').innerText(), /Concurrent games overlap/);
  assert.match(await page.locator('#perf-body').innerText(), /Game engine/);
  assert.match(await page.locator('#perf-body').innerText(), /Evidence and replay/);
  assert.match(await page.locator('#perf-body').innerText(), /Other and unmeasured/);
  assert.equal(await page.getByRole('button', {name: 'During games', exact: true}).getAttribute('aria-pressed'), 'true');
  await page.locator('.fn').filter({hasText: /^Inference ·/}).first().click();
  assert.match(await page.locator('#perf-body').innerText(), /inference_round_trip · 3.0 s/);
  await page.getByRole('button', {name: 'All', exact: true}).first().click();
  assert.equal(await page.locator('.rbtn .tag').innerText(), 'baseline');
  await page.getByRole('button', {name: 'Per 1M decisions', exact: true}).first().click();
  assert.match(await page.locator('#perf-body').innerText(), /per 1M evaluation decisions/);
  assert.doesNotMatch(await page.locator('#perf-body').innerText(), /per 1M learner/);
  await page.locator('.fn').first().click();
  await page.getByRole('button', {name: /forward_host/}).first().click();
  assert.match(await page.locator('#perf-body').innerText(), /request request-1.*batch batch-1/);
  assert.equal(await page.getByRole('link', {name: 'Optimization evidence'}).getAttribute('href'), 'receipt.json');
  assert.equal(await page.getByRole('link', {name: /CPU.GPU diagnostic trace/}).getAttribute('href'), 'receipt.json');
  assert.match(await page.locator('#perf-body').innerText(), /Shared allocation one · charged once/);
  assert.match(await page.locator('#perf-body').innerText(), /3?unavailable|unavailable/);
  // Verify union coverage includes a child that exists only in a later seed.
  const laterChild = await page.evaluate(() => jobTiming({timing: [{run: 0, resource: 'r', decisions: 1, tree: {name: 'x', seconds: 2}}, {run: 1, resource: 'r', decisions: 1, tree: {name: 'x', seconds: 2, children: [{name: 'later', seconds: 1}]}}]}, {runs: [0, 1]}, 'r'));
  assert.match(laterChild.tree.note, /later: 1\/2/);
  assert.equal(laterChild.tree.seconds, 2);
  assert.deepEqual(errors, []);
  console.log('Evaluation Perf report: comparison, coverage, normalization, drill-down, timeline and evidence links passed.');
} finally {
  await browser?.close();
  await rm(directory, {recursive: true, force: true});
}
