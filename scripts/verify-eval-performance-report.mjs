// Headless contract proof of the existing report with uneven real-format timers.
import { chromium } from '../academy/ui/node_modules/@playwright/test/index.mjs';
import { writeFile, readFile, mkdtemp, cp, rm, rename } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

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
  assert.match(await page.locator('#perf-body').innerText(), /Historical queue interval.*includes cold loading/);
  assert.match(await page.locator('#perf-body').innerText(), /3?unavailable|unavailable/);
  // Verify union coverage includes a child that exists only in a later seed.
  const laterChild = await page.evaluate(() => jobTiming({timing: [{run: 0, resource: 'r', decisions: 1, tree: {name: 'x', seconds: 2}}, {run: 1, resource: 'r', decisions: 1, tree: {name: 'x', seconds: 2, children: [{name: 'later', seconds: 1}]}}]}, {runs: [0, 1]}, 'r'));
  assert.match(laterChild.tree.note, /later: 1\/2/);
  assert.equal(laterChild.tree.seconds, 2);
  assert.deepEqual(errors, []);
  // The mixed proof consumes the actual Python CLI output. Controlled raw
  // receipt fixtures share their producer with the focused reader tests.
  execFileSync('uv', ['run', '--extra', 'dev', 'python', '-c', `
import runpy, sys
from pathlib import Path
fixtures = runpy.run_path('tests/eval/test_inference_receipts.py')
root = Path(sys.argv[1])
fixtures['_shared_fixture'](root, llm_failure=True)
`, directory], {timeout: 30000, stdio: 'pipe'});
  const build = name => execFileSync('uv', ['run', '--extra', 'dev', 'python', '-m', 'experiments.report_prototype.build',
    '--baseline-manifest', join(directory, 'baseline.json'), '--out', join(directory, name)], {timeout: 30000, stdio: 'pipe'});
  assert.throws(() => build('contradiction'), error => {
    assert.equal(error.status, 2);
    assert.match(error.stderr.toString(), /attempt llm-1 source .*llm: declared completed contradicts observed failed/);
    return true;
  });
  // Correct the declaration, never the authoritative failed-game receipts.
  const baseline = JSON.parse(await readFile(join(directory, 'baseline.json'), 'utf8'));
  baseline.attempts[0].status = 'failed';
  const declarationBytes = JSON.stringify(baseline);
  await writeFile(join(directory, 'baseline.json'), declarationBytes);
  for (const source of ['llm', 'policy', 'policy-2']) await rename(join(directory, source), join(directory, `${source}-absent`));
  build('missing');
  const missing = JSON.parse(await readFile(join(directory, 'missing', 'data.json'), 'utf8')).experiments[0];
  assert.deepEqual(missing.runs, []);
  assert.ok(missing.inference_performance.attempts.every(a => a.run === null && a.observed_status === 'unknown' && a.evidence_status === 'unavailable'));
  await page.goto(`file://${join(directory, 'missing', 'report.html')}`);
  await page.locator('#attempt-policy-1').waitFor();
  assert.match(await page.locator('#attempt-policy-1').innerText(), /declared completed.*evidence unavailable.*observed unknown/);
  assert.equal(await page.getByRole('link', {name: 'Linked measurements'}).count(), 0);
  for (const source of ['llm', 'policy', 'policy-2']) await rename(join(directory, `${source}-absent`), join(directory, source));
  const executionPath = join(directory, 'shared', 'execution.json');
  const executionBytes = await readFile(executionPath);
  await writeFile(executionPath, Buffer.concat([executionBytes, Buffer.from(' ')]));
  assert.throws(() => build('mutated'), error => {
    assert.equal(error.status, 2);
    assert.match(error.stderr.toString(), /dependency digest\/size mismatch/);
    return true;
  });
  await rm(executionPath);
  build('missing-cost');
  const missingCost = JSON.parse(await readFile(join(directory, 'missing-cost', 'data.json'), 'utf8')).experiments[0];
  assert.equal(missingCost.eval_performance.allocations[0].games_per_measured_dollar, null);
  assert.deepEqual(missingCost.eval_performance.runs.map(r => r.valid_games), [1, 1]);
  await page.goto(`file://${join(directory, 'missing-cost', 'report.html')}`);
  await page.locator('#allocation-allocation-1').waitFor();
  assert.match(await page.locator('#allocation-allocation-1').innerText(), /execution.json: unavailable.*dependency file is missing/);
  await writeFile(executionPath, executionBytes);
  build('mixed');
  const mixed = JSON.parse(await readFile(join(directory, 'mixed', 'data.json'), 'utf8')).experiments[0];
  assert.deepEqual(mixed.inference_performance.attempts.map(a => a.id), missing.inference_performance.attempts.map(a => a.id));
  for (const projection of [...mixed.eval_performance.runs, ...mixed.inference_performance.runs]) {
    const attempt = mixed.inference_performance.attempts.find(a => a.id === projection.attempt_id);
    assert.equal(attempt.run, projection.run);
    assert.equal(attempt.observed_status, projection.observed_status);
    assert.equal(attempt.evidence_status, 'available');
  }
  assert.equal(await readFile(join(directory, 'baseline.json'), 'utf8'), declarationBytes);
  assert.equal(mixed.eval_performance.allocations.length, 1);
  const allocation = mixed.eval_performance.allocations[0];
  assert.equal(allocation.included_cells, 2);
  assert.equal(allocation.cost.measured_usd, 2);
  assert.equal(allocation.games_per_measured_dollar, 1);
  assert.deepEqual(allocation.tree.children.map(c => c.name), ['model_load', 'unattributed']);
  assert.equal(allocation.lifecycle.measurements.service_start_to_ready.seconds, 1);
  // Portability: retain the complete explanation after original sources vanish.
  for (const source of ['llm', 'policy', 'policy-2', 'shared', 'diagnostics']) await rm(join(directory, source), {recursive: true});
  for (const file of allocation.dependencies) {
    assert.equal((await readFile(join(directory, 'mixed', file.href))).length, file.bytes);
  }

  await page.goto(`file://${join(directory, 'mixed', 'report.html')}`);
  const llm = page.locator('[data-inference-receipt]');
  await llm.waitFor();
  assert.equal(await page.getByRole('heading', {name: 'Shared allocation allocation-1 · charged once'}).count(), 1);
  assert.match(await page.locator('#allocation-allocation-1').innerText(), /Measured USD: 2/);
  assert.match(await page.locator('#allocation-allocation-1').innerText(), /Maximum dispatch queue wait.*3 seconds/);
  assert.match(await page.locator('#allocation-allocation-1').innerText(), /Service process start-to-ready: 1 seconds/);
  assert.equal(await page.locator('#allocation-allocation-1 a').count(), 5);

  assert.match(await page.locator('#attempt-llm-1').innerText(), /declared failed.*evidence available.*observed failed/);
  assert.match(await page.locator('#attempt-policy-1').innerText(), /declared completed.*evidence available.*observed completed/);
  assert.equal(await page.locator('#attempt-llm-1').getByRole('link', {name: 'Linked measurements'}).getAttribute('href'), '#perf-run-2');
  assert.equal(await page.locator('#attempt-policy-1').getByRole('link', {name: 'Linked measurements'}).getAttribute('href'), '#perf-run-0');
  await page.locator('#attempt-llm-1').getByRole('link', {name: 'Linked measurements'}).click();
  assert.equal(await page.evaluate(() => location.hash), '#perf-run-2');
  await llm.getByRole('link', {name: 'Declared attempt llm-1'}).click();
  assert.equal(await page.evaluate(() => location.hash), '#attempt-llm-1');
  assert.match(await llm.innerText(), /4\/4 selections.*3\/3 committed Commands.*0 replay-verified games/);
  assert.match(await llm.innerText(), /1 initial.*1 repaired.*1 fallback.*1 failed selections/);
  assert.match(await llm.innerText(), /p95 2.000000 s/);
  assert.match(await llm.innerText(), /inclusive cost.*unavailable/s);
  assert.match(await page.locator('#perf-body').innerText(), /policy-cuda.*unattempted.*unavailable/);
  assert.match(await page.locator('#perf-body').innerText(), /1 valid games/);
  const workerLatency = page.getByText('Game-worker request latency · individual games, no pooled quantiles', {exact: true});
  assert.equal(await workerLatency.count(), 2);
  await workerLatency.first().click();
  assert.match(await workerLatency.first().locator('..').innerText(), /"request_p95_seconds": 0.75/);
  const corpus = await page.getByRole('link', {name: 'Evidence and ordered corpus manifest'}).getAttribute('href');
  assert.ok(corpus.startsWith('corpus-') && corpus.endsWith('.json'));
  await llm.getByText(/immutable evidence files/).click();
  const source = llm.getByRole('link', {name: 'game/summary.json', exact: true});
  assert.match(await source.getAttribute('href'), /^evidence\/[0-9a-f]{64}-summary.json$/);
  await page.getByRole('button', {name: /^LLM llm/}).click();
  assert.equal(await page.locator('[data-inference-receipt]').count(), 0);
  assert.match(await page.locator('#perf-body').innerText(), /1 valid games/);
  assert.deepEqual(errors, []);
  console.log('Evaluation Perf report: activity controls and CLI-generated mixed inference receipts passed.');
} finally {
  await browser?.close();
  await rm(directory, {recursive: true, force: true});
}
