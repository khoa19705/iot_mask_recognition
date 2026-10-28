const test = require('node:test');
const assert = require('node:assert/strict');
const LabMetrics = require('../static/js/metrics.js');
const sample = value => ({ inference_ms: value, response_ms: value + 5, end_to_end_ms: value + 10 });
test('warmup and cold starts are excluded from mean and P95 but preserved in CSV', () => {
  const metrics = new LabMetrics();
  for (let n = 1; n <= 23; n++) metrics.add(sample(n));
  const result = metrics.summary();
  assert.equal(result.count, 20);
  assert.equal(result.inference_ms.mean, 13.5);
  assert.equal(result.inference_ms.p95, 22);
  metrics.add({ ...sample(1000), cold_start: true });
  assert.equal(metrics.summary().count, 20);
  assert.equal(metrics.rows.at(-1).warmup, true);
  assert.match(metrics.csv(), /cold_start,warmup/);
  assert.match(metrics.csv(), /"true","true"/);
});
test('new session resets warmup; invalid samples rejected; CSV escapes metadata', () => {
  const metrics = new LabMetrics();
  assert.throws(() => metrics.add(sample(NaN)));
  metrics.add({ ...sample(1), environment: 'a,"b' });
  assert.match(metrics.csv(), /"a,""b"/);
  metrics.reset();
  assert.equal(metrics.summary().count, 0);
  assert.equal(metrics.summary().inference_ms, null);
  assert.equal(metrics.add(sample(2)).warmup, true);
});
test('retains at most 10000 recent samples', () => {
  const metrics = new LabMetrics();
  for (let n = 0; n <= 10000; n++) metrics.add(sample(n));
  assert.equal(metrics.rows.length, 10000);
  assert.equal(metrics.rows[0].inference_ms, 1);
});
