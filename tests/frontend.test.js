const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const LabMetrics = require('../static/js/metrics.js');

function setup(failInference = false) {
  const nodes = new Map();
  const context2d = { clearRect() {}, drawImage() {} };
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, {
      textContent: '', value: '35', style: {}, disabled: false,
      classList: { add() {}, toggle() {} }, handlers: {},
      addEventListener(name, fn) { this.handlers[name] = fn; },
      querySelector: name => node(`${selector} ${name}`),
      getContext: () => context2d,
      toBlob: callback => callback(new Blob(['JPEG'])),
      videoWidth: 640, videoHeight: 480, play: async () => {}, click() {},
    });
    return nodes.get(selector);
  };
  let stopped = false, clock = 0, savedCsv;
  const response = data => ({ ok: true, status: 200,
    headers: { get: () => 'application/json' }, json: async () => data });
  const context = {
    LabMetrics, Blob, FormData, AbortController, Date, Math, Number, String, Promise,
    document: { hidden: false, querySelector: node, createElement: name => node(name) },
    window: { addEventListener() {} },
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() { stopped = true; } }] }) } },
    performance: { now: () => ++clock },
    requestAnimationFrame: callback => queueMicrotask(callback),
    setTimeout: () => 1, clearTimeout() {},
    URL: { createObjectURL(blob) { savedCsv = blob; return 'blob:test'; }, revokeObjectURL() {} },
    fetch: async url => {
      if (url === '/api/logs') return response({ total: 0, events: [] });
      if (url === '/api/health') return response({ ready: true, environment: 'local' });
      if (url === '/api/model') return response({ labels: ['with_mask', 'without_mask', 'mask_weared_incorrect'], parameters: 100, three_classes_ready: true });
      if (failInference) throw new Error('Network unavailable');
      return response({ detections: [], summary: { total: 0 }, width: 640, height: 480,
        environment: 'local', timing: { inference_ms: .2, server_ms: .3, cold_start: true } });
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/js/app.js'), 'utf8'), context);
  return { node, stopped: () => stopped, csv: () => savedCsv };
}

const settle = async () => { for (let i = 0; i < 5; i++) await new Promise(setImmediate); };

test('camera-to-API-to-render flow records timings and exports CSV; reset requires stopping', async () => {
  const page = setup();
  await settle();
  await page.node('#toggle-camera').handlers.click();
  await settle();
  assert.equal(page.node('#metric-count').textContent, '0 mẫu thống kê / 1 mẫu lưu');
  assert.match(page.node('#metric-inference_ms').textContent, /0.2 ms/);
  assert.match(page.node('#metric-response_ms').textContent, /ms/);
  assert.match(page.node('#metric-end_to_end_ms').textContent, /ms/);
  assert.equal(page.node('#download-metrics').disabled, false);
  page.node('#download-metrics').handlers.click();
  assert.match(await page.csv().text(), /inference_ms,response_ms,end_to_end_ms/);
  page.node('#reset-metrics').handlers.click();
  assert.match(page.node('#toast').textContent, /Tắt webcam/);
  await page.node('#toggle-camera').handlers.click();
  assert.equal(page.stopped(), true);
  page.node('#reset-metrics').handlers.click();
  assert.equal(page.node('#metric-count').textContent, '0 mẫu thống kê / 0 mẫu lưu');
});

test('network failure stops camera and does not record an invalid sample', async () => {
  const page = setup(true);
  await page.node('#toggle-camera').handlers.click();
  await settle();
  assert.equal(page.stopped(), true);
  assert.equal(page.node('#metric-count').textContent, '0 mẫu thống kê / 0 mẫu lưu');
  assert.equal(page.node('#toast').textContent, 'Network unavailable');
});
