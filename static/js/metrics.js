/* Shared by the browser and Node's built-in test runner. */
(() => {
  class LabMetrics {
    constructor() { this.reset(); }
    reset() { this.rows = []; this.seen = 0; }
    add(sample) {
      for (const key of ["inference_ms", "response_ms", "end_to_end_ms"]) {
        if (!Number.isFinite(sample[key]) || sample[key] < 0) throw new Error(`Invalid ${key}`);
      }
      const row = { ...sample, warmup: this.seen < 3 || Boolean(sample.cold_start) };
      this.seen += 1;
      this.rows.push(row);
      if (this.rows.length > 10000) this.rows.shift();
      return row;
    }
    summary() {
      const rows = this.rows.filter(row => !row.warmup);
      const result = { count: rows.length, total: this.rows.length };
      for (const key of ["inference_ms", "response_ms", "end_to_end_ms"]) {
        const values = rows.map(row => row[key]).sort((a, b) => a - b);
        result[key] = values.length ? {
          mean: values.reduce((a, b) => a + b, 0) / values.length,
          p95: values[Math.ceil(values.length * 0.95) - 1],
          min: values[0], max: values[values.length - 1],
        } : null;
      }
      return result;
    }
    csv() {
      const columns = ["timestamp", "environment", "width", "height", "confidence_threshold", "objects",
        "inference_ms", "response_ms", "end_to_end_ms", "server_ms", "queue_ms", "model_load_ms", "cold_start", "warmup"];
      const cell = value => `"${String(value ?? "").replace(/"/g, '""')}"`;
      return "\uFEFF" + [columns.join(","), ...this.rows.map(row => columns.map(key => cell(row[key])).join(","))].join("\r\n");
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = LabMetrics;
  else globalThis.LabMetrics = LabMetrics;
})();
