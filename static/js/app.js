(() => {
  const $ = (selector) => document.querySelector(selector);
  const video = $("#camera-video");
  const overlay = $("#overlay-canvas");
  const captureCanvas = $("#capture-canvas");
  const confidence = $("#confidence");
  const toast = $("#toast");
  let stream = null;
  let running = false;
  let requestPending = false;
  let nextFrameTimer = null;
  let toastTimer = null;
  let controller = null;
  let cameraStarting = false;
  let cameraGeneration = 0;
  const metrics = new LabMetrics();
  const ms = value => Number.isFinite(value) ? `${value.toFixed(1)} ms` : "—";

  function renderMetrics(row) {
    const stats = metrics.summary();
    for (const key of ["inference_ms", "response_ms", "end_to_end_ms"]) {
      $(`#metric-${key}`).textContent = row ? ms(row[key]) : "—";
      $(`#avg-${key}`).textContent = ms(stats[key]?.mean);
      $(`#p95-${key}`).textContent = ms(stats[key]?.p95);
    }
    $("#metric-count").textContent = `${stats.count} mẫu thống kê / ${stats.total} mẫu lưu`;
    $("#metric-phase").textContent = row?.warmup ? "Mẫu khởi động · không tính vào trung bình/P95" : "Bỏ 3 mẫu đầu và mẫu cold start khỏi thống kê";
    $("#download-metrics").disabled = !stats.total;
  }

  $("#download-metrics").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([metrics.csv()], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `masklens-lab3-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $("#reset-metrics").addEventListener("click", () => {
    if (running || requestPending) return notify("Tắt webcam trước khi bắt đầu phiên đo mới.");
    metrics.reset(); renderMetrics();
  });

  const colors = { mask: "#29a46b", no_mask: "#e36c64", incorrect: "#e2ad42", other: "#9aa69c" };
  const labels = { mask: "ĐEO KHẨU TRANG", no_mask: "KHÔNG ĐEO", incorrect: "ĐEO CHƯA ĐÚNG", other: "KHÁC" };

  function notify(message) {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 4200);
  }
  function updateConfidence() {
    const value = Number(confidence.value);
    $("#confidence-value").textContent = `${value}%`;
    confidence.style.background = `linear-gradient(to right,#00c853 ${value}%,rgba(255,255,255,.1) ${value}%)`;
  }
  function setActivity(message) { $("#activity-text").textContent = message; }
  function setCameraState(isOn) {
    running = isOn;
    $("#camera-placeholder").classList.toggle("hidden", isOn);
    $("#live-indicator").classList.toggle("on", isOn);
    $("#live-indicator").querySelector("span").textContent = isOn ? "LIVE" : "ĐANG TẮT";
    $("#frame-status").textContent = isOn ? "ĐANG NHẬN DIỆN" : "CHỜ CAMERA";
    $("#button-label").textContent = isOn ? "Tắt webcam" : "Bật webcam";
    $("#toggle-camera").classList.toggle("stop", isOn);
    $("#toggle-camera .button-icon").textContent = isOn ? "■" : "◉";
  }
  function clearOverlay() {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
  function drawDetections(detections, resultWidth, resultHeight) {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;
    if (overlay.width !== width || overlay.height !== height) {
      overlay.width = width;
      overlay.height = height;
    }
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    const sx = width / resultWidth;
    const sy = height / resultHeight;
    const lineWidth = Math.max(2, Math.round(Math.min(width, height) / 240));
    const fontSize = Math.max(13, Math.round(Math.min(width, height) / 34));
    ctx.lineWidth = lineWidth;
    ctx.font = `700 ${fontSize}px "DM Sans", sans-serif`;
    ctx.textBaseline = "top";
    for (const detection of detections) {
      const [rawX1, rawY1, rawX2, rawY2] = detection.box;
      const x1 = rawX1 * sx, y1 = rawY1 * sy, x2 = rawX2 * sx, y2 = rawY2 * sy;
      const color = colors[detection.category] || colors.other;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const title = `${labels[detection.category] || detection.label}  ${Math.round(detection.confidence * 100)}%`;
      const textWidth = ctx.measureText(title).width;
      const tagHeight = fontSize + 10;
      const tagY = Math.max(0, y1 - tagHeight);
      ctx.fillStyle = color;
      ctx.fillRect(x1, tagY, textWidth + 14, tagHeight);
      ctx.fillStyle = "#fff";
      ctx.fillText(title, x1 + 7, tagY + 5);
    }
  }
  function updateSummary(summary) {
    $("#sum-total").textContent = summary.total ?? 0;
    $("#sum-mask").textContent = summary.mask ?? 0;
    $("#sum-no-mask").textContent = summary.no_mask ?? 0;
    $("#sum-incorrect").textContent = summary.incorrect ?? 0;
    $("#frame-time").textContent = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  }
  function formatLogTime(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString("vi-VN", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  function renderLogs(data) {
    $("#log-total").textContent = `${data.total} sự kiện`;
    const list = $("#log-list");
    if (!data.events.length) {
      list.innerHTML = '<div class="empty-logs">Chưa có sự kiện nào được ghi nhận.</div>';
      return;
    }
    list.innerHTML = data.events.map(event => {
      const counts = event.summary;
      const tags = [
        counts.mask && `<span class="log-tag">Đeo đúng: ${counts.mask}</span>`,
        counts.no_mask && `<span class="log-tag no-mask">Không đeo: ${counts.no_mask}</span>`,
        counts.incorrect && `<span class="log-tag incorrect">Đeo sai: ${counts.incorrect}</span>`,
      ].filter(Boolean).join("");
      return `<article class="log-item"><div class="log-item-head"><time class="log-time">${escapeHtml(formatLogTime(event.timestamp))}</time><span class="log-count">${counts.total} người</span></div><div class="log-tags">${tags}</div></article>`;
    }).join("");
  }
  async function refreshLogs() {
    try {
      const response = await fetch("/api/logs", { cache: "no-store" });
      if (response.ok) renderLogs(await response.json());
    } catch (_error) { /* Log history remains available on the next refresh. */ }
  }
  async function readJson(response) {
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error(`Máy chủ không trả JSON (HTTP ${response.status}). Kiểm tra Cloud hoặc thử lại.`);
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Không thể phân tích khung hình webcam.");
    return data;
  }

  async function analyzeNextFrame() {
    if (!running || requestPending) return;
    if (document.hidden) {
      nextFrameTimer = setTimeout(analyzeNextFrame, 600);
      return;
    }
    const generation = cameraGeneration;
    requestPending = true;
    let timeout;
    try {
      const captureStarted = performance.now();
      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;
      if (!sourceWidth || !sourceHeight) throw new Error("Camera chưa gửi được hình ảnh.");
      const scale = Math.min(1, 960 / sourceWidth);
      captureCanvas.width = Math.round(sourceWidth * scale);
      captureCanvas.height = Math.round(sourceHeight * scale);
      captureCanvas.getContext("2d").drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, "image/jpeg", 0.82));
      if (!blob || !running || generation !== cameraGeneration) return;
      const body = new FormData();
      body.append("file", blob, "webcam.jpg");
      body.append("confidence", Number(confidence.value) / 100);
      const sentAt = performance.now();
      controller = new AbortController();
      timeout = setTimeout(() => controller?.abort(), 120000);
      const result = await readJson(await fetch("/api/detect", { method: "POST", body, signal: controller.signal }));
      const receivedAt = performance.now();
      clearTimeout(timeout);
      if (!running || generation !== cameraGeneration || document.hidden) return;
      drawDetections(result.detections, result.width, result.height);
      updateSummary(result.summary);
      $("#frame-status").textContent = result.summary.total ? `${result.summary.total} ĐỐI TƯỢNG · LIVE` : "ĐANG QUÉT · CHƯA THẤY ĐỐI TƯỢNG";
      setActivity(result.summary.total ? "Đang nhận diện trên webcam trực tiếp." : "Webcam đang hoạt động · chưa phát hiện đối tượng.");
      // Two animation callbacks give the updated result a paint opportunity.
      // This is a display estimate, not a measurement of physical screen latency.
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      if (!running || generation !== cameraGeneration || document.hidden) return;
      const row = metrics.add({
        ...result.timing,
        timestamp: new Date().toISOString(), environment: result.environment,
        width: result.width, height: result.height,
        confidence_threshold: Number(body.get("confidence")), objects: result.summary.total,
        response_ms: receivedAt - sentAt,
        end_to_end_ms: performance.now() - captureStarted,
      });
      renderMetrics(row);
      if (result.logged) refreshLogs();
    } catch (error) {
      if (running && generation === cameraGeneration) {
        setActivity("Không thể phân tích webcam.");
        notify(error.name === "AbortError" ? "Máy chủ chưa phản hồi sau 120 giây. Hãy thử lại." : error.message);
        await stopCamera();
      }
    } finally {
      clearTimeout(timeout);
      controller = null;
      requestPending = false;
      if (running) nextFrameTimer = setTimeout(analyzeNextFrame, 600);
    }
  }
  async function startCamera() {
    if (cameraStarting || requestPending) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      notify("Trình duyệt không hỗ trợ webcam. Hãy mở bằng localhost hoặc HTTPS.");
      return;
    }
    cameraStarting = true;
    $("#toggle-camera").disabled = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      video.srcObject = stream;
      await video.play();
      const ratio = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 4 / 3;
      $("#camera-stage").style.aspectRatio = String(ratio);
      setCameraState(true);
      cameraGeneration += 1;
      setActivity("Đang khởi động nhận diện...");
      analyzeNextFrame();
    } catch (error) {
      if (stream) stream.getTracks().forEach(track => track.stop());
      stream = null;
      notify(error.name === "NotAllowedError" ? "Bạn chưa cấp quyền truy cập webcam." : "Không thể mở webcam. Kiểm tra camera và quyền truy cập.");
      setActivity("Webcam chưa sẵn sàng.");
    } finally {
      cameraStarting = false;
      $("#toggle-camera").disabled = false;
    }
  }
  async function stopCamera() {
    cameraGeneration += 1;
    controller?.abort();
    if (nextFrameTimer) clearTimeout(nextFrameTimer);
    nextFrameTimer = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    clearOverlay();
    updateSummary({ total: 0, mask: 0, no_mask: 0, incorrect: 0 });
    setCameraState(false);
    setActivity("Webcam đã tắt.");
  }

  confidence.addEventListener("input", updateConfidence);
  $("#toggle-camera").addEventListener("click", () => running ? stopCamera() : startCamera());
  window.addEventListener("beforeunload", () => { if (stream) stream.getTracks().forEach(track => track.stop()); });
  updateConfidence();
  renderMetrics();
  refreshLogs();

  fetch("/api/health", { cache: "no-store" }).then(response => response.json()).then(status => {
    const dot = $("#model-dot");
    $("#execution-environment").textContent = status.environment === "cloud" ? "Cloud (cấu hình máy chủ)" : "Local · chưa phải số đo Cloud";
    if (status.ready) { dot.classList.add("ready"); $("#model-status").textContent = "Đã tìm thấy model YOLO"; }
    else { dot.classList.add("error"); $("#model-status").textContent = "Chưa tìm thấy model YOLO"; }
  }).catch(() => { $("#model-dot").classList.add("error"); $("#model-status").textContent = "Không kết nối được máy chủ"; });
  fetch("/api/model").then(readJson).then(info => {
    $("#model-details").textContent = `${info.labels.join(" · ")} | ${(info.parameters / 1e6).toFixed(1)} triệu tham số`;
    $("#model-status").textContent = info.three_classes_ready ? "YOLO đủ 3 nhóm khẩu trang" : "Model chưa đủ 3 nhóm khẩu trang";
  }).catch(error => { $("#model-details").textContent = error.message; });
})();
