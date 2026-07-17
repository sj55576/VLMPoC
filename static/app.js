/* Browser dashboard. API paths deliberately match the FastAPI application. */
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const camera = document.createElement("video");
camera.autoplay = true; camera.muted = true; camera.playsInline = true;

const state = {
  running: false, stream: null, requestInFlight: false, socket: null, reconnectTimer: null,
  pollingTimer: null, frameTimer: null, wsConnected: false, session: null, steps: [],
  progress: 0, events: [], vlmCalls: 0, lastFrame: null, lastVlmTimestamp: null,
  previewUrl: null, source: "待機中",
};
const SKELETON = [["nose","left_shoulder"],["nose","right_shoulder"],["left_shoulder","right_shoulder"],["left_shoulder","left_elbow"],["left_elbow","left_wrist"],["right_shoulder","right_elbow"],["right_elbow","right_wrist"],["left_shoulder","left_hip"],["right_shoulder","right_hip"],["left_hip","right_hip"],["left_hip","left_knee"],["left_knee","left_ankle"],["right_hip","right_knee"],["right_knee","right_ankle"]];

function text(value, fallback = "—") { return value === undefined || value === null || value === "" ? fallback : String(value); }
function escapeHtml(value) { const node = document.createElement("span"); node.textContent = text(value, ""); return node.innerHTML; }
function timestamp(value) { try { return new Intl.DateTimeFormat("ja-JP", {hour:"2-digit", minute:"2-digit", second:"2-digit"}).format(new Date(value)); } catch { return "—"; } }
function isSocketOpen() { return state.socket && state.socket.readyState === WebSocket.OPEN; }

async function call(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text() || `${response.status} ${response.statusText}`);
  return response.json();
}
function setNotice(message, isError = false) {
  const node = $("upload-status"); node.textContent = message; node.style.color = isError ? "#ffabb3" : "";
}
function setConnection(connected, label) {
  state.wsConnected = connected;
  $("connection-dot").className = `connection-dot ${connected ? "online" : "offline"}`;
  $("connection-status").textContent = label;
}
function updateButtons() {
  $("start").disabled = state.running;
  $("pause").disabled = !state.running;
  $("stop").disabled = !state.session || state.session.status === "STOPPED";
}

function draw(data = {}) {
  ctx.fillStyle = "#050a13"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (camera.readyState >= 2) ctx.drawImage(camera, 0, 0, canvas.width, canvas.height);
  ctx.font = "12px system-ui"; ctx.textBaseline = "bottom";
  for (const object of data.objects || []) {
    const [x1, y1, x2, y2] = object.bbox || []; if (![x1,y1,x2,y2].every(Number.isFinite)) continue;
    const color = object.class_name === "person" ? "#48c9ff" : "#ffd34e";
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${object.class_name}  #${object.track_id ?? "—"}  ${Math.round((object.confidence || 0) * 100)}%`;
    const width = ctx.measureText(label).width + 8; ctx.fillStyle = color; ctx.fillRect(x1, Math.max(0, y1 - 18), width, 17);
    ctx.fillStyle = "#07101f"; ctx.fillText(label, x1 + 4, Math.max(14, y1 - 4));
  }
  for (const pose of data.poses || []) {
    const points = pose.keypoints || {};
    const point = (name) => points[name] && points[name].confidence > .15 ? [points[name].x * canvas.width, points[name].y * canvas.height] : null;
    ctx.strokeStyle = "#f58cce"; ctx.lineWidth = 2;
    for (const [a,b] of SKELETON) { const one = point(a), two = point(b); if (one && two) { ctx.beginPath(); ctx.moveTo(...one); ctx.lineTo(...two); ctx.stroke(); } }
    for (const keypoint of Object.values(points)) if (keypoint.confidence > .15) { ctx.fillStyle = "#ff9bd8"; ctx.beginPath(); ctx.arc(keypoint.x * canvas.width, keypoint.y * canvas.height, 3.5, 0, Math.PI * 2); ctx.fill(); }
  }
}

function renderSteps() {
  const completed = state.steps.filter((step) => step.status === "COMPLETED").length;
  $("step-count").textContent = `${completed} / ${state.steps.length}`;
  $("steps").innerHTML = state.steps.length ? state.steps.map((step, index) => {
    const icon = step.status === "COMPLETED" ? "✓" : step.status === "ACTIVE" ? "●" : index + 1;
    return `<li class="step-item ${String(step.status || "").toLowerCase()}"><span class="step-icon">${icon}</span><span><span class="step-title">${escapeHtml(step.step_id)}</span><span class="step-description">${escapeHtml(step.reason || "判定待機中")}</span></span><span class="step-status">${escapeHtml(step.status)}</span></li>`;
  }).join("") : '<li class="muted">工程情報を待機中です。</li>';
}
function renderCurrentStep(current, condition) {
  $("step-name").textContent = current?.name || "セッション未開始";
  $("step-description").textContent = current?.description || "SOPを選択して開始してください。";
  const conditionNode = $("condition-result");
  if (condition) {
    conditionNode.className = `condition ${condition.passed ? "passed" : "failed"}`;
    conditionNode.textContent = `${condition.passed ? "条件成立" : "条件待機"} · ${Math.round((condition.confidence || 0) * 100)}% · ${condition.reason || "根拠なし"}`;
  } else { conditionNode.className = "condition muted"; conditionNode.textContent = "判定根拠を待機中"; }
}
function renderVlm(result) {
  if (!result) return;
  $("vlm-status").textContent = text(result.step_status, "解析済み");
  $("vlm-summary").textContent = text(result.scene_summary, "VLMのシーン要約はありません。");
  const details = [["アクション",result.detected_action],["工程",result.current_step_id],["信頼度",result.confidence !== undefined ? `${Math.round(result.confidence * 100)}%` : null],["安全違反",result.safety_violation === undefined ? null : result.safety_violation ? "あり" : "なし"]];
  $("vlm-details").innerHTML = details.filter(([,value]) => value !== undefined && value !== null).map(([key,value]) => `<dt>${key}</dt><dd>${escapeHtml(value)}</dd>`).join("");
  const entries = [...(result.evidence || []).map((item) => ({label:"根拠", value:item.description || JSON.stringify(item)})), ...(result.uncertainties || []).map((item) => ({label:"不確実性", value:item}))];
  $("vlm-evidence").innerHTML = entries.map((item) => `<div class="evidence"><strong>${item.label}:</strong> ${escapeHtml(item.value)}</div>`).join("");
  $("vlm-result").textContent = JSON.stringify(result, null, 2);
}
function renderActivity(activity) {
  if (!activity) { $("activity-label").textContent = "未検出"; $("activity-confidence").textContent = "—"; $("activity-duration").textContent = "—"; return; }
  $("activity-label").textContent = text(activity.label);
  $("activity-confidence").textContent = `${Math.round((activity.confidence || 0) * 100)}%`;
  $("activity-duration").textContent = `継続時間: ${Math.round(activity.duration_seconds || 0)} 秒`;
}
function renderEvents(events) {
  state.events = events || [];
  $("events").innerHTML = state.events.length ? state.events.map((event) => {
    const severity = String(event.severity || "").toLowerCase();
    const evidence = event.evidence || event.evidence_json;
    const evidenceText = typeof evidence === "string" ? evidence : evidence ? JSON.stringify(evidence) : "";
    return `<article class="event ${severity}"><div class="event-head"><span class="event-type">${escapeHtml(event.event_type || "event")}</span><span class="event-time">${timestamp(event.timestamp)}</span></div><p class="event-message">${escapeHtml(event.message || event.reason || "詳細なし")}</p>${evidenceText ? `<p class="event-evidence">根拠: ${escapeHtml(evidenceText)}</p>` : ""}</article>`;
  }).join("") : '<p class="muted">イベントはまだありません。</p>';
}
function renderMetrics(data) {
  $("metric-fps").textContent = data.fps !== undefined ? `${data.fps} fps` : "—";
  $("metric-inference").textContent = data.inference_ms !== undefined ? `${data.inference_ms} ms` : "—";
  $("metric-vlm").textContent = text(state.vlmCalls, "0");
  $("metric-time").textContent = data.timestamp ? timestamp(data.timestamp) : "—";
  $("metrics").textContent = `FPS ${text(data.fps)}, 推論 ${text(data.inference_ms)} ms, VLM ${state.vlmCalls}`;
}
function renderProgress(progress) {
  state.progress = Number(progress || 0); const percent = Math.round(state.progress * 100);
  $("progress-bar").style.width = `${percent}%`; $("progress-label").textContent = `${percent}%`;
  $("progress-bar").parentElement.setAttribute("aria-valuenow", String(percent));
}
function handleFrame(data, source = "websocket") {
  state.lastFrame = data; state.source = source === "websocket" ? "ライブ" : "HTTP";
  $("source-badge").textContent = state.source;
  $("canvas-empty").classList.add("hidden"); draw(data); renderCurrentStep(data.current_step, data.condition);
  if (data.recent_events) renderEvents(data.recent_events); if (data.vlm_result) renderVlm(data.vlm_result);
  if (data.activity !== undefined) renderActivity(data.activity);
  if (data.vlm_calls !== undefined) state.vlmCalls = data.vlm_calls;
  else if (data.vlm_result && data.timestamp !== state.lastVlmTimestamp) {
    state.vlmCalls += 1; state.lastVlmTimestamp = data.timestamp;
  }
  renderMetrics(data);
}
function applyStatus(data) {
  state.session = data.session || null; state.steps = data.steps || []; state.vlmCalls = data.vlm_calls ?? state.vlmCalls;
  renderProgress(data.progress); renderSteps(); renderCurrentStep(data.current_step, state.lastFrame?.condition); updateButtons();
}

function stopPolling() { if (state.pollingTimer) clearInterval(state.pollingTimer); state.pollingTimer = null; }
async function refreshFallback() {
  try { const [status, events] = await Promise.all([call("/api/session/status"), call("/api/events")]); applyStatus(status); renderEvents(events); }
  catch (error) { setNotice(`状態を取得できません: ${error.message}`, true); }
}
function startPolling() { if (state.pollingTimer || isSocketOpen()) return; refreshFallback(); state.pollingTimer = setInterval(refreshFallback, 2500); }
function connectWebSocket() {
  if (isSocketOpen() || state.socket?.readyState === WebSocket.CONNECTING) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/api/ws`); state.socket = socket;
  socket.onopen = () => { setConnection(true, "ライブ接続中"); stopPolling(); };
  socket.onmessage = (event) => { try { const data = JSON.parse(event.data); if (data.type === "frame_result") handleFrame(data); } catch { setNotice("ライブデータの解析に失敗しました。", true); } };
  socket.onerror = () => socket.close();
  socket.onclose = () => { if (state.socket !== socket) return; setConnection(false, "接続断：HTTPで状態を更新中"); startPolling(); clearTimeout(state.reconnectTimer); state.reconnectTimer = setTimeout(connectWebSocket, 2000); };
}
function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null; camera.srcObject = null; camera.pause();
  if (state.previewUrl) { URL.revokeObjectURL(state.previewUrl); state.previewUrl = null; }
  camera.removeAttribute("src"); camera.load();
}
function scheduleFrame() { clearTimeout(state.frameTimer); if (state.running) state.frameTimer = setTimeout(sendCameraFrame, 150); }
async function sendCameraFrame() {
  if (!state.running || state.requestInFlight) return scheduleFrame();
  if (camera.readyState < 2) return scheduleFrame();
  state.requestInFlight = true;
  try {
    const image = document.createElement("canvas"); image.width = canvas.width; image.height = canvas.height; image.getContext("2d").drawImage(camera, 0, 0, image.width, image.height);
    const blob = await new Promise((resolve) => image.toBlob(resolve, "image/jpeg", .85)); if (!blob) throw new Error("カメラフレームを作成できませんでした");
    const form = new FormData(); form.append("file", blob, "camera.jpg"); const data = await call("/api/analyze/image", {method:"POST", body:form});
    if (!isSocketOpen()) handleFrame(data, "http");
  } catch (error) { state.running = false; stopCamera(); setNotice(`カメラエラー: ${error.message}`, true); updateButtons(); }
  finally { state.requestInFlight = false; scheduleFrame(); }
}
async function startSession(sourceType, sourceName) {
  const data = await call("/api/session/start", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({sop_id:$("sop").value, source_type:sourceType, source_name:sourceName})}); applyStatus(data); return data;
}

$("start").addEventListener("click", async () => {
  try { state.stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}}, audio:false}); camera.srcObject = state.stream; await camera.play(); await startSession("camera", "browser camera"); state.running = true; setNotice("カメラを解析中です。WebSocketで結果を受信します。"); updateButtons(); sendCameraFrame(); }
  catch (error) { stopCamera(); setNotice(`カメラを開始できません: ${error.message}`, true); }
});
$("pause").addEventListener("click", () => { state.running = false; stopCamera(); setNotice("カメラ解析を一時停止しました。"); updateButtons(); });
$("stop").addEventListener("click", async () => { state.running = false; stopCamera(); try { applyStatus(await call("/api/session/stop", {method:"POST"})); setNotice("セッションを停止しました。"); } catch (error) { setNotice(`停止できません: ${error.message}`, true); } updateButtons(); });
$("reset").addEventListener("click", async () => { state.running = false; stopCamera(); try { await call("/api/session/stop", {method:"POST"}); state.lastFrame = null; renderEvents([]); await startSession("mock", "dashboard reset"); setNotice("セッションをリセットしました。カメラまたは動画を開始できます。"); } catch (error) { setNotice(`リセットできません: ${error.message}`, true); } updateButtons(); });
$("vlm").addEventListener("click", async () => { try { $("vlm").disabled = true; const result = await call("/api/vlm/analyze", {method:"POST"}); handleFrame(result, "http"); await refreshFallback(); setNotice("VLMを手動実行しました。"); } catch (error) { setNotice(`VLM解析に失敗しました: ${error.message}`, true); } finally { $("vlm").disabled = false; } });
$("upload").addEventListener("click", () => $("video").click());
$("video").addEventListener("change", async (event) => {
  const file = event.target.files?.[0]; if (!file) return; state.running = false; stopCamera(); updateButtons();
  try {
    state.previewUrl = URL.createObjectURL(file); camera.src = state.previewUrl; camera.load();
    await camera.play().catch(() => undefined);
    setNotice(`${file.name} をアップロード・解析中…`); await startSession("video", file.name);
    const form = new FormData(); form.append("file", file); const result = await call("/api/analyze/video", {method:"POST", body:form});
    handleFrame(result.last, "http"); await refreshFallback(); setNotice(`${file.name}: ${result.frames_processed} フレームを解析しました。`);
  }
  catch (error) { setNotice(`動画を解析できません: ${error.message}`, true); }
  finally { event.target.value = ""; updateButtons(); }
});
$("refresh-events").addEventListener("click", refreshFallback);
window.addEventListener("beforeunload", () => { stopCamera(); clearTimeout(state.reconnectTimer); stopPolling(); state.socket?.close(); });

(async () => { connectWebSocket(); try { const [sops, status, events] = await Promise.all([call("/api/sops"), call("/api/session/status"), call("/api/events")]); $("sop").innerHTML = sops.map((sop) => `<option value="${escapeHtml(sop.id)}">${escapeHtml(sop.name)} (${escapeHtml(sop.version)})</option>`).join("") || $("sop").innerHTML; applyStatus(status); renderEvents(events); } catch (error) { setNotice(`初期情報を取得できません: ${error.message}`, true); startPolling(); } updateButtons(); })();
