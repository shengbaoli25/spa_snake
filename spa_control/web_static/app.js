(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const ui = {port:$("port-select"), refresh:$("refresh-ports"), connect:$("connect"), disconnect:$("disconnect"), light:$("light"), badge:$("camera-badge"), frame:$("camera-frame"), empty:$("video-empty"), sequence:$("video-sequence"), frames:$("frames"), age:$("age"), link:$("link-state"), status:$("camera-status"), serviceError:$("service-error"), seed:$("seed"), reset:$("sim-reset"), stepButton:$("sim-step-btn"), train:$("sim-train"), simStep:$("sim-step"), reward:$("sim-reward"), action:$("sim-action"), policyVersion:$("policy-version"), trainBatches:$("train-batches"), lastLoss:$("last-loss"), simStatus:$("sim-status"), chart:$("trend"), chartEmpty:$("chart-empty"), chartLabel:$("chart-label")};
  let polling = false, connected = false, imageKey = null, lastCamera = null, history = [], commandBusy = false, simulationBusy = false, recoveringSimulation = false;
  let motorBusy = false, recordBusy = false, lastMotor = {}, lastRecord = {}, motorErrorUntil = 0, recordErrorUntil = 0, clearErrorUntil = 0, clearFeedback = "";
  let lastExperiment = {}, experimentBusy = false, experimentDirty = true, experimentHydrated = false, experimentTouched = false, experimentErrorUntil = 0, motorTestErrorUntil = 0, stopBusy = false, heartbeatBusy = false;
  const finite = x => typeof x === "number" && Number.isFinite(x);
  function number(x, digits = 2) {return finite(x) ? x.toFixed(digits) : "—";}
  function displayAction(x) {if (Array.isArray(x)) return x.map(v => number(v, 2)).join(" / "); return number(x, 2);}
  async function api(url, options = {}, timeoutMs = 5000) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {cache:"no-store", ...options, signal:controller.signal});
      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("json") ? await response.json() : {};
      if (!response.ok) throw new Error(data.error || data.message || `请求失败 (${response.status})`);
      return data;
    } finally { clearTimeout(timer); }
  }
  const post = (url, data, timeoutMs = 5000) => api(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)}, timeoutMs);
  async function ports() {
    ui.refresh.disabled = true;
    try {
      const data = await api("/api/ports"), chosen = ui.port.value;
      ui.port.replaceChildren(new Option("选择串口", ""));
      for (const item of (Array.isArray(data.ports) ? data.ports : [])) {
        if (!item || !item.device || item.available_for_camera === false) continue;
        const option = new Option(`${item.device} · ${item.friendly_name || item.description || "串口"}`, item.device);
        option.title = item.hwid || "";
        option.dataset.preferred = item.preferred === true ? "true" : "false";
        ui.port.add(option);
      }
      if ([...ui.port.options].some(option => option.value === chosen)) ui.port.value = chosen;
      if (!ui.port.value) {
        const preferred = [...ui.port.options].find(option => option.dataset.preferred === "true");
        if (preferred) ui.port.value = preferred.value;
      }
      ui.status.textContent = ui.port.options.length > 1 ? "选择 SPA-CAM 串口，点击连接。" : "未发现串口，请检查相机连接。";
    } catch (error) { ui.status.textContent = `无法读取串口：${error.message}`; }
    finally {ui.refresh.disabled = false;}
  }
  function renderCamera(camera = {}) {
    lastCamera = camera;
    connected = !!(camera.active ?? camera.connected);
    const hasFrame = finite(camera.received) && camera.received > 0;
    const stale = camera.frame_stale === true || (finite(camera.age_s) && camera.age_s > 2);
    ui.badge.classList.toggle("live", connected && hasFrame && !stale);
    const stateLabel = !connected ? "未连接" : !hasFrame ? "连接中" : stale ? "画面过时" : "已连接";
    ui.badge.innerHTML = `<i></i> ${stateLabel}`;
    ui.link.textContent = stateLabel;
    ui.connect.disabled = connected || commandBusy;
    ui.disconnect.disabled = !connected || commandBusy;
    ui.port.disabled = connected || commandBusy;
    ui.light.disabled = !connected || commandBusy;
    if (typeof camera.light === "boolean" && !commandBusy) ui.light.checked = camera.light;
    ui.status.textContent = camera.status || (connected ? "相机已连接，等待图像。" : "未连接相机");
    const count = camera.received ?? camera.frames ?? camera.frame_count;
    ui.frames.innerHTML = `${finite(count) ? Math.floor(count) : "—"} <em>帧</em>`;
    const age = camera.age_s ?? camera.frame_age_s;
    ui.age.innerHTML = `${finite(age) ? number(age, 1) : "—"} <em>秒</em>`;
    const seq = camera.sequence ?? camera.frame_sequence;
    ui.sequence.textContent = finite(seq) ? `FRAME ${seq}` : "等待画面";
    const key = finite(count) && count > 0 ? `${camera.boot_id ?? ""}:${seq ?? count}:${count}` : null;
    if (connected && !stale && key && key !== imageKey) {
      imageKey = key;
      ui.frame.src = `/api/camera/frame?t=${encodeURIComponent(key)}`;
    }
    if (!connected || stale) {imageKey = null; ui.frame.hidden = true; ui.empty.hidden = false; ui.frame.removeAttribute("src");}
  }
  ui.frame.addEventListener("load", () => {if (connected && lastCamera && !lastCamera.frame_stale && (!finite(lastCamera.age_s) || lastCamera.age_s <= 2)) {ui.frame.hidden = false; ui.empty.hidden = true;}});
  ui.frame.addEventListener("error", () => {ui.frame.hidden = true; ui.empty.hidden = false; imageKey = null;});
  function renderSimulation(sim = {}) {
    const step = sim.step ?? sim.steps ?? sim.step_count;
    const reward = sim.reward ?? sim.last_reward;
    const action = sim.action ?? sim.last_action ?? sim.residual;
    ui.simStep.textContent = finite(step) ? String(step) : "—";
    ui.reward.textContent = number(reward, 3);
    ui.action.textContent = displayAction(action);
    ui.policyVersion.textContent = finite(sim.policy_version) ? String(sim.policy_version) : "—";
    ui.trainBatches.textContent = finite(sim.train_batches) ? String(sim.train_batches) : "—";
    ui.lastLoss.textContent = number(sim.last_loss, 4);
    if (Array.isArray(sim.history)) {
      history = sim.history.map(item => typeof item === "number" ? item : (item?.reward ?? item?.value)).filter(finite).slice(-120);
    }
    if (sim.status || sim.notice) ui.simStatus.textContent = sim.status || sim.notice;
    draw();
  }
  function motorFaultSummary() {
    if (!lastMotor.fault || lastMotor.fault === "NONE") return "";
    const details = [String(lastMotor.fault)];
    if (lastMotor.fault === "OVERCURRENT") {
      const mask = Number.isInteger(lastMotor.trip_mask) ? lastMotor.trip_mask : 0;
      const channels = [1,2,3,4].filter(id => mask & (1 << (id - 1))).map(id => `M${id}`);
      if (channels.length) details.push(channels.join(" / "));
      if (finite(lastMotor.trip_current)) details.push(`电流原始值 ${lastMotor.trip_current}`);
    }
    return `故障：${details.join(" · ")}。`;
  }
  function renderClearControls() {
    const fresh = lastMotor.state === "connected" && lastMotor.frame_fresh === true;
    const clearing = ["checking", "waiting_ack", "verifying"].includes(lastMotor.clear_state);
    const disabled = motorBusy || experimentBusy || stopBusy || clearing || controlActive() || !fresh || lastMotor.clear_supported !== true;
    for (const id of ["motor-clear-usb", "experiment-clear-fault"]) $(id).disabled = disabled;
    const fault = fresh ? motorFaultSummary() : "";
    let message;
    if (Date.now() < clearErrorUntil) message = clearFeedback;
    else if (clearing || lastMotor.clear_state === "failed") message = lastMotor.clear_message || "等待清除结果。";
    else if (!fresh) message = lastMotor.state === "connected" ? "反馈过时，等待 C 板状态。" : "连接 C 板后可清除故障。";
    else if (lastMotor.clear_state === "succeeded" && !fault) message = "故障已清除，保持停机。";
    else if (fault) message = lastMotor.clear_supported === true ? "点击清除故障；清除后保持停机。" : "当前故障不支持网页清除。";
    else message = "清除后保持停机。";
    for (const id of ["motor-clear-status", "experiment-clear-status"]) $(id).textContent = fault + message;
  }
  function renderMotor(motor = {}) {
    lastMotor = motor || {};
    const state = lastMotor.state || "disconnected";
    const active = state === "connected", pending = state === "connecting" || state === "disconnecting";
    const fresh = active && lastMotor.frame_fresh === true;
    const labels = {disconnected:"未连接", connecting:"连接中", disconnecting:"断开中", connected:fresh ? "反馈在线" : "遥测过时", error:"连接异常"};
    $("motor-badge").classList.toggle("live", fresh);
    $("motor-badge").innerHTML = `<i></i> ${labels[state] || "未知"}`;
    $("motor-connect").disabled = motorBusy || pending || active;
    $("motor-disconnect").disabled = motorBusy || !["connected", "connecting"].includes(state);
    const clearing = ["checking", "waiting_ack", "verifying"].includes(lastMotor.clear_state);
    const moving = motionActive(lastMotor.motion), experimentActive = !!lastExperiment.active;
    renderClearControls();
    if (motorBusy || clearing || stopBusy) for (const id of ["experiment-baseline", "experiment-collect", "experiment-pretrain", "experiment-train"]) $(id).disabled = true;
    if (Date.now() >= motorErrorUntil) $("motor-status").textContent = fresh && motorFaultSummary() ? `${motorFaultSummary()}${lastMotor.message || ""}` : lastMotor.message || "未连接 C 板";
    $("motor-port").textContent = lastMotor.port || "—";
    $("motor-stop").textContent = lastMotor.stop_acknowledged ? "已确认" : lastMotor.stop_sent ? "已发送，未确认" : "—";
    $("motor-can").textContent = !fresh ? "未知" : !lastMotor.can_ready ? "未就绪" : lastMotor.can_healthy ? "正常" : "异常";
    $("motor-fault").textContent = fresh && lastMotor.fault != null ? String(lastMotor.fault) : "未知";
    const dirs = lastMotor.directions;
    $("motor-directions").textContent = fresh && dirs ? `${dirs["3"] ?? "—"} / ${dirs["4"] ?? "—"}` : "—";
    for (const channel of [3,4]) {
      const m = lastMotor.motors?.[String(channel)] || {};
      const available = fresh && m.fresh === true;
      $(`m${channel}-fresh`).textContent = available ? "在线" : m.seen ? "反馈过时" : "无反馈";
      $(`m${channel}-fresh`).classList.toggle("fresh", available);
      for (const [field, key] of [["speed","spd"],["current","torque"],["target","target"],["output","setV"],["temp","temp"]]) {
        $(`m${channel}-${field}`).textContent = available && finite(m[key]) ? String(m[key]) : "—";
      }
      $(`m${channel}-age`).textContent = finite(m.feedback_age_s) ? number(m.feedback_age_s, 2) : "—";
    }
    const testReady = fresh && lastMotor.fault === "NONE" && lastMotor.can_ready && lastMotor.can_healthy && [3,4].every(id => lastMotor.motors?.[String(id)]?.fresh === true);
    $("test-m3").disabled = $("test-m4").disabled = motorBusy || experimentBusy || moving || experimentActive || clearing || !testReady;
    $("test-rpm").disabled = $("test-duration").disabled = moving || experimentActive;
    if (Date.now() >= motorTestErrorUntil) {
      const motion = lastMotor.motion || {}, labels = {idle:"尚未测试", preparing:"正在准备", running:"正在运行", stopping:"正在停止并确认零输出", completed:"测试完成", failed:"测试已终止"};
      const remaining = finite(motion.remaining_s) && moving ? `；剩余 ${number(motion.remaining_s, 1)} 秒` : "";
      const detail = ["failed", "stopping"].includes(motion.state) ? `。${motion.message || ""}` : motion.state === "completed" ? (motion.stop_confirmed ? "，已停止。" : "；停止尚未确认。") : "。";
      $("motor-test-status").textContent = motion.state && motion.state !== "idle" ? `${labels[motion.state] || motion.state}${remaining}${detail}` : testReady ? "已就绪，点击 M3 或 M4 测试。" : "请连接 C 板，确认故障已清除、两路在线。";
    }
  }
  function motionActive(motion = {}) {return ["preparing","running","stopping"].includes(motion.state) || motion.active === true;}
  function controlActive() {return motionActive(lastMotor.motion) || lastExperiment.active === true;}
  async function heartbeat() {
    if (heartbeatBusy || !controlActive()) return;
    heartbeatBusy = true;
    try {await post("/api/control/heartbeat", {}, 1800);} catch (_) { /* The server stops if the page lease expires. */ }
    finally {heartbeatBusy = false;}
  }
  function renderRecord(record = {}) {
    lastRecord = record || {};
    const active = !!lastRecord.active;
    $("record-start").disabled = recordBusy || active;
    $("record-stop").disabled = recordBusy || !active || lastRecord.state === "stopping";
    $("record-duration").disabled = recordBusy || active;
    const labels = {idle:"尚未记录", recording:"正在记录", stopping:"正在停止", completed:"记录完成", stopped:"记录已停止", error:"记录失败"};
    if (Date.now() >= recordErrorUntil) $("record-status").textContent = (labels[lastRecord.state] || "尚未记录") + (lastRecord.error ? `：${lastRecord.error}` : "");
    $("record-samples").textContent = finite(lastRecord.samples) ? String(lastRecord.samples) : "0";
    $("record-images").textContent = finite(lastRecord.images) ? String(lastRecord.images) : "0";
    $("record-directory").textContent = lastRecord.directory || "—";
  }
  async function deviceCommand(kind, url, data) {
    if (kind === "motor" ? motorBusy : recordBusy) return;
    if (kind === "motor") { motorBusy = true; renderMotor(lastMotor); }
    else { recordBusy = true; renderRecord(lastRecord); }
    try {await post(url, data); await poll();}
    catch (error) {
      const field = kind === "motor" ? "motor-status" : "record-status";
      $(field).textContent = `操作失败或状态未确认：${error.message}`;
      if (kind === "motor") motorErrorUntil = Date.now() + 8000;
      else recordErrorUntil = Date.now() + 8000;
    } finally {
      if (kind === "motor") {motorBusy = false; renderMotor(lastMotor);}
      else {recordBusy = false; renderRecord(lastRecord);}
    }
  }
  function draw() {
    const box = ui.chart.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (!box.width || !box.height) return;
    ui.chart.width = Math.round(box.width * ratio); ui.chart.height = Math.round(box.height * ratio);
    const ctx = ui.chart.getContext("2d"); ctx.scale(ratio, ratio);
    ui.chartEmpty.hidden = history.length > 0;
    ui.chartLabel.textContent = history.length ? `${history.length} 个样本 · 模拟奖励` : "等待模拟";
    if (!history.length) return;
    const min = Math.min(...history), max = Math.max(...history), spread = Math.max(max - min, 0.001), pad = 12, w = box.width - 2 * pad, h = box.height - 2 * pad;
    const points = history.map((v, i) => [pad + (history.length === 1 ? 0 : i / (history.length - 1) * w), pad + h - ((v - min) / spread) * h]);
    ctx.beginPath(); points.forEach(([x,y],i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.strokeStyle = "#298969"; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
    const [x,y] = points[points.length - 1]; ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fillStyle="#298969"; ctx.fill();
  }
  async function poll() {
    if (polling) return;
    polling = true;
    try {const data = await api("/api/status"); ui.serviceError.classList.add("hidden"); renderCamera(data.camera); renderGlobal(data.global_camera || {}); renderSimulation(data.simulation); renderExperiment(data.experiment || {}); renderMotor(data.motor); renderRecord(data.recorder); heartbeat(); if (referenceInput && !referenceBusy && Date.now()-referenceLast > 900) updateReference(); if (recoveringSimulation) {recoveringSimulation = false; simulationButtons();}}
    catch (error) {ui.serviceError.textContent = `本地服务连接失败：${error.message}。正在重试；运行连接中断会触发自动停止。`; ui.serviceError.classList.remove("hidden"); ui.frame.hidden = true; ui.empty.hidden = false; renderGlobal({}); clearMeasured(); renderMotor({...lastMotor, frame_fresh:false}); for (const id of ["experiment-baseline","experiment-collect","experiment-pretrain","experiment-train","experiment-configure"]) $(id).disabled=true;}
    finally {polling = false;}
  }
  async function command(url, data, status) {
    if (commandBusy) return;
    commandBusy = true; renderCamera(lastCamera || {});
    let failure = null;
    try {await post(url, data); await poll();}
    catch (error) {failure = `操作失败：${error.message}`;}
    finally {commandBusy = false; renderCamera(lastCamera || {}); if (failure) status.textContent = failure;}
  }
  ui.refresh.addEventListener("click", ports);
  ui.connect.addEventListener("click", () => {if (!ui.port.value) {ui.status.textContent = "请先选择相机串口。"; return;} command("/api/camera/connect", {port:ui.port.value}, ui.status);});
  ui.disconnect.addEventListener("click", () => command("/api/camera/disconnect", {}, ui.status));
  ui.light.addEventListener("change", () => command("/api/camera/light", {enabled:ui.light.checked}, ui.status));
  let globalCamera = {}, globalBusy = false, pointMode = false, roiMode = false, dragStart = null, points = [], globalPointsEdited = false, globalDirectionEdited = false, referenceInput = null, referenceBusy = false, referenceLast = 0, origin = null, referenceGeneration = 0;
  const globalFrame = $("global-frame"), globalWrap = $("global-frame-wrap"), overlay = $("global-overlay");
  function validGlobal(g) {return g && g.valid === true && finite(g.sample_age_s) && g.sample_age_s <= 0.5 && finite(g.position_m) && finite(g.velocity_mps);}
  function clearMeasured() {for (const id of ["reference-actual","reference-error","reference-reward","reference-travel"]) $(id).textContent = "—";}
  function renderGlobal(g) {
    globalCamera = g || {};
    if (!globalBusy && !globalDirectionEdited && ["left","right"].includes(g.direction)) $("global-direction").value=g.direction;
    if (!globalBusy && !globalPointsEdited && !pointMode && Array.isArray(g.points)) {points=g.points.map(p=>({...p}));updatePoints();}
    const active = g.state === "connected", live = active && finite(g.frame_age_s) && g.frame_age_s <= 2;
    $("global-badge").classList.toggle("live", live);
    $("global-badge").innerHTML = `<i></i> ${live ? "画面在线" : (g.state === "connecting" ? "连接中" : "未连接")}`;
    $("global-connect").disabled = globalBusy || ["connecting","connected","disconnecting"].includes(g.state);
    $("global-disconnect").disabled = globalBusy || !["connecting","connected"].includes(g.state);
    $("global-status").textContent = g.message || "未连接全局相机";
    $("global-valid").textContent = validGlobal(g) ? "有效" : (g.note || "不可用");
    $("global-position").textContent = validGlobal(g) ? number(g.position_m, 4) : "—";
    $("global-speed").textContent = validGlobal(g) ? number(g.velocity_mps, 4) : "—";
    if (live && !dragStart) {globalFrame.src = `/api/global-camera/frame?t=${Date.now()}`;}
    if (!live) {globalFrame.hidden = true; $("global-empty").hidden = false; if (g.state !== "connected") origin = null;}
    if (!validGlobal(g)) {origin = null; clearMeasured();}
    else if (origin !== null) $("reference-travel").textContent = number(g.position_m-origin, 4);
    drawGlobalOverlay();
  }
  globalFrame.addEventListener("load", () => {globalFrame.hidden = false; $("global-empty").hidden = true; drawGlobalOverlay();});
  globalFrame.addEventListener("error", () => {globalFrame.hidden = true; $("global-empty").hidden = false;});
  async function globalCommand(path, data) {
    if (globalBusy) return;
    globalBusy = true; renderGlobal(globalCamera);
    try {await post(`/api/global-camera/${path}`, data); if (["calibration","load-calibration"].includes(path)) {globalPointsEdited=false;globalDirectionEdited=false;} $("global-action-status").textContent = "操作已提交"; await poll();}
    catch (e) {$("global-action-status").textContent = `操作失败：${e.message}`;}
    finally {globalBusy = false; renderGlobal(globalCamera);}
  }
  $("global-connect").addEventListener("click", () => {const n = Number($("global-index").value); if (!Number.isInteger(n) || n < 0 || n > 20) {$("global-action-status").textContent="相机编号需为 0–20 的整数。"; return;} origin = null; globalCommand("connect", {camera_index:n});});
  $("global-disconnect").addEventListener("click", () => {origin = null; globalCommand("disconnect", {});});
  $("global-direction").addEventListener("change", () => {globalDirectionEdited=true; $("global-action-status").textContent="头部前沿已更改，应用标定后生效。";});
  $("global-mark").addEventListener("click", () => {if (!$("global-point-position").value.trim() || !Number.isFinite(Number($("global-point-position").value))) {$("global-action-status").textContent="先输入标定点对应的实际轨道位置。"; return;} pointMode = true; roiMode = false; $("global-action-status").textContent="请点击画面中该位置对应的点。";});
  $("global-select-roi").addEventListener("click", () => {pointMode = false; roiMode = true; $("global-action-status").textContent="请在画面中拖框选取静止机器人的头部。";});
  $("global-reset-points").addEventListener("click", () => {points=[];globalPointsEdited=true;origin=null;updatePoints();drawGlobalOverlay();});
  $("global-save-calibration").addEventListener("click", () => {if (points.length < 2) {$("global-action-status").textContent="请先选择至少两个标定点。";return;} origin=null; globalCommand("calibration", {points, direction:$("global-direction").value});});
  $("global-load-calibration").addEventListener("click", () => {origin=null;globalCommand("load-calibration", {direction:$("global-direction").value});});
  $("global-clear-roi").addEventListener("click", () => {origin=null; globalCommand("clear-roi", {});});
  function imagePoint(e) {
    if (globalFrame.hidden || !Array.isArray(globalCamera.frame_size)) return null;
    const r = globalFrame.getBoundingClientRect(), [w,h] = globalCamera.frame_size;
    if (!r.width || !r.height || e.clientX<r.left || e.clientX>r.right || e.clientY<r.top || e.clientY>r.bottom) return null;
    return [Math.max(0,Math.min(w-1,(e.clientX-r.left)*w/r.width)), Math.max(0,Math.min(h-1,(e.clientY-r.top)*h/r.height))];
  }
  function updatePoints() {$("global-points").textContent = points.length ? `已选 ${points.length} 点：${points.map(p => `${Number(Number(p.position_m).toFixed(3))} m`).join("、")}` : "尚未选择标定点";}
  globalWrap.addEventListener("pointerdown", e => {const p=imagePoint(e); if (!p) return;
    if (pointMode) {const pos=Number($("global-point-position").value);globalPointsEdited=true;points.push({image_x_px:p[0], image_y_px:p[1], position_m:pos});pointMode=false;updatePoints();drawGlobalOverlay();$("global-action-status").textContent="已添加标定点；输入下一点位置后继续点选。";}
    else if (roiMode) {dragStart=p; globalWrap.setPointerCapture(e.pointerId); e.preventDefault();}
  });
  globalWrap.addEventListener("pointerup", e => {if (!dragStart) return; const end=imagePoint(e), start=dragStart; dragStart=null; roiMode=false;
    if (!end) {$("global-action-status").textContent="请在画面内完成拖框。";return;} const x=Math.round(Math.min(start[0],end[0])), y=Math.round(Math.min(start[1],end[1])), w=Math.round(Math.abs(end[0]-start[0])), h=Math.round(Math.abs(end[1]-start[1]));
    if (w<5 || h<5) {$("global-action-status").textContent="跟踪框太小，请重新拖框。";return;} origin=null; globalCommand("roi", {bbox:[x,y,w,h]});
  });
  globalWrap.addEventListener("pointercancel", () => {dragStart=null;roiMode=false;});
  function drawGlobalOverlay() {
    const size=globalCamera.frame_size; if (!Array.isArray(size) || globalFrame.hidden) {overlay.replaceChildren();return;}
    overlay.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`); overlay.setAttribute("preserveAspectRatio", "none"); overlay.replaceChildren();
    const ns="http://www.w3.org/2000/svg", selected=globalPointsEdited ? points : (points.length ? points : (globalCamera.points || []));
    for (const p of selected) {const c=document.createElementNS(ns,"circle"); c.setAttribute("cx",p.image_x_px);c.setAttribute("cy",p.image_y_px);c.setAttribute("r",Math.max(4,size[0]/180));c.setAttribute("fill","#ffcf5b");c.setAttribute("stroke","#26372f");overlay.append(c);}
    const b=globalCamera.bbox; if (Array.isArray(b) && b.length===4) {const r=document.createElementNS(ns,"rect");["x","y","width","height"].forEach((key,i)=>r.setAttribute(key,b[i]));r.setAttribute("fill","none");r.setAttribute("stroke","#5fffb5");r.setAttribute("stroke-width",Math.max(2,size[0]/500));overlay.append(r);}
  }
  function inputNum(id) {const value=$(id).value.trim(); return value ? Number(value) : null;}
  $("reference-preview").addEventListener("click", () => {const rpm=inputNum("reference-rpm"), lead=inputNum("reference-lead"), ratio=inputNum("reference-ratio"), distance=inputNum("reference-distance");
    if (![rpm,lead,ratio].every(finite) || !Number.isInteger(rpm) || Math.abs(rpm)>500 || lead<=0 || ratio<=0 || (distance!==null && (!finite(distance)||distance<=0))) {$("reference-status").textContent="请填写 ±500 内的整数转速、正数导程和传动比；距离如填写须为正数。"; referenceInput=null; referenceGeneration++; clearMeasured(); $("reference-theory").textContent="—"; $("reference-goal").textContent="—"; return;}
    referenceInput={motor_rpm:rpm,lead_mm:lead,motor_to_helix_ratio:ratio,forward_sign:Number($("reference-sign").value),distance_m:distance}; referenceGeneration++; updateReference();
  });
  for (const id of ["reference-rpm","reference-lead","reference-ratio","reference-distance","reference-sign"]) $(id).addEventListener("input", () => {referenceInput=null; referenceGeneration++; clearMeasured(); $("reference-theory").textContent="—"; $("reference-goal").textContent="—"; $("reference-status").textContent="参数已更改，点击计算目标速度。";});
  $("reference-origin").addEventListener("click", () => {if (!validGlobal(globalCamera)) {$("reference-status").textContent="先连接相机，完成标定并取得新鲜有效的跟踪。";return;} origin=globalCamera.position_m; $("reference-travel").textContent=number(0,4); $("reference-status").textContent="已设置相机起点；显示的是有符号位移。";});
  async function updateReference() {if (!referenceInput || referenceBusy) return; referenceBusy=true; referenceLast=Date.now(); const generation=referenceGeneration;
    try {const d=await post("/api/kinematics/preview", referenceInput); if (generation!==referenceGeneration) return;
      $("reference-theory").textContent=number(d.theory_speed_mps,4); $("reference-goal").textContent=number(d.distance_m,3);
      if (validGlobal(globalCamera) && d.tracking) {$("reference-actual").textContent=number(d.tracking.measured_mps,4); $("reference-error").textContent=number(d.tracking.error_mps,4); $("reference-reward").textContent=number(d.tracking.tracking_reward,4);} else clearMeasured();
      if (validGlobal(globalCamera) && origin!==null) $("reference-travel").textContent=number(globalCamera.position_m-origin,4);
      $("reference-status").textContent="理论速度按无滑移计算；此处奖励仅含速度项，不启动电机。";
    } catch(e) {if (generation===referenceGeneration) {clearMeasured(); $("reference-theory").textContent="—"; $("reference-status").textContent=`计算失败：${e.message}`;}}
    finally {referenceBusy=false;}
  }
  $("motor-connect").addEventListener("click", () => deviceCommand("motor", "/api/motor/connect", {}));
  $("motor-disconnect").addEventListener("click", () => deviceCommand("motor", "/api/motor/disconnect", {}));
  async function clearMotorFault(event) {
    renderClearControls();
    if (event.currentTarget.disabled) return;
    motorBusy = true; clearErrorUntil = Date.now() + 8000; clearFeedback = "正在请求清除，等待板端确认…";
    renderMotor(lastMotor); renderExperiment(lastExperiment);
    try { await post("/api/motor/clear-fault", {}); clearErrorUntil = 0; await poll(); }
    catch (error) { clearErrorUntil = Date.now() + 8000; clearFeedback = `清除失败或结果未确认：${error.message}`; }
    finally { motorBusy = false; renderMotor(lastMotor); renderExperiment(lastExperiment); }
  }
  for (const id of ["motor-clear-usb", "experiment-clear-fault"]) $(id).addEventListener("click", clearMotorFault);
  for (const motor of [3,4]) $(`test-m${motor}`).addEventListener("click", async () => {
    if ($(`test-m${motor}`).disabled || motorBusy) return;
    const rpm=inputNum("test-rpm"), duration=inputNum("test-duration");
    if (!finite(rpm) || !Number.isInteger(rpm) || !rpm || Math.abs(rpm)>500 || !finite(duration) || duration<0.5 || duration>5) {
      motorTestErrorUntil=Date.now()+8000; $("motor-test-status").textContent="填写非零整数转速（−500 至 500 rpm）和 0.5–5 秒时长。"; return;
    }
    motorBusy=true; motorTestErrorUntil=Date.now()+5000; renderMotor(lastMotor);
    $("motor-test-status").textContent=`准备测试 M${motor}：${rpm} rpm，${duration} 秒…`;
    try {await post("/api/motor/test", {motor,rpm,duration_s:duration}); motorTestErrorUntil=0; await poll();}
    catch (e) {motorTestErrorUntil=Date.now()+10000; $("motor-test-status").textContent=`测试未开始或结果未确认：${e.message}。可随时按停止。`;}
    finally {motorBusy=false; renderMotor(lastMotor);}
  });
  async function stopAll() {
    if (stopBusy) return;
    stopBusy=true; renderClearControls(); $("stop-status").textContent="正在发送停止，等待板端确认…";
    try {await post("/api/motor/stop", {}); $("stop-status").textContent="已请求停止，请查看停止确认与零输出。"; await poll();}
    catch(e) {$("stop-status").textContent=`停止结果未确认：${e.message}。若仍在运动，请立即断开电机电源。`;}
    finally {stopBusy=false; renderClearControls();}
  }
  for (const id of ["all-stop","test-stop","experiment-stop"]) $(id).addEventListener("click", stopAll);
  window.addEventListener("pagehide", () => {
    if (!controlActive()) return;
    fetch("/api/motor/stop", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}", keepalive:true}).catch(() => {});
  });
  const experimentFields=["front-motor","experiment-direction-m3","experiment-direction-m4","experiment-forward-sign","experiment-rpm","experiment-speed","experiment-distance","experiment-duration","experiment-group"];
  function experimentConfiguration() {
    const front=Number($("front-motor").value), rpm=inputNum("experiment-rpm"), speed=inputNum("experiment-speed"), distance=inputNum("experiment-distance"), duration=inputNum("experiment-duration"), group=$("experiment-group").value.trim();
    if (![3,4].includes(front)) throw new Error("请选择前节电机；不确定时用下方单机测试确认。");
    if (!finite(rpm) || !Number.isInteger(rpm) || rpm<1 || rpm>500) throw new Error("标称转速须为 1–500 rpm 的整数。");
    if (!finite(speed) || speed<=0 || Math.abs(speed-rpm*0.0005)>0.000001) throw new Error("目标速度应与转速一致：速度 = rpm × 0.0005。");
    if (!finite(distance) || distance<0.01 || distance>10 || !finite(duration) || duration<2 || duration>30) throw new Error("距离须为 0.01–10 m，最长时长须为 2–30 秒。");
    if (!group || group.length>64) throw new Error("填写 1–64 字符的数据准备组名称。");
    const sign=Number($("experiment-forward-sign").value);
    return {front_motor:front,base_rpm:rpm,target_speed_mps:sign*speed,direction_m3:Number($("experiment-direction-m3").value),direction_m4:Number($("experiment-direction-m4").value),forward_sign:sign,distance_m:distance,duration_s:duration,hold_s:0.5,lead_mm:30,motor_to_helix_ratio:1,preparation_group:group};
  }
  function experimentSummary() {
    const rpm=inputNum("experiment-rpm"), sign=Number($("experiment-forward-sign").value), m3=Number($("experiment-direction-m3").value), m4=Number($("experiment-direction-m4").value);
    $("experiment-config-summary").textContent=finite(rpm) ? `理论速度 ${number(sign*rpm*0.0005,4)} m/s · M3 / M4：${number(m3*rpm,0)} / ${number(m4*rpm,0)} rpm` : "请填写转速。";
  }
  for (const id of experimentFields) $(id).addEventListener("input", () => {
    if (id==="experiment-rpm") {const rpm=inputNum(id); $("experiment-speed").value=finite(rpm)?String(Number((rpm*0.0005).toFixed(6))):"";}
    if (id==="experiment-speed") {const speed=inputNum(id); $("experiment-rpm").value=finite(speed)?String(Number((speed/0.0005).toFixed(6))):"";}
    experimentDirty=true; experimentTouched=true; $("experiment-config-status").textContent="修改后请保存设置。"; experimentSummary(); renderExperiment(lastExperiment);
  });
  function renderExperiment(experiment={}) {
    lastExperiment=experiment || {};
    const e=lastExperiment, c=e.config || {}, active=!!e.active, busy=active || experimentBusy || motorBusy || stopBusy || motionActive(lastMotor.motion) || ["checking", "waiting_ack", "verifying"].includes(lastMotor.clear_state), configured=[3,4].includes(Number(c.front_motor));
    if (!experimentHydrated && !experimentTouched && configured && !experimentFields.some(id=>$(id)===document.activeElement)) {
      const values={"front-motor":c.front_motor,"experiment-direction-m3":c.direction_m3??c.m3_sign??-1,"experiment-direction-m4":c.direction_m4??c.m4_sign??-1,"experiment-forward-sign":c.forward_sign??1,"experiment-rpm":c.base_rpm??60,"experiment-speed":Math.abs(c.target_speed_mps??(c.base_rpm??60)*0.0005),"experiment-distance":c.distance_m??0.3,"experiment-duration":c.duration_s??10,"experiment-group":c.preparation_group??"preparation-1"};
      for (const [id,value] of Object.entries(values)) $(id).value=String(value);
      experimentHydrated=true; experimentDirty=false; $("experiment-config-status").textContent="已载入设置。"; experimentSummary();
    }
    for (const id of experimentFields) $(id).disabled=busy;
    $("experiment-configure").disabled=busy || !e.state;
    const canRun=!busy && !experimentDirty && configured;
    $("experiment-baseline").disabled=!canRun || e.ready_baseline!==true;
    $("experiment-collect").disabled=!canRun || e.ready_collect!==true;
    $("experiment-train").disabled=!canRun || e.ready_train!==true;
    const groups=e.samples_per_group || {}, groupEntries=Object.entries(groups).filter(([,n])=>finite(n) && n>0);
    $("experiment-pretrain").disabled=busy || experimentDirty || !configured || (e.ready_pretrain !== true && !(e.ready_pretrain===undefined && groupEntries.length>=2));
    const labels={idle:"等待运行",ready:"预测模型已准备",baseline:"基线运行中",collect:"预训练数据采集中",train:"强化学习运行中",running:"运行中",preparing:"正在准备",pretraining:"预测模型训练中",training:"正在更新策略",updating:"正在更新策略",stopping:"正在停止",completed:"本次实验完成",stopped:"已停止",error:"实验异常",failed:"实验已终止"};
    $("experiment-badge").classList.toggle("live",active);
    $("experiment-badge").innerHTML=`<i></i> ${e.state ? labels[e.state] || "等待状态" : "等待服务"}`;
    if (Date.now()>=experimentErrorUntil) $("experiment-status").textContent=["idle", "ready"].includes(e.state) && !e.error ? (configured ? (experimentDirty ? "请先保存设置。" : "点击双机试运行。") : "请选择前节并保存；不确定时用下方单机测试确认。") : e.message || "等待服务。";
    const learningOpen=$("learning-details")?.open===true;
    const sourceBlockers=e.blockers_by_mode?.[learningOpen?"train":"baseline"] ?? e.blockers;
    const blockers=Array.isArray(sourceBlockers)?sourceBlockers:typeof sourceBlockers==="object" && sourceBlockers?Object.values(sourceBlockers).flat():[];
    $("experiment-blockers").replaceChildren();
    const notes=[...(experimentDirty?["请先保存设置。"]:[]),...blockers.map(x=>typeof x==="string"?x:x?.message||"").filter(Boolean)];
    const uniqueNotes=[...new Set(notes)], visibleNotes=e.error || ["error", "failed", "stopping"].includes(e.state) ? uniqueNotes : uniqueNotes.slice(0,1);
    for (const text of visibleNotes) {const li=document.createElement("li");li.textContent=text;$("experiment-blockers").append(li);}
    $("experiment-reward").textContent=number(typeof e.reward==="object" ? e.reward?.total??e.reward?.reward : e.reward,4);
    $("experiment-travel").textContent=number(e.progress_m??e.distance_traveled_m,4);
    $("experiment-samples").textContent=finite(e.samples)?String(e.samples):"0";
    $("experiment-groups").textContent=groupEntries.length?groupEntries.map(([group,n])=>`${group}: ${n}`).join(" / "):"尚无数据";
    $("experiment-model").textContent=e.pretrained?"已训练":"未训练";
    $("experiment-policy").textContent=finite(e.policy_version)?String(e.policy_version):"0";
    const mode=e.mode || (["baseline","collect","train"].includes(e.state)?e.state:null);
    $("experiment-run").textContent=mode?({baseline:"基线",collect:"监督采集",train:"强化学习"}[mode]||mode):"—";
    $("experiment-directory").textContent=e.directory || "尚未开始";
    renderClearControls();
  }
  async function experimentCommand(path,data,label) {
    if (experimentBusy) return;
    experimentBusy=true; experimentErrorUntil=Date.now()+6000; renderExperiment(lastExperiment);
    $("experiment-status").textContent=`${label}请求已提交，等待服务确认…`;
    try {
      await post(`/api/experiment/${path}`,data,15000);
      if (path==="configure") {experimentDirty=false;experimentHydrated=true;$("experiment-config-status").textContent="已保存，未启动电机。";}
      experimentErrorUntil=0; await poll();
    } catch(e) {experimentErrorUntil=Date.now()+12000; $("experiment-status").textContent=`${label}失败或状态未确认：${e.message}`;}
    finally {experimentBusy=false; renderExperiment(lastExperiment);}
  }
  $("experiment-configure").addEventListener("click",()=>{try {experimentCommand("configure",experimentConfiguration(),"保存设置");}catch(e){$("experiment-config-status").textContent=e.message;}});
  $("learning-details")?.addEventListener("toggle",()=>renderExperiment(lastExperiment));
  for (const mode of ["baseline","collect","train"]) $(`experiment-${mode}`).addEventListener("click",()=>{if (!$(`experiment-${mode}`).disabled) experimentCommand("start",{mode},{baseline:"基线运行",collect:"采集预训练数据",train:"真机强化学习"}[mode]);});
  $("experiment-pretrain").addEventListener("click",()=>{if (!$("experiment-pretrain").disabled) experimentCommand("pretrain",{epochs:10},"训练预测模型");});
  experimentSummary();
  $("record-start").addEventListener("click", () => {
    const duration = Number($("record-duration").value);
    if (!Number.isFinite(duration) || duration < 2 || duration > 30) {recordErrorUntil = Date.now() + 8000; $("record-status").textContent = "时长须为 2–30 秒。"; return;}
    recordErrorUntil = 0; deviceCommand("record", "/api/record/start", {duration_s:duration});
  });
  $("record-stop").addEventListener("click", () => deviceCommand("record", "/api/record/stop", {}));
  function simulationButtons() {ui.reset.disabled = ui.stepButton.disabled = ui.train.disabled = simulationBusy || recoveringSimulation;}
  async function runSimulation(url, data, label, timeoutMs = 5000) {
    if (simulationBusy || recoveringSimulation) return;
    simulationBusy = true; simulationButtons(); ui.simStatus.textContent = `${label}中，请稍候…`;
    try {await post(url, data, timeoutMs); if (url.endsWith("/reset")) history = []; ui.simStatus.textContent = `${label}完成`; await poll();}
    catch(e) {
      // An aborted client request may still be running on the server. Keep
      // controls locked until a later successful status proves it is responsive.
      if (e.name === "AbortError" || e instanceof TypeError) {recoveringSimulation = true; ui.simStatus.textContent = `${label}状态未确认，等待服务恢复…`;}
      else ui.simStatus.textContent = `${label}失败：${e.message}`;
    } finally {simulationBusy = false; simulationButtons();}
  }
  ui.reset.addEventListener("click", () => {const seed = Number(ui.seed.value); if (!Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) {ui.simStatus.textContent = "种子需为 0–2147483647 的整数。"; return;} runSimulation("/api/simulation/reset", {seed}, "重置模拟");});
  ui.stepButton.addEventListener("click", () => runSimulation("/api/simulation/step", {}, "离线步骤"));
  ui.train.addEventListener("click", () => runSimulation("/api/simulation/train", {}, "合成训练", 30000));
  ui.modelCheck = $("model-check");
  ui.modelCheck.addEventListener("click", async () => {
    if (ui.modelCheck.disabled) return;
    const front = Number($("front-motor").value);
    if (front !== 3 && front !== 4) {$("model-check-status").textContent = "请先选择一个前节假设。"; return;}
    ui.modelCheck.disabled = true;
    $("model-check-status").textContent = "正在检查，可能需要约 30 秒…";
    $("model-result").hidden = true;
    let unknown = false;
    try {
      const data = await post("/api/model/two", {front_motor:front}, 60000);
      const result = data.result || data;
      $("model-notice").textContent = result.notice || "整链检查已完成";
      $("model-order").textContent = `M${front} 在前 / M${7-front} 在后（假设）`;
      $("model-action").textContent = displayAction(result.action);
      $("model-null").textContent = number(result.null_weight, 3);
      $("model-history").textContent = Array.isArray(result.history_available) ? `${result.history_available.filter(Boolean).length} / ${result.history_available.length}` : "—";
      $("model-result").hidden = false;
      $("model-check-status").textContent = "检查完成";
    } catch (error) {
      unknown = error.name === "AbortError" || error instanceof TypeError;
      $("model-check-status").textContent = unknown ? "检查状态未确认，请等待服务恢复后刷新页面。" : `检查失败：${error.message}`;
    } finally {ui.modelCheck.disabled = unknown;}
  });
  window.addEventListener("resize", draw);
  ports(); poll(); setInterval(poll, 500);
})();
