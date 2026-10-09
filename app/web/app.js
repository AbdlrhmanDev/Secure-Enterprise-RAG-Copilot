"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { token: null, user: null, users: [], defaults: null, pollTimer: null };

// --- helpers --------------------------------------------------------------------------------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) node.setAttribute(key, value);
  }
  // Text is always inserted as text nodes: document content is never interpreted as HTML.
  for (const child of children.flat()) node.append(child instanceof Node ? child : String(child ?? ""));
  return node;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, { ...options, headers });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = body.error || {};
    throw new Error(error.message ? `${error.message} (trace ${error.trace_id})` : `Request failed: ${response.status}`);
  }
  return body;
}

const fmt = (value, digits = 3) => (value == null ? "-" : Number(value).toFixed(digits));
const hasRole = (...roles) => state.user && state.user.roles.some((r) => r === "admin" || roles.includes(r));

// --- session --------------------------------------------------------------------------------

async function signIn(userId) {
  const body = await api("/auth/token", { method: "POST", json: { user_id: userId } });
  state.token = body.access_token;
  state.user = body.user;
  try { sessionStorage.setItem("user_id", userId); } catch (_) { /* storage unavailable */ }
  state.defaults = await api("/config");
  for (const key of ["mode", "fusion", "top_k", "rerank_candidates", "final_context_k"]) $(`#cfg-${key}`).value = state.defaults[key];
  $("#cfg-rerank").checked = state.defaults.rerank;
  $("#messages").replaceChildren($("#chat-empty"));
  $("#chat-empty").hidden = false;
  closeDrawer();
  renderRoleChoices();
  await loadCollections();
  await Promise.all([loadDocuments(), loadRuns()]);
}

async function loadCollections() {
  const collections = await api("/collections");
  const select = $("#collection");
  const previous = select.value;
  select.replaceChildren(...collections.map((c) => el("option", { value: c.id }, `${c.id} (${c.document_count})`)));
  if (!collections.length) select.append(el("option", { value: "" }, "no documents yet"));
  if (collections.some((c) => c.id === previous)) select.value = previous;
  if (!$("#upload-collection").value) $("#upload-collection").value = select.value || "policies";
}

function renderRoleChoices() {
  const all = ["employee", "finance", "engineering", "admin"];
  const allowed = state.user.roles.includes("admin") ? all : state.user.roles;
  const boxes = allowed.map((role, i) =>
    el("label", { class: "check" }, el("input", { type: "checkbox", value: role, checked: i === 0 ? "" : false }), role));
  $("#upload-roles").replaceChildren(el("legend", {}, "Who may read it"), ...boxes);
}

// --- chat -----------------------------------------------------------------------------------

function retrievalSettings() {
  return {
    mode: $("#cfg-mode").value,
    fusion: $("#cfg-fusion").value,
    rerank: $("#cfg-rerank").checked,
    top_k: Number($("#cfg-top_k").value),
    rerank_candidates: Number($("#cfg-rerank_candidates").value),
    final_context_k: Number($("#cfg-final_context_k").value),
  };
}

function renderAnswer(result) {
  const byIndex = new Map(result.citations.map((c) => [c.index, c]));
  const body = el("div");
  // Split the answer on [n] markers and turn each valid one into a button that opens the source.
  for (const part of result.answer.split(/(\[\d+\])/)) {
    const match = part.match(/^\[(\d+)\]$/);
    const citation = match && byIndex.get(Number(match[1]));
    if (citation) {
      body.append(el("button", { class: "cite", title: citation.filename, onclick: () => openDrawer(result.citations, citation.index) }, match[1]));
    } else {
      body.append(part);
    }
  }
  const meta = el("div", { class: "meta" },
    el("span", {}, `${Math.round(result.latency_ms)} ms${result.cached ? " (cached)" : ""}`),
    el("span", {}, `retrieval ${Math.round(result.timings.retrieval_ms)} / rerank ${Math.round(result.timings.rerank_ms)} / generation ${Math.round(result.timings.generation_ms)} ms`),
    el("span", {}, result.usage.model),
    result.usage.estimated_cost_usd ? el("span", {}, `$${result.usage.estimated_cost_usd.toFixed(4)}`) : "");
  if (result.citations.length) {
    meta.append(el("button", { class: "ghost", onclick: () => openDrawer(result.citations) }, `${result.citations.length} source${result.citations.length > 1 ? "s" : ""}`));
  }
  const rate = (rating) => async (event) => {
    await api("/feedback", { method: "POST", json: { trace_id: result.trace_id, rating } });
    event.target.parentElement.querySelectorAll(".rate").forEach((b) => (b.disabled = true));
    event.target.textContent = "Thanks";
  };
  meta.append(el("button", { class: "ghost rate", onclick: rate(1) }, "Helpful"), el("button", { class: "ghost rate", onclick: rate(-1) }, "Not helpful"));
  return el("div", { class: `msg${result.insufficient_evidence ? " refusal" : ""}` }, body, meta);
}

function openDrawer(citations, activeIndex) {
  const items = citations.map((c) =>
    el("div", { class: `source${c.index === activeIndex ? " active" : ""}` },
      el("div", { class: "source-title" }, `[${c.index}] ${c.filename}`),
      el("div", { class: "source-meta" },
        [c.section && `Section: ${c.section}`, c.page != null && `Page ${c.page}`, `Score ${c.score.toFixed(3)}`, c.chunk_id].filter(Boolean).join(" · ")),
      el("p", {}, c.text)));
  $("#drawer-body").replaceChildren(...items);
  $("#drawer").hidden = false;
  const active = $("#drawer .source.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function closeDrawer() { $("#drawer").hidden = true; }

async function ask(event) {
  event.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  $("#chat-empty").hidden = true;
  const messages = $("#messages");
  messages.append(el("div", { class: "msg user" }, question));
  const pending = el("div", { class: "msg" }, "Searching your documents...");
  messages.append(pending);
  pending.scrollIntoView({ block: "end" });
  $("#question").value = "";
  $("#ask-button").disabled = true;
  try {
    const result = await api("/query", {
      method: "POST",
      json: { query: question, collection_id: $("#collection").value || undefined, retrieval: retrievalSettings() },
    });
    pending.replaceWith(renderAnswer(result));
  } catch (error) {
    pending.replaceWith(el("div", { class: "msg error" }, error.message));
  } finally {
    $("#ask-button").disabled = false;
    $("#question").focus();
  }
}

// --- documents ------------------------------------------------------------------------------

async function loadDocuments() {
  const documents = await api("/documents");
  const rows = documents.map((d) => {
    const canRetry = d.status === "failed" && (hasRole() || d.uploaded_by === state.user.id);
    return el("tr", {},
      el("td", {}, d.filename, d.error ? el("div", { class: "err" }, d.error) : ""),
      el("td", {}, d.collection_id),
      el("td", {}, el("span", { class: `badge ${d.status}` }, d.status)),
      el("td", {}, d.allowed_roles.join(", ")),
      el("td", { class: "num" }, d.version),
      el("td", { class: "num" }, d.page_count),
      el("td", { class: "num" }, d.chunk_count),
      el("td", {}, canRetry ? el("button", { class: "ghost", onclick: () => retry(d.id) }, "Retry") : ""));
  });
  $("#documents tbody").replaceChildren(...(rows.length ? rows : [el("tr", {}, el("td", { colspan: 8 }, "No documents you can read yet."))]));

  $("#admin-panel").hidden = !hasRole();
  if (hasRole()) {
    const status = await api("/admin/ingestion");
    const stat = (label, value) => el("div", { class: "stat" }, el("b", {}, value), el("span", {}, label));
    $("#ingestion-summary").replaceChildren(
      ...["ready", "processing", "pending", "failed"].map((s) => stat(s, status.counts[s] || 0)),
      stat("pages indexed", status.total_pages), stat("chunks indexed", status.total_chunks));
  }
  // Keep polling while anything is still being ingested.
  clearTimeout(state.pollTimer);
  if (documents.some((d) => d.status === "pending" || d.status === "processing")) {
    state.pollTimer = setTimeout(() => loadDocuments().catch(() => {}), 2000);
  }
}

async function retry(documentId) {
  await api(`/documents/${documentId}/retry`, { method: "POST" }).catch((error) => ($("#upload-status").textContent = error.message));
  await loadDocuments();
}

async function uploadDocument(event) {
  event.preventDefault();
  const roles = [...document.querySelectorAll("#upload-roles input:checked")].map((box) => box.value);
  const status = $("#upload-status");
  if (!roles.length) { status.textContent = "Pick at least one role."; return; }
  const form = new FormData();
  form.append("file", $("#file").files[0]);
  form.append("collection_id", $("#upload-collection").value);
  form.append("allowed_roles", roles.join(","));
  status.textContent = "Uploading...";
  try {
    const result = await api("/documents", { method: "POST", body: form });
    status.textContent = result.deduplicated ? "Already ingested (identical content)." : `Queued: ${result.document.filename}`;
    $("#file").value = "";
    await loadCollections();
    await loadDocuments();
  } catch (error) {
    status.textContent = error.message;
  }
}

// --- evaluation -----------------------------------------------------------------------------

async function loadRuns() {
  const allowed = hasRole("engineering");
  $("#eval button[type=submit]").disabled = !allowed;
  if (!allowed) {
    $("#runs tbody").replaceChildren(el("tr", {}, el("td", { colspan: 13 }, "The evaluation dashboard needs the engineering or admin role.")));
    $("#eval-chart").replaceChildren();
    $("#eval-note").textContent = "";
    return;
  }
  const runs = await api("/eval/runs");
  const describe = (r) => `${r.mode}${r.mode === "hybrid" ? "/" + r.fusion : ""}${r.rerank ? " + rerank" : ""}`;
  const rows = runs.map((run) => {
    const s = run.metrics_json.summary || {};
    return el("tr", {},
      el("td", {}, run.name, " ", el("span", { class: `badge ${run.status}` }, run.status), run.error ? el("div", { class: "err" }, run.error) : ""),
      el("td", {}, describe(run.config_json.retrieval)),
      ...[s.recall_at_k, s.precision_at_k, s.mrr, s.ndcg_at_k, s.answer_correctness, s.groundedness, s.citation_correctness, s.refusal_accuracy]
        .map((v) => el("td", { class: "num" }, fmt(v))),
      el("td", { class: "num" }, s.acl_leaked_chunks ?? "-"),
      el("td", { class: "num" }, fmt(s.latency_p95_ms, 0)),
      el("td", { class: "num" }, fmt(s.cost_per_query_usd, 5)));
  });
  $("#runs tbody").replaceChildren(...(rows.length ? rows : [el("tr", {}, el("td", { colspan: 13 }, "No runs yet."))]));

  const completed = runs.filter((r) => r.status === "completed" && r.recall_at_k != null).slice(0, 10);
  $("#eval-chart").replaceChildren(...completed.map((run) =>
    el("div", { class: "bar-row" },
      el("span", { class: "name", title: run.name }, run.name),
      el("div", { class: "bar-track" }, el("div", { class: "bar", style: `width:${(run.recall_at_k * 100).toFixed(1)}%` })),
      el("span", { class: "value" }, fmt(run.recall_at_k)))));
  const first = completed[0] && completed[0].metrics_json.summary;
  $("#eval-note").textContent = first
    ? `Bars show Recall@${first.k}. Latest run: ${first.cases} cases, generator ${first.generator || "off"}, judge ${first.judge || "off"}.`
    : "";
  if (runs.some((r) => r.status === "running")) setTimeout(() => loadRuns().catch(() => {}), 2500);
}

async function startEvaluation(event) {
  event.preventDefault();
  const status = $("#eval-status");
  status.textContent = "Starting...";
  try {
    await api("/eval/run", {
      method: "POST",
      json: {
        name: $("#eval-name").value,
        k: Number($("#eval-k").value),
        generate: $("#eval-generate").checked,
        retrieval: { mode: $("#eval-mode").value, rerank: $("#eval-rerank").checked },
      },
    });
    status.textContent = "Running in the background...";
    await loadRuns();
    status.textContent = "";
  } catch (error) {
    status.textContent = error.message;
  }
}

// --- wiring ---------------------------------------------------------------------------------

function showView(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  if (name === "documents") loadDocuments().catch(() => {});
  if (name === "evaluation") loadRuns().catch(() => {});
}

async function init() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));
  $("#ask").addEventListener("submit", ask);
  $("#upload").addEventListener("submit", uploadDocument);
  $("#eval").addEventListener("submit", startEvaluation);
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#refresh-docs").addEventListener("click", () => loadDocuments());
  $("#refresh-runs").addEventListener("click", () => loadRuns());
  $("#user").addEventListener("change", (event) => signIn(event.target.value));

  state.users = await api("/auth/users");
  $("#user").replaceChildren(...state.users.map((u) => el("option", { value: u.id }, `${u.name} - ${u.roles.join(", ")}`)));
  let remembered = null;
  try { remembered = sessionStorage.getItem("user_id"); } catch (_) { /* storage unavailable */ }
  const userId = state.users.some((u) => u.id === remembered) ? remembered : state.users[0].id;
  $("#user").value = userId;
  await signIn(userId);
}

init().catch((error) => {
  $("#messages").append(el("div", { class: "msg error" }, `Could not start: ${error.message}`));
});
