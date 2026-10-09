/**
 * Reanimator Bridge — ComfyUI frontend panel.
 *
 * This is the half of pairing that runs *on the machine*. reanimator.app can
 * present a signed assertion, but nothing happens until a human clicks Allow
 * here. That is the whole point: a stolen assertion is useless on its own.
 *
 * Deliberately plain DOM and relative `fetch`. ComfyUI's frontend API has
 * churned a lot between versions, so the only surface we depend on is
 * `app.registerExtension`. Same-origin with ComfyUI means no CORS involved.
 */

import { app } from "../../scripts/app.js";

const POLL_INTERVAL_MS = 2000;
const PANEL_API = "/reanimator/panel";

const state = {
  port: null,
  deviceLabel: null,
  bridgeVersion: null,
  pending: [],
  paired: [],
  // Requests the user already acted on, so a slow poll cannot re-open a dialog
  // they just dismissed.
  handled: new Set(),
  dialogOpen: false,
  pollTimer: null,
};

/* ------------------------------------------------------------------ */
/* utilities                                                          */
/* ------------------------------------------------------------------ */

async function api(path, options) {
  const response = await fetch(`${PANEL_API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function relativeTime(epochSeconds) {
  if (!epochSeconds) return "never";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000) - epochSeconds);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

/* ------------------------------------------------------------------ */
/* styles                                                             */
/* ------------------------------------------------------------------ */

function injectStyles() {
  if (document.getElementById("reanimator-styles")) return;
  document.head.append(
    el("style", {
      id: "reanimator-styles",
      textContent: `
.rb-overlay{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:10000;
  display:flex;align-items:center;justify-content:center;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.rb-card{background:#1f1f24;color:#eee;border:1px solid #3a3a42;border-radius:10px;
  width:min(520px,92vw);max-height:86vh;overflow:auto;
  box-shadow:0 18px 50px rgba(0,0,0,.55)}
.rb-card header{padding:18px 22px 6px;font-size:17px;font-weight:600}
.rb-card .rb-body{padding:6px 22px 18px;font-size:13px;line-height:1.55;color:#c8c8d0}
.rb-card footer{display:flex;gap:10px;justify-content:flex-end;
  padding:14px 22px 18px;border-top:1px solid #2c2c33}
.rb-facts{margin:14px 0;border:1px solid #3a3a42;border-radius:7px;overflow:hidden}
.rb-fact{display:flex;gap:12px;padding:9px 13px;font-size:13px}
.rb-fact+.rb-fact{border-top:1px solid #2c2c33}
.rb-fact span:first-child{color:#8b8b96;min-width:74px;flex-shrink:0}
.rb-fact span:last-child{color:#fff;word-break:break-all}
.rb-warn{background:#3a2a12;border:1px solid #6b4c1d;color:#f0d9a8;
  border-radius:7px;padding:10px 13px;font-size:12.5px;margin-top:12px}
.rb-btn{border:1px solid #4a4a55;background:#2b2b33;color:#eee;border-radius:6px;
  padding:8px 17px;font-size:13px;cursor:pointer;font-family:inherit}
.rb-btn:hover{background:#35353f}
.rb-btn.primary{background:#2f7d4f;border-color:#3a9c63;color:#fff}
.rb-btn.primary:hover{background:#379059}
.rb-btn.danger{background:#7d2f2f;border-color:#9c3a3a;color:#fff}
.rb-btn.danger:hover{background:#903737}
.rb-row{display:flex;align-items:center;gap:10px;padding:9px 0;
  border-bottom:1px solid #2c2c33;font-size:13px}
.rb-row:last-child{border-bottom:none}
.rb-row .rb-grow{flex:1;min-width:0}
.rb-row small{color:#8b8b96;display:block;font-size:11.5px}
.rb-input{width:100%;background:#141418;border:1px solid #3a3a42;border-radius:6px;
  color:#eee;padding:8px 11px;font-size:13px;font-family:inherit;box-sizing:border-box}
.rb-empty{color:#8b8b96;font-style:italic;padding:10px 0;font-size:12.5px}
.rb-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.rb-dot.on{background:#3a9c63}.rb-dot.off{background:#9c3a3a}
`,
    })
  );
}

/* ------------------------------------------------------------------ */
/* pairing approval dialog                                            */
/* ------------------------------------------------------------------ */

function showApprovalDialog(request) {
  if (state.dialogOpen) return;
  state.dialogOpen = true;

  const overlay = el("div", { className: "rb-overlay" });

  const close = () => {
    overlay.remove();
    state.dialogOpen = false;
    document.removeEventListener("keydown", onKey);
  };

  const resolve = async (approve) => {
    // Mark handled immediately: the poll must not re-open this dialog while
    // the request is in flight.
    state.handled.add(request.requestId);
    try {
      await api("/resolve", {
        method: "POST",
        body: JSON.stringify({ requestId: request.requestId, approve }),
      });
    } catch (error) {
      console.error("[reanimator] could not resolve pairing:", error);
    }
    close();
    refresh();
  };

  // Escape rejects rather than dismissing. Leaving a request silently pending
  // would be worse: the user would think they declined when they had not.
  const onKey = (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      resolve(false);
    }
  };
  document.addEventListener("keydown", onKey);

  overlay.append(
    el(
      "div",
      { className: "rb-card" },
      el("header", {}, "Reanimator wants to connect"),
      el(
        "div",
        { className: "rb-body" },
        el(
          "div",
          {},
          "A browser session is asking to use this computer's GPU for generation."
        ),
        el(
          "div",
          { className: "rb-facts" },
          el(
            "div",
            { className: "rb-fact" },
            el("span", {}, "Account"),
            el("span", {}, request.email)
          ),
          el(
            "div",
            { className: "rb-fact" },
            el("span", {}, "Website"),
            el("span", {}, request.origin)
          ),
          el(
            "div",
            { className: "rb-fact" },
            el("span", {}, "Device"),
            el("span", {}, request.device)
          )
        ),
        el(
          "div",
          { className: "rb-warn" },
          "Only allow this if you just clicked \u201cConnect this device\u201d " +
            "on reanimator.app and the account above is yours."
        )
      ),
      el(
        "footer",
        {},
        el(
          "button",
          { className: "rb-btn", onclick: () => resolve(false) },
          "Reject"
        ),
        el(
          "button",
          { className: "rb-btn primary", onclick: () => resolve(true) },
          "Allow"
        )
      )
    )
  );

  // No click-outside-to-dismiss: an accidental approval is not recoverable
  // without noticing, so every outcome must be a deliberate button press.
  document.body.append(overlay);
}

/* ------------------------------------------------------------------ */
/* settings panel                                                     */
/* ------------------------------------------------------------------ */

function showSettingsPanel() {
  const overlay = el("div", { className: "rb-overlay" });
  const close = () => overlay.remove();
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  const body = el("div", { className: "rb-body" });
  const card = el(
    "div",
    { className: "rb-card" },
    el("header", {}, "Reanimator Bridge"),
    body,
    el(
      "footer",
      {},
      el("button", { className: "rb-btn", onclick: close }, "Close")
    )
  );

  const render = () => {
    body.replaceChildren();

    body.append(
      el(
        "div",
        { className: "rb-row" },
        el("div", { className: `rb-dot ${state.port ? "on" : "off"}` }),
        el(
          "div",
          { className: "rb-grow" },
          state.port
            ? `Listening on 127.0.0.1:${state.port}`
            : "Bridge is not listening",
          el("small", {}, `Bridge ${state.bridgeVersion || "?"}`)
        )
      )
    );

    // Device label -------------------------------------------------
    const labelInput = el("input", {
      className: "rb-input",
      value: state.deviceLabel || "",
      placeholder: "Studio PC",
    });
    body.append(
      el("div", { style: "margin:16px 0 6px;font-weight:600" }, "Device name"),
      el(
        "div",
        { style: "color:#8b8b96;font-size:12px;margin-bottom:7px" },
        "Shown in the editor and in the approval dialog."
      ),
      el(
        "div",
        { style: "display:flex;gap:8px" },
        labelInput,
        el(
          "button",
          {
            className: "rb-btn",
            onclick: async () => {
              try {
                await api("/device-label", {
                  method: "POST",
                  body: JSON.stringify({ label: labelInput.value.trim() }),
                });
                await refresh();
                render();
              } catch (error) {
                alert(`Could not save: ${error.message}`);
              }
            },
          },
          "Save"
        )
      )
    );

    // Development origins ------------------------------------------
    const devToggle = el("input", {
      type: "checkbox",
      checked: !!state.devOrigins,
      style: "width:16px;height:16px;cursor:pointer",
    });
    devToggle.onchange = async () => {
      try {
        await api("/dev-origins", {
          method: "POST",
          body: JSON.stringify({ enabled: devToggle.checked }),
        });
        await refresh();
        render();
      } catch (error) {
        alert(error.message);
        devToggle.checked = !devToggle.checked;
      }
    };
    body.append(
      el("div", { style: "margin:20px 0 6px;font-weight:600" }, "Development"),
      el(
        "div",
        { className: "rb-row" },
        devToggle,
        el(
          "div",
          { className: "rb-grow" },
          "Allow localhost",
          el(
            "small",
            {},
            "Only needed when running Reanimator on this machine. " +
              (state.allowedOrigins || []).join(", ")
          )
        )
      )
    );

    // Paired browsers ----------------------------------------------
    body.append(
      el("div", { style: "margin:22px 0 6px;font-weight:600" }, "Paired browsers")
    );
    if (!state.paired.length) {
      body.append(el("div", { className: "rb-empty" }, "None paired yet."));
    } else {
      for (const entry of state.paired) {
        body.append(
          el(
            "div",
            { className: "rb-row" },
            el(
              "div",
              { className: "rb-grow" },
              entry.email,
              el(
                "small",
                {},
                `${entry.origin} \u00b7 last seen ${relativeTime(entry.last_seen)}`
              )
            ),
            el(
              "button",
              {
                className: "rb-btn danger",
                onclick: async () => {
                  if (!confirm(`Revoke access for ${entry.email}?`)) return;
                  await api("/revoke", {
                    method: "POST",
                    body: JSON.stringify({ tokenPrefix: entry.tokenPrefix }),
                  });
                  await refresh();
                  render();
                },
              },
              "Revoke"
            )
          )
        );
      }
    }
  };

  render();
  overlay.append(card);
  document.body.append(overlay);
}

/* ------------------------------------------------------------------ */
/* polling                                                            */
/* ------------------------------------------------------------------ */

async function refresh() {
  try {
    const status = await api("/status");
    state.port = status.port;
    state.deviceLabel = status.deviceLabel;
    state.bridgeVersion = status.bridgeVersion;
    state.devOrigins = !!status.devOrigins;
    state.allowedOrigins = status.allowedOrigins || [];
    state.pending = status.pending || [];
    state.paired = status.paired || [];
  } catch {
    state.port = null;
    return;
  }

  const next = state.pending.find((r) => !state.handled.has(r.requestId));
  if (next) showApprovalDialog(next);
}

function startPolling() {
  const tick = async () => {
    // Only poll a visible tab: this runs for the whole ComfyUI session.
    if (document.visibilityState === "visible") await refresh();
    state.pollTimer = setTimeout(tick, POLL_INTERVAL_MS);
  };
  tick();
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") refresh();
  });
}

/* ------------------------------------------------------------------ */
/* registration                                                       */
/* ------------------------------------------------------------------ */

app.registerExtension({
  name: "reanimator.bridge",
  async setup() {
    injectStyles();

    try {
      app.ui?.menuContainer?.append(
        el(
          "button",
          {
            className: "rb-btn",
            style: "margin:2px 0;width:100%",
            textContent: "Reanimator Bridge",
            onclick: showSettingsPanel,
          }
        )
      );
    } catch (error) {
      // Menu layout differs across ComfyUI versions; the dialog still works
      // and the settings panel is reachable from the console fallback below.
      console.warn("[reanimator] could not add menu button:", error);
    }
    window.reanimatorBridgePanel = showSettingsPanel;

    startPolling();
    console.log("[reanimator] bridge panel ready");
  },
});
