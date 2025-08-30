// assets/orion.js
(function () {
  // 防重复 + 重试机制 + 调试日志
  let bound = false;
  let trying = false;
  let retryTimer = null;
  let attempts = 0;
  const MAX_ATTEMPTS = 80;      // ~24s at 300ms
  const TICK_MS = 300;

  function debug(...args) {
    // 将下面这行注释去掉即可开启详细日志
    // console.debug("[orion]", ...args);
  }

  function findGd() {
    const container = document.getElementById("price-graph");
    if (!container) {
      debug("price-graph container not found");
      return null;
    }
    const gd = container.getElementsByClassName("js-plotly-plot")[0] || null;
    if (!gd) debug("js-plotly-plot not found inside #price-graph");
    return gd;
  }

  function hasSetProps() {
    const sp = window.dash_clientside && window.dash_clientside.set_props;
    if (typeof sp !== "function") {
      debug("dash_clientside.set_props not ready");
      return false;
    }
    return true;
  }

  function pxYtoDataY(gd, clientY) {
    try {
      const layout = gd._fullLayout;
      if (!layout) return null;
      const rect = gd.getBoundingClientRect();
      const plotTop = rect.top + (layout.margin?.t || 0);
      const yPixInPlot = clientY - plotTop;
      const ya = layout.yaxis;
      if (!ya || typeof ya.p2l !== "function") return null;
      return ya.p2l(yPixInPlot);
    } catch (e) {
      debug("pxYtoDataY error", e);
      return null;
    }
  }

  function setStore(value) {
    const sp = window.dash_clientside && window.dash_clientside.set_props;
    if (typeof sp === "function") {
      sp("cursor-y-store", { data: value });
    }
  }

  function bindOnce(gd) {
    if (!gd || bound) return;
    bound = true;
    debug("binding event handlers to gd");

    // Plotly 事件（优先）
    if (typeof gd.on === "function") {
      gd.on("plotly_mousemove", (e) => {
        if (!e || !e.event) return;
        const y = pxYtoDataY(gd, e.event.clientY);
        if (y == null || !isFinite(y)) return;
        setStore(y);
      });

      gd.on("plotly_unhover", () => {
        setStore(null);
      });

      // 交互后重绘，确保仍处于绑定状态（幂等）
      gd.on("plotly_relayout", () => {
        debug("relayout (already bound)");
      });

      gd.on("plotly_afterplot", () => {
        debug("afterplot (already bound)");
      });
    }

    // 兜底：原生事件（某些浏览器/设置下 plotly_* 事件可能不触发）
    gd.addEventListener(
      "mousemove",
      (e) => {
        const y = pxYtoDataY(gd, e.clientY);
        if (y == null || !isFinite(y)) return;
        setStore(y);
      },
      { passive: true }
    );
    gd.addEventListener(
      "mouseleave",
      () => setStore(null),
      { passive: true }
    );

    // 一旦绑定成功，停止重试
    if (retryTimer) {
      clearInterval(retryTimer);
      retryTimer = null;
      trying = false;
      debug("stop retry after bound");
    }
  }

  function tryBind() {
    if (bound) return;
    const gd = findGd();
    if (!gd) return;
    if (!gd._fullLayout) {
      debug("gd found but _fullLayout not ready");
      return;
    }
    if (!hasSetProps()) {
      return;
    }
    bindOnce(gd);
  }

  function scheduleRetry() {
    if (trying || bound) return;
    trying = true;
    attempts = 0;
    retryTimer = setInterval(() => {
      attempts += 1;
      tryBind();
      if (bound || attempts >= MAX_ATTEMPTS) {
        if (retryTimer) {
          clearInterval(retryTimer);
          retryTimer = null;
          trying = false;
        }
        if (!bound) {
          debug("stop retry: max attempts reached");
        }
      }
    }, TICK_MS);
  }

  // 启动时机：DOM 就绪 + 延时轮询 + Dash 渲染事件
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleRetry, { once: true });
  } else {
    scheduleRetry();
  }
  window.addEventListener("load", scheduleRetry);
  window.addEventListener("dash:rendered", scheduleRetry);

  // 手动入口：window.orion.rebind()
  window.orion = window.orion || {};
  window.orion.rebind = function () {
    debug("manual rebind invoked");
    bound = false;
    scheduleRetry();
  };
})();