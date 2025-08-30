(function () {
  function bind() {
    const container = document.getElementById("price-graph");
    if (!container) return;
    const gd = container.getElementsByClassName("js-plotly-plot")[0];
    if (!gd || !gd._fullLayout) return;

    function toDataY(clientY) {
      const rect = gd.getBoundingClientRect();
      const ly = gd._fullLayout;
      const yPixInPlot = clientY - (rect.top + ly.margin.t);
      return ly.yaxis && typeof ly.yaxis.p2l === "function"
        ? ly.yaxis.p2l(yPixInPlot)
        : null;
    }

    function setStore(value) {
      if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
        window.dash_clientside.set_props("cursor-y-store", { data: value });
      }
    }

    gd.addEventListener("mousemove", (ev) => {
      const y = toDataY(ev.clientY);
      if (y != null) setStore(y);
    }, { passive: true });

    gd.addEventListener("mouseleave", () => {
      setStore(null);
    }, { passive: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
