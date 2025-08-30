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
      const ya = ly.yaxis;
      return ya && typeof ya.p2l === "function" ? ya.p2l(yPixInPlot) : null;
    }

    function setStore(value) {
      const setter = window.dash_clientside && window.dash_clientside.set_props;
      if (setter) setter("cursor-y-store", { data: value });
    }

    if (typeof gd.on === "function") {
      gd.on("plotly_mousemove", (ev) => {
        const y = toDataY(ev.event.clientY);
        if (y != null) setStore(y);
      });
      gd.on("plotly_unhover", () => {
        setStore(null);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
