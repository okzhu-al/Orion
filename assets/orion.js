window.orion = window.orion || {};
window.orion.cursorYtoData = function(evt, fig) {
  if (!evt) { return window.dash_clientside.no_update; }
  if (evt.type === "plotly_unhover") { return null; }
  if (!evt.event) { return window.dash_clientside.no_update; }
  const container = document.getElementById("price-graph");
  if (!container) { return window.dash_clientside.no_update; }
  const gd = container.getElementsByClassName("js-plotly-plot")[0];
  if (!gd || !gd._fullLayout) { return window.dash_clientside.no_update; }

  const clientY = evt.event.clientY;
  const rect = gd.getBoundingClientRect();

  const layout = gd._fullLayout;
  const plotTop = rect.top + layout.margin.t;
  const yPixInPlot = clientY - plotTop;

  const ya = layout.yaxis;
  if (!ya || typeof ya.p2l !== "function") { return window.dash_clientside.no_update; }
  return ya.p2l(yPixInPlot);
};
