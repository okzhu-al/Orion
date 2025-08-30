window.orion = window.orion || {};
window.orion.cursorYtoData = function(evt, fig) {
  if (!evt || !evt.event) { return window.dash_clientside.no_update; }
  const gd = document.getElementById("price-graph");
  if (!gd || !gd._fullLayout) { return window.dash_clientside.no_update; }

  const clientY = evt.event.clientY;
  const rect = gd.getBoundingClientRect();

  const layout = gd._fullLayout;
  const plotTop = rect.top + layout.margin.t;
  const yPixInPlot = clientY - plotTop;

  const ya = layout.yaxis;
  if (!ya || typeof ya.p2l !== "function") { return window.dash_clientside.no_update; }
  const dataY = ya.p2l(yPixInPlot);
  return dataY;
};
