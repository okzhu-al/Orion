window.orion = window.orion || {};
window.orion.cursorYtoData = function(evt, fig){
  if (!evt || !evt.event) { return window.dash_clientside.no_update; }
  var gdParent = document.getElementById('price-graph');
  if (!gdParent) { return window.dash_clientside.no_update; }
  var gd = gdParent.getElementsByClassName('js-plotly-plot')[0];
  if (!gd || !gd._fullLayout) { return window.dash_clientside.no_update; }
  var py = evt.event.clientY;
  var rect = gd.getBoundingClientRect();
  var yPixInPlot = py - rect.top - gd._fullLayout.margin.t;
  var ya = gd._fullLayout.yaxis;
  var dataY = ya.p2l(yPixInPlot);
  return dataY;
};
