window.orion = window.orion || {};
window.orion.cursorYtoData = function(evt) {
  if (!evt || !evt.event) { return window.dash_clientside && window.dash_clientside.no_update; }
  const container = document.getElementById('price-graph');
  if (!container) { return window.dash_clientside && window.dash_clientside.no_update; }
  const gd = container.getElementsByClassName('js-plotly-plot')[0];
  if (!gd || !gd._fullLayout) { return window.dash_clientside && window.dash_clientside.no_update; }
  const clientY = evt.event.clientY;
  const rect = gd.getBoundingClientRect();
  const layout = gd._fullLayout;
  const yPixInPlot = clientY - (rect.top + layout.margin.t);
  const ya = layout.yaxis;
  if (!ya || typeof ya.p2l !== 'function') { return window.dash_clientside && window.dash_clientside.no_update; }
  return ya.p2l(yPixInPlot);
};

(function () {
  function setup(gd) {
    if (gd._orionBound) return;
    function move(ev) {
      const y = window.orion.cursorYtoData(ev);
      if (y != null && y !== window.dash_clientside?.no_update) {
        window.dash_clientside?.set_props && window.dash_clientside.set_props('cursor-y-store', { data: y });
      }
    }
    function leave() {
      window.dash_clientside?.set_props && window.dash_clientside.set_props('cursor-y-store', { data: null });
    }
    gd.on('plotly_mousemove', move);
    gd.on('plotly_unhover', leave);
    gd._orionBound = true;
  }

  function bind() {
    const container = document.getElementById('price-graph');
    if (!container) return;
    const gd = container.getElementsByClassName('js-plotly-plot')[0];
    if (!gd) return;
    setup(gd);
    gd.on('plotly_afterplot', () => { gd._orionBound = false; setup(gd); });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bind);
  } else {
    bind();
  }
})();
