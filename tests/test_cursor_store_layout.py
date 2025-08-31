from app.ui.dash_app import app
from dash.development.base_component import Component


def collect_ids(comp, ids):
    if isinstance(comp, Component) and getattr(comp, 'id', None):
        ids.add(comp.id)
    children = getattr(comp, 'children', None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for ch in children:
        collect_ids(ch, ids)


def test_cursor_store_present():
    ids = set()
    collect_ids(app.layout, ids)
    assert 'cursor-y-store' in ids
