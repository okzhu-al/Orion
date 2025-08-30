from app.ui.dash_app import app

def test_orion_js_served():
    client = app.server.test_client()
    resp = client.get('/assets/orion.js')
    assert resp.status_code == 200
    assert b'cursorYtoData' in resp.data
