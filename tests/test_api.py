from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_chat_workflow_returns_agents():
    client = TestClient(app)
    response = client.post('/api/chat', json={'message': 'Plan a travel trip to Paris with budget 1200 and convert 100'})
    assert response.status_code == 200
    data = response.json()
    assert 'session_id' in data
    assert 'Travel Agent' in data['answer'] or 'Travel Planner' in data['answer']
    assert len(data['trace']) >= 4
