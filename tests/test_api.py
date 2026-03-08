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
    assert '3-day itinerary for Paris' in data['answer']
    assert len(data['trace']) >= 4


def test_chat_research_only_does_not_include_coding_output():
    client = TestClient(app)
    response = client.post('/api/chat', json={'message': 'What is machine learning?'})
    assert response.status_code == 200
    data = response.json()
    assert 'Developer Agent' not in data['answer']
    assert 'Starter snippet' not in data['answer']
