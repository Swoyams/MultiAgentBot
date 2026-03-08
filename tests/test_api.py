from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_chat_workflow_returns_structured_output():
    client = TestClient(app)
    response = client.post('/api/chat', json={'message': 'Plan a travel trip to Paris with budget 1200 and convert 100'})
    assert response.status_code == 200
    data = response.json()
    assert 'session_id' in data
    assert 'answer' in data
    assert 'structured_output' in data
    assert data['structured_output']['mode'] in {'full', 'general_search_only', 'fallback', 'memory'}


def test_chat_research_only_does_not_include_coding_output():
    client = TestClient(app)
    response = client.post('/api/chat', json={'message': 'Only do a general search: what is machine learning?'})
    assert response.status_code == 200
    data = response.json()
    assert data['structured_output']['mode'] == 'general_search_only'
    assert 'Developer Agent' not in data['answer']
    assert 'Starter snippet' not in data['answer']


def test_chat_memory_recall_uses_previous_turns():
    client = TestClient(app)
    first = client.post('/api/chat', json={'message': 'I want to travel to Rome with budget 900'})
    assert first.status_code == 200
    session_id = first.json()['session_id']

    recall = client.post('/api/chat', json={'message': 'What did I ask previously?', 'session_id': session_id})
    assert recall.status_code == 200
    data = recall.json()
    assert data['structured_output']['mode'] == 'memory'
    assert 'rome' in data['answer'].lower()
