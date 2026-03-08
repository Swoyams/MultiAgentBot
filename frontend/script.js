let sessionId = null;
const chat = document.getElementById('chat');
const form = document.getElementById('chat-form');
const messageInput = document.getElementById('message');

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage('user', message);
  messageInput.value = '';

  const loading = document.createElement('div');
  loading.className = 'msg bot';
  loading.textContent = 'Thinking...';
  chat.appendChild(loading);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId })
    });
    const data = await res.json();
    sessionId = data.session_id;
    loading.textContent = data.answer;
  } catch (err) {
    loading.textContent = `Error: ${err.message}`;
  }
});
