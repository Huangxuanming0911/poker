// Thin fetch wrapper for the server API.

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = data.error || `HTTP ${res.status}`;
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return data;
}

async function apiNewGame(mode, names, aiLevel) {
  return apiPost('/api/new_game', { mode, names, ai_level: aiLevel });
}
async function apiSubmitAction(gameId, action, amount) {
  return apiPost('/api/action', { game_id: gameId, action, amount });
}
async function apiNextHand(gameId) {
  return apiPost('/api/next_hand', { game_id: gameId });
}
async function apiEndGame(gameId) {
  return apiPost('/api/end_game', { game_id: gameId });
}

window.api = { apiNewGame, apiSubmitAction, apiNextHand, apiEndGame };
