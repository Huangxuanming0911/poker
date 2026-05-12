// Main app state machine. Handles screen transitions, server I/O, and rendering.

const STATE = {
  gameId: null,
  mode: null,             // 'pve' | 'pvp'
  names: [null, null],
  aiLevel: 'advanced',
  serverState: null,      // latest state snapshot from server
  prevServerState: null,
  pendingHandoffFor: null, // PVP: idx of player who needs to "take the screen"
};

const SCREENS = ['screen-start', 'screen-table', 'screen-handoff'];

function showScreen(id) {
  for (const s of SCREENS) {
    const el = document.getElementById(s);
    if (el) el.hidden = s !== id;
  }
}

function openModal(id) { document.getElementById(id).hidden = false; }
function closeModal(id) { document.getElementById(id).hidden = true; }

function updateModeBadge(mode) {
  const badge = document.getElementById('mode-badge');
  if (!badge) return;
  if (!mode) {
    badge.hidden = true;
    badge.removeAttribute('data-mode');
    return;
  }
  badge.hidden = false;
  badge.dataset.mode = mode;
  document.getElementById('mode-badge-text').textContent =
    mode === 'pve' ? t('mode_badge_pve') : t('mode_badge_pvp');
}

// ----------------------------------------------------------------------
// Boot
// ----------------------------------------------------------------------

async function boot() {
  await loadLocale(window.__i18n.locale);
  document.getElementById('select-language').value = window.__i18n.locale;
  document.getElementById('select-card-style').value = getCardStyle();

  wireStartScreen();
  wireSettings();
  wireModals();

  document.getElementById('lang-btn').addEventListener('click', toggleLanguage);

  showScreen('screen-start');
}

async function toggleLanguage() {
  const next = window.__i18n.locale === 'zh' ? 'en' : 'zh';
  await loadLocale(next);
  document.getElementById('select-language').value = next;
  if (STATE.serverState) renderTable();
}

// ----------------------------------------------------------------------
// Settings drawer
// ----------------------------------------------------------------------

function wireSettings() {
  const panel = document.getElementById('settings-panel');
  document.getElementById('settings-btn').addEventListener('click', () => { panel.hidden = false; });
  document.getElementById('settings-close').addEventListener('click', () => { panel.hidden = true; });
  document.getElementById('select-card-style').addEventListener('change', (e) => {
    setCardStyle(e.target.value);
    if (STATE.serverState) renderTable();
  });
  document.getElementById('select-language').addEventListener('change', async (e) => {
    await loadLocale(e.target.value);
    if (STATE.serverState) renderTable();
  });
}

// ----------------------------------------------------------------------
// Start screen
// ----------------------------------------------------------------------

function wireStartScreen() {
  document.querySelectorAll('.mode-card').forEach(btn => {
    btn.addEventListener('click', () => selectMode(btn.dataset.mode));
  });
  document.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', (e) => handleSetupAction(e.currentTarget.dataset.action));
  });
}

function selectMode(mode) {
  STATE.mode = mode;
  document.querySelectorAll('.mode-card').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  document.getElementById('setup-pve').hidden = mode !== 'pve';
  document.getElementById('setup-pvp').hidden = mode !== 'pvp';

  if (mode === 'pve') {
    const a = document.getElementById('input-human-name');
    const b = document.getElementById('input-ai-name');
    if (!a.value) a.value = t('default_human_a');
    if (!b.value) b.value = t('default_ai_name');
  } else {
    const a = document.getElementById('input-name-a');
    const b = document.getElementById('input-name-b');
    if (!a.value) a.value = t('default_human_a');
    if (!b.value) b.value = t('default_human_b');
  }
}

async function handleSetupAction(action) {
  if (action === 'setup-back') {
    STATE.mode = null;
    document.querySelectorAll('.mode-card').forEach(b => b.classList.remove('active'));
    document.getElementById('setup-pve').hidden = true;
    document.getElementById('setup-pvp').hidden = true;
    return;
  }
  if (action === 'setup-start-pve') {
    const human = document.getElementById('input-human-name').value.trim() || t('default_human_a');
    const ai = document.getElementById('input-ai-name').value.trim() || t('default_ai_name');
    STATE.aiLevel = document.getElementById('select-ai-level').value;
    STATE.names = [human, ai];
    await startGame('pve');
  } else if (action === 'setup-start-pvp') {
    const a = document.getElementById('input-name-a').value.trim() || t('default_human_a');
    const b = document.getElementById('input-name-b').value.trim() || t('default_human_b');
    STATE.names = [a, b];
    await startGame('pvp');
  }
}

async function startGame(mode) {
  STATE.mode = mode;
  try {
    const data = await api.apiNewGame(mode, STATE.names, STATE.aiLevel);
    STATE.gameId = data.game_id;
    STATE.prevServerState = null;
    handleNewServerState(data.state);
  } catch (err) {
    showError(err.message);
  }
}

// ----------------------------------------------------------------------
// State transitions
// ----------------------------------------------------------------------

async function handleNewServerState(state) {
  const prev = STATE.serverState;
  STATE.prevServerState = prev;
  STATE.serverState = state;
  updateModeBadge(state.mode);

  if (state.phase === 'hand_done') {
    showScreen('screen-table');
    // Render everything EXCEPT the final community — we may animate the reveal first.
    const animatedCount = await maybeAnimateAllInReveal(state, prev);
    renderTable({ communityCount: animatedCount });
    showHandDoneDrawer(state);
    return;
  }

  if (state.phase === 'human_turn') {
    if (STATE.mode === 'pvp' && shouldShowHandoff(state)) {
      showHandoffScreen(state.current_player_idx);
      return;
    }
    showScreen('screen-table');
    renderTable();
  }
}

function shouldShowHandoff(state) {
  // In PVP, when control moves to a different human seat than before, cover the screen.
  if (STATE.mode !== 'pvp') return false;
  const prev = STATE.prevServerState;
  const newSeat = state.current_player_idx;
  if (newSeat < 0) return false;
  if (!prev) return true; // first hand: always cover
  const prevSeat = prev.phase === 'human_turn' ? prev.current_player_idx : -1;
  return newSeat !== prevSeat;
}

function showHandoffScreen(seatIdx) {
  STATE.pendingHandoffFor = seatIdx;
  const name = STATE.names[seatIdx];
  document.getElementById('handoff-title').textContent = t('pass_screen_to', { name });
  showScreen('screen-handoff');
}

// ----------------------------------------------------------------------
// All-in: animate the remaining street cards revealing one by one.
// Returns a Promise resolving to the FINAL number of community cards rendered
// (caller uses it as `communityCount` for the final renderTable).
// ----------------------------------------------------------------------

const REVEAL_INTERVAL_MS = 800;

async function maybeAnimateAllInReveal(newState, prevState) {
  const prevLen = prevState && prevState.community ? prevState.community.length : 0;
  const newLen = newState.community.length;
  if (!prevState || newLen - prevLen < 2) {
    return newLen;  // no animation needed
  }

  return new Promise((resolve) => {
    let i = prevLen;
    let cancelled = false;
    const skipHandler = () => { cancelled = true; };
    document.addEventListener('click', skipHandler, { capture: true });

    const cleanup = () => {
      document.removeEventListener('click', skipHandler, { capture: true });
    };

    const renderStep = (count) => {
      const visible = newState.community.slice(0, count);
      const placeholders = Array.from({length: Math.max(0, 5 - count)}, () =>
        '<div class="card card-placeholder" style="opacity:0.15;background:rgba(255,255,255,0.05);width:64px;height:96px;border-radius:8px;"></div>'
      ).join('');
      document.getElementById('community-cards').innerHTML =
        visible.map(c => cardHTML(c)).join('') + placeholders;
      // Animate the LAST newly added card.
      const nodes = document.querySelectorAll('#community-cards .card');
      const justAdded = nodes[count - 1];
      if (justAdded && !justAdded.classList.contains('card-placeholder')) {
        justAdded.classList.remove('card-reveal');
        void justAdded.offsetWidth;
        justAdded.classList.add('card-reveal');
      }
    };

    const step = () => {
      if (cancelled) {
        renderStep(newLen);
        cleanup();
        resolve(newLen);
        return;
      }
      i += 1;
      renderStep(i);
      if (i >= newLen) {
        cleanup();
        setTimeout(() => resolve(newLen), 400);
        return;
      }
      setTimeout(step, REVEAL_INTERVAL_MS);
    };

    // Render starting position then schedule the first reveal.
    renderStep(prevLen);
    setTimeout(step, 400);
  });
}

// ----------------------------------------------------------------------
// Render: poker table
// ----------------------------------------------------------------------

function renderTable(opts) {
  opts = opts || {};
  const s = STATE.serverState;
  if (!s) return;

  document.getElementById('pot-display').textContent = `${t('pot')}: ${s.pot}`;
  const limit = (typeof opts.communityCount === 'number') ? opts.communityCount : s.community.length;
  const shown = s.community.slice(0, limit);
  document.getElementById('community-cards').innerHTML =
    shown.map(c => cardHTML(c)).join('') ||
    Array.from({length: 3}, () => '<div class="card card-placeholder" style="opacity:0.15;background:rgba(255,255,255,0.05);width:64px;height:96px;border-radius:8px;"></div>').join('');

  // Player slots (top = idx 1, bottom = idx 0)
  for (const player of s.players) {
    const slot = document.getElementById('slot-' + player.idx);
    if (!slot) continue;
    slot.classList.toggle('current', player.idx === s.current_player_idx);
    slot.classList.toggle('folded', player.folded);
    const buttonMarker = player.is_button ? `<span class="button-marker">${t('button')}</span>` : '';
    const aiBadge = !player.is_human ? `<span class="ai-badge">${t('ai_badge')}</span>` : '';
    const statusLine = player.folded
      ? `<span class="player-status">${t('folded')}</span>`
      : (player.all_in ? `<span class="player-status">${t('all_in_label')}</span>` : '');
    const betLine = player.current_bet > 0
      ? `<div class="player-bet">${t('current_bet')}: ${player.current_bet}</div>`
      : '';
    const holeLine = renderPlayerHole(player, s);
    slot.innerHTML = `
      <div class="player-name">${escapeHtml(player.name)}${aiBadge}${buttonMarker}</div>
      <div class="player-stack">${t('stack')}: ${player.stack}</div>
      ${betLine}
      ${statusLine}
      <div class="player-hole">${holeLine}</div>
    `;
  }

  // Flash any seat that just had an AI action
  if (s.recent_actions && s.recent_actions.length > 0) {
    for (const a of s.recent_actions) {
      const slot = document.getElementById('slot-' + a.player_idx);
      if (slot) {
        slot.classList.remove('flash-action');
        // re-trigger animation
        void slot.offsetWidth;
        slot.classList.add('flash-action');
      }
    }
  }

  // Recent action log line
  renderRecentLog(s);

  // Status text + action bar
  if (s.phase === 'human_turn') {
    const name = STATE.names[s.current_player_idx] || '';
    if (STATE.mode === 'pve' || STATE.serverState.players[s.current_player_idx].is_human) {
      document.getElementById('status-text').className = 'status-text your-turn';
      document.getElementById('status-text').textContent =
        STATE.mode === 'pvp'
          ? t('waiting_for', { name })
          : t('your_turn');
    }
    renderActionBar(s);
  } else if (s.phase === 'ai_turn') {
    document.getElementById('status-text').className = 'status-text';
    document.getElementById('status-text').textContent = t('ai_thinking');
    document.getElementById('action-bar').innerHTML = '';
  } else {
    document.getElementById('status-text').textContent = '';
    document.getElementById('action-bar').innerHTML = '';
  }
}

function renderPlayerHole(player, state) {
  // PVE: human's cards always shown, AI cards hidden until showdown.
  // PVP: only the acting seat's cards are shown server-side.
  // For hand_done, server exposes both if not folded.
  if (player.folded) return '<div class="card card-placeholder" style="opacity:0.15;"></div>'.repeat(2);
  if (player.hole_cards) {
    return player.hole_cards.map(c => cardHTML(c)).join('');
  }
  return cardHTML(null) + cardHTML(null);
}

function renderRecentLog(state) {
  const log = document.getElementById('recent-log');
  if (!state.recent_actions || state.recent_actions.length === 0) {
    log.innerHTML = '';
    return;
  }
  log.innerHTML = state.recent_actions.map(a => {
    const labelKey = 'ai_action_' + a.action.replace('-', '_');
    return `<span class="log-entry">${t(labelKey, { name: a.name })}</span>`;
  }).join('');
}

function renderActionBar(state) {
  const bar = document.getElementById('action-bar');
  bar.innerHTML = '';
  const targets = state.action_targets || {};
  for (const action of state.available_actions) {
    const btn = document.createElement('button');
    btn.dataset.action = action;
    btn.classList.add('btn-' + action.replace('-', '-'));
    btn.textContent = actionButtonLabel(action, targets, state);
    btn.addEventListener('click', () => submitAction(action));
    bar.appendChild(btn);
  }
}

function actionButtonLabel(action, targets, state) {
  if (action === 'fold') return t('fold');
  if (action === 'check') return t('check');
  if (action === 'call') {
    const target = targets.call ?? 0;
    return t('call_to', { amount: target });
  }
  if (action === 'raise-2x') return t('raise_2x', { target: targets['raise-2x'] ?? '?' });
  if (action === 'raise-3x') return t('raise_3x', { target: targets['raise-3x'] ?? '?' });
  if (action === 'all-in') return t('all_in', { amount: targets['all-in'] ?? '?' });
  return action;
}

async function submitAction(action) {
  // Disable buttons while waiting
  document.querySelectorAll('#action-bar button').forEach(b => b.disabled = true);
  document.getElementById('status-text').className = 'status-text';
  document.getElementById('status-text').textContent = t('ai_thinking');
  try {
    const data = await api.apiSubmitAction(STATE.gameId, action);
    handleNewServerState(data.state);
  } catch (err) {
    showError(err.message);
  }
}

// ----------------------------------------------------------------------
// Bottom result drawer (replaces full-screen modals)
// ----------------------------------------------------------------------

function clearDrawer() {
  document.getElementById('result-title').textContent = '';
  document.getElementById('result-winners').innerHTML = '';
  document.getElementById('result-showdown').innerHTML = '';
  document.getElementById('result-message').textContent = '';
  document.getElementById('result-actions').innerHTML = '';
}

function openDrawer() { document.getElementById('result-drawer').hidden = false; }
function closeDrawer() { document.getElementById('result-drawer').hidden = true; clearDrawer(); }

function showHandDoneDrawer(state) {
  clearDrawer();
  document.getElementById('result-title').textContent = t('hand_done_title');

  const winners = state.winner_info || [];
  document.getElementById('result-winners').innerHTML = winners.map(w =>
    `<div class="winner-line">${t('hand_winner', { name: escapeHtml(w.name), amount: w.amount })}</div>`
  ).join('');

  const showdownEl = document.getElementById('result-showdown');
  if (state.showdown_holes && Object.keys(state.showdown_holes).length > 0) {
    const descs = state.hand_descriptions || {};
    showdownEl.innerHTML = Object.entries(state.showdown_holes).map(([name, cards]) => {
      const d = descs[name];
      const descText = d ? t(d.key, { high: d.high || '' }) : '';
      return `
        <div class="showdown-row">
          <span class="name">${escapeHtml(name)}</span>
          <span class="hand-cards">${cards.map(c => cardHTML(c)).join('')}</span>
          <span class="description">${escapeHtml(descText)}</span>
        </div>
      `;
    }).join('');
  }

  const actions = document.getElementById('result-actions');
  const btn = document.createElement('button');
  btn.className = 'primary-btn';
  btn.textContent = t('next_hand');
  btn.addEventListener('click', onClickNextHand);
  actions.appendChild(btn);

  openDrawer();
}

function showGameOverDrawer() {
  clearDrawer();
  const state = STATE.serverState;
  const bustedPlayer = state ? state.players.find(p => p.stack < state.big_blind) : null;
  document.getElementById('result-title').textContent = t('game_over_title');
  document.getElementById('result-message').textContent =
    t('game_over_text', { name: bustedPlayer ? bustedPlayer.name : '?' });

  const actions = document.getElementById('result-actions');
  const endBtn = document.createElement('button');
  endBtn.className = 'ghost-btn';
  endBtn.textContent = t('end_game');
  endBtn.addEventListener('click', onClickEndGame);
  actions.appendChild(endBtn);

  const restartBtn = document.createElement('button');
  restartBtn.className = 'primary-btn';
  restartBtn.textContent = t('restart_game');
  restartBtn.addEventListener('click', onClickRestartGame);
  actions.appendChild(restartBtn);

  openDrawer();
}

function showError(msg) {
  clearDrawer();
  document.getElementById('result-title').textContent = t('error_title');
  document.getElementById('result-message').textContent = msg || t('error_generic');
  const actions = document.getElementById('result-actions');
  const retryBtn = document.createElement('button');
  retryBtn.className = 'primary-btn';
  retryBtn.textContent = t('retry');
  retryBtn.addEventListener('click', closeDrawer);
  actions.appendChild(retryBtn);
  openDrawer();
}

async function onClickNextHand() {
  closeDrawer();
  try {
    const data = await api.apiNextHand(STATE.gameId);
    if (data.state.game_over) {
      STATE.prevServerState = STATE.serverState;
      STATE.serverState = data.state;
      showGameOverDrawer();
    } else {
      STATE.prevServerState = null; // force handoff cover on PVP first action
      handleNewServerState(data.state);
    }
  } catch (err) {
    showError(err.message);
  }
}

async function onClickRestartGame() {
  closeDrawer();
  try {
    const data = await api.apiNextHand(STATE.gameId);
    handleNewServerState(data.state);
  } catch (err) {
    showError(err.message);
  }
}

async function onClickEndGame() {
  closeDrawer();
  if (STATE.gameId) {
    try { await api.apiEndGame(STATE.gameId); } catch (_) {}
  }
  STATE.gameId = null; STATE.serverState = null; STATE.prevServerState = null;
  updateModeBadge(null);
  showScreen('screen-start');
  document.querySelectorAll('.mode-card').forEach(b => b.classList.remove('active'));
  document.getElementById('setup-pve').hidden = true;
  document.getElementById('setup-pvp').hidden = true;
}

function wireModals() {
  document.getElementById('handoff-ready').addEventListener('click', () => {
    STATE.pendingHandoffFor = null;
    showScreen('screen-table');
    renderTable();
  });
}

// ----------------------------------------------------------------------
// Utils
// ----------------------------------------------------------------------

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// Re-render when locale changes mid-game
onLocaleChange(() => {
  if (STATE.serverState) {
    updateModeBadge(STATE.serverState.mode);
    renderTable();
  }
});

document.addEventListener('DOMContentLoaded', boot);
