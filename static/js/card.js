// Card rendering for both CSS mode and SVG mode.
// State is read from window.__cardStyle ('css' | 'svg'), persisted in localStorage.

window.__cardStyle = localStorage.getItem('poker_card_style') || 'css';

const SUIT_TO_SYMBOL = { S: '♠', H: '♥', D: '♦', C: '♣' };

function setCardStyle(style) {
  if (style !== 'css' && style !== 'svg') return;
  window.__cardStyle = style;
  localStorage.setItem('poker_card_style', style);
}

function getCardStyle() {
  return window.__cardStyle;
}

function isRedSuit(suit) {
  return suit === 'H' || suit === 'D';
}

function rankDisplay(rank) {
  return rank === 'T' ? '10' : rank;
}

function svgFilename(card) {
  // Vector-Playing-Cards (Byron Knoll) uses short names like "AS.svg", "10D.svg", "JC.svg"
  const rank = card.rank === 'T' ? '10' : card.rank;
  return `${rank}${card.suit}.svg`;
}

// Render a single card (or its back if `card` is null).
// `opts.large` = use bigger size; `opts.style` = override globally selected mode.
function cardHTML(card, opts) {
  opts = opts || {};
  const mode = opts.style || getCardStyle();
  const sizeClass = opts.large ? ' card-lg' : '';
  if (!card) {
    // Byron Knoll deck doesn't ship a back image, so we always use the CSS back
    return `<div class="card card-back${sizeClass}"></div>`;
  }
  const rank = rankDisplay(card.rank);
  const suitSymbol = SUIT_TO_SYMBOL[card.suit] || card.suit;
  const altText = `${rank}${suitSymbol}`;

  if (mode === 'svg') {
    // onerror: if SVG fails to load (file missing or 404), gracefully fall back to CSS.
    return `<img class="card card-img${sizeClass}"
                 src="/static/img/cards/${svgFilename(card)}"
                 alt="${altText}"
                 onerror="this.outerHTML = window.cardCssHTML(${JSON.stringify(card)}, '${sizeClass}');" />`;
  }
  return cardCssHTML(card, sizeClass);
}

function cardCssHTML(card, sizeClass) {
  const rank = rankDisplay(card.rank);
  const suitSymbol = SUIT_TO_SYMBOL[card.suit] || card.suit;
  const colorClass = isRedSuit(card.suit) ? 'red' : 'black';
  return `
    <div class="card css-card ${colorClass}${sizeClass || ''}">
      <div class="rank-top">${rank}<div class="suit-mini">${suitSymbol}</div></div>
      <div class="suit-center">${suitSymbol}</div>
      <div class="rank-bottom">${rank}<div class="suit-mini">${suitSymbol}</div></div>
    </div>`;
}

function cardBackCssHTML(sizeClass) {
  return `<div class="card card-back${sizeClass || ''}"></div>`;
}

// Expose for inline onerror fallback handlers
window.cardCssHTML = cardCssHTML;
window.cardBackCssHTML = cardBackCssHTML;
