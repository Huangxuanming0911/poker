// Minimal i18n: load JSON for a locale, expose t(key, params).
// State stored in window.__i18n so it's globally accessible without ES module gymnastics.

window.__i18n = {
  locale: localStorage.getItem('poker_locale') || 'zh',
  translations: {},
  listeners: [],
};

async function loadLocale(locale) {
  const res = await fetch(`/static/i18n/${locale}.json`);
  if (!res.ok) throw new Error(`failed to load locale ${locale}`);
  const data = await res.json();
  window.__i18n.translations = data;
  window.__i18n.locale = locale;
  localStorage.setItem('poker_locale', locale);
  document.documentElement.lang = locale;
  applyI18nToDom();
  for (const fn of window.__i18n.listeners) {
    try { fn(); } catch (e) { console.error(e); }
  }
}

function t(key, params) {
  const dict = window.__i18n.translations;
  let s = key in dict ? dict[key] : key;
  if (params) {
    for (const k of Object.keys(params)) {
      s = s.split('{' + k + '}').join(String(params[k]));
    }
  }
  return s;
}

function applyI18nToDom(root) {
  const scope = root || document;
  scope.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  scope.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
}

function onLocaleChange(fn) {
  window.__i18n.listeners.push(fn);
}

function currentLocale() {
  return window.__i18n.locale;
}
