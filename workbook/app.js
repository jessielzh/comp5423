/* COMP5423 workbook — no dependencies, on purpose.
   Grading happens here, in the browser; the answers are public anyway. The only
   things that cross the network are one sign-in and one batched write per round. */

const CFG = (() => {
  const q = new URLSearchParams(location.search).get('env');
  const name = q === 'test' ? 'dev'
    : q || (location.hostname === 'localhost' || location.hostname === '127.0.0.1'
        ? 'dev' : window.COMP5423.defaultEnv);
  return { name, ...window.COMP5423.envs[name] };
})();

const SESSION_KEY = 'comp5423.session';
const QUEUE_KEY = 'comp5423.queue';
const app = document.getElementById('app');

let BANK = null;        // questions.json
let session = null;     // { access_token, refresh_token, expires_at, id, nickname, known_for }
let cleared = new Map();// question_id -> { passed, tries }
let flagged = new Set();// question_ids the student marked to come back to
let flagsOK = true;     // false if the flags table is unreachable; the app carries on
let filter = null;      // class id, or null for all
let round = null;

/* ── tiny helpers ─────────────────────────────────────────────────────────── */

const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const md = s => esc(s)
  .replace(/`([^`]+)`/g, '<code>$1</code>')
  .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');

const store = {
  get: k => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
  del: k => localStorage.removeItem(k),
};

/* Deterministic shuffle, seeded from the student's own id, so the order of a pile is
   the same on their phone and their laptop and does not jump around between visits. */
function seeded(list, seed) {
  let h = 2166136261;
  for (const c of String(seed)) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  const rnd = () => { h += 0x6D2B79F5; let t = h; t = Math.imul(t ^ t >>> 15, t | 1);
                      t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const a = list.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/* ── network ──────────────────────────────────────────────────────────────── */

async function api(path, { method = 'GET', body, auth = true } = {}) {
  const h = { apikey: CFG.key, 'Content-Type': 'application/json' };
  if (auth && session) h.Authorization = 'Bearer ' + session.access_token;
  const r = await fetch(CFG.url + path, { method, headers: h, body: body && JSON.stringify(body) });
  if (!r.ok) throw Object.assign(new Error('http ' + r.status), { status: r.status, detail: await r.text() });
  const text = await r.text();          // 201 and 204 come back with no body at all
  return text ? JSON.parse(text) : null;
}

async function signIn(raw) {
  const code = raw.replace(/[^a-z0-9]/gi, '').toLowerCase();
  const t = await api('/auth/v1/token?grant_type=password', {
    auth: false, method: 'POST',
    body: { email: `${code}@comp5423.invalid`, password: code + '-comp5423-2026' },
  });
  session = { access_token: t.access_token, refresh_token: t.refresh_token,
              expires_at: Date.now() + t.expires_in * 1000, id: t.user.id };
  const [me] = await api(`/rest/v1/students?select=nickname,known_for&id=eq.${session.id}`);
  Object.assign(session, me);
  store.set(SESSION_KEY, session);
}

async function refreshIfNeeded() {
  if (!session || Date.now() < session.expires_at - 60000) return;
  const t = await api('/auth/v1/token?grant_type=refresh_token', {
    auth: false, method: 'POST', body: { refresh_token: session.refresh_token } });
  Object.assign(session, { access_token: t.access_token, refresh_token: t.refresh_token,
                           expires_at: Date.now() + t.expires_in * 1000 });
  store.set(SESSION_KEY, session);
}

/* Answers queue locally and go up in one request per round. A dropped connection in a
   lecture theatre costs nothing: the queue survives a reload and flushes later. */
async function flush() {
  const q = store.get(QUEUE_KEY) || [];
  if (!q.length || !session) return;
  try {
    await refreshIfNeeded();
    await api('/rest/v1/attempts', { method: 'POST', body: q });
    store.del(QUEUE_KEY);
  } catch (e) { /* keep the queue; try again after the next round */ }
}

function record(q, answer, correct) {
  cleared.set(q.id, { passed: (cleared.get(q.id)?.passed) || correct,
                      tries: ((cleared.get(q.id)?.tries) || 0) + 1 });
  if (!session) return;
  const pending = store.get(QUEUE_KEY) || [];
  pending.push({ student_id: session.id, question_id: q.id, answer,
                 correct, question_version: q.version });
  store.set(QUEUE_KEY, pending);
}

/* A flag is mutable state, so it goes straight to the server rather than into the
   answer queue: there is no round boundary to flush it at. */
async function toggleFlag(id) {
  if (!session) return;
  const on = flagged.has(id);
  on ? flagged.delete(id) : flagged.add(id);
  try {
    await refreshIfNeeded();
    if (on) await api(`/rest/v1/flags?student_id=eq.${session.id}&question_id=eq.${id}`, { method: 'DELETE' });
    else await api('/rest/v1/flags', { method: 'POST', body: { student_id: session.id, question_id: id } });
  } catch (e) {
    on ? flagged.add(id) : flagged.delete(id);   // put it back; the server is the truth
    console.warn('flag failed', e.status, e.detail);
  }
}

const flagButton = id => !flagsOK ? '' :
  `<button class="flag" data-flag="${id}" aria-pressed="${flagged.has(id)}">${
    flagged.has(id) ? '\u2691 Flagged' : '\u2690 Flag'}</button>`;

function wireFlags(after) {
  app.querySelectorAll('[data-flag]').forEach(b => b.onclick = async () => {
    await toggleFlag(b.dataset.flag);
    after ? after() : (b.setAttribute('aria-pressed', flagged.has(b.dataset.flag)),
                       b.textContent = flagged.has(b.dataset.flag) ? '\u2691 Flagged' : '\u2690 Flag');
  });
}

/* ── piles ────────────────────────────────────────────────────────────────── */

const inFilter = q => !filter || q.class === filter;
const pile = name => BANK.questions.filter(q => inFilter(q) && (
  name === 'new'     ? !cleared.has(q.id)
  : name === 'pending' ? cleared.has(q.id) && !cleared.get(q.id).passed
  : cleared.get(q.id)?.passed));

/* ── views ────────────────────────────────────────────────────────────────── */

function viewSignIn(err) {
  app.innerHTML = `
    <h1>COMP5423 workbook</h1>
    <p class="dim">A question bank for the course. The final exam is drawn from it.</p>
    <div class="card">
      <label class="dim" for="code">Class code</label>
      <input id="code" placeholder="k7m2-qx4f" autocapitalize="off" autocorrect="off" spellcheck="false">
      <button class="primary" id="go">Continue</button>
      ${err ? `<p class="err">${esc(err)}</p>` : ''}
    </div>
    <p class="dim">No code? <a href="#" id="guest">Practice without saving progress</a>.</p>`;
  const input = document.getElementById('code');
  const submit = async () => {
    const btn = document.getElementById('go');
    btn.disabled = true; btn.textContent = 'Checking…';
    try { await signIn(input.value); await load(); viewHome(); }
    catch (e) {
      viewSignIn(e.status === 400 ? 'That code was not recognised. Check for a typo.'
                                  : 'Could not reach the server. Try again.');
    }
  };
  document.getElementById('go').onclick = submit;
  input.onkeydown = e => { if (e.key === 'Enter') submit(); };
  document.getElementById('guest').onclick = e => { e.preventDefault(); viewHome(); };
  input.focus();
}

function viewHome() {
  const counts = { new: pile('new').length, pending: pile('pending').length, passed: pile('passed').length };
  const bars = BANK.classes.map(c => {
    const total = BANK.questions.filter(q => q.class === c.id).length;
    const done = BANK.questions.filter(q => q.class === c.id && cleared.get(q.id)?.passed).length;
    return `<div style="margin:.6rem 0">
      <div class="top"><span>${esc(c.id)} · ${esc(c.title)}</span><span class="dim">${done}/${total}</span></div>
      <div class="bar"><i style="width:${total ? done / total * 100 : 0}%"></i></div></div>`;
  }).join('');

  app.innerHTML = `
    <div class="top">
      <div>
        <h1>${session ? esc(session.nickname) : 'Practicing as a guest'}</h1>
        <p class="dim">${session ? esc(session.known_for || '') : 'Nothing is saved. Sign in to keep progress.'}</p>
      </div>
      <button id="out">${session ? 'Sign out' : 'Sign in'}</button>
    </div>
    <div class="card">${bars}</div>
    <div class="pills">
      <button data-f="" aria-pressed="${!filter}">All</button>
      ${BANK.classes.map(c => `<button data-f="${c.id}" aria-pressed="${filter === c.id}">${esc(c.id)}</button>`).join('')}
    </div>
    <button data-p="new"     ${counts.new ? '' : 'disabled'}>New questions <span class="dim">· ${counts.new}</span></button>
    <button data-p="pending" ${counts.pending ? '' : 'disabled'}>Pending <span class="dim">· ${counts.pending}</span></button>
    <button data-p="passed"  ${counts.passed ? '' : 'disabled'}>Passed <span class="dim">· ${counts.passed}</span></button>
    <p class="dim">${BANK.questions.length} exercises · built ${esc(BANK.built)}</p>`;

  app.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { filter = b.dataset.f || null; viewHome(); });
  app.querySelectorAll('[data-p]').forEach(b => b.onclick = () => startRound(b.dataset.p));
  document.getElementById('out').onclick = () => {
    if (session) { store.del(SESSION_KEY); session = null; cleared = new Map(); flagged = new Set(); }
    viewSignIn();
  };
}

function startRound(name) {
  const pool = seeded(pile(name), session ? session.id : 'guest');
  if (name === 'passed') return viewPassed(pool);
  round = { name, list: pool.slice(0, 10), i: 0, right: 0, missed: [] };
  viewQuestion();
}

/* Review, not practice: the whole exercise, with the answer written out below it
   rather than highlighted — so the eye can re-think before it reads. */
let passedFlaggedOnly = false;

function viewPassed(all) {
  const list = passedFlaggedOnly ? all.filter(q => flagged.has(q.id)) : all;
  const card = q => {
    const body = q.format === 'scq'
      ? Object.entries(q.options).map(([k, v]) => `<div class="opt"><b>${k}.</b> ${md(v)}</div>`).join('')
      : '';
    return `<div class="card">
      <div class="top"><strong>${md(q.title)}</strong><span class="dim">${esc(q.class)}</span></div>
      <p class="dim" style="margin:.1rem 0 .8rem">${esc(q.topic)}</p>
      <p class="q" style="margin:0 0 .8rem">${md(q.q)}</p>
      ${body}
      <p style="margin:.9rem 0 0"><strong>Answer.</strong> ${esc(q.answer)}</p>
      <div class="why">${md(q.why)}</div>
      ${session ? flagButton(q.id) : ''}</div>`;
  };
  app.innerHTML = `
    <div class="top"><h1>Passed · ${all.length}</h1><button id="back">Back</button></div>
    ${session && flagsOK ? `<div class="pills">
      <button data-pf="0" aria-pressed="${!passedFlaggedOnly}">All</button>
      <button data-pf="1" aria-pressed="${passedFlaggedOnly}">\u2691 Flagged · ${all.filter(q => flagged.has(q.id)).length}</button>
    </div>` : ''}
    ${list.map(card).join('') ||
      `<p class="dim">${passedFlaggedOnly ? 'Nothing flagged yet.' : 'Nothing yet.'}</p>`}
    <button id="back2">Back</button>`;
  app.querySelectorAll('[data-pf]').forEach(b => b.onclick = () => {
    passedFlaggedOnly = b.dataset.pf === '1'; viewPassed(all);
  });
  wireFlags(() => { const y = scrollY; viewPassed(all); scrollTo(0, y); });
  document.getElementById('back').onclick = viewHome;
  document.getElementById('back2').onclick = viewHome;
}

function viewQuestion() {
  const q = round.list[round.i];
  const opts = q.format === 'scq'
    ? Object.entries(q.options).map(([k, v]) => `<button data-a="${k}"><b>${k}.</b> ${md(v)}</button>`).join('')
    : `<div class="row"><button data-a="True">True</button><button data-a="False">False</button></div>`;
  app.innerHTML = `
    <div class="top"><span class="dim">${esc(q.class)} · ${esc(round.name)} · ${round.i + 1} of ${round.list.length}</span>
      <button id="stop">Stop</button></div>
    <div class="bar"><i style="width:${round.i / round.list.length * 100}%"></i></div>
    <p class="q">${md(q.q)}</p>
    <div id="opts">${opts}</div>`;
  document.getElementById('stop').onclick = viewHome;
  app.querySelectorAll('[data-a]').forEach(b => b.onclick = () => answer(q, b.dataset.a));
}

function answer(q, given) {
  const correct = given === q.answer;
  record(q, given, correct);
  if (correct) round.right++; else round.missed.push(q);

  app.querySelectorAll('#opts [data-a]').forEach(b => {
    b.disabled = true;
    if (b.dataset.a === q.answer) b.classList.add('good');
    else if (b.dataset.a === given) b.classList.add('bad');
  });
  const label = q.format === 'scq' ? q.answer : q.answer;
  app.insertAdjacentHTML('beforeend', `
    <p style="margin:1rem 0 0;color:${correct ? 'var(--good)' : 'var(--bad)'}">
      ${correct ? '✓ correct' : '✗ the answer is ' + esc(label)}</p>
    <div class="why">${md(q.why)}</div>
    ${session ? flagButton(q.id) : ''}
    <button class="primary" id="next">${round.i + 1 < round.list.length ? 'Next →' : 'Finish round'}</button>`);
  wireFlags();
  document.getElementById('next').onclick = () => {
    round.i++;
    if (round.i < round.list.length) viewQuestion(); else viewSummary();
  };
  document.getElementById('next').scrollIntoView({ block: 'nearest' });
}

async function viewSummary() {
  await flush();
  const more = pile(round.name).length;
  app.innerHTML = `
    <h1 style="font-size:2rem;text-align:center;margin:1.5rem 0 .2rem">${round.right} / ${round.list.length}</h1>
    <p class="dim" style="text-align:center">${session ? 'saved' : 'not saved — you are a guest'}</p>
    <div class="card">
      ${round.missed.length
        ? `<p class="dim">${round.missed.length} now pending:</p>` +
          round.missed.map(q => `<div style="margin:.35rem 0">${md(q.title)}
            <div class="dim">${esc(q.topic)}</div></div>`).join('')
        : '<p class="dim">Everything in this round is passed.</p>'}
    </div>
    ${more ? `<button class="primary" id="again">Another 10 <span style="opacity:.7">· ${more} left</span></button>` : ''}
    <button id="done">Done</button>`;
  if (more) document.getElementById('again').onclick = () => startRound(round.name);
  document.getElementById('done').onclick = viewHome;
}

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function load() {
  if (!BANK) BANK = await (await fetch('data/questions.json', { cache: 'no-cache' })).json();
  cleared = new Map(); flagged = new Set();
  if (!session) return;
  await refreshIfNeeded();
  // Flags are a study aid, not a requirement. If the table is missing, or the request
  // fails, sign-in must still succeed — never let an extra fetch lock a student out.
  try {
    for (const f of await api('/rest/v1/flags?select=question_id')) flagged.add(f.question_id);
  } catch (e) { flagsOK = false; }
  const rows = await api('/rest/v1/attempts?select=question_id,correct');
  for (const r of rows) {
    const c = cleared.get(r.question_id) || { passed: false, tries: 0 };
    cleared.set(r.question_id, { passed: c.passed || r.correct, tries: c.tries + 1 });
  }
}

(async () => {
  session = store.get(SESSION_KEY);
  try {
    await load();
    await flush();
    session ? viewHome() : viewSignIn();
  } catch (e) {
    store.del(SESSION_KEY); session = null;
    try { await load(); viewSignIn(); }
    catch { app.innerHTML = '<p class="err">Could not load the question bank. Check your connection and reload.</p>'; }
  }
})();
