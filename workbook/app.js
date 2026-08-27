/* COMP5423 workbook — no dependencies, on purpose.
   Grading happens here, in the browser; the answers are public anyway. What crosses
   the network is one sign-in, and one small write per answer — queued in localStorage
   first, so a dropped connection costs nothing and a reload resends. */

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

/* A stem may carry one block of sample text — three model replies, a policy table.
   It is the one place a question has real line breaks, so it gets a <pre> of its own
   rather than being flattened into the paragraph. */
const stem = (q, cls) => `<p class="${cls}">${md(q.q)}</p>` +
  (q.example ? `<pre class="ex">${esc(q.example)}</pre>` : '') +
  (q.q_after ? `<p class="${cls}">${md(q.q_after)}</p>` : '');

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
  // The admin account signs in here too and has no students row — an account that is
  // not a student must still get in, so never assume the lookup found anything.
  const [me] = await api(`/rest/v1/students?select=nickname,known_for&id=eq.${session.id}`);
  Object.assign(session, { nickname: 'Signed in', known_for: '' }, me || {});
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
/* The queue is emptied BEFORE the write, not after, and put back only if the write
   fails. Clearing afterwards means anything that interrupts the delete — a second
   flush already in flight, a reload, a thrown error after the rows landed — leaves
   the batch queued and posts it again on the next visit, forever. The attempts log
   is append-only, so every repeat is permanent. */
let flushing = false;

async function flush() {
  if (flushing || !session) return;
  const q = store.get(QUEUE_KEY) || [];
  if (!q.length) return;
  flushing = true;
  store.del(QUEUE_KEY);
  try {
    await refreshIfNeeded();
    await api('/rest/v1/attempts', { method: 'POST', body: q });
  } catch (e) {
    // put them back in front of anything recorded while we were away
    store.set(QUEUE_KEY, q.concat(store.get(QUEUE_KEY) || []));
  } finally { flushing = false; }
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
    <p class="dim">A question bank for the course.</p>
    <div class="card">
      <label class="dim" for="code">Class code</label>
      <div class="reveal">
        <input id="code" type="password" placeholder="k7m2-qx4f" autocapitalize="off"
               autocorrect="off" spellcheck="false" autocomplete="off">
        <button type="button" id="peek" aria-pressed="false" aria-label="Show the code">Show</button>
      </div>
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
  // Hidden by default: this screen gets projected, and the code on it is a credential.
  const peek = document.getElementById('peek');
  peek.onclick = () => {
    const shown = input.type === 'text';
    input.type = shown ? 'password' : 'text';
    peek.textContent = shown ? 'Show' : 'Hide';
    peek.setAttribute('aria-pressed', String(!shown));
    peek.setAttribute('aria-label', shown ? 'Show the code' : 'Hide the code');
    input.focus();
  };
  document.getElementById('guest').onclick = e => { e.preventDefault(); viewHome(); };
  input.focus();
}

function viewHome() {
  const counts = { new: pile('new').length, pending: pile('pending').length, passed: pile('passed').length,
                   flagged: BANK.questions.filter(q => inFilter(q) && flagged.has(q.id)).length };
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
    ${session && flagsOK ? `<button data-p="flagged" ${counts.flagged ? '' : 'disabled'}>\u2691 Flagged <span class="dim">· ${counts.flagged}</span></button>` : ''}
    <p class="dim">${BANK.questions.length} exercises · built ${esc(BANK.built)}</p>`;

  app.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { filter = b.dataset.f || null; viewHome(); });
  app.querySelectorAll('[data-p]').forEach(b => b.onclick = () => startRound(b.dataset.p));
  document.getElementById('out').onclick = () => {
    if (session) { store.del(SESSION_KEY); session = null; cleared = new Map(); flagged = new Set(); }
    viewSignIn();
  };
}

function startRound(name) {
  if (name === 'flagged') return viewFlagged();
  const pool = seeded(pile(name), session ? session.id : 'guest');
  if (name === 'passed') return viewPassed(pool);
  round = { name, list: pool.slice(0, 10), i: 0, right: 0, missed: [] };
  viewQuestion();
}

/* Review, not practice: the whole exercise, with the answer written out below it
   rather than highlighted — so the eye can re-think before it reads. The ref
   (L1A-kmr) is on every card: it is how a question gets named out loud. */
function reviewCard(q) {
  const body = q.format === 'scq'
    ? Object.entries(q.options).map(([k, v]) => `<div class="opt"><b>${k}.</b> ${md(v)}</div>`).join('')
    : '';
  return `<div class="card">
    <div class="top"><strong>${md(q.title)}</strong><code class="ref">${esc(q.ref || q.class)}</code></div>
    <p class="dim" style="margin:.1rem 0 .8rem">${esc(q.topic)}</p>
    ${stem(q, "q")}
    ${body}
    <p style="margin:.9rem 0 0"><strong>Answer.</strong> ${esc(q.answer)}</p>
    <div class="why">${md(q.why)}</div>
    ${session ? flagButton(q.id) : ''}</div>`;
}

function viewPassed(all) {
  app.innerHTML = `
    <div class="top"><h1>Passed · ${all.length}</h1><button id="back">Back</button></div>
    ${all.map(reviewCard).join('') || '<p class="dim">Nothing yet.</p>'}
    <button id="back2">Back</button>`;
  wireFlags(() => { const y = scrollY; viewPassed(all); scrollTo(0, y); });
  document.getElementById('back').onclick = viewHome;
  document.getElementById('back2').onclick = viewHome;
}

/* Every flagged question in one place, whatever pile it is in — the list you work
   through when you sit down to fix the bank. Unflagging removes it from here. */
function viewFlagged() {
  const list = BANK.questions.filter(q => inFilter(q) && flagged.has(q.id));
  app.innerHTML = `
    <div class="top"><h1>\u2691 Flagged · ${list.length}</h1><button id="back">Back</button></div>
    <p class="dim">Questions you marked to come back to${filter ? ' in ' + esc(filter) : ''}.
       Quote the ref when you report one.</p>
    ${list.map(reviewCard).join('') ||
      '<p class="dim">Nothing flagged. Use \u2690 Flag on any question to add it here.</p>'}
    <button id="back2">Back</button>`;
  wireFlags(() => { const y = scrollY; viewFlagged(); scrollTo(0, y); });
  document.getElementById('back').onclick = viewHome;
  document.getElementById('back2').onclick = viewHome;
}

function viewQuestion() {
  const q = round.list[round.i];
  const opts = q.format === 'scq'
    ? Object.entries(q.options).map(([k, v]) => `<button data-a="${k}"><b>${k}.</b> ${md(v)}</button>`).join('')
    : `<div class="row"><button data-a="True">True</button><button data-a="False">False</button></div>`;
  app.innerHTML = `
    <div class="top"><span class="dim"><code class="ref">${esc(q.ref || q.class)}</code> · ${esc(round.name)} · ${round.i + 1} of ${round.list.length}</span>
      <button id="stop">Stop</button></div>
    <div class="bar"><i style="width:${round.i / round.list.length * 100}%"></i></div>
    ${stem(q, "q")}
    <div id="opts">${opts}</div>`;
  document.getElementById('stop').onclick = viewHome;
  app.querySelectorAll('[data-a]').forEach(b => b.onclick = () => answer(q, b.dataset.a));
}

function answer(q, given) {
  const correct = given === q.answer;
  record(q, given, correct);
  // Send it now rather than at the end of the round: the class report calls its top
  // section "Right now", and a whole round of lag makes that a lie. Not awaited — the
  // queue already holds the answer, so the screen never waits on the network.
  flush();
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
  // Tells the admin page "this student got in", which is a different fact from
  // "this student answered something". Failure here is irrelevant to the student.
  api(`/rest/v1/students?id=eq.${session.id}`, { method: 'PATCH',
       body: { last_seen: new Date().toISOString() } }).catch(() => {});
  const rows = await api('/rest/v1/attempts?select=question_id,correct');
  // Anything still queued has been answered but not yet written; it counts too.
  for (const r of rows.concat(store.get(QUEUE_KEY) || [])) {
    const c = cleared.get(r.question_id) || { passed: false, tries: 0 };
    cleared.set(r.question_id, { passed: c.passed || r.correct, tries: c.tries + 1 });
  }
}

(async () => {
  session = store.get(SESSION_KEY);
  try {
    // Send first, then read: loading before flushing rebuilds local state from a
    // server that has not yet been told about the answers sitting in the queue.
    await flush();
    await load();
    session ? viewHome() : viewSignIn();
  } catch (e) {
    store.del(SESSION_KEY); session = null;
    try { await load(); viewSignIn(); }
    catch { app.innerHTML = '<p class="err">Could not load the question bank. Check your connection and reload.</p>'; }
  }
})();
