/* COMP5423 class report. Same sign-in as the students use; one database rule
   (is_admin()) is what lets this account read every row instead of its own. */

const CFG = (() => {
  const q = new URLSearchParams(location.search).get('env');
  const name = q === 'test' ? 'dev'
    : q || (['localhost', '127.0.0.1'].includes(location.hostname) ? 'dev' : window.COMP5423.defaultEnv);
  return { name, ...window.COMP5423.envs[name] };
})();

const KEY = 'comp5423.admin';
const app = document.getElementById('app');
let session = store(KEY), BANK = null, timer = null;

function store(k, v) {
  if (v === undefined) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } }
  v === null ? localStorage.removeItem(k) : localStorage.setItem(k, JSON.stringify(v));
}
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (a, b) => b ? Math.round(100 * a / b) : 0;
/* Question text is markdown in the bank, so it has to be rendered here too —
   the same three rules the workbook applies, escaping first. Math arrives from
   publish.py already rendered to HTML and fenced in \x01, exactly as in app.js:
   odd pieces of the split are that HTML and pass through unescaped. Text with no
   math splits into one piece and behaves as it always did. */
const md = s => String(s ?? '').split('\x01').map((part, i) => i % 2 ? part : esc(part)
  .replace(/`([^`]+)`/g, '<code>$1</code>')
  .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>')).join('');

/* A stem may carry one block of sample text — three model replies, a policy table.
   It is the one place a question has real line breaks, so it gets a <pre> of its own
   rather than being flattened into the paragraph. */
const stem = (q, cls) => `<p class="${cls}">${md(q.q)}</p>` +
  (q.example ? `<pre class="ex">${esc(q.example)}</pre>` : '') +
  (q.q_after ? `<p class="${cls}">${md(q.q_after)}</p>` : '');

function ago(iso) {
  if (!iso) return '—';
  const m = Math.floor((Date.now() - new Date(iso)) / 60000);
  return m < 1 ? 'just now' : m < 60 ? m + ' min' : m < 1440 ? Math.floor(m / 60) + ' h' : Math.floor(m / 1440) + ' d';
}

async function api(path, opts = {}) {
  const h = { apikey: CFG.key, 'Content-Type': 'application/json' };
  if (session) h.Authorization = 'Bearer ' + session.access_token;
  Object.assign(h, opts.headers || {});
  const r = await fetch(CFG.url + path, { ...opts, headers: h });
  if (!r.ok) throw Object.assign(new Error('http ' + r.status), { status: r.status, detail: await r.text() });
  const t = await r.text();
  return { body: t ? JSON.parse(t) : null, headers: r.headers };
}

const PAGE = 1000;   // Supabase caps a response at this many rows

/* PostgREST caps a response, so a long log arrives in pages and a busy Monday
   must not silently truncate the report. Ask for the row count along with the
   first page, then fetch the remaining pages a handful at a time: a term of
   attempts is a dozen pages, and walking them in a chain pays a dozen round
   trips of latency before anything can be drawn. */
async function all(path) {
  const { body, headers } = await api(path, {
    headers: { Range: `0-${PAGE - 1}`, Prefer: 'count=exact' } });
  if (body.length < PAGE) return body;

  const total = Number(String(headers.get('content-range') || '').split('/')[1]);
  // No count to work from — an older server, or the header kept from us by CORS.
  // Fall back to the one-at-a-time walk: slower, but it cannot be wrong.
  if (!Number.isFinite(total)) {
    const out = [...body];
    for (let from = PAGE; ; from += PAGE) {
      const { body: b } = await api(path, { headers: { Range: `${from}-${from + PAGE - 1}` } });
      out.push(...b);
      if (b.length < PAGE) return out;
    }
  }

  const starts = [];
  for (let from = PAGE; from < total; from += PAGE) starts.push(from);
  const out = [body];
  // Six at a time: a browser will not open more connections to one host anyway,
  // and firing fifty at once only invites the server to start refusing them.
  for (let i = 0; i < starts.length; i += 6) {
    out.push(...await Promise.all(starts.slice(i, i + 6).map(from =>
      api(path, { headers: { Range: `${from}-${from + PAGE - 1}` } }).then(r => r.body))));
  }
  return out.flat();
}

/* The attempts log only ever grows — a reset deletes nothing — so it is
   downloaded once and then topped up: every refresh after the first asks only
   for rows newer than the newest one already held. That turns the 30-second
   tick from "the whole term again" into a handful of rows.

   The five-second overlap costs a few duplicate rows and buys away any worry
   about two attempts sharing a created_at; the map is keyed on the row id, so a
   row that arrives twice lands in the same slot. */
const ATTEMPT_COLS = 'id,student_id,question_id,answer,correct,created_at,question_version';
let LOG = null, LOG_AT = 0;   // rows by id, and the newest created_at held, in ms

/* The log also outlives the page. Class report, students and questions are three
   separate pages, and without this every click between them — and every reopened
   tab — downloaded the whole term again before anything could be drawn. The copy
   lives in this browser's IndexedDB, is thrown away on sign-out, and is trusted
   for half a day at most: a student deleted outright takes their rows with them
   (on delete cascade), and a periodic full download is what notices that. */
const CACHE_DB = 'comp5423.admin', CACHE_TTL = 12 * 3600e3;

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(CACHE_DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore('log');
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function cacheGet() {
  try {
    const db = await idb();
    return await new Promise(res => {
      const g = db.transaction('log').objectStore('log').get(CFG.name);
      g.onsuccess = () => res(g.result || null);
      g.onerror = () => res(null);
    });
  } catch { return null; }
}
async function cachePut(v) {
  try { (await idb()).transaction('log', 'readwrite').objectStore('log').put(v, CFG.name); } catch {}
}
function cacheClear() { try { indexedDB.deleteDatabase(CACHE_DB); } catch {} }

function keep(rows) {
  for (const a of rows) {
    LOG.set(a.id, a);
    const t = +new Date(a.created_at);
    if (t > LOG_AT) LOG_AT = t;
  }
}

async function attemptLog() {
  if (!LOG) {
    LOG = new Map(); LOG_AT = 0;
    const c = await cacheGet();
    if (c && Date.now() - c.saved < CACHE_TTL) { keep(c.rows); LOG.saved = c.saved; }
  }
  const since = LOG_AT ? `&created_at=gte.${encodeURIComponent(new Date(LOG_AT - 5000).toISOString())}` : '';
  // Paging by Range is only sound over a fixed order: without one Postgres may hand
  // back the same row on two pages and skip another entirely.
  const rows = await all(`/rest/v1/attempts?select=${ATTEMPT_COLS}${since}&order=id`);
  keep(rows);
  const log = [...LOG.values()];
  if (rows.length || !LOG.saved) {
    // The TTL runs from the last full download, not from the last top-up.
    LOG.saved = LOG.saved || Date.now();
    cachePut({ saved: LOG.saved, rows: log });
  }
  return log;
}

/* ── data ─────────────────────────────────────────────────────────────────── */

async function report() {
  const [students, log] = await Promise.all([
    all('/rest/v1/students?select=id,nickname,last_seen&active=is.true&order=nickname'),
    attemptLog(),
  ]);
  const byId = new Map(BANK.questions.map(q => [q.id, q]));
  const now = Date.now();

  const per = new Map(students.map(s => [s.id, { ...s, n: 0, passed: new Set(), last: null }]));
  const perQ = new Map();
  let last15 = 0, today = 0;
  const midnight = new Date(); midnight.setHours(0, 0, 0, 0);

  for (const a of log) {
    const t = new Date(a.created_at);
    if (now - t < 15 * 60000) last15++;
    if (t >= midnight) today++;
    const s = per.get(a.student_id);
    // Only questions still in the bank count as passed — a retired one is no longer
    // something anyone can pass, and counting it would disagree with the status board.
    // Any wording of a live question counts, though: cleared stays cleared.
    if (s) { s.n++; if (a.correct && byId.has(a.question_id)) s.passed.add(a.question_id);
             if (!s.last || t > new Date(s.last)) s.last = a.created_at; }
    // An attempt answered against an older wording says nothing about the question
    // as it stands now — mixing them reports options that no longer exist. It still
    // counts as a pass for the student, above; it just does not count as evidence here.
    const q = perQ.get(a.question_id) || { n: 0, wrong: 0, stale: 0, picks: {} };
    const now_v = byId.get(a.question_id)?.version;
    if (now_v && a.question_version && a.question_version !== now_v) q.stale++;
    else { q.n++; q.picks[a.answer] = (q.picks[a.answer] || 0) + 1; if (!a.correct) q.wrong++; }
    perQ.set(a.question_id, q);
  }

  // Status is about right now, not about the term: an answer in the last 30 minutes,
  // or else the app opened in the last hour, or neither. The hour is wider because
  // last_seen is written only when the app is opened or signed into, not while it
  // stays open. What they did earlier is in the Attempts and Last columns.
  const within = (iso, min) => iso && now - new Date(iso) < min * 60000;
  const roster = [...per.values()].map(s => ({
    ...s, passed: s.passed.size, never: !s.n && !s.last_seen,
    state: within(s.last, 30) ? 'answering' : within(s.last_seen, 60) ? 'browsing' : 'idle',
  }));
  const rank = { answering: 0, browsing: 1, idle: 2 };
  roster.sort((a, b) => rank[a.state] - rank[b.state] || b.passed - a.passed || a.nickname.localeCompare(b.nickname));

  // Every live question, not a top ten: this feeds a page of its own now, and a
  // question nobody has answered is itself worth seeing.
  const missed = BANK.questions.map(q => {
    const a = perQ.get(q.id) || { n: 0, wrong: 0, stale: 0, picks: {} };
    const top = Object.entries(a.picks).filter(([k]) => k !== q.answer)
                      .sort((x, y) => y[1] - x[1])[0];
    return { q, n: a.n, wrong: a.wrong, stale: a.stale, picks: a.picks,
             wrongPct: pct(a.wrong, a.n), top };
  }).sort((a, b) => b.wrong - a.wrong || b.wrongPct - a.wrongPct || a.q.class.localeCompare(b.q.class));

  return { roster, missed, attempts: log.length, today, last15 };
}

/* The dashboard asks the server to count and never downloads the log. Three
   attempt counts come back in a header (a HEAD request, no rows), and the class
   is one small request: each active account with its number of attempts. Active
   means a real person — the class, the instructor and the TA — not the unissued
   nicknames in the pool (assign_ids.py marks those inactive) or anyone who dropped. */
async function summary() {
  const at = ms => encodeURIComponent(new Date(ms).toISOString());
  const count = async filter => {
    const { headers } = await api(`/rest/v1/attempts?select=id${filter}`,
      { method: 'HEAD', headers: { Prefer: 'count=exact' } });
    const n = Number(String(headers.get('content-range') || '').split('/')[1]);
    if (!Number.isFinite(n)) throw new Error('no count in the reply');
    return n;
  };
  const midnight = new Date(); midnight.setHours(0, 0, 0, 0);
  const [people, total, last15, today] = await Promise.all([
    api('/rest/v1/students?select=last_seen,attempts(count)&active=is.true').then(r => r.body),
    count(''),
    count(`&created_at=gte.${at(Date.now() - 15 * 60000)}`),
    count(`&created_at=gte.${at(+midnight)}`),
  ]);
  const n = s => s.attempts?.[0]?.count || 0;
  return {
    people: people.length,
    signedIn: people.filter(s => s.last_seen || n(s)).length,
    answered: people.filter(n).length,
    total, last15, today,
  };
}

/* ── views ────────────────────────────────────────────────────────────────── */

function viewSignIn(err) {
  app.innerHTML = `<h1>COMP5423 · class report</h1>
    <p class="dim">Instructor sign-in.</p>
    <p class="reveal">
      <input id="code" type="password" placeholder="class code" autocapitalize="off"
             autocorrect="off" spellcheck="false" autocomplete="off">
    </p>
    <p><button id="go">Continue</button></p>
    ${err ? `<p class="err">${esc(err)}</p>` : ''}`;
  const go = async () => {
    const c = document.getElementById('code').value.replace(/[^a-z0-9]/gi, '').toLowerCase();
    try {
      const { body } = await api('/auth/v1/token?grant_type=password', { method: 'POST',
        body: JSON.stringify({ email: `${c}@comp5423.invalid`, password: c + '-comp5423-2026' }) });
      session = { access_token: body.access_token, expires_at: Date.now() + body.expires_in * 1000 };
      store(KEY, session); boot();
    } catch (e) { viewSignIn(e.status === 400 ? 'Code not recognised.' : 'Could not reach the server.'); }
  };
  document.getElementById('go').onclick = go;
  // Must not be a concise arrow: `e.key === 'Enter' && go()` returns false for every
  // other key, and returning false from an on* handler cancels the keypress itself.
  document.getElementById('code').onkeydown = e => { if (e.key === 'Enter') go(); };
  // The admin code reads every student's attempts, and this screen gets projected.
  // There is deliberately no reveal button: one stray click puts the code on the wall.
  document.getElementById('code').focus();
}

function tile(value, of, label) {
  return `<div class="tile"><b>${value}${of !== null ? `<span class="of"> / ${of}</span>` : ''}</b>
    <span>${esc(label)}</span></div>`;
}

const LIVE = window.PAGE === 'admin';   // only the dashboard refreshes itself

const shell = (title, crumbs, body) => `
    <div class="head">
      <h1>COMP5423 · ${title}</h1>
      <span class="faint">${CFG.name} · updated ${new Date().toLocaleTimeString()}
        ${LIVE ? '' : '<button id="again" style="margin-left:.5rem">Refresh</button>'}
        <button id="out" style="margin-left:.5rem">Sign out</button></span>
    </div>
    ${crumbs ? `<p class="crumb">${crumbs}</p>` : ''}
    ${body}`;

function paint(html) {
  app.innerHTML = html;
  const out = document.getElementById('out');
  if (out) out.onclick = () => { store(KEY, null); session = null; LOG = null; cacheClear(); clearInterval(timer); viewSignIn(); };
  const again = document.getElementById('again');
  if (again) again.onclick = async () => {
    again.disabled = true; again.textContent = 'Refreshing…';
    try { await load(); } catch { again.disabled = false; again.textContent = 'Refresh failed — retry'; }
  };
}

/* The dashboard holds the numbers you want at 12:30 and nothing that takes a
   moment to draw. The two long tables live behind links, and are only fetched
   and built when you ask for them. */
function viewDashboard(d) {
  paint(shell('class report', '', `
    <h2>Overview</h2>
    <div class="tiles">
      ${tile(d.total, null, 'attempts in total')}
    </div>

    <h2>Right now</h2>
    <div class="tiles">
      ${tile(d.signedIn, null, 'have signed in')}
      ${tile(d.answered, null, 'have answered something')}
      ${tile(d.last15, null, 'answers in the last 15 min')}
      ${tile(d.today, null, 'answers today')}
    </div>

    <h2>Look closer</h2>
    <div class="jump">
      <a href="students.html"><b>Students · ${d.people}</b></a>
      <a href="questions.html"><b>Questions · ${BANK.questions.length}</b></a>
    </div>

    <p class="faint" style="margin-top:2rem">refreshes every 30 s</p>`));
}

/* ── sorting ──────────────────────────────────────────────────────────────── */

/* Click a column heading to sort by it; click it again to reverse. The choice
   survives the 30-second refresh, like the class filter, so a table sorted at
   12:31 is still in that order at 12:32. null means the view's own order. */
const sorts = { students: null, questions: null };
let absentOpen = false;   // likewise: a heading click inside the fold must not shut it

/* Missing values sink whichever way the column points — a student who has never
   answered is not the most recent one in either direction. */
function bySort(page, rows, cols) {
  const s = sorts[page];
  if (!s) return rows;
  const val = cols[s.i].val;
  return [...rows].sort((a, b) => {
    const x = val(a), y = val(b);
    const xn = x === null || x === undefined || x === '';
    const yn = y === null || y === undefined || y === '';
    if (xn || yn) return xn && yn ? 0 : xn ? 1 : -1;
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * s.dir;
  });
}

/* A column with no `val` is not sortable. The idle arrow points the way the
   first click will sort, and CSS keeps it invisible until the heading is
   hovered or active. */
const sortHead = (page, cols) => `<thead><tr>${cols.map((c, i) => {
  const s = sorts[page], on = s && s.i === i, dir = on ? s.dir : 0;
  const attrs = c.val ? ` data-sort="${i}" tabindex="0" role="button"` +
    ` aria-sort="${on ? (dir < 0 ? 'descending' : 'ascending') : 'none'}"` : '';
  return `<th class="${[c.cls, c.val && 'sort', on && 'on'].filter(Boolean).join(' ')}"` +
    `${c.style ? ` style="${c.style}"` : ''}${attrs}>${c.label}` +
    `${c.val ? `<i class="ar">${(on ? dir < 0 : c.desc) ? '▾' : '▴'}</i>` : ''}</th>`;
}).join('')}</tr></thead>`;

/* First click sorts the way you actually want to read the column: names from A,
   counts and times from the top. */
function wireSort(page, cols, redraw) {
  app.querySelectorAll('[data-sort]').forEach(th => {
    const go = () => {
      const i = +th.dataset.sort, s = sorts[page];
      sorts[page] = s && s.i === i ? { i, dir: -s.dir } : { i, dir: cols[i].desc ? -1 : 1 };
      redraw();
    };
    th.onclick = go;
    th.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
  });
}

function viewStudents(d) {
  const word = { answering: 'answering', browsing: 'browsing', idle: 'not active' };
  const tone = { answering: 'good', browsing: 'warn', idle: 'idle' };
  // Status sorts by who is here now, not alphabetically.
  const rank = { answering: 0, browsing: 1, idle: 2 };
  const cols = [
    { label: 'Nickname', val: s => s.nickname },
    { label: 'Status', val: s => rank[s.state] },
    { label: 'Passed', cls: 'num', desc: true, val: s => s.passed },
    { label: 'Attempts', cls: 'num', desc: true, val: s => s.n },
    { label: 'Last answer', cls: 'num', desc: true, val: s => s.last && +new Date(s.last) },
    { label: 'Last seen', cls: 'num', desc: true, val: s => s.last_seen && +new Date(s.last_seen) },
  ];
  // Accounts never signed in would bury the ones that matter. They are one click
  // away, never gone. Both tables follow the same sort.
  const here = bySort('students', d.roster.filter(s => !s.never), cols);
  const absent = bySort('students', d.roster.filter(s => s.never), cols);
  const row = s => `<tr>
        <td>${esc(s.nickname)}</td>
        <td><span class="st ${tone[s.state]}"><i></i>${word[s.state]}</span></td>
        <td class="num">${s.passed}</td>
        <td class="num">${s.n || '—'}</td>
        <td class="num faint">${ago(s.last)}</td>
        <td class="num faint">${ago(s.last_seen)}</td></tr>`;
  // Two different facts, so two columns. "Last answer" comes from the attempts table;
  // "Last seen" is written on every sign-in and page load, and is the only signal a
  // student who has read but answered nothing leaves behind.
  const head = sortHead('students', cols);

  paint(shell('students', '<a href="admin.html">← Class report</a>', `
    <h2>Signed in · ${here.length} of ${d.roster.length}</h2>
    <div class="scroll"><table>${head}
      <tbody>${here.map(row).join('') || '<tr><td colspan="6" class="dim">Nobody yet.</td></tr>'}</tbody>
    </table></div>
    ${absent.length ? `<details class="fold"${absentOpen ? ' open' : ''}>
      <summary>${absent.length} ${absent.length === 1 ? 'account has' : 'accounts have'} never signed in</summary>
      <div class="scroll"><table>${head}<tbody>${absent.map(row).join('')}</tbody></table></div>
    </details>` : ''}`));

  wireSort('students', cols, () => viewStudents(d));
  const fold = app.querySelector('details.fold');
  if (fold) fold.ontoggle = () => { absentOpen = fold.open; };
}

let classFilter = null;   // survives the 30-second refresh

/* A modal, not an inline strip: this gets projected in class, so it wants the
   whole question at reading size and nothing else on screen competing with it. */
function optionKeys(q, picks) {
  if (q.format === 'scq') return Object.keys(q.options);
  if (q.format === 'tf') return ['True', 'False'];
  return [...new Set([q.answer, ...Object.keys(picks)])];   // calc: whatever was typed
}

function showQuestion(m) {
  const { q, picks } = m;
  const worst = m.top && m.top[0];
  let dlg = document.getElementById('qm');
  if (!dlg) { dlg = document.createElement('dialog'); dlg.id = 'qm'; document.body.appendChild(dlg); }
  dlg.innerHTML = `
    <div class="qm-top">
      <span class="tag">${esc(q.ref || q.class)} · ${esc(q.topic)}</span>
      <button id="qmx" aria-label="Close">&times;</button>
    </div>
    <h3>${md(q.title)}</h3>
    ${stem(q, "qm-q")}
    <ul class="qm-opts">${optionKeys(q, picks).map(k => {
      const cls = k === q.answer ? 'ok' : k === worst ? 'no' : '';
      const n = picks[k] || 0;
      return `<li class="${cls}">
        <b>${esc(k)}</b><span class="t">${md(q.options?.[k] || '')}</span>
        <span class="n">${n || ''}</span></li>`;
    }).join('')}</ul>
    <p class="qm-foot">${m.n ? `${m.n} answer${m.n === 1 ? '' : 's'} · ${m.wrong} wrong (${m.wrongPct}%)`
                             : 'Nobody has answered this yet'}${
      m.stale ? ` · ${m.stale} on an earlier wording, not counted` : ''}</p>`;
  dlg.showModal();
  document.getElementById('qmx').onclick = () => dlg.close();
  dlg.onclick = e => { if (e.target === dlg) dlg.close(); };   // click the backdrop to dismiss
}

function viewQuestions(d) {
  const classes = [...new Set(d.missed.map(m => m.q.class))].sort();
  // A question with no answers against its current wording has nothing to say about
  // wrong, n or rate, so it sinks to the bottom however the column points.
  const cols = [
    { label: 'Question', val: m => m.q.title },
    { label: 'Wrong', cls: 'num', desc: true, val: m => m.n ? m.wrong : null },
    { label: 'n', cls: 'num', desc: true, val: m => m.n || null },
    { label: 'Rate', style: 'width:8rem', desc: true, val: m => m.n ? m.wrongPct : null },
  ];
  const list = bySort('questions',
    d.missed.filter(m => !classFilter || m.q.class === classFilter), cols);
  const answered = list.filter(m => m.n > 0);

  paint(shell('questions', '<a href="admin.html">← Class report</a>', `
    <div class="pills">
      <button data-c="" aria-pressed="${!classFilter}">All</button>
      ${classes.map(c => `<button data-c="${esc(c)}" aria-pressed="${classFilter === c}">${esc(c)}</button>`).join('')}
    </div>
    <h2>${list.length} questions${sorts.questions ? '' : ' · most wrong answers first'}</h2>
    <div class="scroll"><table>
      ${sortHead('questions', cols)}
      <tbody>${list.map((m, i) => `<tr class="qrow" tabindex="0" role="button" data-i="${i}">
        <td class="miss">${md(m.q.title)}<br><span class="tag">${esc(m.q.ref || m.q.class)} · ${esc(m.q.topic)}</span></td>
        <td class="num">${m.n ? m.wrong : '—'}</td>
        <td class="num faint">${m.n || '—'}</td>
        <td>${m.n ? `<div class="bar"><i style="width:${m.wrongPct}%"></i></div>
            <span class="faint" style="font-size:.78rem">${m.wrongPct}%</span>`
          // n counts only attempts against the current wording. A question that was
          // reworded after people answered it has n === 0 with stale > 0: answered,
          // but not answered against the question as it now stands.
          : m.stale ? `<span class="faint">reworded since</span>`
          : '<span class="faint">untouched</span>'}</td>
      </tr>`).join('')}</tbody></table></div>
    <p class="faint" style="margin-top:1rem">${answered.length} of ${list.length} have been answered
      at least once against their current wording.</p>`));

  app.querySelectorAll('[data-c]').forEach(b => b.onclick = () => {
    classFilter = b.dataset.c || null; viewQuestions(d);
  });
  wireSort('questions', cols, () => viewQuestions(d));
  app.querySelectorAll('[data-i]').forEach(r => {
    const show = () => showQuestion(list[+r.dataset.i]);
    r.onclick = show;
    r.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } };
  });
}

/* The dashboard is counts only, so it is cheap to redraw every 30 seconds. The
   students and questions pages need the whole log and are drawn once; their
   Refresh button tops the log up with whatever is new. */
async function load() {
  if (window.PAGE === 'students') return viewStudents(await report());
  if (window.PAGE === 'questions') return viewQuestions(await report());
  return viewDashboard(await summary());
}

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function boot() {
  if (!session) return viewSignIn();
  try {
    if (!BANK) BANK = await (await fetch('data/questions.json', { cache: 'no-cache' })).json();
    await load();
    clearInterval(timer);
    if (LIVE) timer = setInterval(async () => { try { await load(); } catch {} }, 30000);
  } catch (e) {
    if (e.status === 401 || e.status === 403) { store(KEY, null); session = null; LOG = null; cacheClear(); return viewSignIn('Signed out — sign in again.'); }
    app.innerHTML = `<h1>COMP5423 · class report</h1><p class="err">Could not load: ${esc(e.message)}</p>`;
  }
}
boot();
