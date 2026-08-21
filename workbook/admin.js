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

/* PostgREST caps a response; walk it in pages so a busy Monday cannot silently
   truncate the report. Replaced by an aggregate view when the leaderboard lands. */
async function all(path) {
  const out = [];
  for (let from = 0; ; from += 1000) {
    const { body } = await api(path, { headers: { Range: `${from}-${from + 999}` } });
    out.push(...body);
    if (body.length < 1000) return out;
  }
}

/* ── data ─────────────────────────────────────────────────────────────────── */

async function report() {
  const [students, attempts] = await Promise.all([
    all('/rest/v1/students?select=id,nickname,last_seen&order=nickname'),
    all('/rest/v1/attempts?select=student_id,question_id,answer,correct,created_at'),
  ]);
  const byId = new Map(BANK.questions.map(q => [q.id, q]));
  const now = Date.now();

  const per = new Map(students.map(s => [s.id, { ...s, n: 0, passed: new Set(), last: null }]));
  const perQ = new Map();
  let last15 = 0, today = 0;
  const midnight = new Date(); midnight.setHours(0, 0, 0, 0);

  for (const a of attempts) {
    const t = new Date(a.created_at);
    if (now - t < 15 * 60000) last15++;
    if (t >= midnight) today++;
    const s = per.get(a.student_id);
    if (s) { s.n++; if (a.correct) s.passed.add(a.question_id);
             if (!s.last || t > new Date(s.last)) s.last = a.created_at; }
    const q = perQ.get(a.question_id) || { n: 0, wrong: 0, picks: {} };
    q.n++; if (!a.correct) { q.wrong++; q.picks[a.answer] = (q.picks[a.answer] || 0) + 1; }
    perQ.set(a.question_id, q);
  }

  const roster = [...per.values()].map(s => ({
    ...s, passed: s.passed.size,
    state: s.n ? 'good' : s.last_seen ? 'warn' : 'crit',
  }));
  const rank = { crit: 0, warn: 1, good: 2 };
  roster.sort((a, b) => rank[a.state] - rank[b.state] || b.passed - a.passed || a.nickname.localeCompare(b.nickname));

  const missed = [...perQ.entries()]
    .filter(([id, q]) => q.n >= 3 && byId.has(id))
    .map(([id, q]) => ({ q: byId.get(id), n: q.n, wrongPct: pct(q.wrong, q.n),
      top: Object.entries(q.picks).sort((a, b) => b[1] - a[1])[0] }))
    .sort((a, b) => b.wrongPct - a.wrongPct).slice(0, 10);

  const topics = new Map();
  for (const [id, q] of perQ) {
    const t = byId.get(id)?.topic; if (!t) continue;
    const e = topics.get(t) || { n: 0, ok: 0 };
    e.n += q.n; e.ok += q.n - q.wrong; topics.set(t, e);
  }

  return { roster, missed, attempts: attempts.length, today, last15,
           topics: [...topics].map(([t, e]) => ({ t, pct: pct(e.ok, e.n), n: e.n }))
                              .sort((a, b) => a.pct - b.pct) };
}

/* ── views ────────────────────────────────────────────────────────────────── */

function viewSignIn(err) {
  app.innerHTML = `<h1>COMP5423 · class report</h1>
    <p class="dim">Instructor sign-in.</p>
    <p><input id="code" placeholder="class code" autocapitalize="off" autocorrect="off" spellcheck="false"></p>
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
}

function tile(value, of, label) {
  return `<div class="tile"><b>${value}${of !== null ? `<span class="of"> / ${of}</span>` : ''}</b>
    <span>${esc(label)}</span></div>`;
}

function render(d) {
  const signedIn = d.roster.filter(s => s.last_seen || s.n).length;
  const answering = d.roster.filter(s => s.n).length;
  const stuck = d.roster.filter(s => s.state !== 'good');

  const word = { good: 'answering', warn: 'signed in, no answers', crit: 'never signed in' };

  app.innerHTML = `
    <div class="head">
      <h1>COMP5423 · class report</h1>
      <span class="faint">${CFG.name} · updated ${new Date().toLocaleTimeString()}
        <button id="out" style="margin-left:.5rem">Sign out</button></span>
    </div>

    <h2>Right now</h2>
    <div class="tiles">
      ${tile(signedIn, d.roster.length, 'have signed in')}
      ${tile(answering, d.roster.length, 'have answered something')}
      ${tile(d.last15, null, 'answers in the last 15 min')}
      ${tile(d.today, null, 'answers today')}
    </div>
    ${stuck.length ? `<p class="dim" style="margin-top:.8rem">
      <strong>${stuck.length}</strong> ${stuck.length === 1 ? 'student needs' : 'students need'} help getting started —
      listed first below.</p>` : ''}

    <h2>Students · ${d.roster.length}</h2>
    <div class="scroll"><table>
      <thead><tr><th>Nickname</th><th>Status</th><th class="num">Passed</th>
        <th class="num">Answers</th><th class="num">Last active</th></tr></thead>
      <tbody>${d.roster.map(s => `<tr>
        <td>${esc(s.nickname)}</td>
        <td><span class="st ${s.state}"><i></i>${word[s.state]}</span></td>
        <td class="num">${s.passed}</td>
        <td class="num">${s.n || '—'}</td>
        <td class="num faint">${ago(s.last || s.last_seen)}</td></tr>`).join('')}</tbody>
    </table></div>

    <h2>Most missed</h2>
    ${d.missed.length ? `<div class="scroll"><table>
      <thead><tr><th>Question</th><th class="num">n</th><th style="width:9rem">Wrong</th>
        <th>Most-picked wrong</th></tr></thead>
      <tbody>${d.missed.map(m => `<tr>
        <td class="miss">${esc(m.q.title)}<br><span class="tag">${esc(m.q.class)} · ${esc(m.q.topic)}</span></td>
        <td class="num">${m.n}</td>
        <td><div class="bar"><i style="width:${m.wrongPct}%"></i></div>
            <span class="faint" style="font-size:.78rem">${m.wrongPct}%</span></td>
        <td>${m.top ? `<strong>${esc(m.top[0])}</strong> ${esc(
              m.q.options?.[m.top[0]] ? m.q.options[m.top[0]].slice(0, 60) : '')}` : '—'}</td>
      </tr>`).join('')}</tbody></table></div>`
      : '<p class="dim">Not enough answers yet.</p>'}

    <h2>By topic · weakest first</h2>
    ${d.topics.length ? `<div class="scroll"><table>
      <thead><tr><th>Topic</th><th style="width:9rem">Pass rate</th><th class="num">Answers</th></tr></thead>
      <tbody>${d.topics.map(t => `<tr><td>${esc(t.t)}</td>
        <td><div class="bar"><i style="width:${t.pct}%"></i></div>
            <span class="faint" style="font-size:.78rem">${t.pct}%</span></td>
        <td class="num">${t.n}</td></tr>`).join('')}</tbody></table></div>`
      : '<p class="dim">Nothing yet.</p>'}

    <p class="faint" style="margin-top:2rem">${d.attempts} attempts total · refreshes every 30 s</p>`;

  document.getElementById('out').onclick = () => { store(KEY, null); session = null; clearInterval(timer); viewSignIn(); };
}

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function boot() {
  if (!session) return viewSignIn();
  try {
    if (!BANK) BANK = await (await fetch('data/questions.json', { cache: 'no-cache' })).json();
    render(await report());
    clearInterval(timer);
    timer = setInterval(async () => { try { render(await report()); } catch {} }, 30000);
  } catch (e) {
    if (e.status === 401 || e.status === 403) { store(KEY, null); session = null; return viewSignIn('Signed out — sign in again.'); }
    app.innerHTML = `<h1>COMP5423 · class report</h1><p class="err">Could not load: ${esc(e.message)}</p>`;
  }
}
boot();
