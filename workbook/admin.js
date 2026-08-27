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
   the same three rules the workbook applies, escaping first. */
const md = s => esc(s)
  .replace(/`([^`]+)`/g, '<code>$1</code>')
  .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');

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
    all('/rest/v1/attempts?select=student_id,question_id,answer,correct,created_at,question_version'),
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

  const roster = [...per.values()].map(s => ({
    ...s, passed: s.passed.size,
    state: s.n ? 'good' : s.last_seen ? 'warn' : 'crit',
  }));
  const rank = { crit: 0, warn: 1, good: 2 };
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

  return { roster, missed, attempts: attempts.length, today, last15 };
}

/* ── views ────────────────────────────────────────────────────────────────── */

function viewSignIn(err) {
  app.innerHTML = `<h1>COMP5423 · class report</h1>
    <p class="dim">Instructor sign-in.</p>
    <p class="reveal">
      <input id="code" type="password" placeholder="class code" autocapitalize="off"
             autocorrect="off" spellcheck="false" autocomplete="off">
      <button type="button" id="peek" aria-pressed="false" aria-label="Show the code">Show</button>
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
  // Masked by default, every time — never remembered as revealed.
  const input = document.getElementById('code');
  const peek = document.getElementById('peek');
  peek.onclick = () => {
    const shown = input.type === 'text';
    input.type = shown ? 'password' : 'text';
    peek.textContent = shown ? 'Show' : 'Hide';
    peek.setAttribute('aria-pressed', String(!shown));
    peek.setAttribute('aria-label', shown ? 'Show the code' : 'Hide the code');
    input.focus();
  };
  input.focus();
}

function tile(value, of, label) {
  return `<div class="tile"><b>${value}${of !== null ? `<span class="of"> / ${of}</span>` : ''}</b>
    <span>${esc(label)}</span></div>`;
}

const shell = (title, crumbs, body) => `
    <div class="head">
      <h1>COMP5423 · ${title}</h1>
      <span class="faint">${CFG.name} · updated ${new Date().toLocaleTimeString()}
        <button id="out" style="margin-left:.5rem">Sign out</button></span>
    </div>
    ${crumbs ? `<p class="crumb">${crumbs}</p>` : ''}
    ${body}`;

function paint(html) {
  app.innerHTML = html;
  const out = document.getElementById('out');
  if (out) out.onclick = () => { store(KEY, null); session = null; clearInterval(timer); viewSignIn(); };
}

/* The dashboard holds the numbers you want at 12:30 and nothing that takes a
   moment to draw. The two long tables live behind links, and are only fetched
   and built when you ask for them. */
function viewDashboard(d) {
  const signedIn = d.roster.filter(s => s.last_seen || s.n).length;
  const answering = d.roster.filter(s => s.n).length;
  const stuck = d.roster.filter(s => s.state === 'warn');

  paint(shell('class report', '', `
    <h2>Right now</h2>
    <div class="tiles">
      ${tile(signedIn, d.roster.length, 'have signed in')}
      ${tile(answering, d.roster.length, 'have answered something')}
      ${tile(d.last15, null, 'answers in the last 15 min')}
      ${tile(d.today, null, 'answers today')}
    </div>
    ${stuck.length ? `<p class="dim" style="margin-top:.8rem">
      <strong>${stuck.length}</strong> ${stuck.length === 1 ? 'student has' : 'students have'} signed in
      without answering anything.</p>` : ''}

    <h2>Look closer</h2>
    <div class="jump">
      <a href="students.html"><b>Students · ${d.roster.length}</b></a>
      <a href="questions.html"><b>Questions · ${d.missed.length}</b></a>
    </div>

    <p class="faint" style="margin-top:2rem">${d.attempts} attempts total · refreshes every 30 s</p>`));
}

function viewStudents(d) {
  const word = { good: 'answering', warn: 'signed in, no answers', crit: 'never signed in' };
  // With 200-odd seeded accounts, the untouched ones would bury the handful that
  // matter. They are one click away, never gone.
  const here = d.roster.filter(s => s.state !== 'crit');
  const absent = d.roster.filter(s => s.state === 'crit');
  const row = s => `<tr>
        <td>${esc(s.nickname)}</td>
        <td><span class="st ${s.state}"><i></i>${word[s.state]}</span></td>
        <td class="num">${s.passed}</td>
        <td class="num">${s.n || '—'}</td>
        <td class="num faint">${ago(s.last)}</td>
        <td class="num faint">${ago(s.last_seen)}</td></tr>`;
  // Two different facts, so two columns. "Last answer" comes from the attempts table;
  // "Last seen" is written on every sign-in and page load, and is the only signal a
  // student who has read but answered nothing leaves behind.
  const head = `<thead><tr><th>Nickname</th><th>Status</th><th class="num">Passed</th>
        <th class="num">Answers</th><th class="num">Last answer</th>
        <th class="num">Last seen</th></tr></thead>`;

  paint(shell('students', '<a href="admin.html">← Class report</a>', `
    <h2>Active · ${here.length} of ${d.roster.length}</h2>
    <div class="scroll"><table>${head}
      <tbody>${here.map(row).join('') || '<tr><td colspan="6" class="dim">Nobody yet.</td></tr>'}</tbody>
    </table></div>
    ${absent.length ? `<details class="fold">
      <summary>${absent.length} ${absent.length === 1 ? 'account has' : 'accounts have'} never signed in</summary>
      <div class="scroll"><table>${head}<tbody>${absent.map(row).join('')}</tbody></table></div>
    </details>` : ''}`));
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
    <p class="qm-q">${md(q.q)}</p>
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
  const list = d.missed.filter(m => !classFilter || m.q.class === classFilter);
  const answered = list.filter(m => m.n > 0);

  paint(shell('questions', '<a href="admin.html">← Class report</a>', `
    <div class="pills">
      <button data-c="" aria-pressed="${!classFilter}">All</button>
      ${classes.map(c => `<button data-c="${esc(c)}" aria-pressed="${classFilter === c}">${esc(c)}</button>`).join('')}
    </div>
    <h2>${list.length} questions · most wrong answers first</h2>
    <div class="scroll"><table>
      <thead><tr><th>Question</th><th class="num">Wrong</th><th class="num">n</th>
        <th style="width:8rem">Rate</th></tr></thead>
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
  app.querySelectorAll('[data-i]').forEach(r => {
    const show = () => showQuestion(list[+r.dataset.i]);
    r.onclick = show;
    r.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } };
  });
}

const render = d => ({ students: viewStudents, questions: viewQuestions })[window.PAGE]
  ? ({ students: viewStudents, questions: viewQuestions })[window.PAGE](d)
  : viewDashboard(d);

/* ── boot ─────────────────────────────────────────────────────────────────── */

async function boot() {
  if (!session) return viewSignIn();
  try {
    if (!BANK) BANK = await (await fetch('data/questions.json', { cache: 'no-cache' })).json();
    render(await report());
    clearInterval(timer);
    timer = setInterval(async () => {
      if (document.getElementById('qm')?.open) return;   // not while a question is on screen
      try { render(await report()); } catch {}
    }, 30000);
  } catch (e) {
    if (e.status === 401 || e.status === 403) { store(KEY, null); session = null; return viewSignIn('Signed out — sign in again.'); }
    app.innerHTML = `<h1>COMP5423 · class report</h1><p class="err">Could not load: ${esc(e.message)}</p>`;
  }
}
boot();
