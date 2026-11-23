// ei-organism on Cloudflare.
//
// Every hour the cron asks the Organism Durable Object whether to live again. It does when
// organism_control has key 'request' (set it to ask for a run), or once a day when the database has
// grown since the last run. A life is a small state machine driven by the Durable Object's alarm:
//   ingest  -> read D1 in pages (messages, commits, published writing) and post them to the container
//   seed    -> hand the container the last seed from R2 (none on the first life: a newborn)
//   run     -> start the run with the struck term pairs; poll until done
//   collect -> write the run and its links to D1, the new seed to R2
// No route serves data: the Worker answers /health only. Links are read from D1.

import { DurableObject } from "cloudflare:workers";

const PORT = 8080;
const PAGE = 2000;
const PAGES_PER_ALARM = 15;
const SEED_LATEST = "seed/latest.npz";
const RUN_PARAMS = { gap: 6, parts: 20, top: 300, vocab: 4000, min_df: 3, max_df: 0.3, min_msgs: 6 };

type Phase = "idle" | "ingest" | "seed" | "run" | "collect";
interface Job { phase: Phase; source: number; cursor: number; started: string; maxRow: number; reason: string; error?: string; runStarted?: number }

const SOURCES: { threads?: string; events: string }[] = [
  { // chats, mail, WhatsApp, Telegram: one thread per conversation, account = source (+ AI account label)
    threads: `SELECT c.id, COALESCE(c.title, ''), s.name || COALESCE(':' || a.label, ''), COALESCE(c.last_message_at, '')
              FROM conversations c LEFT JOIN sources s ON s.id = c.source_id LEFT JOIN ai_accounts a ON a.id = c.ai_account_id`,
    events: `SELECT m.rowid, m.conversation_id, COALESCE(m.sent_at_iso, m.sent_at, ''), COALESCE(m.role, ''),
             substr(COALESCE(m.subject || char(10), '') || COALESCE(m.body_text, ''), 1, 4000)
             FROM messages m WHERE m.rowid > ?1 ORDER BY m.rowid LIMIT ?2`,
  },
  { // each repository is one thread of commits
    threads: `SELECT 'gh:' || id, repo_full_name, 'github', COALESCE(created_at, '') FROM github_repos`,
    events: `SELECT rowid, 'gh:' || repo_id, COALESCE(committed_at, ''), 'commit', substr(COALESCE(message, ''), 1, 4000)
             FROM github_commits WHERE rowid > ?1 ORDER BY rowid LIMIT ?2`,
  },
  { // published writing: one thread per piece
    threads: `SELECT 'pub:' || id, COALESCE(title, ''), COALESCE(platform, 'published'), COALESCE(published_at, '') FROM authored_content`,
    events: `SELECT rowid, 'pub:' || id, COALESCE(published_at, ''), 'author', substr(COALESCE(full_text, ''), 1, 4000)
             FROM authored_content WHERE rowid > ?1 ORDER BY rowid LIMIT ?2`,
  },
];

const now = () => new Date().toISOString();

export class Organism extends DurableObject<Env> {
  private async job(): Promise<Job> {
    return (await this.ctx.storage.get<Job>("job")) ?? { phase: "idle", source: 0, cursor: 0, started: "", maxRow: 0, reason: "" };
  }
  private async save(j: Job) { await this.ctx.storage.put("job", j); }

  private async container(path: string, init?: RequestInit): Promise<Response> {
    const c = this.ctx.container!;
    if (!c.running) {
      c.start({ image: c.images.base, instance: "standard-3", enableInternet: false });
      await c.setInactivityTimeout(20 * 60 * 1000);
      for (let i = 0; i < 150; i++) {
        try {
          const r = await c.getTcpPort(PORT).fetch("http://container/health", { signal: AbortSignal.timeout(1000) });
          await r.body?.cancel();
          if (r.ok) break;
        } catch { /* not listening yet */ }
        await scheduler.wait(400);
      }
    }
    const r = await c.getTcpPort(PORT).fetch(`http://container${path}`, init);
    if (!r.ok && r.status !== 202) throw new Error(`container ${path}: ${r.status} ${await r.text()}`);
    return r;
  }

  private async post(path: string, body: unknown) {
    return this.container(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  }

  // called by the cron: start a life if one is due
  async maybeStart(): Promise<string> {
    const j = await this.job();
    if (j.phase !== "idle") return `busy: ${j.phase}`;
    const db = this.env.DB;
    for (const s of (await schemaStatements())) await db.prepare(s).run();
    const req = await db.prepare("SELECT value FROM organism_control WHERE key = 'request'").first<{ value: string }>();
    const maxRow = (await db.prepare("SELECT MAX(rowid) AS m FROM messages").first<{ m: number }>())?.m ?? 0;
    const last = await db.prepare("SELECT run_id, created_at, params FROM organism_runs ORDER BY run_id DESC LIMIT 1")
      .first<{ run_id: string; created_at: string; params: string }>();
    const lastMax = last ? (JSON.parse(last.params || "{}").max_row ?? 0) : -1;
    const dayOld = !last || Date.now() - Date.parse(last.created_at) > 23 * 3600 * 1000;
    const err = await db.prepare("SELECT updated_at FROM organism_control WHERE key = 'last_error'").first<{ updated_at: string }>();
    const failedToday = !!err && Date.now() - Date.parse(err.updated_at) < 23 * 3600 * 1000;   // don't retry a failure every hour
    const reason = req ? `requested ${req.value}` : (dayOld && !failedToday && maxRow !== lastMax ? "database grew" : "");
    if (!reason) return "nothing to do";
    if (req) await db.prepare("DELETE FROM organism_control WHERE key = 'request'").run();
    await this.save({ phase: "ingest", source: 0, cursor: 0, started: now(), maxRow, reason });
    await this.post("/reset", {});
    await this.ctx.storage.setAlarm(Date.now() + 1000);
    return `started: ${reason}`;
  }

  async alarm() {
    const j = await this.job();
    try {
      if (j.phase === "ingest") await this.ingest(j);
      else if (j.phase === "seed") await this.seed(j);
      else if (j.phase === "run") await this.poll(j);
      else if (j.phase === "collect") await this.collect(j);
    } catch (e) {
      j.error = String(e); j.phase = "idle"; await this.save(j);
      await this.env.DB.prepare("INSERT OR REPLACE INTO organism_control VALUES ('last_error', ?1, ?2)").bind(j.error, now()).run();
    }
  }

  private async ingest(j: Job) {
    const db = this.env.DB;
    for (let p = 0; p < PAGES_PER_ALARM; p++) {
      const src = SOURCES[j.source];
      if (!src) { j.phase = "seed"; break; }
      let threads: unknown[] = [];
      if (j.cursor === 0 && src.threads) threads = (await db.prepare(src.threads).raw()) as unknown[];
      const rows = (await db.prepare(src.events).bind(j.cursor, PAGE).raw()) as [number, string, string, string, string][];
      await this.post("/ingest", { threads, events: rows.map(r => [r[1], r[2], r[3], r[4]]) });
      if (rows.length < PAGE) { j.source++; j.cursor = 0; }
      else j.cursor = rows[rows.length - 1][0];
    }
    await this.save(j);
    await this.ctx.storage.setAlarm(Date.now() + 500);
  }

  private async seed(j: Job) {
    const obj = await this.env.SEEDS.get(SEED_LATEST);
    if (obj) await this.container("/seed", { method: "PUT", body: obj.body, headers: { "content-length": String(obj.size) } });
    const strikes = (await this.env.DB.prepare("SELECT term_a, term_b FROM organism_strikes").raw()) as [string, string][];
    await this.post("/run", { ...RUN_PARAMS, strikes });
    j.phase = "run"; j.runStarted = Date.now(); await this.save(j);
    await this.ctx.storage.setAlarm(Date.now() + 30_000);
  }

  private async poll(j: Job) {
    const s = await (await this.container("/status")).json<{ state: string; error: string | null; log: string[]; events: number }>();
    console.log(`organism ${s.state}: ${s.log?.[s.log.length - 1] ?? ""}`);
    if (s.state === "error") throw new Error(`run failed: ${s.error}`);
    if (s.state === "idle") throw new Error(`container restarted and lost the run (events held: ${s.events}); last log: ${s.log?.slice(-3).join(" | ")}`);
    if (Date.now() - (j.runStarted ?? Date.now()) > 90 * 60 * 1000) throw new Error(`run took over 90 minutes; last log: ${s.log?.slice(-3).join(" | ")}`);
    if (s.state === "done") { j.phase = "collect"; await this.save(j); }
    await this.ctx.storage.setAlarm(Date.now() + (s.state === "done" ? 500 : 30_000));
  }

  private async collect(j: Job) {
    type Link = { a: string; b: string; a_acct: string; b_acct: string; a_title: string; b_title: string; map: number;
                  words: number; hidden: number; carried_by: unknown; a_msgs: number[]; b_msgs: number[] };
    const res = await (await this.container("/result")).json<{ run: Record<string, any>; links: Link[] }>();
    const run = res.run, id = run.run_id as string;
    // the new seed: this life's corrected weights, kept as latest and under the run id
    const seedRes = await this.container("/seed");
    const size = Number(seedRes.headers.get("content-length"));
    const bytes = await seedRes.arrayBuffer();
    await this.env.SEEDS.put(`seed/${id}.npz`, bytes);
    await this.env.SEEDS.put(SEED_LATEST, bytes);
    const db = this.env.DB;
    await db.prepare("INSERT INTO organism_runs VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)").bind(
      id, now(), run.threads, run.events, run.vocab, run.bits_first, run.bits_last, run.retro_first, run.retro_last,
      run.seeded ? SEED_LATEST : null, `seed/${id}.npz`, run.strikes_applied,
      JSON.stringify({ ...run.params, max_row: j.maxRow, reason: j.reason, seed_kept: run.seed_kept, seed_bytes: size || bytes.byteLength })).run();
    const ins = db.prepare(`INSERT INTO organism_links (run_id, rank, thread_a, thread_b, acct_a, acct_b, title_a, title_b,
                            map, words, hidden, carried_by, msgs_a, msgs_b) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)`);
    const stmts = res.links.map((l, i) => ins.bind(id, i + 1, l.a, l.b, l.a_acct, l.b_acct, l.a_title, l.b_title, l.map, l.words,
                                                  l.hidden, JSON.stringify(l.carried_by), JSON.stringify(l.a_msgs), JSON.stringify(l.b_msgs)));
    for (let i = 0; i < stmts.length; i += 50) await db.batch(stmts.slice(i, i + 50));
    await db.prepare("INSERT OR REPLACE INTO organism_control VALUES ('last_run', ?1, ?2)").bind(id, now()).run();
    await this.save({ phase: "idle", source: 0, cursor: 0, started: "", maxRow: 0, reason: "" });
    await this.ctx.container?.destroy();
  }
}

async function schemaStatements(): Promise<string[]> {
  return [
    `CREATE TABLE IF NOT EXISTS organism_runs (run_id TEXT PRIMARY KEY, created_at TEXT, threads INTEGER, events INTEGER, vocab INTEGER, bits_first REAL, bits_last REAL, retro_first REAL, retro_last REAL, seed_from TEXT, seed_file TEXT, strikes_applied INTEGER, params TEXT)`,
    `CREATE TABLE IF NOT EXISTS organism_links (run_id TEXT, rank INTEGER, thread_a TEXT, thread_b TEXT, acct_a TEXT, acct_b TEXT, title_a TEXT, title_b TEXT, map REAL, words REAL, hidden REAL, carried_by TEXT, status TEXT DEFAULT 'unstruck', msgs_a TEXT, msgs_b TEXT, PRIMARY KEY (run_id, rank))`,
    `CREATE TABLE IF NOT EXISTS organism_strikes (term_a TEXT, term_b TEXT, run_id TEXT, rank INTEGER, reason TEXT, created_at TEXT, PRIMARY KEY (term_a, term_b))`,
    `CREATE TABLE IF NOT EXISTS organism_control (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)`,
  ];
}

export default {
  async fetch(req: Request): Promise<Response> {
    return new URL(req.url).pathname === "/health" ? new Response("ok") : new Response("not found", { status: 404 });
  },
  async scheduled(_c: ScheduledController, env: Env, ctx: ExecutionContext) {
    ctx.waitUntil(env.ORGANISM.getByName("organism").maybeStart().then(r => console.log(r)));
  },
} satisfies ExportedHandler<Env>;
