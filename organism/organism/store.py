"""The organism's own memory: three tables in plain SQL, same in SQLite, D1 and Postgres."""
import json, sqlite3, datetime, os

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS organism_runs (
        run_id TEXT PRIMARY KEY, created_at TEXT, threads INTEGER, events INTEGER, vocab INTEGER,
        bits_first REAL, bits_last REAL, retro_first REAL, retro_last REAL,
        seed_from TEXT, seed_file TEXT, strikes_applied INTEGER, params TEXT)""",
    """CREATE TABLE IF NOT EXISTS organism_links (
        run_id TEXT, rank INTEGER, thread_a TEXT, thread_b TEXT, acct_a TEXT, acct_b TEXT,
        title_a TEXT, title_b TEXT, map REAL, words REAL, hidden REAL, carried_by TEXT,
        status TEXT DEFAULT 'unstruck', msgs_a TEXT, msgs_b TEXT, PRIMARY KEY (run_id, rank))""",
    """CREATE TABLE IF NOT EXISTS organism_strikes (
        term_a TEXT, term_b TEXT, run_id TEXT, rank INTEGER, reason TEXT, created_at TEXT,
        PRIMARY KEY (term_a, term_b))""",
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Store:
    """'path.db' (SQLite) | 'd1:<uuid>' | 'postgres:DSN'. One method: q(sql, params) -> rows."""
    def __init__(self, spec):
        self.spec = spec
        kind, _, rest = spec.partition(":")
        if kind == "d1":
            from .sources import d1_query
            self.kind, self.db = "d1", rest
            self._d1 = d1_query
        elif kind in ("postgres", "postgresql"):
            import psycopg
            self.kind, self.con = "pg", psycopg.connect(spec if kind == "postgresql" else rest, autocommit=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(spec)), exist_ok=True)
            self.kind, self.con = "sqlite", sqlite3.connect(spec)
        for s in SCHEMA:
            self.q(s)

    def q(self, sql, params=()):
        if self.kind == "d1":
            import urllib.request
            acct, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
            req = urllib.request.Request(
                f"https://api.cloudflare.com/client/v4/accounts/{acct}/d1/database/{self.db}/query",
                data=json.dumps({"sql": sql, "params": list(params)}).encode(), method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                body = json.load(r)
            if not body.get("success"): raise RuntimeError(body.get("errors"))
            return [tuple(x.values()) for x in body["result"][0]["results"]]
        if self.kind == "pg":
            cur = self.con.execute(sql.replace("?", "%s"), params)
            return cur.fetchall() if cur.description else []
        cur = self.con.execute(sql, params); self.con.commit()
        return cur.fetchall()

    # ---- writing a run ----
    def save_run(self, run):
        self.q("INSERT INTO organism_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (run["run_id"], run["created_at"], run["threads"], run["events"], run["vocab"],
                run["bits_first"], run["bits_last"], run.get("retro_first"), run.get("retro_last"),
                run.get("seed_from"), run.get("seed_file"), run.get("strikes_applied", 0),
                json.dumps(run.get("params", {}), ensure_ascii=False)))

    def save_links(self, run_id, links):
        for r, l in enumerate(links, 1):
            self.q("INSERT INTO organism_links (run_id, rank, thread_a, thread_b, acct_a, acct_b, title_a, title_b,"
                   " map, words, hidden, carried_by, msgs_a, msgs_b) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (run_id, r, l["a"], l["b"], l["a_acct"], l["b_acct"], l["a_title"], l["b_title"],
                    l["map"], l["words"], l["hidden"], json.dumps(l["carried_by"], ensure_ascii=False),
                    json.dumps(l.get("a_msgs")), json.dumps(l.get("b_msgs"))))

    # ---- the filter: strikes ----
    def strike(self, run_id, rank, reason=""):
        rows = self.q("SELECT carried_by FROM organism_links WHERE run_id=? AND rank=?", (run_id, rank))
        if not rows: raise SystemExit(f"no link {rank} in run {run_id}")
        pairs = [(u, v) for u, v, _ in json.loads(rows[0][0])]
        for u, v in pairs:
            self.strike_pair(u, v, run_id, rank, reason)
        self.q("UPDATE organism_links SET status='struck' WHERE run_id=? AND rank=?", (run_id, rank))
        return pairs

    def strike_pair(self, u, v, run_id=None, rank=None, reason=""):
        u, v = sorted((u, v))
        self.q("INSERT OR REPLACE INTO organism_strikes VALUES (?,?,?,?,?,?)" if self.kind != "pg" else
               "INSERT INTO organism_strikes VALUES (?,?,?,?,?,?) ON CONFLICT (term_a, term_b) DO NOTHING",
               (u, v, run_id, rank, reason, now()))

    def keep(self, run_id, rank):
        self.q("UPDATE organism_links SET status='kept' WHERE run_id=? AND rank=?", (run_id, rank))

    def struck_pairs(self):
        return [(a, b) for a, b in self.q("SELECT term_a, term_b FROM organism_strikes")]

    def last_run(self):
        r = self.q("SELECT run_id, seed_file FROM organism_runs ORDER BY created_at DESC, run_id DESC LIMIT 1")
        return r[0] if r else None
