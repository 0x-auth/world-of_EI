"""Medium. Every source shows the same shape:
    events(thread_id, time, role, text)          required
    threads(thread_id, title, account, updated)  optional
A source is any callable returning (events, meta). Add a database by writing one function."""
import os, json, zipfile, sqlite3, urllib.request


def claude_export(path):
    """A folder written by `organism clean` (or a raw export folder/zip). Account = folder name."""
    def load(name):
        if os.path.isdir(path):
            p = os.path.join(path, name)
            if os.path.exists(p): return json.load(open(p, encoding="utf-8"))
            zs = sorted(f for f in os.listdir(path) if f.endswith(".zip"))
            if not zs: return None
            zpath = os.path.join(path, zs[0])
        else:
            zpath = path
        with zipfile.ZipFile(zpath) as z:
            hit = [n for n in z.namelist() if n.endswith(name)]
            return json.loads(z.read(hit[0]).decode("utf-8")) if hit else None
    acct = os.path.basename(os.path.normpath(path)).replace(".zip", "")
    events, meta = [], {}
    for c in load("conversations.json") or []:
        tid = c.get("uuid")
        meta[tid] = {"title": c.get("name") or "Untitled", "account": c.get("account") or acct,
                     "updated": c.get("updated_at") or ""}
        for m in c.get("chat_messages", []):
            text = m.get("text") or ""
            if not text and isinstance(m.get("content"), list):   # raw export: text blocks only
                text = "\n".join(b.get("text") or "" for b in m["content"] if b.get("type") == "text")
            events.append((tid, m.get("created_at", ""), m.get("sender") or m.get("role"), text))
    return events, meta


EVENTS_SQL = "SELECT thread_id, time, role, text FROM events ORDER BY thread_id, time"
THREADS_SQL = "SELECT thread_id, title, account, updated FROM threads"


def _split(rows_e, rows_t):
    meta = {r[0]: {"title": r[1], "account": r[2], "updated": r[3]} for r in rows_t}
    return [tuple(r) for r in rows_e], meta


def sqlite_db(path, events_sql=EVENTS_SQL, threads_sql=THREADS_SQL):
    con = sqlite3.connect(path)
    try:
        e = con.execute(events_sql).fetchall()
        try: t = con.execute(threads_sql).fetchall()
        except sqlite3.OperationalError: t = []
    finally:
        con.close()
    return _split(e, t)


def postgres_db(dsn, events_sql=EVENTS_SQL, threads_sql=THREADS_SQL):
    import psycopg   # pip install "psycopg[binary]"
    with psycopg.connect(dsn) as con:
        e = con.execute(events_sql).fetchall()
        try: t = con.execute(threads_sql).fetchall()
        except Exception: con.rollback(); t = []
    return _split(e, t)


def d1_query(database_id, sql, account=None, token=None):
    """Cloudflare D1 over its HTTP API. Credentials come from the environment, never the URL:
    CF_ACCOUNT_ID, CF_API_TOKEN."""
    account = account or os.environ["CF_ACCOUNT_ID"]; token = token or os.environ["CF_API_TOKEN"]
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{database_id}/query",
        data=json.dumps({"sql": sql}).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        body = json.load(r)
    if not body.get("success"):
        raise RuntimeError(body.get("errors"))
    return body["result"][0]["results"]


def d1_db(database_id, events_sql=EVENTS_SQL, threads_sql=THREADS_SQL):
    e = d1_query(database_id, events_sql)
    try: t = d1_query(database_id, threads_sql)
    except Exception: t = []
    return ([(r["thread_id"], r["time"], r["role"], r["text"]) for r in e],
            {r["thread_id"]: {"title": r["title"], "account": r["account"], "updated": r["updated"]} for r in t})


def open_source(spec):
    """'path/to/folder' | 'sqlite:file.db' | 'postgres:DSN' | 'd1:<database-uuid>'"""
    kind, _, rest = spec.partition(":")
    if kind == "sqlite": return sqlite_db(rest)
    if kind in ("postgres", "postgresql"): return postgres_db(spec if kind == "postgresql" else rest)
    if kind == "d1": return d1_db(rest)
    return claude_export(os.path.expanduser(spec))


def merge(sources):
    events, meta = [], {}
    for e, m in sources:
        events += e; meta.update(m)
    return events, meta
