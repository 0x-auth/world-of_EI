#!/usr/bin/env python3
"""EI organism v1.4 — one organism over all your Claude exports, finds hidden links between chats.

v1.4 (7 Nov): the v1.3 run linked everything through talk-to-Claude words (claude, tell, two, real) and two
  short hub chats. Fixed: filler words stopped; terms found in over 30% of chats left out (--max-df); rare terms
  weigh more (idf); each pair is scored against its two chats' average links, so hubs drop out; chats under
  6 messages are not ranked (--min-msgs).
v1.3 (7 Nov): reads the output of clean_exports.py; equations kept as whole tokens (t+1/t, -1/t, Δx, xy=-1);
  --exclude / --since / --focus; plus every v1.2 fix (reproducible vocab order, stable sort, float32 link map,
  first/last-quarter surprise, retro surprise saved in organism_runs.params).

What changed from v0 (from the 3-account toy test):
  * the residue context RESETS at every conversation boundary, so the organism never learns
    "what I asked next in some other chat" as if it were meaning
  * one pair for all accounts (1, 2 and 3 organisms found the same chain once boundaries reset)
  * the pair: a forward organism (cause -> effect) and a retrocausal one living each chat backward
    (effect -> cause). A link is kept only where both hold it. --single runs the forward one alone.

Usage (Mac):
  pip3 install numpy
  python3 ~/Downloads/clean_exports.py ~/Downloads/06-11-2026-claude-exports ~/Downloads/clean
  python3 ~/Downloads/organism_v1.py ~/Downloads/clean/localhost ~/Downloads/clean/phiinverse ~/Downloads/clean/merge \
          --since 2026-08-01 --focus '' --out ~/Downloads/organism_out
  # one folder per account (folder name = account). Learns on every chat since --since; ranks links among --focus.
  # Add --human-only to learn from your words only.

Outputs (in ./organism_out/):
  links.json       ranked chat pairs: high map connection, low word overlap = hidden link
  links.md         the same, readable, with the term pairs that carry each link
  map_top.txt      strongest learned term links
  organism.sql     INSERTs for D1 (tables organism_runs, organism_links). The organism writes back.
                   apply with: npx wrangler d1 execute digital-history --remote --file organism_out/organism.sql
"""
import sys, os, re, json, zipfile, hashlib, argparse, datetime, itertools
import numpy as np

STOP = set("""a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing don down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
on once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves also like one get got let lets yeah okay ok yes right
know think mean really actually basically something thing things want need use using used make made way see say said
im i'm it's that's don't can't you're let's i've i'll""".split()) | set("""claude mr hey hehe haha bro dude man yaar bhai
pls please plz thanks thank sorry wait cos coz sud shud wud cud ur tell told ask asked still sure fine good great nice
cool awesome done okay alright hmm umm uh um gonna wanna kinda lol yep nope""".split())
WORD = re.compile(r"[a-z][a-z0-9_'\-]{2,}|[φπΔ∆∞ψΩτθ√]")
# expressions kept whole so equations can carry links: Δx, t+1/t, t-i/t, -1/t, i/t, xy=-1, √5, 4/3
MATH = re.compile(r"[Δ∆]\s?[a-zA-Zτ]|√\s?\d+|\b[a-z]\s*[+\-−]\s*i?\s*1?\s*/\s*[a-z]\b|[\-−]\s*1\s*/\s*[a-z]\b"
                  r"|\bi\s*/\s*[a-z]\b|\b[a-z]{2}\s*=\s*[\-−]?\s*1\b|\b\d{1,2}\s*/\s*\d{1,2}\b")
MAXCH, MAXART = 4000, 8000
ARTIFACT_TOOLS = {"create_file": ("path", "file_text"), "Write": ("file_path", "content"),
                  "artifacts": ("title", "content")}


# ---------------- load exports ----------------
def read_export(path):
    """Return (account, conversations). A folder (account = folder name) or a zip."""
    def load(name):
        if os.path.isdir(path):
            p = os.path.join(path, name)
            if os.path.exists(p): return json.load(open(p, encoding="utf-8"))
            zs = [f for f in os.listdir(path) if f.endswith(".zip")]
            if not zs: return None
            zpath = os.path.join(path, zs[0])
        else:
            zpath = path
        with zipfile.ZipFile(zpath) as z:
            hit = [n for n in z.namelist() if n.endswith(name)]
            return json.loads(z.read(hit[0]).decode("utf-8")) if hit else None
    users = load("users.json") or []
    email = (users[0].get("email_address") or users[0].get("email")) if users else None
    name = os.path.basename(os.path.normpath(path)).replace(".zip", "")
    return (email or name), (load("conversations.json") or [])


def active_branch(msgs):
    """The export keeps every edit and retry. Follow parent_message_uuid back from the latest message."""
    if not msgs or "parent_message_uuid" not in msgs[0]:
        return sorted(msgs, key=lambda m: m.get("created_at", ""))
    by = {m["uuid"]: m for m in msgs}
    cur, path = max(msgs, key=lambda m: m.get("created_at", "")), []
    while cur is not None and len(path) <= len(msgs):
        path.append(cur); cur = by.get(cur.get("parent_message_uuid"))
    return path[::-1]


def strip_markup(t):
    t = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", t)
    return re.sub(r"<[^>]+>", " ", t)


def msg_parts(m):
    """(text, [(artifact_name, artifact_text)]) — your words, Claude's prose, the files Claude wrote.
    Skipped: thinking, tool results (search pages, bash output), token_budget, flag, injected prompts."""
    role = m.get("sender") or m.get("role")
    blocks = m.get("content") if isinstance(m.get("content"), list) else []
    text, arts = [], []
    for b in blocks:
        t = b.get("type")
        if t == "text" or (t == "voice_note" and role == "human"):
            text.append(b.get("text") or "")
        elif t == "tool_use" and b.get("name") in ARTIFACT_TOOLS:
            k_name, k_body = ARTIFACT_TOOLS[b["name"]]
            inp = b.get("input") or {}
            if inp.get(k_body):
                arts.append((str(inp.get(k_name) or inp.get("id") or "artifact"), strip_markup(inp[k_body])))
    if not blocks and m.get("text"):
        text.append(m["text"])
    for a in m.get("attachments") or []:
        if role == "human" and a.get("extracted_content"):
            text.append(a["extracted_content"][:MAXCH])
    return "\n".join(text), arts


def build_chats(paths, human_only, since=None, exclude=()):
    chats, seen = [], set()
    for p in paths:
        acct, convs = read_export(p)
        acct = acct.split("@")[0]
        n0, skipped = len(chats), 0
        for c in convs:
            title = c.get("name") or "Untitled"
            if any(x.lower() in title.lower() for x in exclude) or (since and (c.get("updated_at") or "") < since):
                skipped += 1; continue
            msgs, files = [], {}
            for m in active_branch(c.get("chat_messages", [])):
                role = m.get("sender") or m.get("role")
                text, arts = msg_parts(m)
                for name, body in arts:
                    files[name] = (m.get("created_at", ""), body[:MAXART])   # last version of each file wins
                if human_only and role not in ("human", "user"):
                    continue
                t = text[:MAXCH]
                if not t.strip(): continue
                h = hashlib.sha1(t.encode()).hexdigest()
                if len(t) >= 40 and h in seen: continue      # same text pasted into another chat/account
                seen.add(h)
                msgs.append((m.get("created_at", ""), role, t))
            if not human_only:
                msgs += [(ts, "artifact", f"{name}\n{body}") for name, (ts, body) in files.items()]
                msgs.sort(key=lambda x: x[0])
            if msgs:
                chats.append({"id": c.get("uuid"), "acct": acct, "title": title, "start": msgs[0][0],
                              "updated": c.get("updated_at") or msgs[-1][0], "msgs": msgs,
                              "n_artifacts": len(files)})
        print(f"  {acct}: {len(chats) - n0} chats with text ({skipped} skipped by --since/--exclude)")
    chats.sort(key=lambda c: c["start"])
    return chats


def tokens(text):
    math = {re.sub(r"\s+", "", x).replace("−", "-").replace("∆", "Δ") for x in MATH.findall(text)}
    return {w for w in WORD.findall(text.lower()) if w not in STOP} | math


# ---------------- the organism ----------------
class Organism:
    """softmax(b + W^T c). Born with W = seed*I ('what I just said, I'll say again').
    Learns only from its own prediction error: dW = eta * c (y - p)^T."""
    def __init__(self, V, eta=1.0, seed=2.0):
        self.W = np.zeros((V, V), np.float32); np.fill_diagonal(self.W, seed)
        self.b = np.zeros(V, np.float32); self.eta = eta

    def step(self, ctx, ids):
        nz = np.flatnonzero(ctx); cv = ctx[nz]
        z = self.b + cv @ self.W[nz]; z -= z.max(); p = np.exp(z); p /= p.sum()
        bits = float(-np.log2(np.maximum(p[ids], 1e-12)).mean())
        y = np.zeros_like(p); y[ids] = 1.0 / len(ids)
        g = (self.eta * (y - p)).astype(np.float32)
        self.W[nz] += np.outer(cv, g); self.b += g
        return bits


def live(chats, vocab, alpha, reverse=False):
    """reverse=True: the retrocausal organism lives each chat from its last message to its first."""
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    org = Organism(V)
    bits, n = [], 0
    for c in chats:
        ctx = np.zeros(V, np.float32)                     # birth of a new conversation: empty context
        for _, _, t in (c["msgs"][::-1] if reverse else c["msgs"]):
            ids = np.array(sorted(idx[w] for w in tokens(t) if w in idx), dtype=np.int64)
            if len(ids) == 0:
                continue
            if ctx.any():
                bits.append(org.step(ctx, ids)); n += 1
            ctx *= alpha; ctx[ids] += (1 - alpha) / len(ids); ctx /= ctx.sum(); ctx[ctx < 1e-3] = 0
    k = min(2000, max(1, n // 4))                          # first and last quarter when there are few predictions
    return org, n, (float(np.mean(bits[:k])) if bits else 0.0), (float(np.mean(bits[-k:])) if bits else 0.0)


# ---------------- reading the map: hidden links between chats ----------------
def link_matrix(Wf, Wb=None):
    """Forward u->v alone, or (two organisms) only what cause->effect AND effect->cause both hold:
    sqrt(forward[u,v] * backward[v,u])."""
    off = lambda W: np.clip(W - np.diag(np.diag(W)), 0, None)       # float32: half the memory
    F = off(Wf)
    L = F if Wb is None else np.sqrt(F * off(Wb).T)
    return (L + L.T) / 2                                    # a link either way connects two chats


def chat_links(chats, vocab, L, top=60, focus=None, min_msgs=6):
    """focus: set of chat indices; only pairs where both chats are in it are ranked (learning still uses all)."""
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    X = np.zeros((len(chats), V))
    for k, c in enumerate(chats):
        for _, _, t in c["msgs"]:
            for w in tokens(t):
                if w in idx: X[k, idx[w]] = 1
    idf = np.log((1 + len(chats)) / (1 + X.sum(0))) + 1     # rare terms count more than common ones
    Xw = X * idf
    Xn = Xw / np.maximum(Xw.sum(1, keepdims=True), 1e-9)
    surface = Xw / np.maximum(np.linalg.norm(Xw, axis=1, keepdims=True), 1e-9)
    surface = surface @ surface.T                           # word overlap (cosine, idf-weighted)
    mapped = Xn @ L @ Xn.T                                  # connection through the learned map
    np.fill_diagonal(mapped, 0)
    # hub correction: a chat that is mildly linked to everything is not linked to anything in particular.
    # Score each pair against what the two chats' average links would predict (lift).
    row = mapped.sum(1) / max(len(chats) - 1, 1)
    mapped = mapped / np.maximum(np.outer(row, row) / max(row.mean(), 1e-12), 1e-12)
    np.fill_diagonal(mapped, 0)
    # rank map strength within its own distribution, then subtract surface rank: hidden = strong map, weak words
    iu = np.triu_indices(len(chats), 1)
    big = np.array([len(c["msgs"]) >= min_msgs for c in chats])
    keep = big[iu[0]] & big[iu[1]]
    iu = (iu[0][keep], iu[1][keep])
    if focus is not None:
        keep = np.array([a in focus and b in focus for a, b in zip(*iu)], dtype=bool)
        iu = (iu[0][keep], iu[1][keep])
    m, s = mapped[iu], surface[iu]
    hidden = (m / max(m.max(), 1e-12)) * (1 - s)          # strong through the map, weak in shared words
    order = np.argsort(-hidden, kind="stable")[:top]
    out = []
    for o in order:
        a, b = iu[0][o], iu[1][o]
        ta, tb = np.flatnonzero(X[a]), np.flatnonzero(X[b])
        sub = L[np.ix_(ta, tb)]
        carry = [(vocab[ta[i]], vocab[tb[j]], float(sub[i, j]))
                 for i, j in zip(*np.unravel_index(np.argsort(-sub, axis=None)[:5], sub.shape)) if sub[i, j] > 0]
        out.append({"a": chats[a]["id"], "a_title": chats[a]["title"], "a_acct": chats[a]["acct"],
                    "b": chats[b]["id"], "b_title": chats[b]["title"], "b_acct": chats[b]["acct"],
                    "map": float(m[o]), "words": float(s[o]), "hidden": float(hidden[o]),
                    "cross_account": chats[a]["acct"] != chats[b]["acct"], "carried_by": carry,
                    "kill_test": "Open both chats. Does the carried-by term pair mean the same object in both? If not, strike it."})
    return out


def sql_escape(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--human-only", action="store_true")
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--min-df", type=int, default=3, help="a term must appear in at least this many chats")
    ap.add_argument("--max-df", type=float, default=0.3, help="leave out terms found in more than this share of chats")
    ap.add_argument("--min-msgs", type=int, default=6, help="rank links only between chats with at least this many messages")
    ap.add_argument("--alpha", type=float, default=0.5, help="context memory inside a chat (0 = last message only)")
    ap.add_argument("--single", action="store_true", help="forward organism only (no retrocausal partner)")
    ap.add_argument("--since", default=None, help="learn only from chats updated on/after this date, e.g. 2026-08-01")
    ap.add_argument("--exclude", default="Cross-account chat synthesis",
                    help="comma-separated title fragments to leave out entirely (default: the chat that built this)")
    ap.add_argument("--focus", default="localhost=10,phiinverse=10,merge=5",
                    help="rank links only among the last N chats per account; '' = all chats")
    ap.add_argument("--out", default="organism_out")
    a = ap.parse_args()

    print("reading exports")
    chats = build_chats(a.exports, a.human_only, a.since, [x.strip() for x in a.exclude.split(",") if x.strip()])
    focus = None
    if a.focus:
        want = dict((k.strip(), int(v)) for k, v in (p.split("=") for p in a.focus.split(",")))
        focus = set()
        print("focus chats:")
        for acct_key, n in want.items():
            mine = sorted([i for i, c in enumerate(chats) if acct_key in c["acct"]],
                          key=lambda i: chats[i]["updated"], reverse=True)[:n]
            focus |= set(mine)
            for i in mine:
                c = chats[i]
                print(f"  {c['acct']:<12} {c['updated'][:10]}  {len(c['msgs']):>5} msgs  {c['n_artifacts']:>3} files  {c['title'][:55]}")
    df = {}
    for c in chats:
        for w in set().union(*(tokens(t) for _, _, t in c["msgs"])):
            df[w] = df.get(w, 0) + 1
    cap = a.max_df * len(chats)
    common = sorted(w for w, n in df.items() if n > cap)
    vocab = [w for w, n in sorted(df.items(), key=lambda kv: (-kv[1], kv[0])) if a.min_df <= n <= cap][:a.vocab]
    print(f"left out {len(common)} terms found in over {a.max_df:.0%} of chats, e.g. {', '.join(common[:12])}")
    print(f"{len(chats)} chats, {sum(len(c['msgs']) for c in chats)} messages, vocabulary {len(vocab)}")

    org, n, b0, b1 = live(chats, vocab, a.alpha)
    k = min(2000, max(1, n // 4))
    print(f"forward organism: {n} predictions; surprise first {k} = {b0:.2f} bits, last {k} = {b1:.2f} bits")
    org_b, c0, c1 = None, None, None
    if not a.single:
        org_b, nb, c0, c1 = live(chats, vocab, a.alpha, reverse=True)
        k = min(2000, max(1, nb // 4))
        print(f"retrocausal organism: {nb} predictions; surprise first {k} = {c0:.2f} bits, last {k} = {c1:.2f} bits")

    os.makedirs(a.out, exist_ok=True)
    links = chat_links(chats, vocab, link_matrix(org.W, org_b.W if org_b else None), focus=focus, min_msgs=a.min_msgs)
    json.dump(links, open(f"{a.out}/links.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{a.out}/links.md", "w") as f:
        f.write(f"# Hidden links ({len(links)})\nStrong through the map, weak in shared words. Each is a candidate: strike what's wrong.\n\n")
        for i, l in enumerate(links, 1):
            tag = "cross-account" if l["cross_account"] else "same account"
            f.write(f"{i}. **{l['a_title']}** ({l['a_acct']}) ↔ **{l['b_title']}** ({l['b_acct']}) · {tag}\n"
                    f"   map {l['map']:.4f} · shared words {l['words']:.2f} · carried by "
                    + ", ".join(f"{u}→{v}" for u, v, _ in l["carried_by"]) + "\n\n")
    D = org.W.copy(); np.fill_diagonal(D, 0)
    flat = np.argsort(-D, axis=None)[:200]
    with open(f"{a.out}/map_top.txt", "w") as f:
        f.write(f"self-persistence (diagonal) mean {np.diag(org.W).mean():.2f}; off-diagonal mean {D.mean():.4f}\n")
        for k in flat:
            i, j = np.unravel_index(k, D.shape); f.write(f"{vocab[i]} -> {vocab[j]}  {D[i, j]:.3f}\n")

    run = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(f"{a.out}/organism.sql", "w") as f:
        f.write("CREATE TABLE IF NOT EXISTS organism_runs (run_id TEXT PRIMARY KEY, created_at TEXT, chats INTEGER, "
                "messages INTEGER, vocab INTEGER, bits_first REAL, bits_last REAL, params TEXT);\n")
        f.write("CREATE TABLE IF NOT EXISTS organism_links (run_id TEXT, rank INTEGER, conv_a TEXT, conv_b TEXT, "
                "acct_a TEXT, acct_b TEXT, map REAL, words REAL, hidden REAL, carried_by TEXT, "
                "status TEXT DEFAULT 'unstruck', PRIMARY KEY (run_id, rank));\n")
        f.write(f"INSERT INTO organism_runs VALUES ({sql_escape(run)}, {sql_escape(run)}, {len(chats)}, {sum(len(c['msgs']) for c in chats)}, {len(vocab)}, "
                f"{b0:.4f}, {b1:.4f}, {sql_escape(json.dumps({**vars(a), 'retro_bits_first': c0, 'retro_bits_last': c1}))});\n")
        for r, l in enumerate(links, 1):
            f.write(f"INSERT INTO organism_links (run_id, rank, conv_a, conv_b, acct_a, acct_b, map, words, hidden, carried_by) "
                    f"VALUES ({sql_escape(run)}, {r}, {sql_escape(l['a'])}, {sql_escape(l['b'])}, {sql_escape(l['a_acct'])}, "
                    f"{sql_escape(l['b_acct'])}, {l['map']:.6f}, {l['words']:.6f}, {l['hidden']:.6f}, "
                    f"{sql_escape(json.dumps(l['carried_by'], ensure_ascii=False))});\n")
    print(f"wrote {a.out}/links.md, links.json, map_top.txt, organism.sql")


if __name__ == "__main__":
    main()
