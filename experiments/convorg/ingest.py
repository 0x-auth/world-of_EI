#!/usr/bin/env python3
"""ingest: turn a conversation export (JSON, see FORMAT.md) into one timeline TSV, one line per message.
   Nothing is thrown away as junk: nulls, deletions, media, files, edits and links stay in as tokens (⟨null⟩ ⟨deleted⟩ ⟨media⟩ ⟨file⟩ ⟨edited⟩ ⟨url⟩).
   Email and WhatsApp are one stream, sorted by time; src is only a tag.
   time<TAB>who<TAB>src<TAB>text      who: me | them    src: w=WhatsApp-like chat, g=email
Usage: python3 ingest.py conversation.json conv.tsv [--me-pattern "you@mail|YourName"] [--max 1500] [--since 2023-08-01]
See FORMAT.md for the input format."""
import json, re, sys, argparse, hashlib

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--max", type=int, default=1500, help="cap chars per message")
ap.add_argument("--since", default="", help="keep messages from this date on (YYYY-MM-DD)")
ap.add_argument("--me-pattern", default="", help="regex for YOUR name/email in the sender field (needed only if messages have no direction field)")
a = ap.parse_args()

ME = re.compile(a.me_pattern, re.I) if a.me_pattern else None
QUOTE_START = re.compile(
    r"^(On .{3,200}wrote:|On .{3,200}\n.{0,200}wrote:|-{2,}\s*Original Message|-{5,} ?Forwarded message|From: .+<.+@.+>|Sent from my )",
    re.I | re.M)

def who(m):
    d = (m.get("direction") or "").lower()
    if d in ("sent", "out", "outgoing", "me"): return "me"
    if d in ("received", "in", "incoming", "them"): return "them"
    s = str(m.get("sender") or m.get("from") or m.get("who") or "")
    if ME is None: sys.exit("no direction field: pass --me-pattern matching your name/email in the sender field")
    return "me" if ME.search(s) else "them"

MARKS = [(re.compile(r"this message was deleted|you deleted this message", re.I), "⟨deleted⟩"),
         (re.compile(r"<?\s*(media|image|video|audio|sticker|gif|document) omitted\s*>?", re.I), "⟨media⟩"),
         (re.compile(r"<attached:[^>]*>|\S*\.(jpe?g|png|opus|mp4|pdf|webp)\b", re.I), "⟨file⟩"),
         (re.compile(r"\bnull\b", re.I), "⟨null⟩"),
         (re.compile(r"<this message was edited>", re.I), "⟨edited⟩")]

def clean(t, src):
    if t is None or str(t).strip() in ("", "null", "None"): return "⟨null⟩"
    t = str(t).replace("\r\n", "\n").replace("\r", "\n")
    for rx, tag in MARKS: t = rx.sub(tag, t)
    if src == "g":
        q = QUOTE_START.search(t)
        if q: t = t[:q.start()]
        t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith(">"))
        t = re.split(r"\n-- ?\n", t)[0]          # signature
    t = re.sub(r"https?://\S+", "⟨url⟩", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:a.max] + ("…" if len(t) > a.max else "")

data = json.load(open(a.inp, encoding="utf-8"))
if isinstance(data, dict):  # tolerate {"messages":[...]}
    data = next(v for v in data.values() if isinstance(v, list))

rows, seen, stats = [], set(), {}
for m in data:
    ts = str(m.get("ts") or m.get("time") or m.get("date") or "")[:16].replace("T", " ")
    if a.since and ts < a.since: continue
    src = "w" if (m.get("source") or "").lower().startswith("wh") else "g"
    w = who(m)
    subj = (m.get("subject") or "").strip()
    fwd = src == "g" and re.match(r"(fwd?|fw):", subj, re.I)
    txt = "[fwd] " + subj if fwd else clean(m.get("text", m.get("body", m.get("message"))), src)
    if src == "g" and subj and not re.match(r"(re|fwd?|fw):", subj, re.I):
        txt = f"({subj}) {txt}"
    if not txt: continue
    key = m.get("message_id") or hashlib.md5(f"{ts}{w}{txt[:120]}".encode()).hexdigest()
    k2 = (ts, w, txt[:120])
    if key in seen or k2 in seen: continue
    seen.add(key); seen.add(k2)
    rows.append((ts, w, src, txt.replace("\t", " ")))
    stats[(src, w)] = stats.get((src, w), 0) + 1

rows.sort()
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows: f.write("\t".join(r) + "\n")

import os
bi, bo = os.path.getsize(a.inp), os.path.getsize(a.out)
print(f"in  {len(data):>7} msgs  {bi/1e6:.1f} MB")
print(f"out {len(rows):>7} msgs  {bo/1e6:.1f} MB  (~{bo//4:,} tokens)")
for (s, w), n in sorted(stats.items()):
    print(f"   {'chat' if s=='w' else 'email':6} {w:5} {n}")
if rows: print(f"range {rows[0][0]} .. {rows[-1][0]}")
